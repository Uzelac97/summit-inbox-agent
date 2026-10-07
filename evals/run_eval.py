"""Score the classifier against the cases in evals/cases.json.

Usage:
    python evals/run_eval.py               # analyze every case, reusing .cache/
    python evals/run_eval.py --all         # also list the cases that passed
    python evals/run_eval.py --cached-only # score only cached cases, no API calls

The first full run costs one API request per uncached case. Later runs cost
nothing, because results are cached by prompt, model, and email text.

Exit code is 1 if any case could not be analyzed at all, 0 otherwise. A low
score is a finding, not a crash.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `src` imports work when run as a script

from src.classifier import active_model, analyze_email, is_cached  # noqa: E402
from src.quote_rules import is_needs_more_draft, is_template_draft  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "cases.json"

# Phrases that promise a time. "as soon as possible" and the company's own
# "within 24 hours" are fine; naming a day or part of a day is not. German is
# included because German drafts are expected, with "guten Morgen" excluded
# since that is a greeting rather than a promise.
TIME_PROMISE_RE = re.compile(
    r"\b(today|tomorrow|tonight|this (?:morning|afternoon|evening)"
    r"|heute|morgen Abend)\b",
    re.IGNORECASE,
)
_GUTEN_MORGEN_RE = re.compile(r"guten morgen", re.IGNORECASE)


def time_promises(draft: str) -> list[str]:
    """Phrases in a draft that promise a specific time."""
    cleaned = _GUTEN_MORGEN_RE.sub("", draft)
    # Bare "morgen" means tomorrow, but "Morgen" also ends "guten Morgen",
    # which the line above has already removed.
    found = set(m.group(0).lower() for m in TIME_PROMISE_RE.finditer(cleaned))
    for m in re.finditer(r"\bmorgen\b", cleaned, re.IGNORECASE):
        found.add(m.group(0).lower())
    return sorted(found)
WIDTH = 86


@dataclass
class Outcome:
    """What one case expected, what it got, and whether it ran at all."""

    case_id: str
    tricky: str | None
    note: str
    expected_category: str
    expected_urgency: str
    got_category: str | None = None
    got_urgency: str | None = None
    error: str | None = None
    # Only set for cases that declare expected_customer_name.
    expected_name: str | None = None
    got_name: str | None = None
    greeting: str = ""
    draft: str = ""
    missing_count: int = 0
    expect_no_questions: bool = False
    expect_template: bool = False
    expect_needs_more: bool = False

    @property
    def ran(self) -> bool:
        return self.error is None

    @property
    def category_ok(self) -> bool:
        return self.ran and self.got_category == self.expected_category

    @property
    def urgency_ok(self) -> bool:
        return self.ran and self.got_urgency == self.expected_urgency

    @property
    def checks_name(self) -> bool:
        return self.expected_name is not None

    @property
    def time_promises(self) -> list[str]:
        return time_promises(self.draft) if self.ran else []

    @property
    def questions_ok(self) -> bool:
        """For a request with nothing missing, the draft must ask nothing."""
        if not self.expect_no_questions:
            return True
        return "?" not in self.draft and self.missing_count == 0

    @property
    def template_ok(self) -> bool:
        """A complete quote gets the fixed acknowledgement, and nothing else.

        The draft must be the template, not the model's text, and must ask
        nothing. Only cases that declare expect_template are checked.
        """
        if not self.expect_template:
            return True
        return (
            self.ran
            and is_template_draft(self.draft)
            and "?" not in self.draft
            and self.missing_count == 0
        )

    @property
    def needs_more_ok(self) -> bool:
        """An incomplete quote request gets the "needs more" template, and
        never promises that the quote will be sent or that we have everything.

        Only cases that declare expect_needs_more are checked.
        """
        if not self.expect_needs_more:
            return True
        return (
            self.ran
            and is_needs_more_draft(self.draft)
            and "24 hours" not in self.draft
        )

    @property
    def signature_ok(self) -> bool:
        """The sign-off must be in the language the draft is written in."""
        if not self.ran or not self.draft.strip():
            return True
        german = "mit freundlichen gr" in self.draft.lower()
        if german:
            return "ihr summit roofing team" in self.draft.lower()
        return "the summit roofing team" in self.draft.lower()

    @property
    def name_ok(self) -> bool:
        return self.ran and self.got_name == self.expected_name

    @property
    def greeting_ok(self) -> bool:
        """True when the draft greets the name that was recorded.

        A record naming one person and a reply greeting another is wrong even
        when the recorded name itself is right, so this is checked separately.
        """
        if not self.ran or not self.got_name:
            return False
        # Any part of the name counts. A formal German reply opens "Guten Tag
        # Herr Brandt," - correct, and it addresses the surname, so requiring
        # the first name would fail a draft that is right.
        greeting = self.greeting.lower()
        return any(part.lower() in greeting for part in self.got_name.split())


def score(outcomes: list[Outcome]) -> dict:
    """Summarize outcomes. Pure function: no I/O, so it is testable on its own."""
    ran = [o for o in outcomes if o.ran]
    return {
        "total": len(outcomes),
        "ran": len(ran),
        "errored": len(outcomes) - len(ran),
        "category_ok": sum(o.category_ok for o in ran),
        "urgency_ok": sum(o.urgency_ok for o in ran),
        "both_ok": sum(o.category_ok and o.urgency_ok for o in ran),
        "mismatches": [o for o in ran if not (o.category_ok and o.urgency_ok)],
        "name_checked": [o for o in ran if o.checks_name],
        "name_bad": [o for o in ran if o.checks_name
                     and not (o.name_ok and o.greeting_ok)],
        "time_bad": [o for o in ran if o.time_promises],
        "questions_bad": [o for o in ran if not o.questions_ok],
        "template_bad": [o for o in ran if not o.template_ok],
        "needs_more_bad": [o for o in ran if not o.needs_more_ok],
        "signature_bad": [o for o in ran if not o.signature_ok],
    }


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "    n/a"


def print_per_category_table(outcomes: list[Outcome]) -> None:
    """Break accuracy down by the category a case was labelled with.

    An overall figure hides which category is actually weak, and with only a
    handful of cases per category one miss moves a column a long way - so the
    case count is printed alongside the rate.
    """
    ran = [o for o in outcomes if o.ran]
    by_category: dict[str, list[Outcome]] = {}
    for outcome in ran:
        by_category.setdefault(outcome.expected_category, []).append(outcome)

    print("ACCURACY BY EXPECTED CATEGORY")
    print(f"  {'category':<15} {'n':>2}   {'category':<14} {'urgency':<14}")
    print(f"  {'-' * 13:<15} {'--':>2}   {'-' * 12:<14} {'-' * 12:<14}")
    for name in sorted(by_category):
        group = by_category[name]
        cat_ok = sum(o.category_ok for o in group)
        urg_ok = sum(o.urgency_ok for o in group)
        print(
            f"  {name:<15} {len(group):>2}   "
            f"{cat_ok}/{len(group)} {percent(cat_ok, len(group))}   "
            f"{urg_ok}/{len(group)} {percent(urg_ok, len(group))}"
        )
    print()


def print_mismatch_table(mismatches: list[Outcome]) -> None:
    print("MISMATCHES")
    if not mismatches:
        print("  none\n")
        return

    rows = []
    for outcome in mismatches:
        if not outcome.category_ok:
            rows.append((outcome.case_id, "category",
                         outcome.expected_category, outcome.got_category))
        if not outcome.urgency_ok:
            rows.append((outcome.case_id, "urgency",
                         outcome.expected_urgency, outcome.got_urgency))

    print(f"  {'case':<26} {'field':<9} {'expected':<15} {'got':<15}")
    print(f"  {'-' * 24:<26} {'-' * 7:<9} {'-' * 13:<15} {'-' * 13:<15}")
    for case_id, fieldname, expected, got in rows:
        print(f"  {case_id:<26} {fieldname:<9} {expected:<15} {got:<15}")
    print()

    print("  Why each case expected what it did:")
    for outcome in mismatches:
        tag = f" [{outcome.tricky}]" if outcome.tricky else ""
        print(f"\n  {outcome.case_id}{tag}")
        print(textwrap.fill(outcome.note, width=WIDTH,
                            initial_indent="    ", subsequent_indent="    "))
    print()


def print_pass_table(outcomes: list[Outcome]) -> None:
    print("ALL CASES")
    print(f"  {'case':<26} {'category':<15} {'urgency':<9} {'result':<8}")
    print(f"  {'-' * 24:<26} {'-' * 13:<15} {'-' * 7:<9} {'-' * 6:<8}")
    for o in outcomes:
        if not o.ran:
            print(f"  {o.case_id:<26} {'-':<15} {'-':<9} {'ERROR':<8}")
            continue
        wrong = [name for name, ok in (("CAT", o.category_ok), ("URG", o.urgency_ok)) if not ok]
        marks = " ".join(wrong) if wrong else "ok"
        print(f"  {o.case_id:<26} {o.got_category:<15} {o.got_urgency:<9} {marks:<8}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all", action="store_true",
                        help="List every case, not just the mismatches.")
    parser.add_argument("--cached-only", action="store_true",
                        help="Skip cases with no cached result, making no API calls.")
    args = parser.parse_args()

    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    cases = data["cases"]
    model = active_model()

    print(f"Model : {model}")
    print(f"Cases : {len(cases)} from {CASES_FILE.relative_to(ROOT)}")
    if args.cached_only:
        print("Mode  : cached-only, no API calls will be made")
    print()

    outcomes: list[Outcome] = []
    skipped = 0
    called = 0

    for index, case in enumerate(cases, start=1):
        text = case["email"]
        cached = is_cached(text)
        outcome = Outcome(
            case_id=case["id"],
            tricky=case.get("tricky"),
            note=case.get("note", ""),
            expected_category=case["expected_category"],
            expected_urgency=case["expected_urgency"],
        )

        if args.cached_only and not cached:
            skipped += 1
            print(f"  [{index:>2}/{len(cases)}] {case['id']:<26} skipped, not cached")
            continue

        if not cached:
            called += 1

        try:
            analysis = analyze_email(text)
            outcome.got_category = analysis.category.value
            outcome.got_urgency = analysis.urgency.value
            outcome.expected_name = case.get("expected_customer_name")
            outcome.got_name = analysis.customer.name
            outcome.greeting = next(
                (l for l in analysis.draft_reply.splitlines() if l.strip()), ""
            )
            outcome.draft = analysis.draft_reply
            outcome.missing_count = len(analysis.missing_info)
            outcome.expect_no_questions = bool(case.get("expect_no_questions"))
            outcome.expect_template = bool(case.get("expect_template"))
            outcome.expect_needs_more = bool(case.get("expect_needs_more"))
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"

        source = "cached" if cached else "api"
        verdict = outcome.error or (
            "ok" if outcome.category_ok and outcome.urgency_ok else "mismatch"
        )
        print(f"  [{index:>2}/{len(cases)}] {case['id']:<26} {source:<7} {verdict}")
        outcomes.append(outcome)

    print()
    print("=" * WIDTH)
    if not outcomes:
        print("Nothing scored: no case had a cached result.")
        print("Run without --cached-only to analyze them.")
        return 0

    summary = score(outcomes)
    if args.all:
        print_pass_table(outcomes)
    print_per_category_table(outcomes)
    print_mismatch_table(summary["mismatches"])

    name_checked = summary["name_checked"]
    if name_checked:
        print("CUSTOMER NAME (cases that declare an expected name)")
        bad = summary["name_bad"]
        if not bad:
            print(
                f"  {len(name_checked)}/{len(name_checked)} correct, and every "
                "draft greets the recorded name\n"
            )
        else:
            print(f"  {'case':<28} {'expected':<18} {'recorded':<18} greeting")
            for o in bad:
                greet = (o.greeting[:28] + "...") if len(o.greeting) > 28 else o.greeting
                print(f"  {o.case_id:<28} {str(o.expected_name):<18} "
                      f"{str(o.got_name):<18} {greet}")
            print()

    print("DRAFT RULES")
    ran = [o for o in outcomes if o.ran]
    for title, bad, detail in (
        ("no time promises", summary["time_bad"],
         lambda o: ", ".join(o.time_promises)),
        ("no questions when nothing is missing",
         summary["questions_bad"],
         lambda o: f"{o.missing_count} missing, "
                   f"{'has' if '?' in o.draft else 'no'} question mark"),
        ("sign-off matches the draft language", summary["signature_bad"],
         lambda o: (o.draft.strip().splitlines() or [""])[-1][:40]),
        ("complete quote gets the fixed acknowledgement", summary["template_bad"],
         lambda o: o.draft.strip().splitlines()[0][:40] if o.draft.strip() else "empty"),
        ("incomplete quote gets the needs-more template", summary["needs_more_bad"],
         lambda o: o.draft.strip().splitlines()[0][:40] if o.draft.strip() else "empty"),
    ):
        if "no questions" in title:
            checked = len([o for o in ran if o.expect_no_questions])
        elif "complete quote" in title:
            checked = len([o for o in ran if o.expect_template])
        elif "incomplete quote" in title:
            checked = len([o for o in ran if o.expect_needs_more])
        else:
            checked = len(ran)
        if not bad:
            print(f"  {title:<38} {checked - len(bad)}/{checked} OK")
        else:
            print(f"  {title:<38} {checked - len(bad)}/{checked} - FAILURES:")
            for o in bad:
                print(f"      {o.case_id:<28} {detail(o)}")
    print()

    scored = summary["ran"]
    print("ACCURACY")
    print(f"  category   {summary['category_ok']:>2}/{scored}   {percent(summary['category_ok'], scored)}")
    print(f"  urgency    {summary['urgency_ok']:>2}/{scored}   {percent(summary['urgency_ok'], scored)}")
    print(f"  both       {summary['both_ok']:>2}/{scored}   {percent(summary['both_ok'], scored)}")

    if skipped:
        print(f"\n  {skipped} case(s) skipped as uncached.")
    if summary["errored"]:
        print(f"  {summary['errored']} case(s) could not be analyzed.")

    caveat = data.get("urgency_rubric", {}).get("caveat")
    if caveat and summary["urgency_ok"] < scored:
        print()
        print(textwrap.fill(f"Note: {caveat}", width=WIDTH,
                            initial_indent="  ", subsequent_indent="  "))

    return 1 if summary["errored"] else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
