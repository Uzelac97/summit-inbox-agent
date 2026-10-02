"""Score the classifier against the cases in evals/cases.json.

Usage:
    python evals/run_eval.py               # analyze every case, reusing .cache/
    python evals/run_eval.py --all         # also list the cases that passed
    python evals/run_eval.py --cached-only # score only cached cases, no API calls

The first full run costs one API request per uncached case. Later runs are free,
because results are cached by prompt, model, and email text.

Exit code is 1 if any case could not be analyzed at all, 0 otherwise. A low
score is a finding, not a crash.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `src` imports work when run as a script

from src.classifier import active_model, analyze_email, is_cached  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "cases.json"
PAUSE_SECONDS = 2
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

    @property
    def ran(self) -> bool:
        return self.error is None

    @property
    def category_ok(self) -> bool:
        return self.ran and self.got_category == self.expected_category

    @property
    def urgency_ok(self) -> bool:
        return self.ran and self.got_urgency == self.expected_urgency


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
    }


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "    n/a"


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
            if called:
                time.sleep(PAUSE_SECONDS)  # stay under the free tier's rate limit
            called += 1

        try:
            analysis = analyze_email(text)
            outcome.got_category = analysis.category.value
            outcome.got_urgency = analysis.urgency.value
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
    print_mismatch_table(summary["mismatches"])

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
