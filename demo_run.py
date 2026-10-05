"""Run every email in samples/ through the classifier and print the results.

Usage: python demo_run.py [--no-cache]
"""

import argparse
import sys
import textwrap
from pathlib import Path

from src.classifier import active_model, analyze_email, is_cached
from src.email_source import SampleSource
from src.models import EmailAnalysis

SAMPLES_DIR = Path(__file__).parent / "samples"
WIDTH = 78

URGENCY_MARK = {"low": "[ low ]", "medium": "[ MED ]", "high": "[ HIGH ]"}


def wrap(text: str, indent: str = "  ") -> str:
    """Wrap text to the report width, keeping the line breaks the model wrote.

    Drafts often contain bulleted lists, so each line is wrapped on its own
    rather than reflowed into one paragraph.
    """
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue
        lines.append(
            textwrap.fill(
                line, width=WIDTH, initial_indent=indent, subsequent_indent=indent
            )
        )
    return "\n".join(lines)


def print_analysis(result: EmailAnalysis) -> None:
    """Print one analysis as a readable block."""
    mark = URGENCY_MARK.get(result.urgency.value, result.urgency.value)
    print(f"  CATEGORY : {result.category.value}")
    print(f"  URGENCY  : {result.urgency.value} {mark}")
    print()
    print("  SUMMARY")
    print(wrap(result.summary, "    "))
    print()

    print("  CUSTOMER")
    fields = result.customer.model_dump()
    if any(fields.values()):
        for key, value in fields.items():
            print(f"    {key:<8}: {value if value else '-'}")
    else:
        print("    (none found)")
    print()

    print("  MISSING INFO")
    if result.missing_info:
        for item in result.missing_info:
            print(textwrap.fill(
                item, width=WIDTH, initial_indent="    - ", subsequent_indent="      "
            ))
    else:
        print("    (nothing missing)")
    print()

    print("  DRAFT REPLY")
    if result.draft_reply.strip():
        print(wrap(result.draft_reply, "    "))
    else:
        print("    (no reply drafted)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Call the API for every email instead of reusing .cache/.",
    )
    args = parser.parse_args()
    use_cache = not args.no_cache

    samples = SampleSource(SAMPLES_DIR).fetch()
    if not samples:
        print(f"No samples found in {SAMPLES_DIR}")
        return 1

    print(f"Model: {active_model()}{'' if use_cache else '  (cache disabled)'}\n")

    failures = 0
    hits = 0
    called = 0
    for index, email in enumerate(samples, start=1):
        text = email.prompt_text
        cached = use_cache and is_cached(text)

        print("=" * WIDTH)
        print(f"[{index}/{len(samples)}] {email.origin}{'  (cached)' if cached else ''}")
        print("=" * WIDTH)

        if not cached:
            called += 1
        try:
            print_analysis(analyze_email(text, use_cache=use_cache))
            if cached:
                hits += 1
        except Exception as exc:  # keep going so one bad call doesn't end the run
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}\n")

    print("=" * WIDTH)
    print(
        f"Done: {len(samples) - failures}/{len(samples)} analyzed, "
        f"{failures} failed, {hits} from cache."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
