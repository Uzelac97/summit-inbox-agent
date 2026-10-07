"""Run the inbox agent over the mail it has not handled yet.

Usage:
    python run_agent.py                   # one pass, up to 25 emails
    python run_agent.py --max-emails 50   # one pass, up to 50 emails
    python run_agent.py --loop            # repeat every POLL_INTERVAL_MINUTES

POLL_INTERVAL_MINUTES is read from .env and defaults to 5. Ctrl+C stops a loop
between passes, or during one, without losing what has already been labelled.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.email_source import GmailSource  # noqa: E402
from src.forward import NOT_QUALIFYING, QuoteForwarder, webhook_url_from_env  # noqa: E402
from src.gmail_client import GmailClient  # noqa: E402
from src.labels import SYSTEM_LABELS, UNPROCESSED_QUERY  # noqa: E402
from src.processor import Processor, Result  # noqa: E402

DEFAULT_MAX_EMAILS = 25
DEFAULT_POLL_MINUTES = 5.0
WIDTH = 78


def poll_interval_minutes() -> float:
    """POLL_INTERVAL_MINUTES from the environment, checked before any work starts."""
    raw = os.getenv("POLL_INTERVAL_MINUTES", "").strip()
    if not raw:
        return DEFAULT_POLL_MINUTES
    try:
        value = float(raw)
    except ValueError:
        sys.exit(f"POLL_INTERVAL_MINUTES must be a number, not {raw!r}.")
    if value <= 0:
        sys.exit("POLL_INTERVAL_MINUTES must be greater than zero.")
    return value


def build_forwarder(client: GmailClient) -> QuoteForwarder | None:
    """A forwarder when QUOTE_WEBHOOK_URL is set, otherwise None."""
    url = webhook_url_from_env(os.environ)
    if not url:
        return None
    secret = (os.environ.get("QUOTE_WEBHOOK_SECRET") or "").strip()
    return QuoteForwarder(client, url, secret)


def actions_taken(result: Result) -> list[str]:
    """Plain-English list of what happened to one email."""
    labels = [name for name in result.labels_applied if name not in SYSTEM_LABELS]
    actions = []
    if labels:
        actions.append("labelled " + ", ".join(labels))
    if SYSTEM_LABELS & set(result.labels_applied):
        actions.append("starred, marked important")
    if result.draft_id:
        actions.append("draft saved")
    elif result.draft_skipped:
        actions.append(f"no draft ({result.draft_skipped})")
    if result.forward and result.forward.status != NOT_QUALIFYING:
        detail = f" ({result.forward.detail})" if result.forward.detail else ""
        actions.append(f"quote forward {result.forward.status}{detail}")
    return actions


def print_result(index: int, total: int, result: Result) -> None:
    """One block per email: who, what, and what the agent did about it."""
    email = result.email
    subject = email.subject or "(no subject)"
    print(f"[{index}/{total}] {email.sender or '(unknown sender)'}")
    print(f"    subject : {subject}")
    if not result.ok:
        print(f"    ERROR   : {result.error}")
        print("    actions : none completed; retried next run")
        return
    analysis = result.analysis
    print(f"    category: {analysis.category.value}")
    print(f"    urgency : {analysis.urgency.value}")
    print(f"    actions : {'; '.join(actions_taken(result)) or 'none'}")


def print_retry_result(result: Result) -> None:
    """One line per retried forward: who it was for, and what happened."""
    email = result.email
    subject = email.subject or "(no subject)"
    if result.error:
        print(f"    {email.sender or '(unknown sender)':<40} {subject:<30} ERROR: {result.error}")
        return
    forward = result.forward
    status = forward.status if forward else "skipped (attempts exhausted)"
    detail = f" ({forward.detail})" if forward and forward.detail else ""
    print(f"    {email.sender or '(unknown sender)':<40} {subject:<30} {status}{detail}")


def run_once(client: GmailClient, max_emails: int, forwarder=None) -> tuple[int, int]:
    """One pass over unprocessed mail. Returns (handled, failed).

    A failure to read the mailbox ends this pass but not the loop, so one bad
    poll does not stop the agent. A failure on one email is recorded and the
    pass moves on to the next.
    """
    print(f"--- pass at {datetime.now():%Y-%m-%d %H:%M:%S} ---")

    processor = Processor(client, forwarder=forwarder)
    try:
        processor.prepare()
    except Exception as exc:
        print(f"Could not prepare labels: {type(exc).__name__}: {exc}\n")
        return 0, 1

    # Retrying previously failed forwards happens before any new mail is
    # fetched, so a webhook that is back up gets a chance before this pass
    # takes on more work.
    try:
        retried = processor.retry_forward_failures()
    except Exception as exc:
        print(f"Could not retry failed forwards: {type(exc).__name__}: {exc}\n")
        retried = []
    if retried:
        print(f"Retrying {len(retried)} previously failed forward(s):")
        for result in retried:
            print_retry_result(result)
        print()

    try:
        source = GmailSource(client=client, query=UNPROCESSED_QUERY)
        emails = source.fetch(limit=max_emails)
    except Exception as exc:
        print(f"Could not read the mailbox: {type(exc).__name__}: {exc}\n")
        return 0, 1

    if not emails:
        print("Nothing new to process.\n")
        return 0, 0

    failed = 0
    for index, email in enumerate(emails, start=1):
        result = processor.process(email)
        print_result(index, len(emails), result)
        if not result.ok:
            failed += 1
        print()

    handled = len(emails) - failed
    print(f"Pass done: {handled} handled, {failed} failed.\n")
    return handled, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--loop", action="store_true",
        help="Keep running, one pass every POLL_INTERVAL_MINUTES, until Ctrl+C.",
    )
    parser.add_argument(
        "--max-emails", type=int, default=DEFAULT_MAX_EMAILS,
        help=f"Most emails to handle in one pass (default {DEFAULT_MAX_EMAILS}).",
    )
    args = parser.parse_args()
    if args.max_emails < 1:
        parser.error("--max-emails must be at least 1")

    # Logging stays quiet unless something needs a person's attention. The
    # per-email lines below are the report.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Add it to .env, then run again.")
        return 1
    interval = poll_interval_minutes()

    try:
        client = GmailClient()
        print(f"Mailbox : {client.address()}")
    except Exception as exc:
        print(f"Could not connect to Gmail: {type(exc).__name__}: {exc}")
        return 1

    forwarder = build_forwarder(client)
    if forwarder is None:
        print("Forward : disabled (QUOTE_WEBHOOK_URL empty)")
    elif not forwarder.secret:
        print("Forward : REFUSED - QUOTE_WEBHOOK_URL is set but QUOTE_WEBHOOK_SECRET is empty")
    else:
        print(f"Forward : quote requests to {forwarder.url}")

    if not args.loop:
        _, failed = run_once(client, args.max_emails, forwarder)
        return 1 if failed else 0

    print(f"Looping every {interval:g} minute(s). Ctrl+C to stop.\n")
    try:
        while True:
            run_once(client, args.max_emails, forwarder)
            time.sleep(interval * 60)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
