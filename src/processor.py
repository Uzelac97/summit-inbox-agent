"""The per-email pipeline: analyze, then act on the result.

Pure orchestration, so it can be driven with a fake client in a test. Drafting
arrives with the step that needs it; for now acting means labelling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.classifier import analyze_email
from src.email_source import GmailSource
from src.gmail_client import build_reply_mime
from src.forward import (
    MAX_FORWARD_ATTEMPTS,
    RETRYABLE_STATUSES,
    Forwarded,
    QuoteForwarder,
    clear_forward_attempts,
    forward_attempts,
    record_forward_attempt,
)
from src.labels import (
    ALL_LABELS,
    EMERGENCY_MARKS,
    FORWARD_FAILED_LABEL,
    PROCESSED_LABEL,
    RETRY_QUERY,
    label_for,
)
from src.models import Category, EmailAnalysis, InboundEmail
from src.notify import NoOpNotifier, Notifier

log = logging.getLogger(__name__)


@dataclass
class Result:
    """What happened to one email."""

    email: InboundEmail
    analysis: EmailAnalysis | None = None
    labels_applied: tuple[str, ...] = ()
    draft_id: str | None = None
    draft_skipped: str = ""
    forward: Forwarded | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Processor:
    """Runs the pipeline over emails, one at a time, never stopping on one."""

    def __init__(
        self,
        client,
        dry_run: bool = False,
        notifier: Notifier | None = None,
        forwarder: QuoteForwarder | None = None,
    ) -> None:
        self.client = client
        self.dry_run = dry_run
        self.notifier = notifier or NoOpNotifier()
        self.forwarder = forwarder

    def prepare(self) -> None:
        """Create the agent's labels up front.

        Done once per run rather than per email, and before any query that
        filters on a label: a Gmail search naming a label that does not exist
        is not a useful way to find unprocessed mail.
        """
        if self.dry_run:
            log.info("dry run: would ensure labels %s", ", ".join(ALL_LABELS))
            return
        self.client.ensure_labels(ALL_LABELS)

    def _notify(self, result: Result) -> None:
        """Tell a person about an emergency. A failure here never fails the email."""
        try:
            self.notifier.emergency(result)
        except Exception:
            log.exception("notify failed: %s", result.email.source_id)

    def _forward(self, email: InboundEmail, analysis: EmailAnalysis) -> Forwarded:
        """Send a complete quote request on. A failure is recorded, not raised."""
        try:
            return self.forwarder.forward(email, analysis)
        except Exception as exc:
            log.exception("forward crashed: %s", email.source_id)
            return Forwarded("failed", f"{type(exc).__name__}: {exc}")

    def _record_failed_forward(self, email: InboundEmail, forward: Forwarded) -> int:
        """Count one more failed attempt and warn loudly once attempts run out."""
        attempts = record_forward_attempt(email.source_id)
        if attempts >= MAX_FORWARD_ATTEMPTS:
            log.warning(
                "forward failed %d time(s), giving up on %s (%s): %s",
                attempts, email.subject or email.source_id, email.source_id,
                forward.detail,
            )
        return attempts

    def process(self, email: InboundEmail) -> Result:
        """Analyze one email, label it, and draft a reply.

        ``agent/processed`` is applied last, after the draft exists and after
        any forward has either succeeded or did not need to happen, so an
        email interrupted part way through is picked up again on the next
        run. A duplicate label is harmless and a duplicate draft is
        recoverable; a silently skipped emergency is not.

        If a qualifying quote request's forward fails, ``agent/processed`` is
        withheld and ``agent/forward-failed`` is applied instead: the draft
        stays, but the email is left for :meth:`retry_forward_failures` to
        pick up on a later pass rather than being marked done.
        """
        result = Result(email=email)
        try:
            result.analysis = analyze_email(email.prompt_text)
            category_label = label_for(result.analysis.category)
            # Starred on the same write as the category label, so an emergency
            # is flagged even if a later step in this email fails.
            marks = (
                EMERGENCY_MARKS
                if result.analysis.category is Category.EMERGENCY
                else ()
            )
            draft_body = result.analysis.draft_reply.strip()

            if not draft_body:
                # Spam gets no reply by design, so there is nothing to draft.
                result.draft_skipped = "no reply drafted for this category"
            elif not email.thread_id:
                # Without a thread there is nothing to reply into; sample
                # emails have no mailbox behind them.
                result.draft_skipped = "no thread to reply into"

            if self.dry_run:
                log.info(
                    "dry run: would label %s with %s, %s, then %s",
                    email.source_id,
                    category_label,
                    result.draft_skipped or "draft a reply",
                    PROCESSED_LABEL,
                )
                result.labels_applied = (category_label, *marks, PROCESSED_LABEL)
                return result

            self.client.add_labels(email.source_id, [category_label, *marks])

            if not result.draft_skipped:
                result.draft_id = self.client.create_draft(
                    email.thread_id,
                    build_reply_mime(
                        to=email.sender,
                        subject=email.subject,
                        body=draft_body,
                        in_reply_to=email.rfc822_message_id,
                        references=email.references,
                    ),
                )

            if self.forwarder:
                result.forward = self._forward(email, result.analysis)

            if result.forward is not None and result.forward.status in RETRYABLE_STATUSES:
                self.client.add_labels(email.source_id, [FORWARD_FAILED_LABEL])
                result.labels_applied = (category_label, *marks, FORWARD_FAILED_LABEL)
                self._record_failed_forward(email, result.forward)
            else:
                # Only now is the email genuinely handled.
                self.client.add_labels(email.source_id, [PROCESSED_LABEL])
                result.labels_applied = (category_label, *marks, PROCESSED_LABEL)

            if marks:
                self._notify(result)

        except Exception as exc:
            # One unprocessable email must not end the run. Without
            # agent/processed it will simply be retried next time.
            result.error = f"{type(exc).__name__}: {exc}"
            log.exception("failed: %s", email.subject or email.source_id)

        return result

    def retry_forward_failures(self) -> list[Result]:
        """Retry the forward for every email still labelled agent/forward-failed.

        Only the forward is retried. Analysis and the draft already exist
        from when it first failed, so neither is redone here - re-analyzing
        would cost an API call for an unchanged email, and a second draft
        would duplicate the one already sitting in the thread. A message
        that has already used up its attempts is left alone and logged again,
        so the warning is not a one-off that scrolls out of view.
        """
        if self.forwarder is None or self.dry_run:
            return []

        results = []
        source = GmailSource(client=self.client, query=RETRY_QUERY, skip_own=False)
        for email in source.fetch(limit=100):
            results.append(self._retry_one_forward(email))
        return results

    def _retry_one_forward(self, email: InboundEmail) -> Result:
        result = Result(email=email)
        attempts = forward_attempts(email.source_id)
        if attempts >= MAX_FORWARD_ATTEMPTS:
            log.warning(
                "forward for %s has failed %d time(s) already; needs manual "
                "review, not retrying automatically",
                email.subject or email.source_id, attempts,
            )
            return result

        try:
            result.analysis = analyze_email(email.prompt_text)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            log.exception("retry: could not re-read analysis for %s", email.source_id)
            return result

        result.forward = self._forward(email, result.analysis)

        if result.forward.status in RETRYABLE_STATUSES:
            self._record_failed_forward(email, result.forward)
            result.labels_applied = (FORWARD_FAILED_LABEL,)
            return result

        # Forwarded, already forwarded, or no longer qualifying: in every
        # case the email is done and should stop being retried.
        try:
            self.client.remove_labels(email.source_id, [FORWARD_FAILED_LABEL])
            self.client.add_labels(email.source_id, [PROCESSED_LABEL])
        except Exception as exc:
            log.warning(
                "retry: forward resolved for %s but labels could not be "
                "updated: %s", email.source_id, exc,
            )
        clear_forward_attempts(email.source_id)
        result.labels_applied = (PROCESSED_LABEL,)
        return result
