"""The per-email pipeline: analyze, then act on the result.

Pure orchestration, so it can be driven with a fake client in a test. Drafting
arrives with the step that needs it; for now acting means labelling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.classifier import analyze_email
from src.gmail_client import build_reply_mime
from src.labels import ALL_LABELS, EMERGENCY_MARKS, PROCESSED_LABEL, label_for
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
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Processor:
    """Runs the pipeline over emails, one at a time, never stopping on one."""

    def __init__(
        self, client, dry_run: bool = False, notifier: Notifier | None = None
    ) -> None:
        self.client = client
        self.dry_run = dry_run
        self.notifier = notifier or NoOpNotifier()

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

    def process(self, email: InboundEmail) -> Result:
        """Analyze one email, label it, and draft a reply.

        ``agent/processed`` is applied last, after the draft exists, so an
        email interrupted part way through is picked up again on the next run.
        A duplicate label is harmless and a duplicate draft is recoverable; a
        silently skipped emergency is not.
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
