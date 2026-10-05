"""The per-email pipeline: analyze, then act on the result.

Pure orchestration, so it can be driven with a fake client in a test. Drafting
arrives with the step that needs it; for now acting means labelling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.classifier import analyze_email
from src.labels import ALL_LABELS, PROCESSED_LABEL, label_for
from src.models import EmailAnalysis, InboundEmail

log = logging.getLogger(__name__)


@dataclass
class Result:
    """What happened to one email."""

    email: InboundEmail
    analysis: EmailAnalysis | None = None
    labels_applied: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Processor:
    """Runs the pipeline over emails, one at a time, never stopping on one."""

    def __init__(self, client, dry_run: bool = False) -> None:
        self.client = client
        self.dry_run = dry_run

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

    def process(self, email: InboundEmail) -> Result:
        """Analyze one email and label it.

        The category label goes on first and ``agent/processed`` last, so an
        email interrupted part way through is picked up again on the next run.
        A duplicate label is harmless; a silently skipped emergency is not.
        """
        result = Result(email=email)
        try:
            result.analysis = analyze_email(email.prompt_text)
            category_label = label_for(result.analysis.category)

            if self.dry_run:
                log.info(
                    "dry run: would label %s with %s then %s",
                    email.source_id, category_label, PROCESSED_LABEL,
                )
                result.labels_applied = (category_label, PROCESSED_LABEL)
                return result

            self.client.add_labels(email.source_id, [category_label])
            # Only now is the email genuinely handled.
            self.client.add_labels(email.source_id, [PROCESSED_LABEL])
            result.labels_applied = (category_label, PROCESSED_LABEL)

        except Exception as exc:
            # One unprocessable email must not end the run. Without
            # agent/processed it will simply be retried next time.
            result.error = f"{type(exc).__name__}: {exc}"
            log.exception("failed: %s", email.subject or email.source_id)

        return result
