"""Where emails come from.

A source yields :class:`~src.models.InboundEmail` objects and knows nothing
about how they are analyzed. :class:`SampleSource` reads the bundled demo
files; a Gmail-backed source will implement the same protocol, which is what
lets DEMO_MODE switch between them without the rest of the app noticing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Protocol, runtime_checkable

from src.models import InboundEmail

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"

# Default ceiling on a single fetch. Keeps a first run over a real mailbox from
# analyzing hundreds of emails before anyone has looked at the output.
DEFAULT_LIMIT = 25

# A header line: a token, a colon, then the value. Deliberately strict, so a
# body opening with "Hi there:" or a German "Guten Tag:" is not mistaken for
# one.
_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z-]{1,30}):\s*(.*)$")


@runtime_checkable
class EmailSource(Protocol):
    """Anything that can hand us emails to triage."""

    def fetch(self, limit: int = DEFAULT_LIMIT) -> List[InboundEmail]:
        """Return up to ``limit`` emails that still need processing."""
        ...


def split_headers(text: str) -> tuple[dict[str, str], str]:
    """Split a raw email into its leading headers and its body.

    Only the run of header lines at the very top is consumed, stopping at the
    first blank or non-header line. Sample files are written this way, and so
    is anything we reconstruct from a provider.
    """
    headers: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    for index, line in enumerate(lines):
        if not line.strip():
            index += 1  # skip the blank separator
            break
        match = _HEADER_RE.match(line)
        if not match:
            break
        headers[match.group(1).lower()] = match.group(2).strip()
    else:
        index = len(lines)

    return headers, "\n".join(lines[index:]).strip()


class GmailSource:
    """The real mailbox, read through :class:`~src.gmail_client.GmailClient`.

    Fetching is read-only: messages are downloaded and parsed, and nothing is
    labelled, drafted, or sent. The caller decides what to do with the result.
    """

    def __init__(
        self,
        client: object | None = None,
        query: str = "in:inbox",
        skip_own: bool = True,
    ) -> None:
        # Imported here so demo mode never pays for the Google libraries, and
        # so a broken OAuth setup cannot stop the sample path from working.
        from src.gmail_client import GmailClient

        self.client = client or GmailClient()
        self.query = query
        self.skip_own = skip_own
        self._own_address: str | None = None

    def own_address(self) -> str:
        """The mailbox's own address, fetched once per source."""
        if self._own_address is None:
            self._own_address = self.client.address()
        return self._own_address

    def _is_own_message(self, sender: str) -> bool:
        """True when the mailbox sent this itself.

        The inbox receives the mailbox's own automated mail - n8n quote and
        error notifications. Those must never be classified, labelled, or
        replied to: drafting a reply to ourselves would be noise at best, and a
        self-sustaining loop at worst.
        """
        from src.gmail_client import normalize_address

        if not self.skip_own:
            return False
        return normalize_address(sender) == normalize_address(self.own_address())

    def fetch(self, limit: int = DEFAULT_LIMIT) -> List[InboundEmail]:
        """Return up to ``limit`` emails, skipping the mailbox's own mail.

        A message that cannot be read is logged and skipped rather than ending
        the run: one malformed email in an inbox must not stop the other
        nineteen from being triaged.
        """
        from src.gmail_client import extract_attachments, extract_body

        emails: List[InboundEmail] = []
        for message_id in self.client.list_message_ids(self.query, limit):
            try:
                message = self.client.get_message(message_id)
                payload = message.get("payload", {})
                headers = {
                    h["name"].lower(): h["value"]
                    for h in payload.get("headers", [])
                }
                sender = headers.get("from", "")
                subject = headers.get("subject", "")

                if self._is_own_message(sender):
                    log.info("skipped: own message (%s)", subject or message_id)
                    continue

                emails.append(
                    InboundEmail(
                        source_id=message["id"],
                        thread_id=message.get("threadId", ""),
                        rfc822_message_id=headers.get("message-id", ""),
                        references=headers.get("references", ""),
                        origin="gmail",
                        sender=sender,
                        subject=subject,
                        body=extract_body(payload),
                        attachments=extract_attachments(payload),
                    )
                )
            except Exception:
                log.exception("skipped: could not read message %s", message_id)
        return emails


class SampleSource:
    """The bundled demo emails in ``samples/``.

    Used for DEMO_MODE and for ``demo_run.py``. Every file is returned on every
    fetch: these are fixtures, so there is no "already processed" state to
    track the way a real mailbox has.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or SAMPLES_DIR

    def fetch(self, limit: int = DEFAULT_LIMIT) -> List[InboundEmail]:
        emails = []
        for path in sorted(self.directory.glob("*.txt"))[:limit]:
            text = path.read_text(encoding="utf-8")
            headers, body = split_headers(text)
            emails.append(
                InboundEmail(
                    origin=path.name,
                    sender=headers.get("from", ""),
                    subject=headers.get("subject", ""),
                    body=body,
                    # Sample files carry no provider ids and no attachments:
                    # there is no mailbox behind them to label or reply into.
                )
            )
        return emails
