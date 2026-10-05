"""Where emails come from.

A source yields :class:`~src.models.InboundEmail` objects and knows nothing
about how they are analyzed. :class:`SampleSource` reads the bundled demo
files; a Gmail-backed source will implement the same protocol, which is what
lets DEMO_MODE switch between them without the rest of the app noticing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Protocol, runtime_checkable

from src.models import InboundEmail

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
