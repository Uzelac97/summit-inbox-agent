"""Forward complete quote requests to the quote generator (n8n).

A quote request that qualifies for the acknowledgement template - a described
job, an address, and a photo - is POSTed to QUOTE_WEBHOOK_URL as
multipart/form-data. The label agent/forwarded is added only after the POST
succeeds, and its presence stops a second forward. A failed POST leaves the
draft in place; :class:`~src.processor.Processor` withholds agent/processed in
that case and applies agent/forward-failed instead, so a later pass retries
just the forward (see Processor.retry_forward_failures) rather than losing it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Dict

import requests

from src.labels import FORWARDED_LABEL
from src.models import EmailAnalysis, InboundEmail
from src.quote_rules import qualifies

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30
PHOTO_FIELD = "roof_photos"

NOT_QUALIFYING = "not qualifying"
ALREADY = "skipped"
FORWARDED = "forwarded"
FAILED = "failed"
REFUSED = "refused"

# A forward in either of these states did not reach the generator and should
# be retried, up to MAX_FORWARD_ATTEMPTS times, rather than treated as done.
RETRYABLE_STATUSES = (FAILED, REFUSED)

KEY_HEADER = "X-Summit-Key"

# How many times a single message's forward is retried before it is left
# alone with agent/forward-failed for a person to look at.
MAX_FORWARD_ATTEMPTS = 5

# Failure counts per message, kept locally rather than in Gmail: a label can
# say "this failed" but not "how many times", and the count must survive
# between passes of run_agent.py.
ATTEMPTS_FILE = Path(__file__).resolve().parent.parent / "data" / "forward_attempts.json"


def _load_attempts() -> Dict[str, int]:
    try:
        return json.loads(ATTEMPTS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_attempts(data: Dict[str, int]) -> None:
    ATTEMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def forward_attempts(message_id: str) -> int:
    """How many times forwarding this message has already failed."""
    return _load_attempts().get(message_id, 0)


def record_forward_attempt(message_id: str) -> int:
    """Record one more failed attempt for a message, returning the new total."""
    data = _load_attempts()
    data[message_id] = data.get(message_id, 0) + 1
    _save_attempts(data)
    return data[message_id]


def clear_forward_attempts(message_id: str) -> None:
    """Drop a message's failure count once it no longer needs retrying."""
    data = _load_attempts()
    if data.pop(message_id, None) is not None:
        _save_attempts(data)

# Where a reply stops being the customer's own words: a signature sign-off, a
# "--" signature delimiter, or the start of quoted history.
_SIGN_OFF = re.compile(
    r"^\s*(best regards|kind regards|regards|many thanks|thanks|thank you|cheers|"
    r"best|sincerely|mit freundlichen gr\S*|viele gr\S*|lg|gr[uü][sß]e?)\s*[,!.]?\s*$",
    re.IGNORECASE,
)
_HISTORY = re.compile(
    r"^\s*(on .+ wrote:?|am .+ schrieb.*:?|-{2,}\s*original message\s*-{2,}|"
    r"-{2,}\s*forwarded message\s*-{2,}|from:\s.+|sent from my .*)$",
    re.IGNORECASE,
)


def own_words(body: str) -> str:
    """The customer's own words: the body with quoted history and signature cut.

    Reading stops at the first sign-off, signature delimiter, or history
    header, so everything after it - the signature, the sender's phone number,
    the earlier messages in the thread - is dropped. Quoted lines ("> ...") are
    dropped wherever they appear.
    """
    kept = []
    for line in body.splitlines():
        if line.strip() == "--" or _SIGN_OFF.match(line) or _HISTORY.match(line):
            break
        if line.lstrip().startswith(">"):
            continue
        kept.append(line.rstrip())
    text = "\n".join(kept).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


@dataclass
class Forwarded:
    """What happened when a quote request was forwarded."""

    status: str
    detail: str = ""


def webhook_url_from_env(environ: Dict[str, str]) -> str:
    """QUOTE_WEBHOOK_URL with surrounding space removed. Empty means disabled."""
    return (environ.get("QUOTE_WEBHOOK_URL") or "").strip()


class QuoteForwarder:
    """Sends qualifying quote requests to the webhook, once each."""

    def __init__(self, client, url: str, secret: str = "") -> None:
        self.client = client
        self.url = url
        self.secret = secret

    def fields(self, email: InboundEmail, analysis: EmailAnalysis) -> Dict[str, str]:
        """The text fields the generator receives."""
        customer = analysis.customer
        return {
            "name": customer.name or "",
            "email": customer.email or parseaddr(email.sender)[1],
            "address": customer.address or "",
            "job_description": own_words(email.body),
        }

    def already_forwarded(self, email: InboundEmail) -> bool:
        label_id = self.client.label_ids().get(FORWARDED_LABEL)
        if not label_id:
            return False
        message: Dict[str, Any] = self.client.get_message(email.source_id)
        return label_id in message.get("labelIds", [])

    def forward(self, email: InboundEmail, analysis: EmailAnalysis) -> Forwarded:
        """Forward one email if it qualifies and has not been forwarded before."""
        if not qualifies(email.prompt_text, analysis):
            return Forwarded(NOT_QUALIFYING)
        if not self.secret:
            # Without a key the generator cannot tell us from anyone else, so
            # nothing is sent. Logged here and shown per email in the console.
            log.error("not forwarding %s: QUOTE_WEBHOOK_SECRET is empty", email.source_id)
            return Forwarded(REFUSED, "QUOTE_WEBHOOK_SECRET is empty")
        try:
            if self.already_forwarded(email):
                return Forwarded(ALREADY, "already has agent/forwarded")
            photos = self.client.get_image_attachments(email.source_id)
            files = [(PHOTO_FIELD, (name, data, mime)) for name, mime, data in photos]
            response = requests.post(
                self.url,
                data=self.fields(email, analysis),
                files=files,
                headers={KEY_HEADER: self.secret},
                timeout=TIMEOUT_SECONDS,
            )
        except Exception as exc:
            log.warning("forward failed for %s: %s", email.source_id, exc)
            return Forwarded(FAILED, f"{type(exc).__name__}: {exc}")

        if not 200 <= response.status_code < 300:
            log.warning("forward rejected for %s: HTTP %s", email.source_id, response.status_code)
            return Forwarded(FAILED, f"HTTP {response.status_code}")

        detail = f"{len(photos)} photo(s), HTTP {response.status_code}"
        try:
            self.client.add_labels(email.source_id, [FORWARDED_LABEL])
        except Exception as exc:
            # The POST has happened, so the record is wrong rather than the
            # forward. Say so; do not pretend it did not go.
            log.warning("forwarded %s but could not label it: %s", email.source_id, exc)
            detail += f"; label not saved ({type(exc).__name__})"
        return Forwarded(FORWARDED, detail)
