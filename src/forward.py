"""Forward complete quote requests to the quote generator (n8n).

A quote request that qualifies for the acknowledgement template - a described
job, an address, and a photo - is POSTed to QUOTE_WEBHOOK_URL as
multipart/form-data. The label agent/forwarded is added only after the POST
succeeds, and its presence stops a second forward. A failed POST leaves the
draft in place and adds nothing, so the next run will not forward it either
(the email is already agent/processed); it is reported in the console.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from email.utils import parseaddr
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

    def __init__(self, client, url: str) -> None:
        self.client = client
        self.url = url

    def fields(self, email: InboundEmail, analysis: EmailAnalysis) -> Dict[str, str]:
        """The text fields the generator receives."""
        customer = analysis.customer
        return {
            "name": customer.name or "",
            "email": customer.email or parseaddr(email.sender)[1],
            "address": customer.address or "",
            "job_description": analysis.summary,
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
        try:
            if self.already_forwarded(email):
                return Forwarded(ALREADY, "already has agent/forwarded")
            photos = self.client.get_image_attachments(email.source_id)
            files = [(PHOTO_FIELD, (name, data, mime)) for name, mime, data in photos]
            response = requests.post(
                self.url,
                data=self.fields(email, analysis),
                files=files,
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
