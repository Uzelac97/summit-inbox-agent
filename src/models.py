"""Provider-agnostic data models for the inbox agent.

Nothing in this module may import an AI SDK or a mail provider SDK: these
models describe *what* an email is and *what* we want out of it, not *who*
delivered it or *who* analyzed it.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Bullet characters a model may prefix list entries with despite being asked
# for plain phrases.
_BULLETS = "-*•· \t"


class AttachmentInfo(BaseModel):
    """What we tell the model about an attachment: its name and type only.

    The file itself is never read or uploaded. A name and a MIME type are
    enough for the model to say "you mentioned photos but attached a
    spreadsheet", and keep us clear of opening customer files.
    """

    filename: str = Field(description="Attachment filename as the sender set it.")
    mime_type: str = Field(description='MIME type, e.g. "application/pdf".')

    def describe(self) -> str:
        return f"{self.filename} ({self.mime_type})"


class InboundEmail(BaseModel):
    """One email awaiting triage, independent of where it came from.

    The first three fields are what a provider needs in order to act on the
    message later - label it, or reply in its thread - and are empty for
    emails that did not come from a real mailbox. Everything after them is
    what the model reads.
    """

    source_id: str = Field(
        default="", description="Provider's own message id, for labelling."
    )
    thread_id: str = Field(
        default="", description="Provider's thread id, so a reply stays in thread."
    )
    rfc822_message_id: str = Field(
        default="",
        description="The Message-ID header, which In-Reply-To must quote.",
    )
    references: str = Field(
        default="",
        description="The References header, so a reply keeps the whole chain.",
    )
    origin: str = Field(
        default="", description="Where this came from, for display: a filename or a mailbox."
    )
    sender: str = Field(default="", description="From header, as written.")
    subject: str = Field(default="", description="Subject header, as written.")
    body: str = Field(default="", description="Plain-text body, HTML already stripped.")
    attachments: List[AttachmentInfo] = Field(default_factory=list)

    @property
    def prompt_text(self) -> str:
        """The canonical text handed to ``analyze_email``.

        Every source renders down to this one shape, so a Gmail message and a
        sample file are indistinguishable by the time the model sees them. It
        is also what the analysis cache keys on, so changing this layout
        retires cached results - deliberately, since it changes the input.
        """
        header = []
        if self.sender:
            header.append(f"From: {self.sender}")
        if self.subject:
            header.append(f"Subject: {self.subject}")
        if self.attachments:
            listed = ", ".join(a.describe() for a in self.attachments)
            header.append(f"Attachments: {listed}")

        body = self.body.strip()
        if not header:
            return f"{body}\n"
        return "\n".join(header) + f"\n\n{body}\n"


class Category(str, Enum):
    """What the email is fundamentally asking for.

    ``EMERGENCY`` outranks the others: an active leak is an emergency whether or
    not the sender is also complaining about work we did.
    """

    EMERGENCY = "emergency"
    QUOTE_REQUEST = "quote_request"
    APPOINTMENT = "appointment"
    COMPLAINT = "complaint"
    GENERAL = "general"
    SPAM = "spam"


class Urgency(str, Enum):
    """How fast a human needs to get involved."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Customer(BaseModel):
    """Contact details pulled out of the email body or signature.

    Every field is optional: real inbound mail is usually incomplete, and a
    guessed phone number is worse than a missing one.
    """

    name: Optional[str] = Field(default=None, description="Full name of the sender.")
    phone: Optional[str] = Field(default=None, description="Phone number, as written.")
    email: Optional[str] = Field(default=None, description="Reply-to email address.")
    address: Optional[str] = Field(
        default=None, description="Street address of the property, if mentioned."
    )


class EmailAnalysis(BaseModel):
    """The complete triage result for a single inbound email."""

    category: Category = Field(description="Primary intent of the email.")
    urgency: Urgency = Field(description="How quickly this needs a human response.")
    customer: Customer = Field(description="Contact details found in the email.")
    summary: str = Field(description="One sentence describing what the sender wants.")
    missing_info: List[str] = Field(
        description="Details we still need from the customer before we can act."
    )
    job_described: bool = Field(
        default=False,
        description=(
            "True only if the email says what work is needed, in the customer's "
            "own words. An address or a photo alone does not count."
        ),
    )
    draft_reply: str = Field(description="Ready-to-send reply, signed as Summit Roofing.")

    @field_validator("missing_info")
    @classmethod
    def _strip_bullets(cls, items: List[str]) -> List[str]:
        """Drop bullet prefixes so the UI can render these as its own list.

        Runs on cached results too, which keeps entries written before the
        prompt forbade bullets from displaying as "- - thing".
        """
        cleaned = (item.strip().lstrip(_BULLETS).strip() for item in items)
        return [item for item in cleaned if item]
