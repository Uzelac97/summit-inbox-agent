"""Provider-agnostic data models for the inbox agent.

Nothing in this module may import an AI SDK: these models describe *what* we
want out of an email, not *who* produces it.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Bullet characters a model may prefix list entries with despite being asked
# for plain phrases.
_BULLETS = "-*•· \t"


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
