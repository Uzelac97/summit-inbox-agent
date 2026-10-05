"""Gmail label names for the agent, and how they map to a category.

Every label the agent uses is namespaced under ``agent/`` so that nothing it
does is confused with a label a human made, and so the whole lot can be found,
inspected, or deleted as a group.
"""

from __future__ import annotations

from src.models import Category

# One label per category. The names are deliberately shorter than the category
# values - "agent/quote" reads better in a Gmail sidebar than
# "agent/quote_request".
CATEGORY_LABELS = {
    Category.EMERGENCY: "agent/emergency",
    Category.QUOTE_REQUEST: "agent/quote",
    Category.APPOINTMENT: "agent/appointment",
    Category.COMPLAINT: "agent/complaint",
    Category.GENERAL: "agent/general",
    Category.SPAM: "agent/spam",
}

# Applied once an email has been fully handled. It is what makes a re-run
# idempotent, so it is applied last - after the category label and, later, the
# draft - so that a crash part way through leaves the email to be retried
# rather than silently skipped.
PROCESSED_LABEL = "agent/processed"

ALL_LABELS = sorted(set(CATEGORY_LABELS.values()) | {PROCESSED_LABEL})


def label_for(category: Category) -> str:
    """The Gmail label for a category."""
    return CATEGORY_LABELS[category]
