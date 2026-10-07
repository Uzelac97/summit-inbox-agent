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

# Applied only after a complete quote request has been successfully POSTed to
# the quote generator. Its presence is what stops a second forward.
FORWARDED_LABEL = "agent/forwarded"

# Applied instead of PROCESSED_LABEL when a qualifying quote request's forward
# POST fails. Its presence is what a retried pass looks for, and what stops
# the email being picked up by the normal unprocessed-mail query in the
# meantime - the draft already exists, so it must not be redrafted.
FORWARD_FAILED_LABEL = "agent/forward-failed"

ALL_LABELS = sorted(
    set(CATEGORY_LABELS.values())
    | {PROCESSED_LABEL, FORWARDED_LABEL, FORWARD_FAILED_LABEL}
)

# Gmail's own flags rather than agent labels. An emergency is starred and
# marked important so it stands out in any view of the inbox, not only one
# the agent's labels are filtered into. These are system labels: Gmail has
# them already, so they are never created or coloured.
EMERGENCY_MARKS = ("STARRED", "IMPORTANT")
SYSTEM_LABELS = frozenset(EMERGENCY_MARKS)

# The mail the agent has not handled yet. Used as the query for a run, so an
# email already processed is not analysed or drafted for a second time. A
# forward-failed email is excluded too: it already has its draft, and is
# retried through RETRY_QUERY instead of being run through the full pipeline
# again.
UNPROCESSED_QUERY = (
    f'in:inbox -label:"{PROCESSED_LABEL}" -label:"{FORWARD_FAILED_LABEL}"'
)

# Emails whose forward still needs retrying.
RETRY_QUERY = f'label:"{FORWARD_FAILED_LABEL}"'

# Gmail accepts only a fixed palette for label colours; an arbitrary hex is
# rejected with a 400. Every value below was checked against the live API.
# Both backgroundColor and textColor are required - sending one alone fails.
LABEL_COLORS = {
    "agent/emergency": {"backgroundColor": "#fb4c2f", "textColor": "#ffffff"},
    "agent/complaint": {"backgroundColor": "#ffad47", "textColor": "#000000"},
    "agent/quote": {"backgroundColor": "#16a766", "textColor": "#ffffff"},
    "agent/appointment": {"backgroundColor": "#a479e2", "textColor": "#ffffff"},
    "agent/general": {"backgroundColor": "#4a86e8", "textColor": "#ffffff"},
    "agent/spam": {"backgroundColor": "#999999", "textColor": "#ffffff"},
    PROCESSED_LABEL: {"backgroundColor": "#cccccc", "textColor": "#000000"},
    # Gmail's palette has no true teal; this mint green is the closest it offers.
    FORWARDED_LABEL: {"backgroundColor": "#42d692", "textColor": "#000000"},
    # A distinct red from agent/emergency's, so the two are never confused at
    # a glance - this one means "the webhook failed", not "call the customer".
    FORWARD_FAILED_LABEL: {"backgroundColor": "#cc3a21", "textColor": "#ffffff"},
}


def label_for(category: Category) -> str:
    """The Gmail label for a category."""
    return CATEGORY_LABELS[category]


def color_for(name: str) -> dict | None:
    """The colour for a label, or None if it should keep Gmail's default."""
    return LABEL_COLORS.get(name)
