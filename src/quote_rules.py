"""Deterministic handling of quote requests.

A quote request with everything we need - a described job, an address, and a
photo - gets a fixed acknowledgement from config/company.yaml, not the model's
draft. A quote request missing any of the three gets a different fixed
template, asking only for what is actually missing. The model cannot be
trusted to stop asking questions, or to stop promising a quote it cannot yet
send, so both rules are enforced here.

The photo is judged from the email's attachment list, not from the model's
reading of the body, since the attachment list is a fact about the message.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.company import MATCH_CUSTOMER, COMPANY_LANGUAGE, load_company
from src.models import Category, EmailAnalysis

TEMPLATE_LANGUAGES = ("en", "de")

# Text that only the complete-quote template contains. Used by the evals to
# tell whether the rule fired, so keep these in step with config/company.yaml.
TEMPLATE_MARKERS = (
    "Thank you for your request and the photo.",
    "Vielen Dank für Ihre Anfrage und das Foto.",
)

# Text that only the needs-more template contains. Kept in step with
# config/company.yaml's quote_needs_more.
NEEDS_MORE_MARKERS = (
    "To prepare your quote we need:",
    "Um Ihr Angebot vorzubereiten, benötigen wir noch:",
)

# What a quote needs, in the order they are listed when missing. Keys match
# config/company.yaml's quote_needs_more.<lang>.items.
MISSING_PHOTO = "photo"
MISSING_ADDRESS = "address"
MISSING_JOB = "job"

# English labels for missing_info, which stays in English for staff regardless
# of the draft's language.
MISSING_INFO_LABELS = {
    MISSING_PHOTO: "a photo of the roof",
    MISSING_ADDRESS: "the property address",
    MISSING_JOB: "a description of the work needed (for guttering, include the length in metres)",
}

_ATTACHMENTS_LINE = re.compile(r"^Attachments:\s*(.*)$", re.MULTILINE)
_GERMAN_HINTS = re.compile(
    r"mit freundlichen|Ihr .{0,40}Team|Guten Tag|Hallo", re.IGNORECASE
)


def has_photo(prompt_text: str) -> bool:
    """True if the email carries an image attachment.

    Reads the ``Attachments:`` line that InboundEmail.prompt_text writes, so the
    answer is the same for a Gmail message and a sample file.
    """
    match = _ATTACHMENTS_LINE.search(prompt_text)
    return bool(match) and "image/" in match.group(1).lower()


def is_template_draft(draft: str) -> bool:
    """True if a draft is the fixed acknowledgement rather than model output."""
    return any(marker in draft for marker in TEMPLATE_MARKERS)


def is_needs_more_draft(draft: str) -> bool:
    """True if a draft is the fixed "needs more" template rather than model output."""
    return any(marker in draft for marker in NEEDS_MORE_MARKERS)


def qualifies(prompt_text: str, analysis: EmailAnalysis) -> bool:
    """A complete quote request: the job is described, we have an address, and
    there is a photo. Anything less keeps the model's draft and its questions."""
    return (
        analysis.category is Category.QUOTE_REQUEST
        and analysis.job_described
        and bool(analysis.customer.address)
        and has_photo(prompt_text)
    )


def missing_quote_items(prompt_text: str, analysis: EmailAnalysis) -> List[str]:
    """Which of the three things a quote needs are missing, in a fixed order.

    Judged the same way ``qualifies`` judges them, so the two can never
    disagree about what is missing.
    """
    missing = []
    if not has_photo(prompt_text):
        missing.append(MISSING_PHOTO)
    if not analysis.customer.address:
        missing.append(MISSING_ADDRESS)
    if not analysis.job_described:
        missing.append(MISSING_JOB)
    return missing


def _language_for(analysis: EmailAnalysis, company: Dict[str, Any]) -> str:
    """Which template language to use, following the reply_language setting."""
    setting = str(company.get("reply_language") or MATCH_CUSTOMER).strip()
    if setting == MATCH_CUSTOMER:
        code = "de" if _GERMAN_HINTS.search(analysis.draft_reply) else "en"
    elif setting == COMPANY_LANGUAGE:
        code = str(company.get("language") or "en")
    else:
        code = setting
    # Templates exist only for en and de. Anything else falls back to English,
    # which is readable to most customers rather than wrong in a language we
    # have not checked.
    return code if code in TEMPLATE_LANGUAGES else "en"


def _greeting(template: Dict[str, str], name: str | None) -> str:
    if not name:
        return template["greeting_anonymous"]
    return template["greeting"].format(name=name, first_name=name.split()[0])


def render_template(analysis: EmailAnalysis, company: Dict[str, Any] | None = None) -> str:
    """The acknowledgement, filled in for this customer."""
    c = company or load_company()
    templates = c.get("quote_acknowledgement") or {}
    lang = _language_for(analysis, c)
    template = templates.get(lang) or templates.get("en")
    if not template:
        raise RuntimeError(
            "config/company.yaml has no quote_acknowledgement templates. Add "
            "entries for en (and de) to use the quote rule."
        )

    greeting = _greeting(template, analysis.customer.name)
    signoff = template["signoff"].format(company=c.get("name", "the company"))
    return f"{greeting}\n\n{template['body']}\n\n{signoff}\n"


def render_needs_more_template(
    analysis: EmailAnalysis,
    missing: List[str],
    company: Dict[str, Any] | None = None,
) -> str:
    """The "needs more" reply, listing only the items in ``missing``."""
    c = company or load_company()
    templates = c.get("quote_needs_more") or {}
    lang = _language_for(analysis, c)
    template = templates.get(lang) or templates.get("en")
    if not template:
        raise RuntimeError(
            "config/company.yaml has no quote_needs_more templates. Add "
            "entries for en (and de) to use the quote rule."
        )

    items = template.get("items") or {}
    greeting = _greeting(template, analysis.customer.name)
    bullets = "\n".join(f"- {items.get(key, key)}" for key in missing)
    signoff = template["signoff"].format(company=c.get("name", "the company"))
    return (
        f"{greeting}\n\n{template['intro']}\n\n{bullets}\n\n"
        f"{template['outro']}\n\n{signoff}\n"
    )


def apply_quote_rules(
    prompt_text: str,
    analysis: EmailAnalysis,
    company: Dict[str, Any] | None = None,
) -> EmailAnalysis:
    """Return the analysis with a fixed template in place of the draft.

    A complete quote request gets the acknowledgement, with missing_info
    emptied - a draft that asks nothing and a list that still asks for things
    would contradict each other. Any other quote request gets the needs-more
    template instead of the model's draft, so it can never promise a quote or
    claim to have everything while something is still missing. missing_info is
    rebuilt from the same items the template lists, not from the model.
    """
    if qualifies(prompt_text, analysis):
        return analysis.model_copy(
            update={
                "draft_reply": render_template(analysis, company),
                "missing_info": [],
            }
        )
    if analysis.category is not Category.QUOTE_REQUEST:
        return analysis
    missing = missing_quote_items(prompt_text, analysis)
    return analysis.model_copy(
        update={
            "draft_reply": render_needs_more_template(analysis, missing, company),
            "missing_info": [MISSING_INFO_LABELS[key] for key in missing],
        }
    )
