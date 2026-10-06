"""Deterministic handling of complete quote requests.

A quote request with everything we need - a described job, an address, and a
photo - gets a fixed acknowledgement from config/company.yaml, not the model's
draft. The model cannot be trusted to stop asking questions on its own, so the
rule is enforced here.

The photo is judged from the email's attachment list, not from the model's
reading of the body, since the attachment list is a fact about the message.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from src.company import MATCH_CUSTOMER, COMPANY_LANGUAGE, load_company
from src.models import Category, EmailAnalysis

TEMPLATE_LANGUAGES = ("en", "de")

# Text that only the template contains. Used by the evals to tell whether the
# rule fired, so keep these in step with config/company.yaml.
TEMPLATE_MARKERS = (
    "Thank you for your request and the photo.",
    "Vielen Dank für Ihre Anfrage und das Foto.",
)

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


def qualifies(prompt_text: str, analysis: EmailAnalysis) -> bool:
    """A complete quote request: the job is described, we have an address, and
    there is a photo. Anything less keeps the model's draft and its questions."""
    return (
        analysis.category is Category.QUOTE_REQUEST
        and analysis.job_described
        and bool(analysis.customer.address)
        and has_photo(prompt_text)
    )


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


def apply_quote_rules(
    prompt_text: str,
    analysis: EmailAnalysis,
    company: Dict[str, Any] | None = None,
) -> EmailAnalysis:
    """Return the analysis with the template in place of the draft, if it applies.

    When it applies, missing_info is emptied too: a draft that asks nothing and
    a list that still asks for things would contradict each other.
    """
    if not qualifies(prompt_text, analysis):
        return analysis
    return analysis.model_copy(
        update={
            "draft_reply": render_template(analysis, company),
            "missing_info": [],
        }
    )
