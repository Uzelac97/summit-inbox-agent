"""Business facts, loaded from config/company.yaml.

Everything company-specific lives in that file, so a new business adopts the
agent by editing it and nothing else. The rendered text is folded into the
system prompt and therefore into its fingerprint, so editing a fact retires
cached analyses rather than serving stale answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPANY_FILE = ROOT / "config" / "company.yaml"

MATCH_CUSTOMER = "match_customer"
COMPANY_LANGUAGE = "company"

# Enough to name the common cases in the prompt. An unlisted code is passed
# through as-is, which still reads sensibly: "reply in fr".
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "pl": "Polish",
    "sr": "Serbian",
}


def load_company(path: Path | None = None) -> Dict[str, Any]:
    """Read the facts file."""
    source = path or COMPANY_FILE
    if not source.exists():
        raise RuntimeError(
            f"{source} is missing. The agent answers customer questions from "
            "that file and must not invent facts in its place."
        )
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{source} did not parse to a mapping.")
    return data


def language_name(code: str) -> str:
    """A human name for a language code, falling back to the code itself."""
    return LANGUAGE_NAMES.get((code or "").lower(), code)


def has_service_area(company: Dict[str, Any]) -> bool:
    """True only when a service area is actually configured.

    An empty mapping, an empty list, or a missing key all mean "serve
    anywhere", which is the default. Getting this wrong in the other
    direction - treating absent as restricted - would have the agent decline
    work for a business that never set a boundary.
    """
    area = company.get("service_area") or {}
    if isinstance(area, str):
        return bool(area.strip())
    if isinstance(area, list):
        return bool(area)
    return bool(area.get("description") or area.get("areas"))


def _bullets(items: List[str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}- {item}" for item in items)


def _render_language_rule(company: Dict[str, Any]) -> List[str]:
    """How the draft should choose its language, per configuration."""
    setting = str(company.get("reply_language") or MATCH_CUSTOMER).strip()
    own = str(company.get("language") or "en")

    if setting == MATCH_CUSTOMER:
        return [
            "Reply language: write the draft in the same language the customer",
            "wrote in. Reply to a German email in German, to a French email in",
            "French, and to an English email in English. These instructions are",
            "in English, which does not change that: follow the customer's",
            "language, not this prompt's.",
        ]

    target = own if setting == COMPANY_LANGUAGE else setting
    name = language_name(target)
    return [
        f"Reply language: always write the draft in {name}, whatever language",
        "the customer wrote in. Do not switch languages to match them.",
    ]


def _render_service_area(company: Dict[str, Any]) -> List[str]:
    """The coverage rule, which depends on whether an area is configured."""
    if not has_service_area(company):
        return [
            "Service area: not restricted. This business serves customers",
            "anywhere, so never decline work because of where the customer is,",
            "never question whether an address is covered, and do not mention",
            "location, distance, region, or travel in the draft at all.",
        ]

    area = company["service_area"]
    if isinstance(area, str):
        return [f"Service area: {area}", "", _COVERAGE_RULE]
    if isinstance(area, list):
        return ["Service area covers:", _bullets(area), "", _COVERAGE_RULE]

    lines = [f"Service area: {area.get('description', 'as listed below')}"]
    if area.get("areas"):
        lines.append(_bullets(area["areas"]))
    lines += ["", _COVERAGE_RULE]
    return lines


_COVERAGE_RULE = (
    "If an address is clearly outside that area, say so politely and do not "
    "offer to quote or schedule. If you cannot tell, do not guess and do not "
    "raise it - let a colleague confirm. Never decline an emergency because of "
    "location: an emergency is always answered and always pointed at the "
    "emergency line."
)


def render_facts(company: Dict[str, Any] | None = None) -> str:
    """Render the facts as the block that goes into the system prompt."""
    c = company or load_company()
    contact = c.get("contact", {})
    hours = c.get("hours", {})
    response = c.get("response_times", {})

    parts = [
        "THESE ARE THE BUSINESS FACTS.",
        "",
        "Answer customer questions from these facts. If a question cannot be",
        "answered from them, say that a colleague will confirm the details,",
        "and do not guess. Never state a service, an opening hour, a price, or",
        "a warranty length that does not appear here.",
        "",
        f"Company: {c.get('name', '')} - {c.get('tagline', '')}".rstrip(" -"),
        f"Main phone: {contact.get('phone', 'not listed')}",
        f"Emergency line, 24/7: {contact.get('emergency_phone', 'not listed')}",
        "",
    ]
    parts += _render_language_rule(c)
    parts += [""]

    if c.get("services"):
        parts += ["Services offered:", _bullets(c["services"]), ""]
    if c.get("not_offered"):
        parts += [
            "Not offered - say so plainly and do not offer to arrange it:",
            _bullets(c["not_offered"]),
            "",
        ]

    parts += _render_service_area(c)
    parts += [""]

    if hours:
        parts += [
            "Opening hours:",
            f"  {hours.get('weekdays', '')}".rstrip(),
            f"  {hours.get('saturday', '')}".rstrip(),
            f"  Sunday: {hours.get('sunday', 'Closed')}",
            f"  {hours.get('emergency', '')}".rstrip(),
            "",
        ]
    if response:
        parts += ["Response times:", _bullets([str(v) for v in response.values()]), ""]

    warranty = c.get("warranty", {})
    if warranty:
        parts += ["Warranty:", _bullets([str(v) for v in warranty.values()]), ""]

    payment = c.get("payment", {})
    if payment.get("note"):
        parts += [f"Pricing: {payment['note'].strip()}"]

    return "\n".join(line for line in parts).strip()


def emergency_phone(company: Dict[str, Any] | None = None) -> str:
    """The 24/7 number, for anything that needs it outside the prompt."""
    return (company or load_company()).get("contact", {}).get("emergency_phone", "")
