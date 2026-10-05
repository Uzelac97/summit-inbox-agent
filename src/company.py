"""Summit Roofing's business facts, loaded from config/company.yaml.

Kept separate from the prompt so the facts can be edited without touching
instructions, and so a non-developer can change opening hours. The rendered
text is folded into the system prompt and therefore into its fingerprint, so
editing the file retires cached analyses rather than serving stale answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPANY_FILE = ROOT / "config" / "company.yaml"


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


def _bullets(items: List[str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}- {item}" for item in items)


def render_facts(company: Dict[str, Any] | None = None) -> str:
    """Render the facts as the block that goes into the system prompt.

    Written as prose headings rather than dumped YAML: the model reads it
    better, and it keeps the prompt readable when debugging a bad draft.
    """
    c = company or load_company()
    contact = c.get("contact", {})
    hours = c.get("hours", {})
    area = c.get("service_area", {})
    response = c.get("response_times", {})

    parts = [
        "THESE ARE SUMMIT ROOFING'S BUSINESS FACTS.",
        "",
        "Answer customer questions from these facts. If a question cannot be",
        "answered from them, say that a colleague will confirm the details,",
        "and do not guess. Never state a service, an area, an opening hour, a",
        "price, or a warranty length that does not appear here.",
        "",
        f"Company: {c.get('name', 'Summit Roofing')} - {c.get('tagline', '')}".rstrip(
            " -"
        ),
        f"Main phone: {contact.get('phone', 'not listed')}",
        f"Emergency line, 24/7: {contact.get('emergency_phone', 'not listed')}",
        "",
        "Services offered:",
        _bullets(c.get("services", [])),
        "",
        "Not offered - say so plainly and do not offer to arrange it:",
        _bullets(c.get("not_offered", [])),
        "",
        f"Service area: {area.get('description', 'not listed')}",
        _bullets(area.get("areas", [])),
        "",
        "Opening hours:",
        f"  {hours.get('weekdays', '')}",
        f"  {hours.get('saturday', '')}",
        f"  Sunday: {hours.get('sunday', 'Closed')}",
        f"  {hours.get('emergency', '')}",
        "",
        "Response times:",
        _bullets([str(v) for v in response.values()]),
    ]

    warranty = c.get("warranty", {})
    if warranty:
        parts += ["", "Warranty:", _bullets([str(v) for v in warranty.values()])]

    payment = c.get("payment", {})
    if payment.get("note"):
        parts += ["", f"Pricing: {payment['note'].strip()}"]

    return "\n".join(parts).strip()


def emergency_phone(company: Dict[str, Any] | None = None) -> str:
    """The 24/7 number, for anything that needs it outside the prompt."""
    return (company or load_company()).get("contact", {}).get("emergency_phone", "")
