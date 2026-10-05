"""Claude-backed email analysis.

This is the only module that knows which AI provider we use. Everything else
talks to ``analyze_email`` and the models in :mod:`src.models`, so swapping
provider means rewriting this file and nothing else.
"""

import hashlib
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from src.models import EmailAnalysis

load_dotenv()

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Enough for a classification plus a short drafted reply; the tool input runs
# to a few hundred tokens in practice.
MAX_TOKENS = 2048

# The SDK retries 429, 5xx (including 529 overloaded), and connection errors
# with exponential backoff. Raising it from the default of 2 covers a provider
# hiccup without us hand-rolling a loop.
MAX_RETRIES = 5

# Re-analyzing an email we have already seen costs money and returns the same
# answer, so results are cached on disk.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

TOOL_NAME = "classify_email"

SYSTEM_INSTRUCTION = """\
You are the inbox assistant for Summit Roofing, a residential and commercial \
roofing company. You triage inbound customer email.

For each email you receive:
- Classify its intent, choosing exactly one category:
  - emergency: active water coming into the building, storm or wind damage, \
exposed roof structure, or anything presenting a safety risk. This applies to \
any sender, whether they are a long-standing customer or have never contacted \
us before, and whether or not they are also complaining about work we did. An \
active leak is an emergency first and a complaint second.
  - quote_request: wants a price or an estimate for work.
  - appointment: wants to book, confirm, move, or cancel a visit.
  - complaint: unhappy with work we did or how we did it, with nothing \
currently leaking, exposed, or unsafe.
  - general: questions, paperwork, and enquiries that need no work scheduled.
  - spam: marketing, sales pitches, and bulk mail, including mail that opens \
like a genuine enquiry before pitching a product or service.
- Judge how urgently a human needs to respond:
  - high: water is entering the building, the structure is exposed, someone \
could be hurt, or damage will get materially worse within a day or so. Every \
emergency is high.
  - medium: we owe the sender a concrete action that carries a time element - \
a visit to schedule or move, a deadline they have named, or a complaint about \
work we did where nothing is currently leaking or unsafe.
  - low: information, paperwork, and budgeting enquiries with no time \
pressure, and all spam.
- Extract the customer's contact details. Only record what the email actually \
states; leave a field empty rather than guessing or inferring it.
- List the information we still need before we can quote, schedule, or resolve \
the request. If nothing is missing, return an empty list.
- Write the draft reply as Summit Roofing, in English: friendly, professional, \
and concise. Thank them, acknowledge their specific situation, ask for exactly \
the missing details you listed, and state the next step. Never invent prices, \
dates, appointment times, or warranty terms.

Lay the draft reply out as a real email, using newline characters:

Hi <first name>,
<blank line>
<short opening paragraph>
<blank line>
<further paragraph, one per idea>
<blank line>
Best regards,
The Summit Roofing Team

The greeting, each paragraph, and each line of the sign-off are separated by \
newlines, with a blank line between blocks. Never return the reply as one \
single run-on paragraph. Keep paragraphs to two or three sentences. If the \
reply asks for several details, write them inside the reply body as one line \
each beginning with "- ".

That bullet formatting applies only to the draft reply. Entries in the missing \
information list are plain phrases: no leading dash, bullet, or numbering.

For spam, classify it as spam with low urgency and leave the draft reply empty.
"""

# Editing the prompt changes what the model returns, so cached results from an
# older prompt are stale. Fingerprinting the prompt into the cache key retires
# those entries automatically instead of leaving them to be cleared by hand.
PROMPT_FINGERPRINT = hashlib.sha256(SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest()[:8]

CLASSIFY_TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Record the triage result for one inbound customer email. "
        "Call this tool exactly once with the complete analysis."
    ),
    "input_schema": EmailAnalysis.model_json_schema(),
}

_client = None


def _get_client() -> anthropic.Anthropic:
    """Build the client lazily, so importing this module needs no credentials."""
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key."
            )
        _client = anthropic.Anthropic(max_retries=MAX_RETRIES)
    return _client


def active_model() -> str:
    """The model this process will call."""
    return os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL


def _cache_file(text: str, model: str) -> Path:
    """Where the result for this email, model, and prompt is stored.

    All three are part of the key: the same email analyzed by a different model
    or under a different prompt is a different result, and must not be served
    from another one's entry. The fingerprint also prefixes the filename, so
    entries from a retired prompt are easy to spot and delete.
    """
    key = hashlib.sha256(
        f"{PROMPT_FINGERPRINT}\n{model}\n{text}".encode("utf-8")
    ).hexdigest()
    return CACHE_DIR / f"{PROMPT_FINGERPRINT}-{key}.json"


def is_cached(text: str) -> bool:
    """True if this email already has a stored result for the active model."""
    return _cache_file(text, active_model()).exists()


def _read_cache(path: Path) -> EmailAnalysis | None:
    """Load a stored result, or None if it is missing, unreadable, or stale."""
    if not path.exists():
        return None
    try:
        return EmailAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        # Written by an older version of the schema, or truncated. Re-analyze.
        return None


def _write_cache(path: Path, result: EmailAnalysis) -> None:
    """Store a result. A cache we cannot write is not worth failing a run over."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError:
        pass


def _tool_input(response: anthropic.types.Message, model: str) -> dict:
    """Pull the forced tool call out of a response, or explain why it is absent.

    Forcing ``tool_choice`` means a healthy response always contains exactly one
    ``tool_use`` block. The ways that can fail are worth naming individually,
    because each has a different fix.
    """
    for block in response.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            return block.input

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{model} hit the {MAX_TOKENS}-token limit before finishing the "
            "analysis. Raise MAX_TOKENS."
        )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"{model} declined to analyze this email.")
    raise RuntimeError(
        f"{model} returned no {TOOL_NAME} call (stop_reason={response.stop_reason!r})."
    )


def analyze_email(text: str, use_cache: bool = True) -> EmailAnalysis:
    """Triage one inbound email and draft a reply to it.

    Args:
        text: The raw email, headers and signature included.
        use_cache: Read and write ``.cache/``. Pass False to force a fresh call.

    Returns:
        The structured analysis, including a ready-to-review draft reply.
    """
    model = active_model()
    cache_file = _cache_file(text, model)

    if use_cache:
        cached = _read_cache(cache_file)
        if cached is not None:
            return cached

    try:
        response = _get_client().messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0.2,
            system=SYSTEM_INSTRUCTION,
            messages=[{"role": "user", "content": f"Analyze this email:\n\n{text}"}],
            tools=[CLASSIFY_TOOL],
            # Forcing the tool is what makes the response structured. Newer
            # models (Opus 5.5, Sonnet 5.5, Fable 5.1) reject a forced
            # tool_choice with a 400, so this pins us to models that allow it.
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
    except anthropic.BadRequestError as exc:
        raise RuntimeError(
            f"{model} rejected the request: {exc}. If this mentions tool_choice, "
            "the model does not support forced tool use - set ANTHROPIC_MODEL "
            f"back to {DEFAULT_MODEL}."
        ) from exc

    result = EmailAnalysis.model_validate(_tool_input(response, model))

    if use_cache:
        _write_cache(cache_file, result)
    return result
