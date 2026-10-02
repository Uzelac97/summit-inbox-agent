"""Gemini-backed email analysis.

This is the only module that knows Gemini exists. Everything else talks to
``analyze_email`` and the models in :mod:`src.models`, so swapping provider
means rewriting this file and nothing else.
"""

import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from src.models import EmailAnalysis

load_dotenv()

# The newest Flash models (3.6, 3.8) answer free-tier traffic with 429/503 for
# a workload this size; 3.5-flash serves it reliably. Point GEMINI_MODEL at a
# newer one once you have paid quota.
DEFAULT_MODEL = "gemini-3.5-flash"

# The free tier returns 429/503 under load often enough that one attempt is not
# a fair test of whether a request works.
MAX_ATTEMPTS = 6
RETRY_STATUSES = frozenset({429, 500, 503})
BACKOFF_SECONDS = 5

# The free tier allows only 20 requests per day per model, so re-analyzing an
# email we have already seen is a real cost. Results are cached on disk.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

SYSTEM_INSTRUCTION = """\
You are the inbox assistant for Summit Roofing, a residential and commercial \
roofing company. You triage inbound customer email.

For each email you receive:
- Classify its intent and how urgently a human needs to respond. Active leaks, \
storm damage, structural concerns, and anything threatening the inside of the \
building are high urgency.
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

For spam or marketing email, classify it as spam with low urgency and leave the \
draft reply empty.
"""

# Editing the prompt changes what the model returns, so cached results from an
# older prompt are stale. Fingerprinting the prompt into the cache key retires
# those entries automatically instead of leaving them to be cleared by hand.
PROMPT_FINGERPRINT = hashlib.sha256(SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest()[:8]

_client = None


def _get_client() -> genai.Client:
    """Build the Gemini client lazily, so importing this module needs no key."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def active_model() -> str:
    """The model this process will call."""
    return os.getenv("GEMINI_MODEL") or DEFAULT_MODEL


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

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=EmailAnalysis,
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _get_client().models.generate_content(
                model=model,
                contents=f"Analyze this email:\n\n{text}",
                config=config,
            )
            break
        except errors.APIError as exc:
            if exc.code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(BACKOFF_SECONDS * attempt)

    # The SDK validates into our model for us; fall back to the raw JSON if a
    # response comes back unparsed (e.g. a truncated generation).
    if isinstance(response.parsed, EmailAnalysis):
        result = response.parsed
    elif response.text:
        result = EmailAnalysis.model_validate_json(response.text)
    else:
        raise RuntimeError(f"{model} returned an empty response.")

    if use_cache:
        _write_cache(cache_file, result)
    return result
