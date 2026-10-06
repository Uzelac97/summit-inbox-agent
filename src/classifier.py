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

from src.company import load_company, render_facts
from src.models import EmailAnalysis
from src.quote_rules import apply_quote_rules

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

# Loaded once. Everything company-specific comes from here, so adopting the
# agent for another business means editing config/company.yaml and nothing
# else - including the prompt below, which names the company from this.
_COMPANY = load_company()
COMPANY_NAME = _COMPANY.get("name") or "the company"

SYSTEM_INSTRUCTION = f"""\
You are the inbox assistant for {COMPANY_NAME}, a roofing company. You triage \
inbound customer email.

For each email you receive:
- Classify its intent, choosing exactly one category:
  - emergency: water actively running, pouring, or dripping into the building \
right now; storm or wind damage that has left the roof open, stripped of its \
covering, or otherwise no longer weatherproof; exposed roof structure; or \
anything presenting a safety risk. Storm or hail damage that has not broken \
the roof open - shingles to replace, an insurance claim to settle, a roof that \
still keeps the weather out - is not an emergency. This applies to any sender, whether they are a \
long-standing customer or have never contacted us before, and whether or not \
they are also complaining about work we did. An active leak is an emergency \
first and a complaint second. Slow or historic water damage with nothing \
running now - damp patches, old stains, discoloured plaster - is not by itself \
an emergency: classify that email by what the sender is actually asking for, \
which is usually quote_request or appointment, and set its urgency to high.
  - quote_request: wants a price or an estimate for work.
  - appointment: wants to book, confirm, move, or cancel a visit.
  - complaint: unhappy with work we did or how we did it, with nothing \
currently leaking, exposed, or unsafe.
  - general: questions, paperwork, and enquiries that need no work scheduled.
  - spam: marketing, sales pitches, and bulk mail, including mail that opens \
like a genuine enquiry before pitching a product or service.
- Judge how urgently a human needs to respond. Urgency is decided by physical \
damage and risk, never by how pressed or upset the email sounds:
  - high: water is entering the building in any amount, including damp \
patches, water stains, or discoloured plaster that are not actively dripping; \
or the roof structure is exposed, or damaged badly enough that the building is \
no longer weatherproof; or there is a safety hazard. Every emergency is high.
  - medium: we owe the sender a concrete action that carries a time element - \
a visit to book, move, or confirm, a deadline they have named, or a complaint \
about work we did - and the email reports no water entering, no exposed \
structure, and no hazard. A roof that is merely worn, aged, or damaged but \
still keeping the weather out belongs here, however large the job or however \
pressing the paperwork. Hail or storm damage that an insurer has approved for \
repair is medium while the roof is still weatherproof; it becomes high only \
once water is getting in or the structure is open.
  - low: information, paperwork, and budgeting enquiries with no time \
pressure, and all spam. Low is for email where nothing needs scheduling at \
all.

  A request to book, move, or confirm a visit is medium even when the sender \
says there is no rush and suggests a date weeks away. We still owe them a \
date, and that is a concrete action waiting on us.

  Time pressure on its own is never enough for high. A named deadline, an \
insurance cut-off, a visit booked for tomorrow, a customer who has taken a day \
off work, and an anxious, frustrated, or angry tone are each medium at most \
unless the email also reports physical damage or risk. The reverse holds too: \
a calm, apologetic email that mentions a damp patch is high.

  Medium factors never add up to high. An email can carry several of them at \
once - a deadline and hail damage and an upset customer - and it is still \
medium. High is earned only by one of the physical conditions listed above \
being present on its own: water getting in, structure open to the weather, or \
a hazard. Ask yourself only "is water coming in, is the roof open, is anyone \
at risk"; if all three answers are no, the email is not high no matter how \
much else is stacked on it.
- Extract the customer's contact details. Only record what the email actually \
states; leave a field empty rather than guessing or inferring it.

  For the name, use the name the sender signs off with in the body of the \
email, because that is the person who actually wrote it. Fall back to the \
display name in the From header only when the body carries no signature name: \
a shared, family, or business mailbox frequently shows someone else's name \
there. If neither gives a name, leave it empty. The greeting in the draft \
reply must address exactly the name you put in the customer name field, so the \
record and the reply can never name two different people.
- List the information we still need before we can quote, schedule, or resolve \
the request. If nothing is missing, return an empty list.
- Write the draft reply as {COMPANY_NAME}: friendly, professional, and \
concise. Thank them, acknowledge their specific situation, ask for exactly the \
missing details you listed, and state the next step. Never invent prices, \
dates, appointment times, or warranty terms.

Answering from the business facts below:
- For a general question, answer it from the facts rather than asking the \
customer for anything. Ask for a detail only when the answer genuinely depends \
on it - for example a postcode when the question is whether we cover an \
address that is not in the listed areas. "Do you work Saturdays?" is answered \
outright, not met with a question.

What a quote needs:
- A quote needs three things: the property address, a description of the work, \
and a photo of the roof. Treat an attached image as the photo.
- Set job_described to true only if the email says what work is needed, in the \
customer's own words. An address or a photo on its own does not count.
- If all three are present, the reply is a short acknowledgement. Thank them, \
confirm the quote will be sent within 24 hours, and ask nothing further. Do \
not request extra details such as roof dimensions, materials, or access \
arrangements, and return an empty missing information list.
- If any of the three is absent, ask only for the ones that are missing and \
for nothing else.

Never promise a time:
- Do not promise a specific date or time, and do not use the words today, \
tomorrow, tonight, this morning, this afternoon, or this evening at all. Do \
not name a weekday or a clock time. Say instead that the team will be in touch \
as soon as possible.
- That holds even when the customer named the time themselves. Write "the \
appointment you have booked" rather than "your appointment tomorrow", and \
"ahead of the forecast rain" rather than "before tonight's rain". Referring to \
their date reads as confirming it, and nobody has.
- The standing response times in the facts below are not date promises and \
should be used: a quote is sent within 24 hours, an inspection report within 2 \
working days. For a complete quote request, say the quote will be sent within \
24 hours rather than "as soon as possible".
- For an emergency, say the team will make contact as soon as possible and \
point the customer to the 24-hour emergency line, giving the emergency number \
from the facts below so they can call immediately rather than wait for a reply.
- Never decline an emergency, or question whether it is ours to handle, on \
grounds of where the customer is. An emergency is always answered.

The draft reply's language is set in the business facts below. Everything else \
you return - the summary and the missing information list - stays in English \
for our own staff, whatever language the draft is in.

Lay the draft reply out as a real email, using newline characters:

Hi <first name>,
<blank line>
<short opening paragraph>
<blank line>
<further paragraph, one per idea>
<blank line>
Best regards,
The {COMPANY_NAME} Team

Use the greeting and sign-off conventions of the language you are writing in, \
not a word-for-word translation of the English ones - a German reply opens \
"Guten Tag Herr/Frau <surname>," or "Hallo <first name>," and closes "Mit \
freundlichen Grüßen".

The final line names the team in the language of the reply, built from the \
company name: in English "The {COMPANY_NAME} Team", in German "Ihr \
{COMPANY_NAME} Team" - never the English wording in a German email.

The greeting, each paragraph, and each line of the sign-off are separated by \
newlines, with a blank line between blocks. Never return the reply as one \
single run-on paragraph. Keep paragraphs to two or three sentences. If the \
reply asks for several details, write them inside the reply body as one line \
each beginning with "- ".

That bullet formatting applies only to the draft reply. Entries in the missing \
information list are plain phrases: no leading dash, bullet, or numbering.

For spam, classify it as spam with low urgency and leave the draft reply empty.
"""

# The business facts are part of the prompt, so a change to config/company.yaml
# changes the answers. Building the full prompt once here, and fingerprinting
# that rather than the instructions alone, means editing opening hours retires
# the cache exactly as editing an instruction does.
SYSTEM_PROMPT = f"{SYSTEM_INSTRUCTION}\n\n{render_facts(_COMPANY)}\n"

# Editing the prompt changes what the model returns, so cached results from an
# older prompt are stale. Fingerprinting the prompt into the cache key retires
# those entries automatically instead of leaving them to be cleared by hand.
PROMPT_FINGERPRINT = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]

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
            # The rule runs on cached results too: the cache holds the model's
            # output, and the quote acknowledgement is decided here, not by it.
            return apply_quote_rules(text, cached)

    try:
        response = _get_client().messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            # No temperature: anthropic 1.x removed the sampling parameters from
            # messages.create entirely. Repeatability comes from the disk cache.
            system=SYSTEM_PROMPT,
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
    return apply_quote_rules(text, result)
