"""Streamlit dashboard for the Summit Roofing inbox agent.

Run with:  streamlit run app.py

Opening the dashboard never calls the API. Sample emails are read from the
on-disk cache in ``.cache/``; analysis only happens when you explicitly paste
an email or press Analyze, which keeps the free-tier daily quota intact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.classifier import active_model, analyze_email, is_cached
from src.models import Category, EmailAnalysis, Urgency

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
DATA_DIR = ROOT / "data"
STATUS_FILE = DATA_DIR / "status.json"
PASTED_FILE = DATA_DIR / "pasted.json"

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"

URGENCY_RANK = {Urgency.HIGH: 0, Urgency.MEDIUM: 1, Urgency.LOW: 2}
URGENCY_COLOR = {Urgency.HIGH: "#dc2626", Urgency.MEDIUM: "#d97706", Urgency.LOW: "#64748b"}
URGENCY_DOT = {
    Urgency.HIGH: ":red[●]",
    Urgency.MEDIUM: ":orange[●]",
    Urgency.LOW: ":grey[●]",
}

STATUS_LABEL = {APPROVED: "Sent", REJECTED: "Rejected", PENDING: "Needs review"}
STATUS_COLOR = {APPROVED: "#059669", REJECTED: "#64748b", PENDING: "#2563eb"}
ARCHIVED_COLOR = "#94a3b8"

CSS = """
<style>
  /* Inbox rows: make the native buttons read as list items, not form controls. */
  div[class*="st-key-open_"] button {
      justify-content: flex-start;
      text-align: left;
      font-weight: 600;
      border: none;
      background: transparent;
      padding: 0.1rem 0;
  }
  div[class*="st-key-open_"] button:hover { background: transparent; color: #1d4ed8; }
  div[class*="st-key-open_"] button p { font-size: 0.95rem; }

  .badge {
      display: inline-block;
      padding: 0.08rem 0.5rem;
      margin-right: 0.3rem;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      color: #fff;
      white-space: nowrap;
  }
  .badge-outline {
      background: #f1f5f9;
      color: #334155;
      border: 1px solid #cbd5e1;
  }
  .brand { line-height: 1.15; margin-bottom: 0.1rem; }
  .brand-name { font-size: 1.35rem; font-weight: 700; color: #0f172a; }
  .brand-sub { font-size: 0.78rem; color: #64748b; text-transform: uppercase;
               letter-spacing: 0.08em; }
  .field-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
                 color: #64748b; font-weight: 600; }
  .field-value { font-size: 0.92rem; color: #0f172a; }
  .field-empty { font-size: 0.92rem; color: #94a3b8; font-style: italic; }
</style>
"""


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def _read_json(path: Path, fallback):
    """Load JSON, falling back if the file is absent or was hand-edited badly."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_status() -> dict:
    return _read_json(STATUS_FILE, {})


def set_status(email_id: str, status: str, draft: str | None = None) -> None:
    """Record a decision for one email, keeping any edited draft alongside it."""
    store = load_status()
    entry = store.get(email_id, {})
    entry["status"] = status
    entry["updated"] = datetime.now().isoformat(timespec="seconds")
    if draft is not None:
        entry["draft"] = draft
    store[email_id] = entry
    _write_json(STATUS_FILE, store)


def load_pasted() -> list[str]:
    pasted = _read_json(PASTED_FILE, [])
    return pasted if isinstance(pasted, list) else []


def add_pasted(text: str) -> None:
    pasted = load_pasted()
    if text not in pasted:
        pasted.append(text)
        _write_json(PASTED_FILE, pasted)


# --------------------------------------------------------------------------
# Inbox assembly
# --------------------------------------------------------------------------

def email_id(text: str) -> str:
    """Stable short id for an email, independent of where it came from."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def header_field(text: str, field: str) -> str:
    """Pull a header such as Subject or From out of the raw email."""
    prefix = f"{field.lower()}:"
    for line in text.splitlines()[:12]:
        if line.lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def subject_of(text: str) -> str:
    subject = header_field(text, "Subject")
    if subject:
        return subject
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return (first[:60] + "...") if len(first) > 60 else (first or "(no subject)")


def load_inbox() -> list[dict]:
    """Build the inbox from sample files plus anything pasted in.

    Analysis is read from the disk cache only: an email with no cached result
    is listed as un-analyzed rather than silently costing a request.
    """
    sources = [(path.name, path.read_text(encoding="utf-8")) for path in sorted(SAMPLES_DIR.glob("*.txt"))]
    sources += [("pasted", text) for text in load_pasted()]

    status_store = load_status()
    inbox = []
    for origin, text in sources:
        analysis = analyze_email(text) if is_cached(text) else None
        item_id = email_id(text)
        saved = status_store.get(item_id, {})
        inbox.append(
            {
                "id": item_id,
                "origin": origin,
                "text": text,
                "subject": subject_of(text),
                "sender": header_field(text, "From"),
                "analysis": analysis,
                "status": saved.get("status", PENDING),
                "draft": saved.get("draft"),
            }
        )
    return inbox


def sort_key(item: dict):
    """High urgency first; un-analyzed emails sink to the bottom."""
    analysis = item["analysis"]
    if analysis is None:
        return (3, item["subject"])
    return (URGENCY_RANK[analysis.urgency], item["subject"])


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

def badge(label: str, color: str | None = None) -> str:
    if color is None:
        return f'<span class="badge badge-outline">{label}</span>'
    return f'<span class="badge" style="background:{color}">{label}</span>'


def category_label(analysis: EmailAnalysis) -> str:
    return analysis.category.value.replace("_", " ").title()


def is_spam(item: dict) -> bool:
    """Spam is archived on arrival and never counts as work to review."""
    analysis = item["analysis"]
    return analysis is not None and analysis.category is Category.SPAM


def item_badges(item: dict) -> str:
    analysis = item["analysis"]
    if analysis is None:
        return badge("Not analyzed")
    parts = [
        badge(category_label(analysis)),
        badge(analysis.urgency.value.upper(), URGENCY_COLOR[analysis.urgency]),
    ]
    if is_spam(item):
        parts.append(badge("Archived", ARCHIVED_COLOR))
    elif item["status"] != PENDING:
        parts.append(badge(STATUS_LABEL[item["status"]], STATUS_COLOR[item["status"]]))
    return "".join(parts)


def field(label: str, value: str | None) -> None:
    st.markdown(f'<div class="field-label">{label}</div>', unsafe_allow_html=True)
    if value:
        st.markdown(f'<div class="field-value">{value}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="field-empty">not provided</div>', unsafe_allow_html=True)


def run_analysis(text: str) -> bool:
    """Analyze an email, surfacing API failures instead of crashing the app."""
    try:
        with st.spinner(f"Analyzing with {active_model()}..."):
            analyze_email(text)
        return True
    except Exception as exc:
        st.error(f"Could not analyze this email ({type(exc).__name__}). {exc}")
        return False


# --------------------------------------------------------------------------
# Panes
# --------------------------------------------------------------------------

def render_sidebar(inbox: list[dict]) -> tuple[list[str], list[str]]:
    with st.sidebar:
        st.markdown(
            '<div class="brand">'
            '<div class="brand-name">Summit Roofing</div>'
            '<div class="brand-sub">Inbox Agent</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption("AI triage for inbound customer email")
        st.divider()

        analyzed = [i for i in inbox if i["analysis"]]
        urgent = sum(1 for i in analyzed if i["analysis"].urgency is Urgency.HIGH)
        waiting = sum(
            1
            for i in analyzed
            if i["status"] == PENDING and not is_spam(i)
        )

        left, right = st.columns(2)
        left.metric("Needs review", waiting)
        right.metric("High urgency", urgent)

        st.divider()
        st.markdown("**Filters**")
        categories = sorted({category_label(i["analysis"]) for i in analyzed})
        picked_categories = st.multiselect("Category", categories, default=categories)
        urgencies = [u.value for u in (Urgency.HIGH, Urgency.MEDIUM, Urgency.LOW)]
        picked_urgencies = st.multiselect("Urgency", urgencies, default=urgencies)

        st.divider()
        st.caption(f"Model: `{active_model()}`")
        st.caption(
            f"{len(analyzed)}/{len(inbox)} emails loaded from the local cache. "
            "Opening this dashboard costs no API quota."
        )
    return picked_categories, picked_urgencies


def render_compose() -> None:
    with st.expander("Paste a new email", expanded=False):
        text = st.text_area(
            "Raw email",
            height=180,
            placeholder="From: customer@example.com\nSubject: ...\n\nMessage body...",
            label_visibility="collapsed",
            key="compose_box",
        )
        st.caption("Analyzing a new email uses one API request.")
        if st.button("Analyze and add to inbox", type="primary"):
            if not text.strip():
                st.warning("Paste an email first.")
            elif run_analysis(text):
                add_pasted(text)
                st.session_state.selected = email_id(text)
                st.rerun()


def render_row(item: dict) -> None:
    """One clickable inbox row."""
    with st.container(border=True):
        is_open = item["id"] == st.session_state.get("selected")
        label = (
            f"{URGENCY_DOT[item['analysis'].urgency]} "
            if item["analysis"]
            else ":grey[○] "
        )
        if st.button(
            f"{label}{item['subject']}",
            key=f"open_{item['id']}",
            width="stretch",
            type="primary" if is_open else "secondary",
        ):
            st.session_state.selected = item["id"]
            st.rerun()

        st.markdown(item_badges(item), unsafe_allow_html=True)
        analysis = item["analysis"]
        if analysis:
            who = analysis.customer.name or item["sender"] or "Unknown sender"
            st.caption(f"**{who}** — {analysis.summary}")
        else:
            st.caption(item["sender"] or item["origin"])
            if st.button("Analyze", key=f"run_{item['id']}"):
                if run_analysis(item["text"]):
                    st.rerun()


def render_list(items: list[dict]) -> None:
    st.markdown("##### Inbox")
    if not items:
        st.info("No emails match the current filters.")
        return

    # Spam is archived on arrival: out of the main flow, but still reachable.
    active = [i for i in items if not is_spam(i)]
    spam = [i for i in items if is_spam(i)]

    if active:
        for item in active:
            render_row(item)
    else:
        st.info("Nothing to review.")

    if spam:
        with st.expander(f"Spam ({len(spam)})", expanded=False):
            st.caption("Archived automatically. No reply drafted.")
            for item in spam:
                render_row(item)


def render_detail(item: dict | None) -> None:
    st.markdown("##### Details")
    if item is None:
        st.info("Select an email from the inbox to review its draft reply.")
        return

    analysis = item["analysis"]
    with st.container(border=True):
        st.markdown(f"**{item['subject']}**")
        st.markdown(item_badges(item), unsafe_allow_html=True)

        if analysis is None:
            st.warning("This email has not been analyzed yet.")
            with st.expander("Original email", expanded=True):
                st.code(item["text"], language=None, wrap_lines=True)
            if st.button("Analyze now", type="primary", key=f"detail_run_{item['id']}"):
                if run_analysis(item["text"]):
                    st.rerun()
            return

        if item["status"] == APPROVED:
            st.success("Sent — this reply was approved.")
        elif item["status"] == REJECTED:
            st.warning("Rejected — this draft was discarded.")

        st.markdown(f'<div class="field-label">Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{analysis.summary}</div>', unsafe_allow_html=True)
        st.write("")

        customer = analysis.customer
        st.markdown("**Extracted customer data**")
        col_a, col_b = st.columns(2)
        with col_a:
            field("Name", customer.name)
            field("Phone", customer.phone)
        with col_b:
            field("Email", customer.email)
            field("Address", customer.address)
        st.write("")

        st.markdown("**Missing information**")
        if analysis.missing_info:
            for missing in analysis.missing_info:
                st.markdown(f"- {missing}")
        else:
            st.markdown('<div class="field-empty">Nothing missing.</div>', unsafe_allow_html=True)
        st.write("")

        with st.expander("Original email"):
            st.code(item["text"], language=None, wrap_lines=True)

        st.markdown("**Draft reply**")
        draft = st.text_area(
            "Draft reply",
            value=item["draft"] if item["draft"] is not None else analysis.draft_reply,
            height=260,
            label_visibility="collapsed",
            key=f"draft_{item['id']}",
        )

        approve, reject, save = st.columns([1, 1, 1])
        if approve.button("Approve", type="primary", width="stretch", key=f"ok_{item['id']}"):
            set_status(item["id"], APPROVED, draft)
            st.rerun()
        if reject.button("Reject", width="stretch", key=f"no_{item['id']}"):
            set_status(item["id"], REJECTED, draft)
            st.rerun()
        if save.button("Save draft", width="stretch", key=f"save_{item['id']}"):
            set_status(item["id"], item["status"], draft)
            st.toast("Draft saved.")

        st.caption("Approving marks the reply as sent. Nothing is emailed yet.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Summit Roofing — Inbox Agent",
        page_icon="🏠",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    inbox = load_inbox()
    picked_categories, picked_urgencies = render_sidebar(inbox)

    st.title("Inbox Agent")
    st.caption(
        "Inbound email, triaged and drafted automatically. Review each draft, "
        "then approve or reject it."
    )
    render_compose()

    def visible(item: dict) -> bool:
        analysis = item["analysis"]
        if analysis is None:
            return True  # never hide work that still needs doing
        return (
            category_label(analysis) in picked_categories
            and analysis.urgency.value in picked_urgencies
        )

    items = sorted([i for i in inbox if visible(i)], key=sort_key)

    if st.session_state.get("selected") not in {i["id"] for i in items}:
        openable = [i for i in items if not is_spam(i)] or items
        st.session_state.selected = openable[0]["id"] if openable else None
    current = next((i for i in items if i["id"] == st.session_state.selected), None)

    list_col, detail_col = st.columns([1, 1.45], gap="large")
    with list_col:
        render_list(items)
    with detail_col:
        render_detail(current)


if __name__ == "__main__":
    main()
