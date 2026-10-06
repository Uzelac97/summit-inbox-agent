"""Gmail access: OAuth and reading.

This is the only module that knows Gmail exists, the same way
:mod:`src.classifier` is the only module that knows Anthropic exists.

Run it directly to check the setup end to end::

    python -m src.gmail_client

That authenticates (opening a browser the first time), prints which mailbox
it reached, and lists recent subjects. It writes nothing to the mailbox.
"""

from __future__ import annotations

import base64
import re
from email.message import EmailMessage
from email.utils import parseaddr
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterator, List

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.labels import SYSTEM_LABELS
from src.models import AttachmentInfo

ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = ROOT / "credentials.json"
TOKEN_FILE = ROOT / "token.json"

# gmail.modify covers reading and labelling; gmail.compose covers creating
# drafts. gmail.send is deliberately absent, so the OAuth grant itself makes it
# impossible for this agent to send mail as the mailbox owner. Changing this
# list invalidates token.json and forces a fresh consent.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
]

DEFAULT_QUERY = "in:inbox"

_SETUP_HELP = f"""\
{CREDENTIALS_FILE.name} was not found at {CREDENTIALS_FILE}

To create it:
  1. console.cloud.google.com, signed in as the mailbox owner
  2. APIs & Services > Credentials > Create Credentials > OAuth client ID
  3. Application type: Desktop app
  4. Download JSON, save it to the path above, keep the filename

The downloaded file is named client_secret_<long-id>.apps.googleusercontent.com
.json, so it has to be renamed. It is already gitignored."""


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------

def normalize_address(value: str) -> str:
    """Reduce an address to a comparable form.

    Gmail ignores dots in a gmail.com local part and anything after a ``+``, so
    ``summit.roofing.demo11+n8n@gmail.com`` and ``summitroofingdemo11@gmail.com``
    are one mailbox. Comparing raw header strings would let a plus-addressed
    notification past the self-sent filter.
    """
    _, address = parseaddr(value)
    address = address.strip().lower()
    if "@" not in address:
        return address

    local, _, domain = address.partition("@")
    local = local.split("+", 1)[0]
    if domain in {"gmail.com", "googlemail.com"}:
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}"


# --------------------------------------------------------------------------
# MIME
# --------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.S | re.I)
_BREAK_RE = re.compile(r"(?i)<\s*(br|/p|/div|/tr|/li|/h[1-6]|/table)\s*/?\s*>")
_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Flatten an HTML body to something worth classifying.

    Deliberately crude - no parser dependency. Block-level tags become line
    breaks so paragraphs survive, scripts and styles are dropped so their
    contents are not mistaken for prose, and entities are unescaped. Marketing
    mail still flattens untidily, but a misread spam body costs a wasted
    classification, not a wrong reply.
    """
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = unescape(text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


def decode_part_data(data: str) -> str:
    """Decode Gmail's base64url part payload, tolerating missing padding."""
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode(
        "utf-8", errors="replace"
    )


def walk_parts(payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield every part of a message payload, depth first."""
    pending = [payload]
    while pending:
        part = pending.pop(0)
        yield part
        pending = list(part.get("parts", [])) + pending


def extract_body(payload: Dict[str, Any]) -> str:
    """Best plain-text body for a message.

    text/plain wins outright; HTML is flattened only when no plain part exists,
    which is the usual shape of marketing mail.
    """
    plain: List[str] = []
    html: List[str] = []

    for part in walk_parts(payload):
        if part.get("filename"):
            continue  # an attachment, not the message body
        data = part.get("body", {}).get("data", "")
        if not data:
            continue
        mime = part.get("mimeType", "")
        if mime == "text/plain":
            plain.append(decode_part_data(data))
        elif mime == "text/html":
            html.append(decode_part_data(data))

    if plain:
        return "\n".join(plain).strip()
    if html:
        return html_to_text("\n".join(html))
    return ""


def extract_attachments(payload: Dict[str, Any]) -> List[AttachmentInfo]:
    """Names and types of attached files. The files themselves are never read."""
    found = []
    for part in walk_parts(payload):
        filename = part.get("filename")
        if filename:
            found.append(
                AttachmentInfo(
                    filename=filename,
                    mime_type=part.get("mimeType") or "application/octet-stream",
                )
            )
    return found


# --------------------------------------------------------------------------
# Replies
# --------------------------------------------------------------------------

def reply_subject(subject: str) -> str:
    """``Re:`` the subject without stacking a second one."""
    subject = subject.strip()
    if not subject:
        return "Re:"
    if subject.lower().startswith("re:"):
        return subject
    return f"Re: {subject}"


def build_reply_mime(
    to: str,
    subject: str,
    body: str,
    in_reply_to: str = "",
    references: str = "",
) -> str:
    """Build a base64url RFC822 reply, ready for drafts.create.

    Threading needs three things to agree, and Gmail's own threadId is only
    one of them: ``In-Reply-To`` must quote the parent's Message-ID, and
    ``References`` must carry the chain so other mail clients thread it too.
    A draft with the right threadId but no headers looks correct through the
    API and shows up detached in a non-Gmail client.
    """
    message = EmailMessage()
    message["To"] = to
    message["Subject"] = reply_subject(subject)
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
        # The parent's own chain, then the parent itself, oldest first.
        chain = f"{references} {in_reply_to}".strip()
        message["References"] = " ".join(chain.split())
    # utf-8 so a German reply keeps its umlauts, and quoted-printable rather
    # than the default 8bit: 8bit needs the 8BITMIME extension to survive a
    # strict SMTP hop, while quoted-printable travels anywhere and leaves the
    # ASCII parts readable.
    message.set_content(body, charset="utf-8", cte="quoted-printable")
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def _load_saved_credentials() -> Credentials | None:
    """Return stored credentials, refreshing them if they have expired."""
    if not TOKEN_FILE.exists():
        return None

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    except (OSError, ValueError):
        # Truncated or hand-edited. Consenting again is cheap; a crash with a
        # JSON error here would send someone hunting for the wrong problem.
        TOKEN_FILE.unlink(missing_ok=True)
        return None

    if creds.valid:
        return creds

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            # While the OAuth app is in Testing, refresh tokens expire after
            # seven days, and revoking access has the same effect. Neither is a
            # bug, so drop the dead token and consent again.
            TOKEN_FILE.unlink(missing_ok=True)
            return None
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        return creds

    return None


def get_credentials() -> Credentials:
    """Load credentials, running the desktop consent flow if needed."""
    creds = _load_saved_credentials()
    if creds is not None:
        return creds

    if not CREDENTIALS_FILE.exists():
        raise RuntimeError(_SETUP_HELP)

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


class GmailClient:
    """Thin wrapper over the Gmail API.

    Reading, and applying labels. Draft creation arrives with the step that
    needs it. Sending is impossible: the scopes above do not permit it.
    """

    def __init__(self, service: Any | None = None) -> None:
        self._service = service
        self._label_ids: Dict[str, str] | None = None

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = build(
                "gmail", "v1", credentials=get_credentials(), cache_discovery=False
            )
        return self._service

    def address(self) -> str:
        """The mailbox these credentials reached, for confirming the account."""
        profile = self.service.users().getProfile(userId="me").execute()
        return profile.get("emailAddress", "unknown")

    def list_message_ids(
        self, query: str = DEFAULT_QUERY, limit: int = 25
    ) -> List[str]:
        """Message ids matching ``query``, newest first."""
        response = (
            self.service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max(1, min(limit, 500)))
            .execute()
        )
        return [item["id"] for item in response.get("messages", [])]

    def get_message(self, message_id: str) -> Dict[str, Any]:
        """Fetch a whole message, body and all."""
        return (
            self.service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )

    # ---- labels ---------------------------------------------------------

    def label_ids(self, refresh: bool = False) -> Dict[str, str]:
        """Map of label name to label id, fetched once and cached."""
        if self._label_ids is None or refresh:
            response = self.service.users().labels().list(userId="me").execute()
            self._label_ids = {
                item["name"]: item["id"] for item in response.get("labels", [])
            }
        return self._label_ids

    def ensure_label(self, name: str) -> str:
        """Return the id of ``name``, creating the label if it is missing.

        Gmail creates the parent of a nested name implicitly, so asking for
        ``agent/quote`` also produces the ``agent`` group in the sidebar. A
        label that already exists has its colour brought up to date, so a
        palette change reaches labels created by an earlier run.
        """
        from src.labels import color_for

        color = color_for(name)
        existing = self.label_ids().get(name)
        if existing:
            if color:
                self.service.users().labels().patch(
                    userId="me", id=existing, body={"color": color}
                ).execute()
            return existing

        body: Dict[str, Any] = {
            "name": name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        }
        if color:
            body["color"] = color

        created = (
            self.service.users().labels().create(userId="me", body=body).execute()
        )
        self.label_ids()[name] = created["id"]
        return created["id"]

    def ensure_labels(self, names: List[str]) -> Dict[str, str]:
        """Create any of ``names`` that do not exist yet, and colour them all."""
        return {name: self.ensure_label(name) for name in names}

    def add_labels(self, message_id: str, names: List[str]) -> None:
        """Add labels to a message."""
        if not names:
            return
        ids = [
            name if name in SYSTEM_LABELS else self.ensure_label(name)
            for name in names
        ]
        self.service.users().messages().modify(
            userId="me", id=message_id, body={"addLabelIds": ids}
        ).execute()

    def remove_labels(self, message_id: str, names: List[str]) -> None:
        """Remove labels from a message, ignoring any that do not exist."""
        ids = [self.label_ids()[n] for n in names if n in self.label_ids()]
        if not ids:
            return
        self.service.users().messages().modify(
            userId="me", id=message_id, body={"removeLabelIds": ids}
        ).execute()

    # ---- drafts ---------------------------------------------------------

    def create_draft(self, thread_id: str, raw_message: str) -> str:
        """Save a draft into ``thread_id`` and return its id. Never sends.

        The scopes in use cannot send, so the worst a bug here can do is leave
        an unwanted draft in the mailbox.
        """
        draft = (
            self.service.users()
            .drafts()
            .create(
                userId="me",
                body={"message": {"threadId": thread_id, "raw": raw_message}},
            )
            .execute()
        )
        return draft["id"]

    def get_headers(self, message_id: str, names: List[str]) -> Dict[str, str]:
        """Fetch only the named headers, leaving the body on the server."""
        message = (
            self.service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=names,
            )
            .execute()
        )
        headers = message.get("payload", {}).get("headers", [])
        return {h["name"].lower(): h["value"] for h in headers}


def main() -> int:
    """Authenticate and list recent subjects. Reads only."""
    try:
        client = GmailClient()
        address = client.address()
    except RuntimeError as exc:
        print(exc)
        return 1
    except HttpError as exc:
        print(f"Gmail refused the request: {exc}")
        if exc.resp.status == 403:
            print(
                "\nA 403 here usually means the Gmail API is not enabled on the "
                "project, or the signed-in account is not listed under Test users."
            )
        return 1

    print(f"Mailbox : {address}")
    print(f"Scopes  : {', '.join(s.rsplit('/', 1)[-1] for s in SCOPES)}")
    print(f"Query   : {DEFAULT_QUERY}\n")

    message_ids = client.list_message_ids(limit=25)
    if not message_ids:
        print("No messages matched. An empty inbox is a valid result.")
        return 0

    print(f"{len(message_ids)} message(s), newest first:\n")
    for index, message_id in enumerate(message_ids, start=1):
        try:
            headers = client.get_headers(message_id, ["Subject"])
            subject = headers.get("subject", "(no subject)")
        except HttpError as exc:
            # One unreadable message must not end the run.
            subject = f"<could not read: {exc.resp.status}>"
        print(f"  [{index:>2}] {subject}")

    print("\nNothing was written to the mailbox.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
