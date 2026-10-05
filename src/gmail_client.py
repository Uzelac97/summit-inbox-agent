"""Gmail access: OAuth and reading.

This is the only module that knows Gmail exists, the same way
:mod:`src.classifier` is the only module that knows Anthropic exists.

Run it directly to check the setup end to end::

    python -m src.gmail_client

That authenticates (opening a browser the first time), prints which mailbox
it reached, and lists recent subjects. It writes nothing to the mailbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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

    Only read operations exist so far. Labelling and draft creation arrive with
    the steps that need them, so nothing here can yet change a mailbox.
    """

    def __init__(self, service: Any | None = None) -> None:
        self._service = service

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
