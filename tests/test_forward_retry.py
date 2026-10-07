"""Unit tests for the forward-failure retry mechanism.

Covers src/forward.py's local attempt counter and the label bookkeeping in
src/processor.py (Processor.process / retry_forward_failures). Everything is
faked: no Gmail API, no Anthropic key, no network.

Run with:  python -m unittest tests.test_forward_retry
"""

from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import forward as forward_module  # noqa: E402
from src.email_source import GmailSource  # noqa: E402
from src.forward import FAILED, FORWARDED, MAX_FORWARD_ATTEMPTS, Forwarded  # noqa: E402
from src.labels import FORWARD_FAILED_LABEL, PROCESSED_LABEL  # noqa: E402
from src.models import Category, Customer, EmailAnalysis, Urgency  # noqa: E402
from src.processor import Processor  # noqa: E402


def make_analysis() -> EmailAnalysis:
    return EmailAnalysis(
        category=Category.QUOTE_REQUEST,
        urgency=Urgency.LOW,
        customer=Customer(name="Test Customer"),
        summary="wants a quote",
        missing_info=[],
        job_described=True,
        draft_reply="Hi there,\n\nThanks.\n\nBest regards,\nThe Summit Roofing Team\n",
    )


def make_message(message_id: str, body: str = "Hello") -> dict:
    """A minimal Gmail API message dict, enough for GmailSource to parse."""
    encoded = base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")
    payload = {
        "headers": [
            {"name": "From", "value": "cust@example.com"},
            {"name": "Subject", "value": "Re: Quote"},
        ],
        "mimeType": "text/plain",
        "body": {"data": encoded},
    }
    return {"id": message_id, "threadId": f"thread-{message_id}", "payload": payload}


class FakeClient:
    """Records label changes and drafts; no real Gmail API involved."""

    def __init__(self, message: dict) -> None:
        self._message = message
        self.added: list[tuple[str, tuple[str, ...]]] = []
        self.removed: list[tuple[str, tuple[str, ...]]] = []

    def list_message_ids(self, query: str, limit: int) -> list[str]:
        return [self._message["id"]]

    def get_message(self, message_id: str) -> dict:
        return self._message

    def add_labels(self, message_id: str, names: list[str]) -> None:
        self.added.append((message_id, tuple(names)))

    def remove_labels(self, message_id: str, names: list[str]) -> None:
        self.removed.append((message_id, tuple(names)))

    def create_draft(self, thread_id: str, raw_message: str) -> str:
        return "draft-1"


class FakeForwarder:
    """Returns one scripted outcome per call; errors if called too often."""

    def __init__(self, outcomes: list[Forwarded]) -> None:
        self._outcomes = list(outcomes)

    def forward(self, email, analysis) -> Forwarded:
        return self._outcomes.pop(0)


def make_email(client: FakeClient):
    return GmailSource(client=client, query="x", skip_own=False).fetch()[0]


class ForwardAttemptsTests(unittest.TestCase):
    """The local JSON-backed failure counter in src/forward.py."""

    def setUp(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        patcher = patch.object(
            forward_module, "ATTEMPTS_FILE", Path(tmpdir.name) / "forward_attempts.json"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_round_trip(self) -> None:
        self.assertEqual(forward_module.forward_attempts("m1"), 0)
        self.assertEqual(forward_module.record_forward_attempt("m1"), 1)
        self.assertEqual(forward_module.record_forward_attempt("m1"), 2)
        self.assertEqual(forward_module.forward_attempts("m1"), 2)
        forward_module.clear_forward_attempts("m1")
        self.assertEqual(forward_module.forward_attempts("m1"), 0)

    def test_clear_of_untracked_message_is_a_no_op(self) -> None:
        forward_module.clear_forward_attempts("never-seen")  # must not raise


class ProcessorForwardFailureTests(unittest.TestCase):
    """agent/processed vs. agent/forward-failed, and the retry step."""

    def setUp(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        patcher = patch.object(
            forward_module, "ATTEMPTS_FILE", Path(tmpdir.name) / "forward_attempts.json"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _processor(self, message_id: str, forwarder: FakeForwarder):
        client = FakeClient(make_message(message_id))
        processor = Processor(client, forwarder=forwarder)
        patcher = patch("src.processor.analyze_email", return_value=make_analysis())
        patcher.start()
        self.addCleanup(patcher.stop)
        return processor, client

    def test_failed_forward_withholds_processed_and_labels_forward_failed(self) -> None:
        processor, client = self._processor("m1", FakeForwarder([Forwarded(FAILED, "boom")]))
        email = make_email(client)

        result = processor.process(email)

        self.assertTrue(result.ok)
        self.assertIn(FORWARD_FAILED_LABEL, result.labels_applied)
        self.assertNotIn(PROCESSED_LABEL, result.labels_applied)
        self.assertEqual(forward_module.forward_attempts("m1"), 1)

    def test_successful_forward_marks_processed_not_forward_failed(self) -> None:
        processor, client = self._processor("m2", FakeForwarder([Forwarded(FORWARDED, "ok")]))
        email = make_email(client)

        result = processor.process(email)

        self.assertIn(PROCESSED_LABEL, result.labels_applied)
        self.assertNotIn(FORWARD_FAILED_LABEL, result.labels_applied)
        self.assertEqual(forward_module.forward_attempts("m2"), 0)

    def test_retry_success_clears_label_and_attempts(self) -> None:
        processor, client = self._processor("m3", FakeForwarder([Forwarded(FORWARDED, "ok")]))
        forward_module.record_forward_attempt("m3")  # one prior failure
        email = make_email(client)

        result = processor._retry_one_forward(email)

        self.assertIn(PROCESSED_LABEL, result.labels_applied)
        self.assertIn(("m3", (FORWARD_FAILED_LABEL,)), client.removed)
        self.assertEqual(forward_module.forward_attempts("m3"), 0)

    def test_retry_failure_increments_attempts_and_keeps_label(self) -> None:
        processor, client = self._processor("m4", FakeForwarder([Forwarded(FAILED, "still down")]))
        forward_module.record_forward_attempt("m4")  # one prior failure
        email = make_email(client)

        result = processor._retry_one_forward(email)

        self.assertIn(FORWARD_FAILED_LABEL, result.labels_applied)
        self.assertEqual(forward_module.forward_attempts("m4"), 2)
        self.assertEqual(client.removed, [])

    def test_retry_gives_up_after_max_attempts_without_calling_forwarder(self) -> None:
        processor, client = self._processor("m5", FakeForwarder([]))  # must not be called
        for _ in range(MAX_FORWARD_ATTEMPTS):
            forward_module.record_forward_attempt("m5")
        email = make_email(client)

        result = processor._retry_one_forward(email)

        self.assertIsNone(result.forward)
        self.assertEqual(forward_module.forward_attempts("m5"), MAX_FORWARD_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
