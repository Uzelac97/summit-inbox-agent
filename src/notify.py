"""Ways of telling a person about an emergency, beyond the mailbox.

An emergency is starred and marked important in Gmail, which is what the
mailbox shows. A push to a phone is a separate channel, and it is not built
yet. The processor calls ``emergency`` once for each email it handles as an
emergency. The default does nothing. A phone push is a new notifier with an
``emergency`` method, passed to ``Processor(notifier=...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.processor import Result


class Notifier(Protocol):
    """Anything that can raise the alarm about an emergency."""

    def emergency(self, result: "Result") -> None: ...


class NoOpNotifier:
    """Does nothing. The default until a real channel is wired in."""

    def emergency(self, result: "Result") -> None:
        return None
