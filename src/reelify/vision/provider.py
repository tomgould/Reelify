from abc import ABC, abstractmethod
from pathlib import Path


class ProviderUnavailableError(Exception):
    """Raised when a vision provider cannot be reached or configured."""


class VisionProvider(ABC):
    """Abstract base class for LLM vision providers."""

    @abstractmethod
    def describe_frame(self, image_path: Path) -> str:
        """Return a one-sentence description of the image at *image_path*."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...


# Import concrete providers at module level so they are patchable in tests.
# The imports are placed after the base classes to avoid circular imports.
from reelify.vision.local import LocalVisionProvider  # noqa: E402
from reelify.vision.gemini import GeminiVisionProvider  # noqa: E402


def get_provider(prefer: str = "local") -> VisionProvider:
    """Return a :class:`VisionProvider` according to *prefer*.

    Args:
        prefer: ``'local'`` → LM Studio only (default); ``'api'`` → Gemini
            (requires REELIFY_PRO=1); ``'auto'`` → local first, fall back to
            Gemini if unavailable (requires REELIFY_PRO=1 for fallback).

    Raises:
        ProviderUnavailableError: if the requested provider cannot be reached.
    """
    import os
    pro = os.environ.get("REELIFY_PRO", "0") == "1"

    if prefer == "api":
        if not pro:
            raise ProviderUnavailableError(
                "API providers require REELIFY_PRO=1"
            )
        return GeminiVisionProvider()

    if prefer == "local":
        return LocalVisionProvider()

    # auto: always try local; only fall back to API if PRO enabled
    try:
        return LocalVisionProvider()
    except ProviderUnavailableError:
        if pro:
            return GeminiVisionProvider()
        raise
