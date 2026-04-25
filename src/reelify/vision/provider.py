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


def get_provider(prefer: str = "auto") -> VisionProvider:
    """Return a :class:`VisionProvider` according to *prefer*.

    Args:
        prefer: ``'local'`` → LM Studio only; ``'api'`` → Gemini only;
            ``'auto'`` → try local first, fall back to API silently.

    Raises:
        ProviderUnavailableError: if the requested provider (or all providers
            when ``prefer='auto'``) cannot be initialised.
    """
    if prefer == "local":
        return LocalVisionProvider()

    if prefer == "api":
        return GeminiVisionProvider()

    # auto: try local first, fall back to Gemini
    try:
        return LocalVisionProvider()
    except ProviderUnavailableError:
        pass

    return GeminiVisionProvider()
