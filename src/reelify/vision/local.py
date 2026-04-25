import base64
from pathlib import Path

from reelify.vision.provider import VisionProvider, ProviderUnavailableError

_PROMPT = (
    "Describe what is happening on this screen in one sentence. "
    "Focus on the main activity visible."
)

_BASE_URL = "http://localhost:1234/v1"
_API_KEY = "lm-studio"

try:
    import openai as _openai_module
except ImportError:
    _openai_module = None  # type: ignore[assignment]

try:
    import httpx as _httpx_module
except ImportError:
    _httpx_module = None  # type: ignore[assignment]


class LocalVisionProvider(VisionProvider):
    """Vision provider backed by a local LM Studio instance."""

    @property
    def name(self) -> str:
        return "local"

    def describe_frame(self, image_path: Path) -> str:
        """Send *image_path* to LM Studio and return a one-sentence description.

        Raises:
            ProviderUnavailableError: if LM Studio is not reachable.
        """
        # Use module-level import (may be None if package absent)
        openai = _openai_module
        if openai is None:
            raise ProviderUnavailableError(
                "openai package is required for LocalVisionProvider"
            )

        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        suffix = image_path.suffix.lower().lstrip(".")
        mime = f"image/{suffix if suffix in ('png', 'gif', 'webp') else 'jpeg'}"
        data_url = f"data:{mime};base64,{image_data}"

        try:
            client = openai.OpenAI(base_url=_BASE_URL, api_key=_API_KEY)
            response = client.chat.completions.create(
                model="local-model",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=128,
            )
            return response.choices[0].message.content or ""
        except ConnectionRefusedError as exc:
            raise ProviderUnavailableError(
                f"LM Studio connection refused at {_BASE_URL}"
            ) from exc
        except Exception as exc:
            httpx = _httpx_module
            if httpx is not None and isinstance(exc, httpx.ConnectError):
                raise ProviderUnavailableError(
                    f"LM Studio not reachable: {exc}"
                ) from exc
            exc_type = type(exc).__name__
            if exc_type in (
                "ConnectError",
                "APIConnectionError",
                "APIStatusError",
                "AuthenticationError",
            ):
                raise ProviderUnavailableError(
                    f"LM Studio provider error ({exc_type}): {exc}"
                ) from exc
            raise
