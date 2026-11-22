import base64
import io
import json
from pathlib import Path

from reelify.vision.provider import VisionProvider, ProviderUnavailableError

_PROMPT = (
    "Describe what is happening on this screen in one sentence. "
    "Focus on the main activity visible."
)

_BASE_URL = "http://localhost:1234/v1"
_MODEL = "qwen/qwen2.5-vl-7b"
_MAX_PX = 1280  # resize longest edge to this before sending

try:
    import requests as _requests_module
except ImportError:
    _requests_module = None  # type: ignore[assignment]

try:
    from PIL import Image as _Image_module
except ImportError:
    _Image_module = None  # type: ignore[assignment]


def _encode_image(image_path: Path) -> str:
    """Return base64 JPEG data URL, resized to _MAX_PX on the longest edge."""
    if _Image_module is None:
        # Fallback: send raw bytes without resize
        return "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode()

    img = _Image_module.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_PX:
        scale = _MAX_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), _Image_module.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class LocalVisionProvider(VisionProvider):
    """Vision provider backed by a local LM Studio instance."""

    @property
    def name(self) -> str:
        return "local"

    def describe_frame(self, image_path: Path) -> str:
        """Send *image_path* to LM Studio and return a one-sentence description.

        Uses requests directly (more compatible with LM Studio than openai SDK).

        Raises:
            ProviderUnavailableError: if LM Studio is not reachable.
        """
        if _requests_module is None:
            raise ProviderUnavailableError(
                "requests package is required for LocalVisionProvider"
            )

        data_url = _encode_image(image_path)

        payload = {
            "model": _MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 128,
        }

        try:
            resp = _requests_module.post(
                f"{_BASE_URL}/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""
        except _requests_module.exceptions.ConnectionError as exc:
            raise ProviderUnavailableError(
                f"LM Studio not reachable at {_BASE_URL}"
            ) from exc
        except _requests_module.exceptions.Timeout as exc:
            raise ProviderUnavailableError("LM Studio request timed out") from exc
        except Exception as exc:
            raise ProviderUnavailableError(
                f"LM Studio request failed: {exc}"
            ) from exc
