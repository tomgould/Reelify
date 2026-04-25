import io
import os
from pathlib import Path

from google import genai
from google.genai import types

from reelify.vision.provider import VisionProvider, ProviderUnavailableError

_PROMPT = (
    "Describe what is happening on this screen in one sentence. "
    "Focus on the main activity visible."
)

_MODEL = "gemini-2.0-flash"

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


class GeminiVisionProvider(VisionProvider):
    """Vision provider backed by Google Gemini Flash."""

    @property
    def name(self) -> str:
        return "gemini"

    def describe_frame(self, image_path: Path) -> str:
        """Send *image_path* to Gemini and return a one-sentence description.

        Raises:
            ProviderUnavailableError: if ``GOOGLE_API_KEY`` is not set or the
                request fails.
        """
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderUnavailableError(
                "GOOGLE_API_KEY environment variable is not set"
            )

        if Image is None:
            raise ProviderUnavailableError(
                "Pillow package is required for GeminiVisionProvider"
            )

        image = Image.open(image_path)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=_MODEL,
                contents=[image_part, _PROMPT],
            )
            return response.text or ""
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Gemini request failed: {exc}"
            ) from exc