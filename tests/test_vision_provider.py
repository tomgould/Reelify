"""Tests for reelify.vision — no real LLM calls."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reelify.vision.provider import (
    VisionProvider,
    ProviderUnavailableError,
    get_provider,
    LocalVisionProvider,
    GeminiVisionProvider,
)


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------

def test_get_provider_local_returns_local_provider() -> None:
    provider = get_provider("local")
    assert isinstance(provider, LocalVisionProvider)


def test_get_provider_api_returns_gemini_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REELIFY_PRO", "1")
    provider = get_provider("api")
    assert isinstance(provider, GeminiVisionProvider)


def test_get_provider_api_raises_without_pro(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REELIFY_PRO", raising=False)
    with pytest.raises(ProviderUnavailableError, match="REELIFY_PRO"):
        get_provider("api")


def test_get_provider_auto_returns_local_when_available() -> None:
    with patch("reelify.vision.provider.LocalVisionProvider") as MockLocal:
        mock_instance = MagicMock(spec=LocalVisionProvider)
        MockLocal.return_value = mock_instance
        provider = get_provider("auto")
        assert provider is mock_instance


def test_get_provider_auto_falls_back_to_gemini_when_local_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REELIFY_PRO", "1")
    with patch("reelify.vision.provider.LocalVisionProvider") as MockLocal, \
         patch("reelify.vision.provider.GeminiVisionProvider") as MockGemini:
        MockLocal.side_effect = ProviderUnavailableError("LM Studio down")
        mock_gemini = MagicMock(spec=GeminiVisionProvider)
        MockGemini.return_value = mock_gemini
        provider = get_provider("auto")
        assert provider is mock_gemini


def test_get_provider_auto_raises_without_pro_when_local_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REELIFY_PRO", raising=False)
    with patch("reelify.vision.provider.LocalVisionProvider") as MockLocal:
        MockLocal.side_effect = ProviderUnavailableError("LM Studio down")
        with pytest.raises(ProviderUnavailableError):
            get_provider("auto")


# ---------------------------------------------------------------------------
# LocalVisionProvider.describe_frame
# ---------------------------------------------------------------------------

def test_local_provider_name() -> None:
    provider = LocalVisionProvider()
    assert provider.name == "local"


def _make_mock_pil() -> MagicMock:
    mock_img = MagicMock()
    mock_img.size = (1920, 1080)
    mock_img.convert.return_value = mock_img
    mock_img.resize.return_value = mock_img
    mock_img.save = MagicMock()
    mock_pil = MagicMock()
    mock_pil.open.return_value = mock_img
    mock_pil.LANCZOS = 1
    return mock_pil


def _mock_requests_success(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock_resp.raise_for_status = MagicMock()
    mock_requests = MagicMock()
    mock_requests.post.return_value = mock_resp
    mock_requests.exceptions.ConnectionError = ConnectionError
    mock_requests.exceptions.Timeout = TimeoutError
    return mock_requests


def test_local_describe_frame_returns_string(tmp_path: Path) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    mock_requests = _mock_requests_success("User is typing in a terminal.")
    with patch("reelify.vision.local._requests_module", mock_requests), \
         patch("reelify.vision.local._Image_module", _make_mock_pil()):
        provider = LocalVisionProvider()
        result = provider.describe_frame(img)

    assert result == "User is typing in a terminal."


def test_local_describe_frame_raises_on_connection_error(tmp_path: Path) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    mock_requests = MagicMock()
    mock_requests.exceptions.ConnectionError = ConnectionError
    mock_requests.exceptions.Timeout = TimeoutError
    mock_requests.post.side_effect = ConnectionError("refused")

    with patch("reelify.vision.local._requests_module", mock_requests), \
         patch("reelify.vision.local._Image_module", _make_mock_pil()):
        provider = LocalVisionProvider()
        with pytest.raises(ProviderUnavailableError, match="not reachable"):
            provider.describe_frame(img)


def test_local_raises_on_timeout(tmp_path: Path) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    mock_requests = MagicMock()
    mock_requests.exceptions.ConnectionError = ConnectionError
    mock_requests.exceptions.Timeout = TimeoutError
    mock_requests.post.side_effect = TimeoutError("timed out")

    with patch("reelify.vision.local._requests_module", mock_requests), \
         patch("reelify.vision.local._Image_module", _make_mock_pil()):
        provider = LocalVisionProvider()
        with pytest.raises(ProviderUnavailableError, match="timed out"):
            provider.describe_frame(img)


def test_local_raises_when_requests_not_installed(tmp_path: Path) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    with patch("reelify.vision.local._requests_module", None):
        provider = LocalVisionProvider()
        with pytest.raises(ProviderUnavailableError, match="requests package"):
            provider.describe_frame(img)


# ---------------------------------------------------------------------------
# GeminiVisionProvider
# ---------------------------------------------------------------------------

def test_gemini_provider_name() -> None:
    provider = GeminiVisionProvider()
    assert provider.name == "gemini"


def test_gemini_raises_when_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    provider = GeminiVisionProvider()
    with pytest.raises(ProviderUnavailableError, match="GOOGLE_API_KEY"):
        provider.describe_frame(img)


def test_gemini_describe_frame_returns_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    mock_response = MagicMock()
    mock_response.text = "The user is scrolling through code."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_image_module = MagicMock()
    mock_image = MagicMock()
    mock_image_module.open.return_value = mock_image

    mock_types = MagicMock()
    mock_part = MagicMock()
    mock_types.Part.from_bytes.return_value = mock_part

    with patch("reelify.vision.gemini.genai", mock_genai), \
         patch("reelify.vision.gemini.Image", mock_image_module), \
         patch("reelify.vision.gemini.types", mock_types):
        provider = GeminiVisionProvider()
        result = provider.describe_frame(img)

    assert result == "The user is scrolling through code."


def test_gemini_raises_provider_unavailable_on_api_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_image_module = MagicMock()
    mock_image = MagicMock()
    mock_image_module.open.return_value = mock_image

    mock_types = MagicMock()
    mock_part = MagicMock()
    mock_types.Part.from_bytes.return_value = mock_part

    with patch("reelify.vision.gemini.genai", mock_genai), \
         patch("reelify.vision.gemini.Image", mock_image_module), \
         patch("reelify.vision.gemini.types", mock_types):
        provider = GeminiVisionProvider()
        with pytest.raises(ProviderUnavailableError, match="quota exceeded"):
            provider.describe_frame(img)


def test_gemini_raises_when_package_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    mock_image_module = MagicMock()
    mock_image = MagicMock()
    mock_image_module.open.return_value = mock_image

    mock_types = MagicMock()
    mock_part = MagicMock()
    mock_types.Part.from_bytes.return_value = mock_part

    with patch("reelify.vision.gemini.genai", None), \
         patch("reelify.vision.gemini.Image", mock_image_module), \
         patch("reelify.vision.gemini.types", mock_types):
        provider = GeminiVisionProvider()
        with pytest.raises(ProviderUnavailableError, match="Gemini request failed"):
            provider.describe_frame(img)


# ---------------------------------------------------------------------------
# VisionProvider ABC
# ---------------------------------------------------------------------------

def test_cannot_instantiate_abstract_provider() -> None:
    with pytest.raises(TypeError):
        VisionProvider()  # type: ignore[abstract]
