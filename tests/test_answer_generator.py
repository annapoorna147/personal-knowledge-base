from types import SimpleNamespace

import pytest

from src import answer_generator


def test_generate_answer_without_chunks_does_not_require_api_key(
    monkeypatch,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    answer = answer_generator.generate_answer(
        "What is in my documents?",
        [],
    )

    assert "couldn't find relevant passages" in answer.lower()


def test_generate_answer_requires_api_key_when_chunks_exist(
    monkeypatch,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing"):
        answer_generator.generate_answer(
            "What is Python?",
            [
                {
                    "text": "Python is a programming language.",
                    "source": "notes.txt",
                    "page": 1,
                }
            ],
        )


def test_generate_answer_uses_retrieved_context(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents):
            captured["model"] = model
            captured["contents"] = contents
            return SimpleNamespace(
                text="Python is a programming language. [Source 1]"
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.models = FakeModels()

    monkeypatch.setattr(
        answer_generator.genai,
        "Client",
        FakeClient,
    )

    answer = answer_generator.generate_answer(
        "What is Python?",
        [
            {
                "text": "Python is a programming language.",
                "source": "notes.txt",
                "page": 2,
            }
        ],
    )

    assert "Python is a programming language" in answer
    assert captured["model"] == answer_generator.MODEL_NAME
    assert "What is Python?" in captured["contents"]
    assert "Python is a programming language." in captured["contents"]
    assert "notes.txt" in captured["contents"]
    assert "page 2" in captured["contents"]
    assert "[Source 1]" in captured["contents"]


def test_generate_answer_handles_empty_gemini_response(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeModels:
        def generate_content(self, *, model, contents):
            return SimpleNamespace(text=None)

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(
        answer_generator.genai,
        "Client",
        FakeClient,
    )

    answer = answer_generator.generate_answer(
        "What is Python?",
        [
            {
                "text": "Python is a programming language.",
                "source": "notes.txt",
                "page": 1,
            }
        ],
    )

    assert "empty response" in answer.lower()
