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


def test_generate_answer_retries_temporary_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(answer_generator.time, "sleep", lambda _: None)

    class FakeAPIError(Exception):
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

    monkeypatch.setattr(answer_generator.errors, "APIError", FakeAPIError)

    class FakeModels:
        calls = 0

        def generate_content(self, *, model, contents):
            self.calls += 1
            if self.calls == 1:
                raise FakeAPIError("Temporary server error", 503)
            return SimpleNamespace(text="Answer recovered. [Source 1]")

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    client = FakeClient()
    monkeypatch.setattr(
        answer_generator.genai,
        "Client",
        lambda **kwargs: client,
    )

    answer = answer_generator.generate_answer(
        "What is Python?",
        [{"text": "Python is a programming language.", "source": "notes.txt", "page": 1}],
    )

    assert answer == "Answer recovered. [Source 1]"
    assert client.models.calls == 2


def test_generate_answer_does_not_retry_permanent_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeAPIError(Exception):
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

    monkeypatch.setattr(answer_generator.errors, "APIError", FakeAPIError)

    class FakeModels:
        calls = 0

        def generate_content(self, *, model, contents):
            self.calls += 1
            raise FakeAPIError("Invalid request", 400)

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    client = FakeClient()
    monkeypatch.setattr(
        answer_generator.genai,
        "Client",
        lambda **kwargs: client,
    )

    answer = answer_generator.generate_answer(
        "What is Python?",
        [{"text": "Python is a programming language.", "source": "notes.txt", "page": 1}],
    )

    assert "API request failed" in answer
    assert client.models.calls == 1


def test_generate_answer_returns_friendly_message_after_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(answer_generator.time, "sleep", lambda _: None)

    class FakeAPIError(Exception):
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

    monkeypatch.setattr(answer_generator.errors, "APIError", FakeAPIError)

    class FakeModels:
        calls = 0

        def generate_content(self, *, model, contents):
            self.calls += 1
            raise FakeAPIError("Service unavailable", 503)

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    client = FakeClient()
    monkeypatch.setattr(
        answer_generator.genai,
        "Client",
        lambda **kwargs: client,
    )

    answer = answer_generator.generate_answer(
        "What is Python?",
        [{"text": "Python is a programming language.", "source": "notes.txt", "page": 1}],
    )

    assert "temporarily busy or unavailable" in answer
    assert client.models.calls == answer_generator.MAX_ATTEMPTS
