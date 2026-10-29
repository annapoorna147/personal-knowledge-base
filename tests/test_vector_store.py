from uuid import uuid4

import pytest
import chromadb
import numpy as np

from src.vector_store import VectorStore, MAX_COSINE_DISTANCE


class FakeEmbeddingModel:
    """Deterministic embeddings for testing without downloading a model."""

    def encode(self, texts, normalize_embeddings=True):
        vectors = {
            "python and machine learning": [1.0, 0.0],
            "unrelated cooking recipe": [0.0, 1.0],
        }
        return np.array([vectors.get(text, [1.0, 0.0]) for text in texts])


@pytest.fixture
def store():
    """Create an isolated in-memory vector store."""
    instance = VectorStore.__new__(VectorStore)
    instance.client = chromadb.EphemeralClient()
    instance.collection = instance.client.create_collection(
        name=f"test_personal_knowledge_{uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )
    instance.model = FakeEmbeddingModel()
    yield instance


def test_add_chunks_returns_zero_for_empty_input(store):
    assert store.add_chunks([]) == 0
    assert store.collection.count() == 0


def test_add_chunks_ignores_empty_text(store):
    chunks = [
        {"text": "   ", "source": "notes.txt", "page": 1},
        {"text": "", "source": "notes.txt", "page": 2},
    ]

    assert store.add_chunks(chunks) == 0
    assert store.collection.count() == 0


def test_add_chunks_stores_text_and_metadata(store):
    chunks = [
        {
            "text": "  python and machine learning  ",
            "source": "notes.txt",
            "page": 2,
        }
    ]

    assert store.add_chunks(chunks) == 1
    assert store.collection.count() == 1

    saved = store.collection.get(
        include=["documents", "metadatas"]
    )
    assert saved["documents"] == ["python and machine learning"]
    assert saved["metadatas"][0]["source"] == "notes.txt"
    assert saved["metadatas"][0]["page"] == 2


def test_search_returns_relevant_passage(store):
    store.add_chunks([
        {
            "text": "python and machine learning",
            "source": "skills.txt",
            "page": 1,
        }
    ])

    results = store.search("python and machine learning", top_k=2)

    assert len(results) == 1
    assert results[0]["text"] == "python and machine learning"
    assert results[0]["source"] == "skills.txt"
    assert results[0]["page"] == 1
    assert results[0]["distance"] <= MAX_COSINE_DISTANCE


def test_search_returns_empty_for_blank_question(store):
    store.add_chunks([
        {"text": "python and machine learning"}
    ])

    assert store.search("") == []
    assert store.search("   ") == []


def test_search_returns_empty_for_empty_collection(store):
    assert store.search("python") == []


def test_search_filters_dissimilar_passages(store):
    store.add_chunks([
        {"text": "unrelated cooking recipe", "source": "food.txt"}
    ])

    assert store.search("python and machine learning") == []


def test_search_removes_duplicate_passages(store):
    store.add_chunks([
        {"text": "python and machine learning", "source": "first.txt"},
        {"text": "python and machine learning", "source": "second.txt"},
    ])

    results = store.search("python and machine learning", top_k=4)

    assert len(results) == 1
    assert results[0]["text"] == "python and machine learning"
