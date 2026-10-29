import io
import pytest

from src.document_processor import extract_text, split_into_chunks


def test_split_into_chunks_short_text():
    chunks = split_into_chunks("Hello world", chunk_size=100, overlap=10)

    assert chunks == ["Hello world"]


def test_split_into_chunks_normalizes_whitespace():
    chunks = split_into_chunks("Hello\n\nworld   again")

    assert chunks == ["Hello world again"]


def test_split_into_chunks_covers_text_with_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = split_into_chunks(text, chunk_size=10, overlap=2)

    assert chunks[0] == text[:10]
    assert chunks[1] == text[8:18]
    assert chunks[2] == text[16:26]
    assert "".join(
        chunk if index == 0 else chunk[2:]
        for index, chunk in enumerate(chunks)
    ) == text


def test_split_into_chunks_uses_overlap():
    chunks = split_into_chunks(
        "abcdefghijklmnopqrst",
        chunk_size=10,
        overlap=3,
    )

    assert chunks[0][-3:] == chunks[1][:3]


def test_split_into_chunks_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_into_chunks("hello", chunk_size=0)


def test_split_into_chunks_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_into_chunks("hello", chunk_size=10, overlap=10)


def test_extract_text_from_txt():
    uploaded_file = io.BytesIO(b"Python and AI")
    uploaded_file.name = "notes.txt"

    pages = extract_text(uploaded_file)

    assert len(pages) == 1
    assert pages[0]["text"] == "Python and AI"
    assert pages[0]["source"] == "notes.txt"
    assert pages[0]["page"] == 1


def test_extract_text_ignores_empty_txt():
    uploaded_file = io.BytesIO(b"   \n")
    uploaded_file.name = "empty.txt"

    assert extract_text(uploaded_file) == []


def test_extract_text_rejects_unsupported_file():
    uploaded_file = io.BytesIO(b"some data")
    uploaded_file.name = "notes.docx"

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(uploaded_file)
