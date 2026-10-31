
from io import BytesIO


def extract_text(uploaded_file):
    """Extract text from an uploaded PDF or TXT file."""
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    pages = []

    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "text": text,
                    "source": filename,
                    "page": page_number,
                })

    elif filename.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="replace")

        if text.strip():
            pages.append({
                "text": text,
                "source": filename,
                "page": 1,
            })

    else:
        raise ValueError(
            f"Unsupported file type: {filename}"
        )

    return pages


def split_into_chunks(text, chunk_size=700, overlap=100):
    """
    Split text into smaller overlapping chunks.

    Smaller chunks help retrieval focus on individual topics.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size."
        )

    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks
