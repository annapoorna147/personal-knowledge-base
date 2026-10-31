
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
    """Split text into overlapping chunks with normalized whitespace."""
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


def split_into_sections(text, chunk_size=700):
    """Split a document by headings and lines to preserve context."""
    import re

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    headings = {
        "PROFILE",
        "TECHNICAL SKILLS",
        "PROJECTS",
        "AI Quiz Generator",
        "RUBIX — Ultimate Edition",
        "DSA & PROBLEM SOLVING",
        "EDUCATION",
        "CERTIFICATIONS & ACHIEVEMENTS",
    }

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    sections = []
    heading = None
    section_lines = []

    for line in lines:
        if line in headings:
            if heading is not None or section_lines:
                sections.append((heading, section_lines))
            heading = line
            section_lines = []
        else:
            section_lines.append(line)

    if heading is not None or section_lines:
        sections.append((heading, section_lines))

    chunks = []

    for heading, section_lines in sections:
        prefix = f"{heading}\n" if heading else ""
        current = prefix

        for line in section_lines:
            candidate = current + ("\n" if current.strip() else "") + line

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current.strip():
                chunks.append(current.strip())

            current = prefix

            # Break unusually long lines at word boundaries.
            words = line.split()
            piece = ""

            for word in words:
                proposed = f"{piece} {word}".strip()

                if len(current) + len(proposed) > chunk_size:
                    if piece:
                        chunks.append((current + piece).strip())
                        current = prefix
                        piece = word
                    elif current.strip():
                        chunks.append(current.strip())
                        current = ""
                        piece = word
                    else:
                        # Handle a single word longer than the limit.
                        for offset in range(0, len(word), chunk_size):
                            part = word[offset:offset + chunk_size]
                            if offset + chunk_size < len(word):
                                chunks.append(part)
                            else:
                                current = part
                        piece = ""
                else:
                    piece = proposed

            if piece:
                current += piece

        if current.strip():
            chunks.append(current.strip())

    return chunks
