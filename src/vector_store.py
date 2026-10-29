
from pathlib import Path
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / ".chroma"
MODEL_NAME = "all-MiniLM-L6-v2"

# Cosine distance: lower values indicate more similar results.
MAX_COSINE_DISTANCE = 0.65


class VectorStore:
    """Store document chunks and retrieve relevant passages."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name="personal_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

        self.model = SentenceTransformer(MODEL_NAME)

    def add_chunks(self, chunks):
        """Add document chunks with source and page metadata."""
        if not chunks:
            return 0

        # Avoid adding empty text chunks.
        valid_chunks = [
            item for item in chunks
            if item.get("text", "").strip()
        ]

        if not valid_chunks:
            return 0

        texts = [item["text"].strip() for item in valid_chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

        ids = [str(uuid4()) for _ in valid_chunks]

        metadatas = [
            {
                "source": item.get("source", "Unknown"),
                "page": int(item.get("page", 1)),
            }
            for item in valid_chunks
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(valid_chunks)

    def search(self, question, top_k=4):
        """Return relevant, non-duplicate document passages."""
        if not question or not question.strip():
            return []

        total = self.collection.count()

        if total == 0:
            return []

        query_embedding = self.model.encode(
            [question.strip()],
            normalize_embeddings=True,
        ).tolist()

        # Retrieve extra candidates so duplicates can be removed.
        candidate_count = min(max(top_k * 3, top_k), total)

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_count,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []
        seen_chunks = set()

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(
            documents, metadatas, distances
        ):
            normalized_text = " ".join(text.split()).casefold()

            # Ignore exact duplicate passages.
            if normalized_text in seen_chunks:
                continue

            # Ignore candidates that are too dissimilar.
            if distance > MAX_COSINE_DISTANCE:
                continue

            seen_chunks.add(normalized_text)

            matches.append({
                "text": text,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", 1),
                "distance": float(distance),
            })

            if len(matches) >= top_k:
                break

        return matches
