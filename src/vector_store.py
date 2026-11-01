
from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / ".chroma"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_COSINE_DISTANCE = 0.65


class VectorStore:
    def __init__(self):
        """Initialize the persistent ChromaDB collection and embedding model."""
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name="personal_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

        self.model = SentenceTransformer(MODEL_NAME)

    def is_file_indexed(self, source, file_hash):
        """Return whether this exact file version has indexed chunks."""
        if not source or not file_hash:
            return False

        result = self.collection.get(
            where={
                "$and": [
                    {"source": source},
                    {"file_hash": file_hash},
                ]
            },
            include=["metadatas"],
        )
        return bool(result["ids"])

    def add_chunks(self, chunks):
        """
        Store document chunks.

        Existing chunks belonging to the same source filename are replaced.
        Chunks from other source filenames are preserved.
        """
        if not chunks:
            return 0

        valid_chunks = [
            item
            for item in chunks
            if item.get("text", "").strip()
        ]

        if not valid_chunks:
            return 0

        # Normalize and validate metadata before changing stored records.
        prepared_chunks = []

        for item in valid_chunks:
            prepared_chunks.append({
                "text": item["text"].strip(),
                "source": item.get("source", "Unknown"),
                "page": int(item.get("page", 1)),
                "file_hash": item.get("file_hash", ""),
            })

        # Generate deterministic IDs so identical chunks have stable IDs.
        ids = []

        for item in prepared_chunks:
            key = (
                f'{item["source"]}|'
                f'{item["page"]}|'
                f'{item["text"]}'
            )

            ids.append(
                hashlib.sha256(
                    key.encode("utf-8")
                ).hexdigest()
            )

        texts = [
            item["text"]
            for item in prepared_chunks
        ]

        metadatas = [
            {
                "source": item["source"],
                "page": item["page"],
                "file_hash": item["file_hash"],
            }
            for item in prepared_chunks
        ]

        # Calculate embeddings before modifying the existing collection.
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

        # Replace existing records for the source filenames being indexed.
        # Other documents in the collection remain untouched.
        sources = {
            item["source"]
            for item in prepared_chunks
        }

        for source in sources:
            existing = self.collection.get(
                where={"source": source},
                include=["metadatas"],
            )

            existing_ids = existing["ids"]

            # Keep records that will be re-created with the same IDs.
            # Delete only obsolete records for this source.
            ids_to_keep = set(ids)
            obsolete_ids = [
                existing_id
                for existing_id in existing_ids
                if existing_id not in ids_to_keep
            ]

            if obsolete_ids:
                self.collection.delete(ids=obsolete_ids)

        # Upsert inserts new chunks and updates matching deterministic IDs.
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(prepared_chunks)

    def search(self, question, top_k=4, source=None):
        """
        Retrieve relevant passages.

        Args:
            question: The user's question.
            top_k: Maximum number of unique passages to return.
            source: Optional exact filename filter.

        Returns:
            A list of dictionaries containing text, source, page, and distance.
        """
        if not question or not question.strip():
            return []

        if top_k <= 0:
            return []

        # Filter by exact source filename when requested.
        filters = {"source": source} if source else None

        # Collection.count() in this environment does not accept `where`.
        # Use get() to count records when filtering by source.
        if filters:
            filtered_records = self.collection.get(
                where=filters,
                include=["metadatas"],
            )
            total = len(filtered_records["ids"])
        else:
            total = self.collection.count()

        if total == 0:
            return []

        query_embedding = self.model.encode(
            [question.strip()],
            normalize_embeddings=True,
        ).tolist()

        query_args = {
            "query_embeddings": query_embedding,
            "n_results": min(top_k * 3, total),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if filters:
            query_args["where"] = filters

        results = self.collection.query(**query_args)

        matches = []
        seen_chunks = set()

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            if not text or not text.strip():
                continue

            normalized_text = " ".join(
                text.split()
            ).casefold()

            # Avoid returning identical text more than once.
            if normalized_text in seen_chunks:
                continue

            # Lower cosine distance means greater similarity.
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
