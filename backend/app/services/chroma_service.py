import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChromaService:
    """
    Service managing persistent vector storage and querying in ChromaDB.
    Persists data locally in backend/data/chroma/.
    """

    _client: Optional[chromadb.PersistentClient] = None
    _collection: Optional[Any] = None

    @classmethod
    def get_client(cls) -> chromadb.PersistentClient:
        """Lazily initialize and return the ChromaDB PersistentClient."""
        if cls._client is None:
            persist_path = Path(settings.CHROMA_PERSIST_DIR).resolve()
            persist_path.mkdir(parents=True, exist_ok=True)
            logger.info("Initializing ChromaDB PersistentClient at '%s'", persist_path)
            cls._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return cls._client

    @classmethod
    def get_collection(cls, name: Optional[str] = None) -> Any:
        """
        Returns or creates the Chroma collection.
        Default collection: settings.CHROMA_COLLECTION_NAME ("documind_chunks").
        Uses cosine distance metric.
        """
        collection_name = name or settings.CHROMA_COLLECTION_NAME
        client = cls.get_client()
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def upsert_chunks(
        cls,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Upserts vectors into ChromaDB.
        Idempotent: updates existing records or inserts new ones.
        Prevents duplicate entries for the same chunk ID.

        Args:
            chunk_ids: List of unique string IDs (e.g. ['chunk_1', 'chunk_2'])
            embeddings: List of embedding vectors (floats)
            documents: Raw text content of the chunks
            metadatas: Metadata dictionaries for each chunk

        Returns:
            Count of upserted records.
        """
        if not chunk_ids:
            return 0

        # Sanitize metadatas: ChromaDB values must be str, int, float, or bool
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for k, v in meta.items():
                if v is None:
                    cleaned[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                else:
                    cleaned[k] = str(v)
            cleaned_metadatas.append(cleaned)

        collection = cls.get_collection(collection_name)
        collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned_metadatas,
        )
        return len(chunk_ids)

    @classmethod
    def delete_document_vectors(
        cls,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Deletes all vector embeddings associated with a given document_id.

        Args:
            document_id: UUID string of the document.

        Returns:
            Number of vectors deleted.
        """
        collection = cls.get_collection(collection_name)
        try:
            # Query existing vectors for this document first to count them
            existing = collection.get(where={"document_id": document_id})
            count = len(existing.get("ids", []))
            if count > 0:
                collection.delete(where={"document_id": document_id})
                logger.info("Deleted %d vectors for document '%s'", count, document_id)
            return count
        except Exception as err:
            logger.error("Failed to delete vectors for document '%s': %s", document_id, err)
            raise

    @classmethod
    def get_document_vector_count(
        cls,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Returns the number of vectors stored in ChromaDB for a specific document."""
        collection = cls.get_collection(collection_name)
        try:
            results = collection.get(where={"document_id": document_id})
            return len(results.get("ids", []))
        except Exception:
            return 0

    @classmethod
    def query_chunks(
        cls,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Queries ChromaDB for the most similar chunks based on cosine distance.
        Gracefully handles empty collections or requesting more results than available.

        Args:
            query_embeddings: List of embedding vectors to query against.
            n_results: Number of nearest neighbor results to retrieve.
            where: Optional metadata filter dict (e.g., {"document_id": doc_id}).
            collection_name: Optional custom collection name.

        Returns:
            Dict containing 'ids', 'distances', 'documents', 'metadatas'.
        """
        collection = cls.get_collection(collection_name)
        total_count = collection.count()

        if total_count == 0 or not query_embeddings:
            return {
                "ids": [[] for _ in query_embeddings] if query_embeddings else [[]],
                "distances": [[] for _ in query_embeddings] if query_embeddings else [[]],
                "documents": [[] for _ in query_embeddings] if query_embeddings else [[]],
                "metadatas": [[] for _ in query_embeddings] if query_embeddings else [[]],
            }

        effective_n = min(n_results, total_count)
        kwargs = {
            "query_embeddings": query_embeddings,
            "n_results": effective_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            return collection.query(**kwargs)
        except Exception as err:
            logger.error("Error querying ChromaDB collection: %s", err)
            raise

    @classmethod
    def get_stats(cls, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns statistics regarding the vector database and collection.
        """
        collection = cls.get_collection(collection_name)
        total_vectors = collection.count()
        persist_dir = str(Path(settings.CHROMA_PERSIST_DIR).resolve())

        return {
            "collection_name": collection.name,
            "total_vectors": total_vectors,
            "persist_directory": persist_dir,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "distance_metric": "cosine",
            "status": "operational",
        }

