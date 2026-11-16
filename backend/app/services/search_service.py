import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.services.embedding_service import EmbeddingService
from app.services.chroma_service import ChromaService
from app.models.document import Document

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service managing semantic search and retrieval across document chunk embeddings.
    Converts natural-language queries into vectors and retrieves top-k matching chunks from ChromaDB.
    """

    @staticmethod
    def validate_query(query: str) -> str:
        """Validates that search query is non-empty and non-whitespace."""
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty or whitespace only.")
        return query.strip()

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        db: Optional[Session] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes semantic search for a given natural-language query across one or multiple documents.

        Args:
            query: User's search query string.
            top_k: Number of most relevant chunks to return (1 to 100).
            document_id: Optional UUID to restrict search to a single document.
            document_ids: Optional list of document UUIDs to restrict search across multiple documents.
            db: Optional database session to validate document existence and auto-index.
            user_id: Optional user UUID to strictly isolate search to documents owned by this user.

        Returns:
            Dict containing query, total_results, document_filter, document_ids_filter, and results list.
        """
        import math

        # 1. Validate query string
        cleaned_query = cls.validate_query(query)

        # 2. Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            raise ValueError("top_k must be an integer between 1 and 100.")

        # 3. Normalize Document Filtering
        target_doc_ids: List[str] = []
        if document_ids is not None:
            if isinstance(document_ids, str):
                target_doc_ids = [document_ids.strip()] if document_ids.strip() else []
            elif isinstance(document_ids, (list, tuple, set)):
                target_doc_ids = [str(d).strip() for d in document_ids if d and str(d).strip()]
        elif document_id:
            cleaned_id = str(document_id).strip()
            if cleaned_id and cleaned_id.lower() != "all":
                target_doc_ids = [cleaned_id]

        # Deduplicate while preserving order
        target_doc_ids = list(dict.fromkeys(target_doc_ids))

        # 3b. Enforce User Data Isolation
        if db is not None and user_id:
            user_docs = db.query(Document.id).filter(Document.user_id == user_id).all()
            allowed_user_doc_ids = {d.id for d in user_docs}
            if target_doc_ids:
                for t_id in target_doc_ids:
                    if t_id not in allowed_user_doc_ids:
                        raise ValueError(f"Access forbidden: You do not have permission to access document '{t_id}'.")
            else:
                target_doc_ids = list(allowed_user_doc_ids)
                if not target_doc_ids:
                    return {
                        "query": cleaned_query,
                        "total_results": 0,
                        "document_filter": None,
                        "document_ids_filter": [],
                        "results": [],
                    }


        # Validate existence & auto-embed chunks if DB is provided
        if db is not None:
            from app.services.document_service import DocumentService
            if target_doc_ids:
                for d_id in target_doc_ids:
                    doc = db.query(Document).filter(Document.id == d_id).first()
                    if not doc:
                        raise ValueError(f"Document with ID '{d_id}' not found.")
                    chroma_count = ChromaService.get_document_vector_count(d_id)
                    if chroma_count == 0 and doc.chunk_count > 0:
                        logger.info("Auto-embedding document '%s' (%d chunks)...", d_id, doc.chunk_count)
                        DocumentService.embed_document_chunks(db, d_id)
            else:
                # If searching across all documents, auto-embed any pending document
                pending_docs = db.query(Document).filter(Document.chunk_count > 0).all()
                for p_doc in pending_docs:
                    if ChromaService.get_document_vector_count(p_doc.id) == 0:
                        logger.info("Auto-embedding document '%s' (%s) for global search...", p_doc.filename, p_doc.id)
                        DocumentService.embed_document_chunks(db, p_doc.id)

        # 4. Generate Query Vector Embedding
        # Uses existing loaded SentenceTransformer (all-MiniLM-L6-v2, 384 dimensions)
        query_embedding = EmbeddingService.generate_embedding(cleaned_query)

        # 5. ChromaDB Query with Balanced Representation
        where_filter: Optional[Dict[str, Any]] = None
        is_multi_doc = len(target_doc_ids) > 1

        if len(target_doc_ids) == 1:
            where_filter = {"document_id": target_doc_ids[0]}
            n_to_retrieve = top_k
        elif is_multi_doc:
            where_filter = {"document_id": {"$in": target_doc_ids}}
            # Retrieve a larger candidate pool to ensure balanced representation across documents
            n_to_retrieve = min(max(top_k * 3, len(target_doc_ids) * 3), 60)
        else:
            n_to_retrieve = top_k

        raw_results = ChromaService.query_chunks(
            query_embeddings=[query_embedding],
            n_results=n_to_retrieve,
            where=where_filter,
        )

        ids_list = raw_results.get("ids", [[]])[0]
        distances_list = raw_results.get("distances", [[]])[0]
        documents_list = raw_results.get("documents", [[]])[0]
        metadatas_list = raw_results.get("metadatas", [[]])[0]

        # 6. Format and De-duplicate Retrieved Chunks
        candidate_chunks: List[Dict[str, Any]] = []
        seen_chunk_ids = set()

        for i in range(len(ids_list)):
            doc_text = documents_list[i] if i < len(documents_list) else ""
            meta = metadatas_list[i] if i < len(metadatas_list) else {}
            dist = float(distances_list[i]) if i < len(distances_list) else 1.0

            chunk_id = int(meta.get("chunk_id", 0))
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            # ChromaDB cosine distance: d in [0, 2].
            # Cosine similarity for normalized vectors: 1 - d, bounded in [0.0, 1.0]
            similarity_score = round(max(0.0, min(1.0, 1.0 - dist)), 4)
            distance_rounded = round(dist, 4)

            # Parse page/slide metadata
            source_pages = meta.get("source_pages")
            if source_pages == "":
                source_pages = None

            slide_number = meta.get("slide_number")
            if slide_number is not None:
                try:
                    slide_int = int(slide_number)
                    slide_number = slide_int if slide_int >= 0 else None
                except (ValueError, TypeError):
                    slide_number = None

            candidate_chunks.append({
                "chunk_id": chunk_id,
                "document_id": meta.get("document_id", ""),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "content": doc_text,
                "filename": meta.get("filename", ""),
                "source_pages": source_pages,
                "slide_number": slide_number,
                "word_count": int(meta.get("word_count", len(doc_text.split()))),
                "score": similarity_score,
                "distance": distance_rounded,
            })

        # 7. Apply Fair Balanced Selection when Multiple Documents are Selected
        # Prevents one document from monopolizing all top_k slots
        if is_multi_doc and candidate_chunks:
            doc_chunks_map: Dict[str, List[Dict[str, Any]]] = {}
            for c in candidate_chunks:
                d_id = c["document_id"]
                doc_chunks_map.setdefault(d_id, []).append(c)

            # Sort chunks in each document by score descending
            for d_id in doc_chunks_map:
                doc_chunks_map[d_id].sort(key=lambda x: x["score"], reverse=True)

            selected_chunks: List[Dict[str, Any]] = []
            selected_ids = set()

            # Pass 1: Guarantee balanced representation
            # Allocate up to fair_quota per document
            fair_quota = max(1, math.ceil(top_k / len(target_doc_ids)))
            for d_id in target_doc_ids:
                if d_id in doc_chunks_map:
                    for chunk in doc_chunks_map[d_id][:fair_quota]:
                        if chunk["chunk_id"] not in selected_ids and len(selected_chunks) < top_k:
                            selected_chunks.append(chunk)
                            selected_ids.add(chunk["chunk_id"])

            # Pass 2: Fill any remaining capacity with highest scoring remaining chunks
            if len(selected_chunks) < top_k:
                remaining_candidates = [c for c in candidate_chunks if c["chunk_id"] not in selected_ids]
                remaining_candidates.sort(key=lambda x: x["score"], reverse=True)
                for chunk in remaining_candidates:
                    if len(selected_chunks) >= top_k:
                        break
                    selected_chunks.append(chunk)
                    selected_ids.add(chunk["chunk_id"])

            # Final sort of selected chunks by relevance score descending
            selected_chunks.sort(key=lambda x: x["score"], reverse=True)
            results = selected_chunks
        else:
            results = candidate_chunks[:top_k]

        logger.info(
            "SearchService: query='%s' | target_doc_ids=%s | top_k=%d | retrieved=%d | "
            "chunk_ids=%s | filenames=%s | scores=%s",
            cleaned_query,
            target_doc_ids or "ALL",
            top_k,
            len(results),
            [r["chunk_id"] for r in results],
            list(dict.fromkeys(r["filename"] for r in results if r.get("filename"))),
            [r["score"] for r in results],
        )

        return {
            "query": cleaned_query,
            "total_results": len(results),
            "document_filter": target_doc_ids[0] if len(target_doc_ids) == 1 else None,
            "document_ids_filter": target_doc_ids,
            "results": results,
        }

