from typing import List, Optional, Dict, Any, Union, Tuple
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.chunking_service import ChunkItem
from app.services.embedding_service import EmbeddingService
from app.services.chroma_service import ChromaService

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service managing persistence, queries, chunk associations,
    and aggregation metrics for documents in SQLite.
    """

    @staticmethod
    def create_document(
        db: Session,
        filename: str,
        original_filename: str,
        file_type: str,
        file_size: int,
        storage_path: Optional[str] = None,
        extracted_text: Optional[str] = None,
        page_count: Optional[int] = None,
        chunk_count: int = 0,
        processing_status: str = "pending",
        user_id: Optional[str] = None,
    ) -> Document:
        """Creates and stores a new Document record."""
        doc = Document(
            user_id=user_id,
            filename=filename,
            original_filename=original_filename,
            file_type=file_type.lower().lstrip("."),
            file_size=file_size,
            storage_path=storage_path,
            extracted_text=extracted_text,
            page_count=page_count,
            chunk_count=chunk_count,
            processing_status=processing_status,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc

    @staticmethod
    def get_document(db: Session, document_id: str) -> Optional[Document]:
        """Retrieves a single document by its UUID string."""
        return db.query(Document).filter(Document.id == document_id).first()

    @staticmethod
    def list_documents(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """Returns documents ordered by upload date descending, filtered by user_id if provided."""
        query = db.query(Document)
        if user_id:
            query = query.filter(Document.user_id == user_id)
        return (
            query.order_by(Document.upload_date.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )


    @staticmethod
    def delete_document(db: Session, document_id: str) -> Optional[Document]:
        """
        Deletes a document by ID. Cascades automatically to associated chunks in SQLite
        and removes vector embeddings from ChromaDB.
        Returns the deleted Document instance (containing storage_path), or None if not found.
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return None

        # Clean up ChromaDB vectors for this document
        try:
            ChromaService.delete_document_vectors(document_id)
        except Exception as chroma_err:
            logger.warning("Error deleting ChromaDB vectors for document %s: %s", document_id, chroma_err)

        db.delete(doc)
        db.commit()
        return doc

    @staticmethod
    def update_document_status(
        db: Session,
        document_id: str,
        status: str,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
        page_count: Optional[int] = None,
        extracted_text: Optional[str] = None,
    ) -> Optional[Document]:
        """Updates document processing status, errors, and counts."""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return None

        doc.processing_status = status
        if error_message is not None:
            doc.error_message = error_message
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if page_count is not None:
            doc.page_count = page_count
        if extracted_text is not None:
            doc.extracted_text = extracted_text

        db.commit()
        db.refresh(doc)
        return doc

    @staticmethod
    def save_chunks(
        db: Session,
        document_id: str,
        chunks_data: List[Union[ChunkItem, Dict[str, Any]]],
    ) -> List[Chunk]:
        """
        Persists a batch of chunk items associated with a document.
        Updates the document's chunk_count.
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document with id '{document_id}' not found.")

        chunk_records: List[Chunk] = []
        for item in chunks_data:
            if isinstance(item, ChunkItem):
                c_idx = item.chunk_index
                c_content = item.content
                c_words = item.word_count
                c_pages = item.source_pages
                c_slide = item.slide_number
            else:
                c_idx = item.get("chunk_index", 0)
                c_content = item.get("content", "")
                c_words = item.get("word_count", len(c_content.split()))
                c_pages = item.get("source_pages")
                c_slide = item.get("slide_number")

            chunk = Chunk(
                document_id=document_id,
                chunk_index=c_idx,
                content=c_content,
                word_count=c_words,
                source_pages=c_pages,
                slide_number=c_slide,
            )
            chunk_records.append(chunk)

        db.add_all(chunk_records)
        doc.chunk_count = len(chunk_records)
        db.commit()

        for c in chunk_records:
            db.refresh(c)
        return chunk_records

    @staticmethod
    def get_document_chunks(
        db: Session,
        document_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[int, List[Chunk]]:
        """
        Returns total chunk count and paginated chunks for a document ordered by chunk_index.
        """
        query = db.query(Chunk).filter(Chunk.document_id == document_id)
        total = query.count()
        chunks = (
            query.order_by(Chunk.chunk_index.asc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return total, chunks

    @staticmethod
    def get_library_stats(db: Session, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Computes aggregate metrics across documents and chunks, isolated by user_id if provided.
        """
        doc_query = db.query(Document)
        if user_id:
            doc_query = doc_query.filter(Document.user_id == user_id)

        total_documents = doc_query.with_entities(func.count(Document.id)).scalar() or 0
        total_bytes = doc_query.with_entities(func.sum(Document.file_size)).scalar() or 0

        # Calculate chunks for user's documents
        if user_id:
            user_doc_ids = [d[0] for d in db.query(Document.id).filter(Document.user_id == user_id).all()]
            total_chunks = (
                db.query(func.count(Chunk.id))
                .filter(Chunk.document_id.in_(user_doc_ids))
                .scalar()
                or 0
            ) if user_doc_ids else 0
        else:
            total_chunks = db.query(func.count(Chunk.id)).scalar() or 0

        processed = (
            doc_query.filter(Document.processing_status == "ready")
            .with_entities(func.count(Document.id))
            .scalar()
            or 0
        )
        processing = (
            doc_query.filter(Document.processing_status == "processing")
            .with_entities(func.count(Document.id))
            .scalar()
            or 0
        )

        total_storage_mb = round(total_bytes / (1024 * 1024), 2)

        return {
            "total_documents": total_documents,
            "processed_documents": processed,
            "processing_documents": processing,
            "total_storage_mb": total_storage_mb,
            "total_chunks": total_chunks,
            "system_status": "healthy",
        }


    @staticmethod
    def embed_document_chunks(db: Session, document_id: str) -> Tuple[int, int]:
        """
        Generates vector embeddings for all SQLite chunks belonging to the document
        and persists them into ChromaDB.
        Returns (document_chunk_count, embedded_vector_count).
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document with id '{document_id}' not found.")

        chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )

        if not chunks:
            doc.embedding_status = "embedded"
            doc.embedded_chunk_count = 0
            db.commit()
            return (0, 0)

        doc.embedding_status = "processing"
        db.commit()

        try:
            texts = [c.content for c in chunks]
            embeddings = EmbeddingService.generate_embeddings(texts)
            chunk_ids = [f"chunk_{c.id}" for c in chunks]
            metadatas = [
                {
                    "document_id": doc.id,
                    "user_id": doc.user_id or "",
                    "chunk_id": c.id,
                    "chunk_index": c.chunk_index,
                    "filename": doc.original_filename or doc.filename,
                    "source_pages": c.source_pages or "",
                    "slide_number": c.slide_number if c.slide_number is not None else -1,
                    "word_count": c.word_count or len(c.content.split()),
                }
                for c in chunks
            ]

            upserted = ChromaService.upsert_chunks(
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            doc.embedding_status = "embedded"
            doc.embedded_chunk_count = upserted
            db.commit()
            db.refresh(doc)
            return (len(chunks), upserted)

        except Exception as e:
            logger.error("Failed to generate embeddings for doc %s: %s", document_id, e)
            doc.embedding_status = "failed"
            db.commit()
            raise e

    @staticmethod
    def embed_all_documents(db: Session) -> Dict[str, Any]:
        """
        Finds all documents and generates embeddings for documents with chunks.
        """
        documents = db.query(Document).all()
        processed = 0
        total_vectors = 0
        errors = []

        for doc in documents:
            if doc.chunk_count > 0:
                try:
                    _, upserted = DocumentService.embed_document_chunks(db, doc.id)
                    processed += 1
                    total_vectors += upserted
                except Exception as err:
                    errors.append({"document_id": doc.id, "error": str(err)})

        return {
            "total_documents": len(documents),
            "documents_embedded": processed,
            "total_vectors_embedded": total_vectors,
            "errors": errors,
        }
