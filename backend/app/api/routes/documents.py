import re
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.document import Document
from app.models.user import User
from app.api.deps import get_current_user, verify_document_ownership
from app.schemas.document import (

    ChunkListResponse,
    ChunkSchema,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSchema,
    DocumentStatsResponse,
    DocumentEmbedResponse,
    EmbedAllResponse,
    EmbeddingStatusResponse,
    EmbeddingStatsResponse,
)
from app.services.chunking_service import ChunkingService
from app.services.document_extractor import DocumentExtractor, ExtractionError, UnsupportedFormatError
from app.services.document_service import DocumentService
from app.services.text_cleaner import TextCleaner
from app.services.embedding_service import EmbeddingService
from app.services.chroma_service import ChromaService

router = APIRouter(prefix="/documents", tags=["Documents"])


def format_file_size(size_bytes: int) -> str:
    """Formats file size into human-readable B, KB, or MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def doc_to_schema(doc: Document) -> DocumentSchema:
    """Converts a Document SQLAlchemy model instance to DocumentSchema."""
    return DocumentSchema(
        id=doc.id,
        name=doc.original_filename or doc.filename,
        file_type=doc.file_type.upper(),
        size_bytes=doc.file_size,
        size_formatted=format_file_size(doc.file_size),
        uploaded_at=doc.upload_date.strftime("%b %d, %Y") if doc.upload_date else "",
        status=doc.processing_status,
        embedding_status=getattr(doc, "embedding_status", "pending") or "pending",
        embedded_chunk_count=getattr(doc, "embedded_chunk_count", 0) or 0,
        pages=doc.page_count,
        chunks_count=doc.chunk_count,
        token_count_estimate=0,
        description=f"Uploaded {doc.original_filename}",
    )


@router.post(
    "/upload",
    response_model=DocumentSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Process Document",
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    """
    Accepts a document upload (PDF, DOCX, PPTX, TXT, MD, CSV), saves it securely,
    extracts its text content, cleans it, chunks it with 500-word size and 100-word overlap,
    and persists metadata and chunks in SQLite.
    """
    # 1. Validate file extension
    raw_filename = Path(file.filename or "document").name
    ext = Path(raw_filename).suffix.lower()

    if not ext or ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {allowed}",
        )

    clean_ext = ext.lstrip(".")

    # 2. Read and validate content size
    try:
        content = await file.read()
    except Exception as read_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read upload payload: {str(read_err)}",
        )

    file_size = len(content)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb}MB.",
        )

    # 3. Generate safe unique filename and persist to backend/uploads/
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_basename = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_filename)
    unique_filename = f"{uuid.uuid4().hex}_{safe_basename}"
    storage_path = settings.UPLOAD_DIR / unique_filename

    try:
        storage_path.write_bytes(content)
    except Exception as write_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store file on disk: {str(write_err)}",
        )

    # 4. Create document record in database
    doc = DocumentService.create_document(
        db=db,
        filename=unique_filename,
        original_filename=raw_filename,
        file_type=clean_ext,
        file_size=file_size,
        storage_path=str(storage_path),
        processing_status="processing",
        user_id=current_user.id,
    )


    # 5. Execute processing pipeline: Extract -> Clean -> Chunk -> Save
    try:
        extraction = DocumentExtractor.extract(storage_path, clean_ext)
        raw_text = extraction.get("raw_text", "")
        page_count = extraction.get("page_count")
        pages_text = extraction.get("pages_text")

        if not raw_text or not raw_text.strip():
            raise ExtractionError("Document contains no readable text content.")

        cleaned_text = TextCleaner.clean(raw_text)
        if not cleaned_text:
            raise ExtractionError("Document text was empty after sanitization.")

        # Chunking: preserve actual page / slide numbers if available
        if clean_ext in {"pdf", "pptx"} and pages_text and len(pages_text) > 0:
            chunks = ChunkingService.chunk_pages_or_slides(
                pages_text=pages_text,
                is_presentation=(clean_ext == "pptx"),
                chunk_size=settings.DEFAULT_CHUNK_SIZE,
                chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
            )
        else:
            chunks = ChunkingService.chunk_text(
                text=cleaned_text,
                chunk_size=settings.DEFAULT_CHUNK_SIZE,
                chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
            )

        if not chunks:
            raise ExtractionError("Could not generate valid text chunks from document.")

        # Persist chunks in SQLite
        DocumentService.save_chunks(db, doc.id, chunks)

        # Update document record to ready
        DocumentService.update_document_status(
            db,
            document_id=doc.id,
            status="ready",
            chunk_count=len(chunks),
            page_count=page_count,
            extracted_text=cleaned_text,
        )
        db.refresh(doc)

        # Automatically generate dense vector embeddings and index in ChromaDB
        try:
            DocumentService.embed_document_chunks(db, doc.id)
            db.refresh(doc)
        except Exception as embed_err:
            logger.warning(
                "Auto-embedding on upload failed for doc '%s' (can be indexed on first query): %s",
                doc.id,
                embed_err,
            )

    except Exception as proc_err:
        # Prevent orphaned files or inconsistent records
        db.rollback()
        storage_path.unlink(missing_ok=True)
        try:
            DocumentService.delete_document(db, doc.id)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document processing failed: {str(proc_err)}",
        )

    return doc_to_schema(doc)


@router.get("", response_model=DocumentListResponse, summary="List Stored Documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the list of documents persisted in SQLite owned by the authenticated user.
    """
    docs = DocumentService.list_documents(db, skip=skip, limit=limit, user_id=current_user.id)
    items = [doc_to_schema(doc) for doc in docs]
    return DocumentListResponse(total=len(items), documents=items)


@router.get("/stats", response_model=DocumentStatsResponse, summary="Get Document Library Statistics")
async def get_document_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns aggregated metrics such as total document count, storage used, and chunk counts
    computed directly from SQLite storage for the authenticated user.
    """
    stats = DocumentService.get_library_stats(db, user_id=current_user.id)
    return DocumentStatsResponse(
        total_documents=stats["total_documents"],
        processed_documents=stats["processed_documents"],
        processing_documents=stats["processing_documents"],
        total_storage_mb=stats["total_storage_mb"],
        total_chunks=stats["total_chunks"],
        system_status=stats["system_status"],
    )


@router.get(
    "/embedding-stats",
    response_model=EmbeddingStatsResponse,
    summary="Get Vector Database and Embedding Statistics",
)
async def get_embedding_stats():
    """
    Returns statistics regarding the ChromaDB vector collection and SentenceTransformer model.
    """
    stats = ChromaService.get_stats()
    model_info = EmbeddingService.get_model_info()
    return EmbeddingStatsResponse(
        collection_name=stats["collection_name"],
        total_vectors=stats["total_vectors"],
        persist_directory=stats["persist_directory"],
        embedding_model=stats["embedding_model"],
        dimension=model_info["dimension"],
        distance_metric=stats["distance_metric"],
        status=stats["status"],
    )


@router.post(
    "/embed-all",
    response_model=EmbedAllResponse,
    summary="Generate Embeddings for All Eligible Documents",
)
async def embed_all_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Iterates through all documents owned by user in SQLite and generates embeddings in ChromaDB.
    """
    result = DocumentService.embed_all_documents(db)
    return EmbedAllResponse(
        total_documents=result["total_documents"],
        documents_embedded=result["documents_embedded"],
        total_vectors_embedded=result["total_vectors_embedded"],
        message=f"Embedded {result['total_vectors_embedded']} vectors across {result['documents_embedded']} documents.",
        errors=result["errors"],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get Document Details",
)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves full details of a single document by its UUID, enforcing ownership.
    """
    doc = verify_document_ownership(db, document_id, current_user.id)
    base_schema = doc_to_schema(doc)
    preview = (doc.extracted_text[:300] + "...") if doc.extracted_text and len(doc.extracted_text) > 300 else doc.extracted_text

    return DocumentDetailResponse(
        **base_schema.model_dump(),
        error_message=doc.error_message,
        extracted_text_preview=preview,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get Document Chunks",
)
async def get_document_chunks(
    document_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves paginated chunks generated for a specific document, enforcing ownership.
    """
    doc = verify_document_ownership(db, document_id, current_user.id)
    total, chunks = DocumentService.get_document_chunks(db, document_id, skip=skip, limit=limit)
    chunk_schemas = [
        ChunkSchema(
            id=c.id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            content=c.content,
            word_count=c.word_count or len(c.content.split()),
            source_pages=c.source_pages,
            slide_number=c.slide_number,
        )
        for c in chunks
    ]

    page = (skip // limit) + 1 if limit > 0 else 1
    return ChunkListResponse(
        total=total,
        document_id=document_id,
        chunks=chunk_schemas,
        page=page,
        page_size=limit,
    )

    return ChunkListResponse(
        total=total,
        document_id=document_id,
        chunks=chunk_schemas,
        page=page,
        page_size=limit,
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete Document",
)
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Deletes a document from SQLite (cascading to its chunks) and safely
    removes the original uploaded file, enforcing ownership.
    """
    doc = verify_document_ownership(db, document_id, current_user.id)

    # Remove file from disk if path exists
    if doc.storage_path:
        file_path = Path(doc.storage_path)
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass  # Best effort disk removal

    # Delete from database (cascades to chunks)
    DocumentService.delete_document(db, document_id)

    return DocumentDeleteResponse(
        message=f"Document '{doc.original_filename}' and associated chunks deleted successfully.",
        id=document_id,
    )


@router.post(
    "/{document_id}/embed",
    response_model=DocumentEmbedResponse,
    summary="Generate Embeddings for Document Chunks",
)
async def embed_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generates dense vector embeddings for all chunks of a specific document, enforcing ownership.
    """
    doc = verify_document_ownership(db, document_id, current_user.id)

    if doc.chunk_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document '{doc.original_filename}' has 0 chunks to embed.",
        )

    try:
        chunk_count, vector_count = DocumentService.embed_document_chunks(db, document_id)
        return DocumentEmbedResponse(
            document_id=document_id,
            status="embedded",
            total_chunks=chunk_count,
            embedded_chunks=vector_count,
            message=f"Successfully generated and stored {vector_count} vector embeddings in ChromaDB.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}",
        )


@router.get(
    "/{document_id}/embedding-status",
    response_model=EmbeddingStatusResponse,
    summary="Get Document Embedding Status",
)
async def get_document_embedding_status(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns current embedding status and vector counts in ChromaDB, enforcing ownership.
    """
    doc = verify_document_ownership(db, document_id, current_user.id)
    chroma_count = ChromaService.get_document_vector_count(document_id)

    return EmbeddingStatusResponse(
        document_id=doc.id,
        filename=doc.original_filename or doc.filename,
        processing_status=doc.processing_status,
        embedding_status=getattr(doc, "embedding_status", "pending") or "pending",
        chunk_count=doc.chunk_count,
        embedded_chunk_count=getattr(doc, "embedded_chunk_count", 0) or 0,
        chroma_vector_count=chroma_count,
    )


@router.post(
    "/compare",
    summary="Compare Two Documents (Alias)",
    status_code=status.HTTP_200_OK,
)
async def compare_documents_alias(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Alias for POST /api/comparison/compare to support /api/documents/compare, enforcing ownership."""
    from app.schemas.comparison import ComparisonRequest
    from app.services.comparison_service import ComparisonService
    req = ComparisonRequest(**request)
    verify_document_ownership(db, req.document_id_1, current_user.id)
    verify_document_ownership(db, req.document_id_2, current_user.id)
    return await ComparisonService.compare_documents(db=db, request=req)


