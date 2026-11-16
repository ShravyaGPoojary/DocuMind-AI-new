from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentSchema(BaseModel):
    id: str
    name: str
    file_type: str
    size_bytes: int
    size_formatted: str
    uploaded_at: str
    status: str = Field(..., description="'ready', 'processing', 'draft', or 'error'")
    embedding_status: Optional[str] = Field("pending", description="'pending', 'processing', 'embedded', or 'failed'")
    embedded_chunk_count: Optional[int] = 0
    pages: Optional[int] = None
    chunks_count: Optional[int] = 0
    token_count_estimate: Optional[int] = 0
    description: Optional[str] = None


class DocumentDetailResponse(DocumentSchema):
    error_message: Optional[str] = None
    extracted_text_preview: Optional[str] = None


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentSchema]


class DocumentStatsResponse(BaseModel):
    total_documents: int
    processed_documents: int
    processing_documents: int
    total_storage_mb: float
    total_chunks: int
    system_status: str


class ChunkSchema(BaseModel):
    id: int
    document_id: str
    chunk_index: int
    content: str
    word_count: int = 0
    source_pages: Optional[str] = None
    slide_number: Optional[int] = None


class ChunkListResponse(BaseModel):
    total: int
    document_id: str
    chunks: List[ChunkSchema]
    page: int = 1
    page_size: int = 50


class DocumentDeleteResponse(BaseModel):
    message: str
    id: str


class DocumentEmbedResponse(BaseModel):
    document_id: str
    status: str
    total_chunks: int
    embedded_chunks: int
    message: str


class EmbedAllResponse(BaseModel):
    total_documents: int
    documents_embedded: int
    total_vectors_embedded: int
    message: str
    errors: List[Dict[str, Any]] = []


class EmbeddingStatusResponse(BaseModel):
    document_id: str
    filename: str
    processing_status: str
    embedding_status: str
    chunk_count: int
    embedded_chunk_count: int
    chroma_vector_count: int


class EmbeddingStatsResponse(BaseModel):
    collection_name: str
    total_vectors: int
    persist_directory: str
    embedding_model: str
    dimension: int
    distance_metric: str
    status: str

