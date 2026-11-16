from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Payload for executing a semantic search query."""

    query: str = Field(..., min_length=1, description="Natural language query string")
    top_k: Optional[int] = Field(5, ge=1, le=100, description="Maximum number of relevant chunks to retrieve")
    document_id: Optional[str] = Field(None, description="Optional document UUID to scope search to a single document")
    document_ids: Optional[List[str]] = Field(None, description="Optional list of document UUIDs to scope search across multiple documents")


class SearchResultItem(BaseModel):
    """A single matched document chunk with content, metadata, and relevance metrics."""

    chunk_id: int = Field(..., description="SQLite primary key ID of chunk")
    document_id: str = Field(..., description="UUID of the parent document")
    chunk_index: int = Field(..., description="Zero-based chunk position index")
    content: str = Field(..., description="Actual chunk text content")
    filename: str = Field(..., description="Original filename of the document")
    source_pages: Optional[str] = Field(None, description="Page number(s) (e.g. '1' or '3-5')")
    slide_number: Optional[int] = Field(None, description="Slide number for presentation files")
    word_count: int = Field(0, description="Word count of the chunk content")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0, higher is more relevant)")
    distance: float = Field(..., description="Raw ChromaDB cosine distance (lower is closer)")


class SearchResponse(BaseModel):
    """Structured response returned by semantic search."""

    query: str = Field(..., description="The original search query executed")
    total_results: int = Field(..., description="Count of matched chunks returned")
    document_filter: Optional[str] = Field(None, description="Document ID filter applied, if any")
    document_ids_filter: List[str] = Field(default_factory=list, description="Document IDs filter applied, if any")
    results: List[SearchResultItem] = Field(default_factory=list, description="Ranked list of relevant chunks")

