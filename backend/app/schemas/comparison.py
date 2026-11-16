from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ComparisonRequest(BaseModel):
    """Payload for initiating a cross-document comparison."""
    document_id_1: str = Field(..., description="ID of the first document to compare")
    document_id_2: str = Field(..., description="ID of the second document to compare")
    focus_area: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional thematic focus or specific comparison topic (e.g. 'Methodology', 'Pricing', 'Security')"
    )
    provider: Optional[str] = Field("groq", description="LLM provider: 'groq' (default) or 'ollama'")
    model: Optional[str] = Field(None, description="Optional model override")


class ComparisonFindingItem(BaseModel):
    """An individual comparison finding (similarity, difference, or agreement)."""
    topic: str = Field(..., description="Topic or category of the finding")
    description: str = Field(..., description="Detailed explanation of the similarity, difference, or agreement")
    source_document: Optional[str] = Field(None, description="Document filename or 'Both Documents'")
    document_id: Optional[str] = Field(None, description="Document ID if specific to one document")
    source_pages: Optional[str] = Field(None, description="Page number(s) if applicable")
    slide_number: Optional[int] = Field(None, description="Slide number if applicable")
    chunk_id: Optional[int] = Field(None, description="Originating chunk ID")
    excerpt: Optional[str] = Field(None, description="Exact or supporting excerpt from the text")


class ComparisonContradictionItem(BaseModel):
    """A direct contradiction, conflict, or divergent stance between the two documents."""
    topic: str = Field(..., description="The subject or claim where the documents contradict")
    doc1_claim: str = Field(..., description="Statement, fact, or conclusion asserted in Document 1")
    doc1_source: str = Field(..., description="Filename of Document 1")
    doc1_pages: Optional[str] = Field(None, description="Page(s) or slide in Document 1")
    doc1_excerpt: Optional[str] = Field(None, description="Direct supporting excerpt from Document 1")
    doc2_claim: str = Field(..., description="Conflicting statement, fact, or conclusion asserted in Document 2")
    doc2_source: str = Field(..., description="Filename of Document 2")
    doc2_pages: Optional[str] = Field(None, description="Page(s) or slide in Document 2")
    doc2_excerpt: Optional[str] = Field(None, description="Direct supporting excerpt from Document 2")
    analysis: str = Field(..., description="Analytical synthesis explaining the nature and significance of the conflict")


class ComparisonCitation(BaseModel):
    """Source citation anchoring comparison findings in grounded document chunks."""
    document_id: str
    document_name: str
    chunk_index: int
    chunk_id: Optional[int] = None
    source_pages: Optional[str] = None
    slide_number: Optional[int] = None
    excerpt: str
    relevance_topic: Optional[str] = None


class ComparisonDocumentMetadata(BaseModel):
    """Metadata summary of a document being compared."""
    id: str
    filename: str
    original_filename: str
    file_type: str
    file_size_formatted: str
    page_count: Optional[int] = None
    chunk_count: int


class ComparisonResponse(BaseModel):
    """Comprehensive structured response for Document Comparison & Cross-Document Analysis."""
    comparison_id: str
    document_1: ComparisonDocumentMetadata
    document_2: ComparisonDocumentMetadata
    focus_area: Optional[str] = None
    comparison_summary: str
    similarities: List[ComparisonFindingItem] = []
    differences: List[ComparisonFindingItem] = []
    shared_concepts: List[str] = []
    unique_to_doc1: List[str] = []
    unique_to_doc2: List[str] = []
    contradictions: List[ComparisonContradictionItem] = []
    agreements: List[ComparisonFindingItem] = []
    citations: List[ComparisonCitation] = []
    provider: str
    model: str
    processing_time_ms: float

