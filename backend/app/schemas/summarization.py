from typing import List, Optional
from pydantic import BaseModel, Field


class SummarySectionItem(BaseModel):
    """A structured section within a generated summary."""

    title: str = Field(..., description="Section title or heading")
    content: Optional[str] = Field(None, description="Narrative paragraph text for this section")
    bullets: Optional[List[str]] = Field(default_factory=list, description="Bulleted takeaway points")


class SummarizationRequest(BaseModel):
    """Request payload for generating an intelligent document summary."""

    summary_type: Optional[str] = Field(
        "brief",
        description="Type of summary to generate: 'brief', 'detailed', 'key_points', or 'executive'",
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider override ('groq' or 'ollama'). Defaults to configured LLM_PROVIDER ('groq').",
    )
    model: Optional[str] = Field(
        None,
        description="Optional model identifier override",
    )
    background: Optional[bool] = Field(
        False,
        description="If True, dispatches summarization as a background job and returns immediately with a job ID.",
    )


class SummarizationResponse(BaseModel):
    """Structured response containing generated summary and metadata."""

    document_id: str = Field(..., description="UUID of the summarized document")
    document_name: str = Field(..., description="Filename of the summarized document")
    summary_type: str = Field(..., description="Normalized summary type requested")
    summary_type_label: str = Field(..., description="Human-readable summary type label")
    summary: str = Field(..., description="Complete generated markdown summary text")
    key_points: List[str] = Field(default_factory=list, description="Extracted core bullet points")
    sections: List[SummarySectionItem] = Field(default_factory=list, description="Structured parsed sections")
    model: str = Field(..., description="LLM model used to generate the summary")
    provider: str = Field(..., description="LLM provider used ('groq' or 'ollama')")
    chunk_count: int = Field(..., description="Number of document chunks processed")
    word_count: int = Field(..., description="Word count of the generated summary")
    strategy_used: str = Field("direct", description="Summarization strategy: 'direct' or 'map_reduce'")
    generation_time_ms: Optional[float] = Field(None, description="Time taken to generate in milliseconds")


class SummaryTypeItem(BaseModel):
    """Metadata describing a supported summary type option."""

    id: str = Field(..., description="Summary type identifier, e.g. 'brief'")
    title: str = Field(..., description="Display title")
    description: str = Field(..., description="Description of the summary depth and focus")


class SummaryTypesResponse(BaseModel):
    """List of available summary types supported by the platform."""

    types: List[SummaryTypeItem] = Field(default_factory=list)


class SummarizationJobResponse(BaseModel):
    """Status and progress response for a background summarization job."""

    job_id: str = Field(..., description="Unique UUID of the summarization task")
    document_id: str = Field(..., description="UUID of the document being summarized")
    document_name: str = Field(..., description="Filename of the target document")
    summary_type: str = Field(..., description="Summary type requested")
    summary_type_label: str = Field(..., description="Human-readable summary type label")
    status: str = Field(..., description="Job status: pending, processing, completed, failed, paused")
    strategy: str = Field("map_reduce", description="Summarization strategy: direct or map_reduce")
    provider: str = Field("groq", description="Provider used")
    model: str = Field(..., description="Model configured")
    total_chunks: int = Field(0, description="Total chunks in document")
    processed_chunks: int = Field(0, description="Number of document chunks processed so far")
    total_batches: int = Field(0, description="Total map batches to process")
    completed_batches: int = Field(0, description="Number of successfully processed batches")
    current_batch: int = Field(0, description="Currently active batch index (1-based)")
    current_stage: str = Field("map", description="Current summarization stage: map, reduce, or complete")
    progress_percent: int = Field(0, description="Completion percentage (0-100)")
    progress_message: str = Field("Initializing...", description="Human-readable progress status")
    error_message: Optional[str] = Field(None, description="Error details if status is failed")
    result: Optional[SummarizationResponse] = Field(None, description="Final parsed summary once completed")
    elapsed_ms: Optional[float] = Field(None, description="Total processing time in ms")


