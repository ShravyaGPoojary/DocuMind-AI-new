from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 1. Information Extraction Schemas
# -----------------------------------------------------------------------------
class ExtractionEntityItem(BaseModel):
    name: str = Field(..., description="Name of the entity")
    category: str = Field(..., description="Entity category: Organization, Technology, Product, Person, Location, Standard")
    description: str = Field(..., description="Contextual role or description of the entity in the document")
    source_page: Optional[str] = Field(None, description="Page number or section reference")


class ExtractionDateItem(BaseModel):
    date_str: str = Field(..., description="Extracted date, year, deadline, or timeframe")
    event_description: str = Field(..., description="Event, milestone, or occurrence tied to this date")
    source_page: Optional[str] = Field(None, description="Page number or section reference")


class ExtractionDefinitionItem(BaseModel):
    term: str = Field(..., description="Technical term, acronym, or concept defined in text")
    definition: str = Field(..., description="Clear explanation or definition of the term as stated in document")
    source_page: Optional[str] = Field(None, description="Page number or section reference")
    context: Optional[str] = Field(None, description="Short supporting excerpt")


class ExtractionMetricItem(BaseModel):
    metric_name: str = Field(..., description="Name of the measured statistic, benchmark, or KPI")
    value: str = Field(..., description="Exact numerical value, percentage, or currency")
    context: str = Field(..., description="What the metric measures or signifies")
    source_page: Optional[str] = Field(None, description="Page number or section reference")


class ExtractionResponse(BaseModel):
    document_id: str
    document_name: str
    entities: List[ExtractionEntityItem] = []
    dates_and_events: List[ExtractionDateItem] = []
    definitions: List[ExtractionDefinitionItem] = []
    key_metrics: List[ExtractionMetricItem] = []
    model: str
    processing_time_ms: float


# -----------------------------------------------------------------------------
# 2. Document Insights Schemas
# -----------------------------------------------------------------------------
class StructuralSectionItem(BaseModel):
    section_name: str
    overview: str
    key_takeaway: Optional[str] = None


class DocumentInsightsResponse(BaseModel):
    document_id: str
    document_name: str
    executive_summary: str
    key_topics: List[str] = []
    major_findings: List[str] = []
    risks_and_challenges: List[str] = []
    structural_roadmap: List[StructuralSectionItem] = []
    strategic_takeaways: List[str] = []
    model: str
    processing_time_ms: float


# -----------------------------------------------------------------------------
# 3. Question Generation Schemas
# -----------------------------------------------------------------------------
class QuestionItem(BaseModel):
    id: str
    question: str
    answer: str
    question_type: str = "conceptual"  # conceptual, analytical, factual
    difficulty: str = "medium"         # easy, medium, hard
    source_pages: Optional[str] = None
    evidence_excerpt: Optional[str] = None


class QuestionGenerationRequest(BaseModel):
    num_questions: int = Field(5, ge=1, le=15, description="Number of questions to generate (1-15)")
    difficulty: Optional[str] = Field("mixed", description="Target difficulty: easy, medium, hard, or mixed")


class QuestionGenerationResponse(BaseModel):
    document_id: str
    document_name: str
    questions: List[QuestionItem] = []
    model: str
    processing_time_ms: float


# -----------------------------------------------------------------------------
# 4. Contextual Analysis Schemas
# -----------------------------------------------------------------------------
class EvidencePoint(BaseModel):
    claim: str
    source_pages: Optional[str] = None
    excerpt: Optional[str] = None
    significance: str


class ContextualAnalysisRequest(BaseModel):
    inquiry: str = Field(..., min_length=3, max_length=1000, description="Hypothesis, topic, or inquiry for cross-section analysis")


class ContextualAnalysisResponse(BaseModel):
    document_id: str
    document_name: str
    inquiry: str
    thesis_assessment: str
    supporting_evidence: List[EvidencePoint] = []
    counter_evidence_or_limitations: List[EvidencePoint] = []
    synthesis_and_implications: str
    citations: List[Dict[str, Any]] = []
    model: str
    processing_time_ms: float

