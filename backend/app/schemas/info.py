from typing import List
from pydantic import BaseModel, Field


class FeatureItem(BaseModel):
    id: str
    title: str
    description: str
    status: str
    stage: str


class SupportedFormat(BaseModel):
    extension: str
    name: str
    mime_type: str
    max_size_mb: int
    description: str


class ProjectInfoResponse(BaseModel):
    name: str = Field(..., description="Project Name")
    tagline: str = Field(..., description="Project Tagline")
    description: str = Field(..., description="Detailed description")
    version: str = Field(..., description="Software version")
    stage: str = Field(..., description="Current development phase")
    architecture: str = Field(default="FastAPI + React 18 + Vite (RAG Ready)")
    supported_formats: List[SupportedFormat] = Field(default_factory=list)
    features: List[FeatureItem] = Field(default_factory=list)
    roadmap: List[str] = Field(default_factory=list)
