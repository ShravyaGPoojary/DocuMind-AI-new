import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.db.base import Base


class SummarizationJob(Base):
    """
    Tracks asynchronous or background document summarization workflows.
    Provides real-time progress tracking, batch counters, and resumability.
    """
    __tablename__ = "summarization_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    summary_type = Column(String(50), nullable=False, default="brief")
    provider = Column(String(50), nullable=False, default="groq")
    model = Column(String(100), nullable=False, default="openai/gpt-oss-120b")
    status = Column(String(30), nullable=False, default="pending", index=True)  # pending, processing, completed, failed, paused
    strategy = Column(String(30), nullable=False, default="direct")  # direct, map_reduce

    total_chunks = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)
    total_batches = Column(Integer, default=0)
    completed_batches = Column(Integer, default=0)
    current_batch = Column(Integer, default=0)
    current_stage = Column(String(50), default="map")
    progress_percent = Column(Integer, default=0)
    progress_message = Column(String(255), default="Initializing summarization...")
    error_message = Column(Text, nullable=True)

    # Serialized JSON of full SummarizationResponse upon successful completion
    result_json = Column(Text, nullable=True)
    elapsed_ms = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("Document", backref="summarization_jobs")
    batches = relationship(
        "SummarizationBatch",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="SummarizationBatch.batch_index",
    )


class SummarizationBatch(Base):
    """
    Stores intermediate section digests from Map-Reduce summarization passes.
    Enables instant resumability for interrupted or rate-limited summarization jobs.
    Uses stable cache_key and content_hash so intermediate summaries can be reused
    even if adaptive batch boundaries shift between runs.
    """
    __tablename__ = "summarization_batches"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey("summarization_jobs.id", ondelete="CASCADE"), nullable=True, index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    batch_index = Column(Integer, nullable=False)
    start_chunk = Column(Integer, nullable=False)
    end_chunk = Column(Integer, nullable=False)
    chunk_count = Column(Integer, nullable=False, default=2)

    # Stable content-aware cache identifiers
    cache_key = Column(String(128), nullable=True, index=True)
    content_hash = Column(String(64), nullable=True, index=True)
    prompt_version = Column(String(30), default="v1", nullable=True)
    stage = Column(String(30), default="map", nullable=True)  # map, reduce_level_1, reduce_final
    level = Column(Integer, default=0, nullable=True)
    token_count = Column(Integer, default=0, nullable=True)

    summary_text = Column(Text, nullable=False)
    status = Column(String(30), nullable=False, default="completed")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    job = relationship("SummarizationJob", back_populates="batches")

    __table_args__ = (
        Index("ix_doc_batch_idx", "document_id", "batch_index"),
        Index("ix_doc_content_hash", "document_id", "content_hash"),
        Index("ix_batch_cache_key", "cache_key"),
    )
