import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    upload_date = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    processing_status = Column(String(50), default="pending", nullable=False)  # pending, processing, ready, error
    embedding_status = Column(String(50), default="pending", nullable=False)   # pending, processing, embedded, failed
    embedded_chunk_count = Column(Integer, default=0, nullable=False)
    extracted_text = Column(Text, nullable=True)
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0, nullable=False)
    storage_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationship to user
    user = relationship("User", back_populates="documents")

    # Relationship to chunks with cascading deletion
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.processing_status}>"

