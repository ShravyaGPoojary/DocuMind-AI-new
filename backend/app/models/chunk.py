from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    document_id = Column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=True, default=0)
    source_pages = Column(String(255), nullable=True)  # e.g., "1", "1,2", or "3-5"
    slide_number = Column(Integer, nullable=True)     # For presentation files (PPTX)

    # Relationship back to Document
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc_id={self.document_id} index={self.chunk_index}>"

