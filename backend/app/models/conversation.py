import uuid
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship
from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title = Column(String(255), nullable=False)
    document_id = Column(
        String(36),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # JSON-encoded array of document UUIDs for multi-document conversations
    document_ids_json = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationship to user
    user = relationship("User", back_populates="conversations")

    # Relationship to Messages with cascading delete

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )

    @property
    def document_ids(self) -> List[str]:
        """Parses document_ids_json into a Python list of document UUID strings."""
        if self.document_ids_json:
            try:
                parsed = json.loads(self.document_ids_json)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [self.document_id] if self.document_id else []

    @document_ids.setter
    def document_ids(self, value: Optional[List[str]]) -> None:
        """Serializes list of document UUID strings into JSON."""
        if value:
            clean_list = [str(d).strip() for d in value if d and str(d).strip()]
            self.document_ids_json = json.dumps(clean_list) if clean_list else None
            # Only assign foreign key document_id when exactly one document is targeted
            if len(clean_list) == 1:
                self.document_id = clean_list[0]
            else:
                self.document_id = None
        else:
            self.document_ids_json = None
            self.document_id = None

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} title={self.title}>"


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(50), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # JSON-encoded array of source citation dicts
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationship back to Conversation
    conversation = relationship("Conversation", back_populates="messages")

    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Parses sources_json into a Python list of dicts."""
        if not self.sources_json:
            return []
        try:
            return json.loads(self.sources_json)
        except Exception:
            return []

    @sources.setter
    def sources(self, value: Optional[List[Dict[str, Any]]]):
        """Serializes list of dicts into JSON string."""
        if value is None:
            self.sources_json = None
        else:
            self.sources_json = json.dumps(value)

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} conv={self.conversation_id}>"

