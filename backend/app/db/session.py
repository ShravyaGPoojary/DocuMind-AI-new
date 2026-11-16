from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# Ensure data, uploads, and chroma persistence directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Connection arguments for SQLite (necessary for multithreading / FastAPI)
connect_args = {}
if "sqlite" in settings.sync_database_url:
    connect_args["check_same_thread"] = False

# Create single SQLAlchemy engine
engine = create_engine(
    settings.sync_database_url,
    connect_args=connect_args,
    echo=False,
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enforce foreign key constraints in SQLite."""
    if "sqlite" in settings.sync_database_url:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency providing a database session per request.
    Ensures safe rollback on exception and guaranteed closure.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initializes the database: ensures data, upload, and chroma directories exist
    and creates all declared tables if they do not already exist.
    Also ensures any new schema columns are added to existing SQLite tables.
    """
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    from app.db.base import Base
    Base.metadata.create_all(bind=engine)

    # Automated lightweight migration for SQLite schema upgrades
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "summarization_batches" in inspector.get_table_names():
        batch_cols = {col["name"] for col in inspector.get_columns("summarization_batches")}
        with engine.begin() as conn:
            if "cache_key" not in batch_cols:
                conn.execute(text("ALTER TABLE summarization_batches ADD COLUMN cache_key VARCHAR(128)"))
            if "content_hash" not in batch_cols:
                conn.execute(text("ALTER TABLE summarization_batches ADD COLUMN content_hash VARCHAR(64)"))
            if "prompt_version" not in batch_cols:
                conn.execute(text("ALTER TABLE summarization_batches ADD COLUMN prompt_version VARCHAR(30) DEFAULT 'v1'"))
            if "stage" not in batch_cols:
                conn.execute(text("ALTER TABLE summarization_batches ADD COLUMN stage VARCHAR(30) DEFAULT 'map'"))
            if "level" not in batch_cols:
                conn.execute(text("ALTER TABLE summarization_batches ADD COLUMN level INTEGER DEFAULT 0"))
            if "token_count" not in batch_cols:
                conn.execute(text("ALTER TABLE summarization_batches ADD COLUMN token_count INTEGER DEFAULT 0"))

    if "summarization_jobs" in inspector.get_table_names():
        job_cols = {col["name"] for col in inspector.get_columns("summarization_jobs")}
        with engine.begin() as conn:
            if "processed_chunks" not in job_cols:
                conn.execute(text("ALTER TABLE summarization_jobs ADD COLUMN processed_chunks INTEGER DEFAULT 0"))
            if "current_stage" not in job_cols:
                conn.execute(text("ALTER TABLE summarization_jobs ADD COLUMN current_stage VARCHAR(50) DEFAULT 'map'"))

    if "conversations" in inspector.get_table_names():
        conv_cols = {col["name"] for col in inspector.get_columns("conversations")}
        with engine.begin() as conn:
            if "document_ids_json" not in conv_cols:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN document_ids_json TEXT"))
            if "user_id" not in conv_cols:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN user_id VARCHAR(36)"))

    if "documents" in inspector.get_table_names():
        doc_cols = {col["name"] for col in inspector.get_columns("documents")}
        with engine.begin() as conn:
            if "user_id" not in doc_cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN user_id VARCHAR(36)"))

    # Safe data migration: ensure a default admin user exists and assign legacy unowned records
    from app.core.security import hash_password
    with SessionLocal() as db_session:
        from app.models.user import User
        from app.models.document import Document
        from app.models.conversation import Conversation

        default_admin = db_session.query(User).filter(User.email == "admin@documind.ai").first()
        if not default_admin:
            default_admin = User(
                email="admin@documind.ai",
                full_name="DocuMind Admin",
                password_hash=hash_password("DocuMind@2026"),
                is_active=True,
            )
            db_session.add(default_admin)
            db_session.commit()
            db_session.refresh(default_admin)

        # Migrate any unowned documents to default_admin
        unowned_docs = db_session.query(Document).filter(
            (Document.user_id.is_(None)) | (Document.user_id == "")
        ).all()
        if unowned_docs:
            for doc in unowned_docs:
                doc.user_id = default_admin.id
            db_session.commit()

        # Migrate any unowned conversations to default_admin
        unowned_convs = db_session.query(Conversation).filter(
            (Conversation.user_id.is_(None)) | (Conversation.user_id == "")
        ).all()
        if unowned_convs:
            for conv in unowned_convs:
                conv.user_id = default_admin.id
            db_session.commit()



