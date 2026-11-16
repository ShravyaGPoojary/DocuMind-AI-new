import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.main import app, root_redirect
from app.core.config import settings
from app.db.session import init_db, SessionLocal, engine
from app.db.base import Base
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.text_cleaner import TextCleaner, clean_text
from app.services.chunking_service import ChunkingService, chunk_text
from app.services.document_extractor import DocumentExtractor, UnsupportedFormatError
from app.services.document_service import DocumentService
from app.api.routes.health import get_health
from app.api.routes.info import get_project_info
from app.api.routes.documents import list_documents, get_document_stats


def run_coro(coro):
    """Safely steps a coroutine to completion without requiring loopback sockets."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


def test_database_initialization():
    print("[1/6] Testing database initialization...")
    init_db()
    assert settings.DATA_DIR.exists(), "Data directory was not created!"
    db_file = settings.DATA_DIR / settings.DATABASE_NAME
    assert db_file.exists(), f"Database file {db_file} does not exist!"
    print(f"  -> SQLite database file verified at: {db_file}")


def test_document_and_chunk_models():
    print("[2/6] Testing Document & Chunk models, relationships and cascading delete...")
    db = SessionLocal()
    try:
        # 1. Create a document
        doc = DocumentService.create_document(
            db=db,
            filename="test_policy.pdf",
            original_filename="Company_Policy_2026.pdf",
            file_type="pdf",
            file_size=2048576,
            storage_path=str(settings.DATA_DIR / "test_policy.pdf"),
            extracted_text="Sample company policy text for testing document extraction.",
            page_count=5,
            processing_status="ready",
        )
        assert doc.id is not None
        assert doc.processing_status == "ready"
        print(f"  -> Created Document: id={doc.id}, name={doc.original_filename}")

        # 2. Add chunks
        chunks_data = [
            {"chunk_index": 0, "content": "Section 1: Working hours and remote policy.", "source_pages": "1"},
            {"chunk_index": 1, "content": "Section 2: Benefits, insurance and leave.", "source_pages": "2"},
            {"chunk_index": 2, "content": "Section 3: Security, access tokens and devices.", "source_pages": "3"},
        ]
        saved_chunks = DocumentService.save_chunks(db, doc.id, chunks_data)
        assert len(saved_chunks) == 3
        assert doc.chunk_count == 3
        print(f"  -> Created {len(saved_chunks)} chunks for document {doc.id}")

        # 3. Retrieve chunks via relationship
        retrieved_doc = DocumentService.get_document(db, doc.id)
        assert retrieved_doc is not None
        assert len(retrieved_doc.chunks) == 3
        assert retrieved_doc.chunks[0].chunk_index == 0
        print("  -> Relationship verified: retrieved_doc.chunks has 3 chunks")

        # 4. Verify cascade delete
        doc_id = doc.id
        deleted = DocumentService.delete_document(db, doc_id)
        assert deleted is True

        # Document must be gone
        assert DocumentService.get_document(db, doc_id) is None

        # Associated chunks must also be deleted via cascade
        orphan_chunks = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
        assert len(orphan_chunks) == 0, f"Expected 0 chunks after cascade delete, found {len(orphan_chunks)}"
        print("  -> Cascade delete verified: chunks automatically deleted when document deleted")

    finally:
        db.close()


def test_text_cleaner():
    print("[3/6] Testing TextCleaner service...")
    dirty_text = "  Hello \x00 world!\r\n\r\n\r\nThis   is   a\ttest   with   excessive\n\n\n\nspaces.  "
    cleaned = clean_text(dirty_text)

    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "Hello world!" in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Hello")
    print(f"  -> Text cleaner successfully sanitized test input:\n     Original: {repr(dirty_text[:30])}...\n     Cleaned:  {repr(cleaned[:30])}...")


def test_chunking_service():
    print("[4/6] Testing ChunkingService...")
    assert ChunkingService.DEFAULT_CHUNK_SIZE == 500
    assert ChunkingService.DEFAULT_CHUNK_OVERLAP == 100

    # 1. Text smaller than chunk size
    short_text = "Word " * 50
    chunks = chunk_text(short_text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].word_count == 50
    assert chunks[0].chunk_index == 0

    # 2. Text larger than chunk size (700 words, chunk_size=500, overlap=100 -> step=400)
    long_words = [f"w{i}" for i in range(700)]
    long_text = " ".join(long_words)
    long_chunks = chunk_text(long_text, chunk_size=500, chunk_overlap=100)
    assert len(long_chunks) == 2
    assert long_chunks[0].word_count == 500
    assert long_chunks[0].chunk_index == 0
    assert long_chunks[1].word_count == 300
    assert long_chunks[1].chunk_index == 1

    c0_words = long_chunks[0].content.split()
    c1_words = long_chunks[1].content.split()
    assert c0_words[-100:] == c1_words[:100], "Chunk overlap did not match expected words!"
    print(f"  -> Chunking verified: 700 words -> {len(long_chunks)} chunks with 100-word overlap confirmed")

    # 3. Empty text
    assert chunk_text("") == []

    # 4. Invalid inputs raise ValueError
    try:
        chunk_text("test", chunk_size=100, chunk_overlap=100)
        assert False, "Should have raised ValueError for overlap >= size"
    except ValueError:
        pass
    print("  -> Boundary conditions and error validation passed")


def test_document_extractor_foundation():
    print("[5/6] Testing DocumentExtractor foundation...")
    temp_txt = settings.DATA_DIR / "sample_temp.txt"
    temp_txt.write_text("This is a sample document for testing the extractor.", encoding="utf-8")
    try:
        result = DocumentExtractor.extract(temp_txt)
        assert "sample document" in result["raw_text"]
        assert result["file_type"] == "txt"
        assert result["page_count"] == 1
        print("  -> Plain text extraction succeeded")
    finally:
        temp_txt.unlink(missing_ok=True)

    temp_bin = settings.DATA_DIR / "sample.exe"
    temp_bin.write_bytes(b"dummy")
    try:
        try:
            DocumentExtractor.extract(temp_bin)
            assert False, "Should raise UnsupportedFormatError"
        except UnsupportedFormatError:
            print("  -> Unsupported format rejection verified")
    finally:
        temp_bin.unlink(missing_ok=True)


def test_api_endpoints():
    print("[6/6] Testing existing and updated FastAPI endpoints...")

    # 1. Root endpoint
    root_res = run_coro(root_redirect())
    assert "DocuMind" in root_res["message"]
    print("  -> GET / verified:", root_res["message"])

    # 2. Health endpoint
    health_res = run_coro(get_health())
    assert health_res.status == "ok"
    assert "DocuMind" in health_res.app
    print("  -> GET /api/health verified: status=", health_res.status)

    # 3. Info endpoint
    info_res = run_coro(get_project_info())
    assert info_res.name == "DocuMind AI"
    assert len(info_res.supported_formats) >= 5
    print("  -> GET /api/info verified: name=", info_res.name)

    # 4. Documents list endpoint with database session
    db = SessionLocal()
    try:
        docs_res = run_coro(list_documents(db=db))
        assert hasattr(docs_res, "total")
        assert hasattr(docs_res, "documents")
        print(f"  -> GET /api/documents verified: total={docs_res.total}")

        # 5. Documents stats endpoint
        stats_res = run_coro(get_document_stats(db=db))
        assert stats_res.system_status == "healthy"
        assert stats_res.total_documents >= 0
        print(f"  -> GET /api/documents/stats verified: system_status={stats_res.system_status}")
    finally:
        db.close()


if __name__ == "__main__":
    print("==================================================")
    print("  DOCUMIND AI - STAGE 2A VERIFICATION TEST SUITE  ")
    print("==================================================")
    test_database_initialization()
    test_document_and_chunk_models()
    test_text_cleaner()
    test_chunking_service()
    test_document_extractor_foundation()
    test_api_endpoints()
    print("==================================================")
    print("  ALL STAGE 2A TESTS PASSED SUCCESSFULLY!          ")
    print("==================================================")
