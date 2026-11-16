import io
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

import fitz
import docx
import pptx
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.db.session import init_db, SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.text_cleaner import TextCleaner, clean_text
from app.services.chunking_service import ChunkingService, chunk_text
from app.services.document_extractor import DocumentExtractor, UnsupportedFormatError
from app.services.document_service import DocumentService


def create_sample_pdf_bytes() -> bytes:
    """Creates an in-memory 2-page PDF document."""
    doc = fitz.open()
    # Page 1
    p1 = doc.new_page()
    p1.insert_text((50, 72), "DocuMind AI Architecture - Page 1\n" + "Artificial intelligence document understanding system. " * 30)
    # Page 2
    p2 = doc.new_page()
    p2.insert_text((50, 72), "Retrieval-Augmented Generation Pipeline - Page 2\n" + "Vector search chunking and source grounding mechanism. " * 30)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_sample_docx_bytes() -> bytes:
    """Creates an in-memory DOCX document."""
    doc = docx.Document()
    doc.add_heading("DocuMind Engineering Specification", level=1)
    doc.add_paragraph("This document outlines the core architecture for multi-format document processing.")
    p2 = doc.add_paragraph("Key features include automated chunking, text cleaning, and persistent SQLite storage. " * 15)
    
    # Add a table
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Component"
    table.cell(0, 1).text = "Status"
    table.cell(1, 0).text = "Document Processing"
    table.cell(1, 1).text = "Production Ready"

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def create_sample_pptx_bytes() -> bytes:
    """Creates an in-memory 2-slide PPTX presentation."""
    prs = pptx.Presentation()
    # Slide 1
    s1 = prs.slides.add_slide(prs.slide_layouts[0])
    s1.shapes.title.text = "DocuMind AI Overview"
    if s1.placeholders[1]:
        s1.placeholders[1].text = "Intelligent Multi-Format Document Understanding Platform."
    
    # Slide 2
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Pipeline Architecture"
    if s2.placeholders[1]:
        s2.placeholders[1].text = "Slide 2 Content: Upload -> Text Extraction -> Normalization -> Chunking -> SQLite."

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def run_tests():
    print("==================================================================")
    print("      DOCUMIND AI — STAGE 2B VERIFICATION TEST SUITE             ")
    print("==================================================================")

    init_db()
    client = TestClient(app)
    created_doc_ids = []

    try:
        # 1. Health check
        print("\n[1/10] Checking System Health Endpoint...")
        r = client.get("/api/health")
        assert r.status_code == 200, f"Health check failed: {r.text}"
        assert r.json()["status"] == "ok"
        print("  -> GET /api/health OK (200)")

        # 2. Text Cleaner & Chunking Unit Tests
        print("\n[2/10] Testing Text Sanitization & Chunking Services...")
        raw_sample = "  DocuMind \x00 AI  test\r\n\r\n\r\nwith   spaces.  "
        cleaned = clean_text(raw_sample)
        assert "\x00" not in cleaned
        assert "\r" not in cleaned
        assert "DocuMind AI test" in cleaned
        print("  -> TextCleaner passed")

        long_sample = " ".join([f"term_{i}" for i in range(700)])
        chunks = chunk_text(long_sample, chunk_size=500, chunk_overlap=100)
        assert len(chunks) == 2
        assert chunks[0].word_count == 500
        assert chunks[1].word_count == 300
        print(f"  -> ChunkingService passed (500 size, 100 overlap: 700 words -> {len(chunks)} chunks)")

        # 3. Reject Empty File
        print("\n[3/10] Testing Empty File Rejection...")
        empty_resp = client.post(
            "/api/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert empty_resp.status_code == 400, f"Expected 400, got {empty_resp.status_code}"
        assert "empty" in empty_resp.json()["detail"].lower()
        print("  -> Rejected 0-byte file with 400 Bad Request:", empty_resp.json()["detail"])

        # 4. Reject Unsupported File Type
        print("\n[4/10] Testing Unsupported File Extension Rejection...")
        invalid_resp = client.post(
            "/api/documents/upload",
            files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
        )
        assert invalid_resp.status_code == 400, f"Expected 400, got {invalid_resp.status_code}"
        assert "unsupported" in invalid_resp.json()["detail"].lower()
        print("  -> Rejected .exe file with 400 Bad Request:", invalid_resp.json()["detail"])

        # 5. TXT Document Upload & Processing
        print("\n[5/10] Testing TXT Document Upload & Processing...")
        txt_content = (
            "DocuMind AI Architecture Whitepaper.\n\n"
            "Section 1: Ingestion Pipeline.\n"
            "The platform ingests heterogeneous document formats and normalizes all characters.\n\n"
            "Section 2: Sliding Window Chunking.\n"
            "Documents are broken into 500-word segments with 100-word overlap for high semantic recall.\n"
        ).encode("utf-8")
        
        txt_resp = client.post(
            "/api/documents/upload",
            files={"file": ("whitepaper.txt", txt_content, "text/plain")},
        )
        assert txt_resp.status_code == 201, f"TXT upload failed: {txt_resp.text}"
        txt_data = txt_resp.json()
        assert txt_data["status"] == "ready"
        assert txt_data["file_type"] == "TXT"
        assert txt_data["chunks_count"] >= 1
        created_doc_ids.append(txt_data["id"])
        print(f"  -> Uploaded TXT doc: ID={txt_data['id']}, chunks={txt_data['chunks_count']}")

        # 6. PDF Document Upload & Processing (Multi-page with page preservation)
        print("\n[6/10] Testing Multi-Page PDF Upload & Processing...")
        pdf_bytes = create_sample_pdf_bytes()
        pdf_resp = client.post(
            "/api/documents/upload",
            files={"file": ("system_design.pdf", pdf_bytes, "application/pdf")},
        )
        assert pdf_resp.status_code == 201, f"PDF upload failed: {pdf_resp.text}"
        pdf_data = pdf_resp.json()
        assert pdf_data["status"] == "ready"
        assert pdf_data["file_type"] == "PDF"
        assert pdf_data["pages"] == 2
        assert pdf_data["chunks_count"] >= 1
        created_doc_ids.append(pdf_data["id"])
        print(f"  -> Uploaded PDF doc: ID={pdf_data['id']}, pages={pdf_data['pages']}, chunks={pdf_data['chunks_count']}")

        # 7. DOCX Document Upload & Processing
        print("\n[7/10] Testing DOCX Document Upload & Processing...")
        docx_bytes = create_sample_docx_bytes()
        docx_resp = client.post(
            "/api/documents/upload",
            files={"file": ("specification.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert docx_resp.status_code == 201, f"DOCX upload failed: {docx_resp.text}"
        docx_data = docx_resp.json()
        assert docx_data["status"] == "ready"
        assert docx_data["file_type"] == "DOCX"
        assert docx_data["chunks_count"] >= 1
        created_doc_ids.append(docx_data["id"])
        print(f"  -> Uploaded DOCX doc: ID={docx_data['id']}, chunks={docx_data['chunks_count']}")

        # 8. PPTX Presentation Upload & Processing (Slide numbers preservation)
        print("\n[8/10] Testing PPTX Presentation Upload & Processing...")
        pptx_bytes = create_sample_pptx_bytes()
        pptx_resp = client.post(
            "/api/documents/upload",
            files={"file": ("deck.pptx", pptx_bytes, "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
        )
        assert pptx_resp.status_code == 201, f"PPTX upload failed: {pptx_resp.text}"
        pptx_data = pptx_resp.json()
        assert pptx_data["status"] == "ready"
        assert pptx_data["file_type"] == "PPTX"
        assert pptx_data["pages"] == 2  # 2 slides
        assert pptx_data["chunks_count"] >= 1
        created_doc_ids.append(pptx_data["id"])
        print(f"  -> Uploaded PPTX doc: ID={pptx_data['id']}, slides={pptx_data['pages']}, chunks={pptx_data['chunks_count']}")

        # 9. Verify Chunks API and Document Details API
        print("\n[9/10] Testing Document Details and Chunks Retrieval APIs...")
        # Inspect PPTX chunks
        chunk_resp = client.get(f"/api/documents/{pptx_data['id']}/chunks")
        assert chunk_resp.status_code == 200, f"Chunk retrieval failed: {chunk_resp.text}"
        chunk_data = chunk_resp.json()
        assert chunk_data["total"] >= 1
        first_chunk = chunk_data["chunks"][0]
        assert "DocuMind" in first_chunk["content"]
        assert first_chunk["word_count"] > 0
        print(f"  -> Retrieved {chunk_data['total']} chunks for PPTX document:")
        print(f"     Chunk 0 preview: {first_chunk['content'][:60]}... (word count: {first_chunk['word_count']}, slide: {first_chunk['slide_number']})")

        # Inspect PDF chunks
        pdf_chunks_resp = client.get(f"/api/documents/{pdf_data['id']}/chunks")
        assert pdf_chunks_resp.status_code == 200
        first_pdf_chunk = pdf_chunks_resp.json()["chunks"][0]
        assert first_pdf_chunk["source_pages"] is not None
        print(f"  -> PDF Chunk 0 preserved source pages: {first_pdf_chunk['source_pages']}")

        # Inspect Document Details API
        detail_resp = client.get(f"/api/documents/{pdf_data['id']}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert detail_data["id"] == pdf_data["id"]
        assert detail_data["extracted_text_preview"] is not None
        print(f"  -> GET /api/documents/{pdf_data['id']} details verified")

        # Check library stats
        stats_resp = client.get("/api/documents/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_documents"] >= 4
        assert stats["processed_documents"] >= 4
        assert stats["total_chunks"] >= 4
        print(f"  -> Real SQLite stats: docs={stats['total_documents']}, processed={stats['processed_documents']}, chunks={stats['total_chunks']}, storage={stats['total_storage_mb']}MB")

        # 10. Deletion and Cascade Deletion
        print("\n[10/10] Testing Document Deletion and Cascade Cleanup...")
        doc_to_delete = created_doc_ids.pop()
        
        # Verify file exists on disk before delete
        db = SessionLocal()
        db_doc = DocumentService.get_document(db, doc_to_delete)
        disk_path = Path(db_doc.storage_path) if db_doc else None
        assert disk_path is not None and disk_path.exists(), f"File {disk_path} does not exist on disk!"
        db.close()

        del_resp = client.delete(f"/api/documents/{doc_to_delete}")
        assert del_resp.status_code == 200, f"Delete failed: {del_resp.text}"
        assert del_resp.json()["id"] == doc_to_delete

        # Verify 404 for deleted doc
        get_deleted = client.get(f"/api/documents/{doc_to_delete}")
        assert get_deleted.status_code == 404, "Expected 404 after deletion"

        # Verify file was removed from disk
        assert not disk_path.exists(), f"File {disk_path} was not deleted from backend/uploads/!"

        # Verify chunks cascade deletion in SQLite
        db = SessionLocal()
        orphan_chunks = db.query(Chunk).filter(Chunk.document_id == doc_to_delete).all()
        assert len(orphan_chunks) == 0, f"Expected 0 orphaned chunks, found {len(orphan_chunks)}"
        db.close()
        print(f"  -> Document {doc_to_delete} deleted: DB record removed, original file unlinked, chunks cascade-deleted.")

    finally:
        # Clean up remaining test documents
        print("\nCleaning up test documents from database and uploads directory...")
        for doc_id in created_doc_ids:
            try:
                client.delete(f"/api/documents/{doc_id}")
            except Exception:
                pass
        print("Cleanup completed.")

    print("\n==================================================================")
    print("      ALL STAGE 2B VERIFICATION TESTS PASSED SUCCESSFULLY!        ")
    print("==================================================================")


if __name__ == "__main__":
    run_tests()

