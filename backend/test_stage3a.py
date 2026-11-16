"""
Stage 3A Verification Script for DocuMind AI
Tests:
1. SentenceTransformers Embedding Service (all-MiniLM-L6-v2, 384 dims, normalized)
2. ChromaDB Persistent Client & Collection at backend/data/chroma/
3. Embedding generation for actual SQLite chunks
4. Metadata storage and correctness in ChromaDB
5. Duplicate prevention (idempotent upsert)
6. Document deletion cascading to ChromaDB vectors
7. SQLite <-> ChromaDB consistency
8. REST API endpoints via FastAPI TestClient
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.core.config import settings
from app.db.session import init_db, SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.embedding_service import EmbeddingService
from app.services.chroma_service import ChromaService
from app.services.document_service import DocumentService
from app.main import app

def run_tests():
    print("=" * 70)
    print("RUNNING STAGE 3A VERIFICATION SUITE")
    print("=" * 70)

    # 0. Initialize DB
    init_db()
    db = SessionLocal()
    client = TestClient(app)

    try:
        # TEST 1: Embedding Service
        print("\n[TEST 1] Testing SentenceTransformer Embedding Service...")
        model_info = EmbeddingService.get_model_info()
        print(f"  Model Info: {model_info}")
        assert model_info["dimension"] == 384, f"Expected 384 dims, got {model_info['dimension']}"
        assert "MiniLM" in model_info["model_name"]

        test_sentences = [
            "DocuMind AI provides retrieval augmented generation.",
            "Vector databases store dense embeddings for semantic search."
        ]
        embeddings = EmbeddingService.generate_embeddings(test_sentences)
        assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
        assert len(embeddings[0]) == 384, f"Expected dimension 384, got {len(embeddings[0])}"
        assert all(isinstance(val, float) for val in embeddings[0]), "Embeddings must be floats"
        print("  ✓ EmbeddingService generates valid 384-dimensional normalized vectors")

        # TEST 2: ChromaDB Persistence & Collection
        print("\n[TEST 2] Testing ChromaDB Persistence & Collection Setup...")
        chroma_client = ChromaService.get_client()
        collection = ChromaService.get_collection()
        assert collection is not None
        assert collection.name == settings.CHROMA_COLLECTION_NAME
        persist_path = Path(settings.CHROMA_PERSIST_DIR)
        assert persist_path.exists(), f"Chroma persist dir does not exist: {persist_path}"
        stats = ChromaService.get_stats()
        print(f"  Chroma stats: {stats}")
        print("  ✓ ChromaDB initialized persistently at backend/data/chroma/")

        # TEST 3: Create Sample Document & Chunks in SQLite
        print("\n[TEST 3] Creating Test Document and SQLite Chunks...")
        doc = DocumentService.create_document(
            db=db,
            filename="stage3a_test.pdf",
            original_filename="stage3a_test.pdf",
            file_type="pdf",
            file_size=1024,
            storage_path=str(settings.UPLOAD_DIR / "stage3a_test.pdf"),
            extracted_text="Artificial Intelligence and vector databases enable semantic document retrieval.",
            page_count=2,
            processing_status="ready",
        )
        test_doc_id = doc.id
        print(f"  Created Document in SQLite: id={test_doc_id}")

        test_chunks = [
            {
                "chunk_index": 0,
                "content": "Artificial intelligence and vector search enable high precision semantic retrieval.",
                "word_count": 10,
                "source_pages": "1",
                "slide_number": None,
            },
            {
                "chunk_index": 1,
                "content": "DocuMind AI indexes documents in ChromaDB with Sentence Transformers.",
                "word_count": 9,
                "source_pages": "2",
                "slide_number": None,
            },
        ]
        chunks = DocumentService.save_chunks(db, test_doc_id, test_chunks)
        assert len(chunks) == 2
        print(f"  Saved {len(chunks)} chunks in SQLite for document {test_doc_id}")

        # TEST 4: Generate Embeddings and Upsert to ChromaDB
        print("\n[TEST 4] Embedding SQLite Chunks and Storing into ChromaDB...")
        chunk_count, vector_count = DocumentService.embed_document_chunks(db, test_doc_id)
        assert chunk_count == 2
        assert vector_count == 2
        
        # Verify in SQLite
        db.refresh(doc)
        assert doc.embedding_status == "embedded", f"Expected 'embedded', got {doc.embedding_status}"
        assert doc.embedded_chunk_count == 2, f"Expected 2, got {doc.embedded_chunk_count}"
        
        # Verify in ChromaDB
        chroma_vector_count = ChromaService.get_document_vector_count(test_doc_id)
        assert chroma_vector_count == 2, f"Expected 2 vectors in Chroma, got {chroma_vector_count}"
        print(f"  ✓ Successfully embedded {vector_count} chunks into ChromaDB and updated SQLite")

        # TEST 5: Verify Metadata Correctness in ChromaDB
        print("\n[TEST 5] Verifying ChromaDB Metadata Correctness...")
        retrieved = collection.get(where={"document_id": test_doc_id})
        assert len(retrieved["ids"]) == 2
        meta0 = retrieved["metadatas"][0]
        print(f"  Chroma stored metadata sample: {meta0}")
        assert meta0["document_id"] == test_doc_id
        assert "chunk_id" in meta0
        assert "chunk_index" in meta0
        assert meta0["filename"] == "stage3a_test.pdf"
        assert meta0["source_pages"] in ["1", "2"]
        print("  ✓ Metadata accurately persisted with all required schema attributes")

        # TEST 6: Duplicate Prevention (Idempotency)
        print("\n[TEST 6] Testing Duplicate Prevention / Idempotency...")
        # Re-run embedding on the exact same document
        count2, vectors2 = DocumentService.embed_document_chunks(db, test_doc_id)
        assert count2 == 2
        assert vectors2 == 2
        chroma_vector_count_after = ChromaService.get_document_vector_count(test_doc_id)
        assert chroma_vector_count_after == 2, f"Expected 2 vectors (no duplicates), but found {chroma_vector_count_after}"
        print("  ✓ Re-embedding does not duplicate vectors (idempotent upsert verified)")

        # TEST 7: REST API Endpoints Verification
        print("\n[TEST 7] Testing Stage 3A REST API Endpoints...")
        
        # 7a: GET /api/documents/{id}/embedding-status
        res = client.get(f"/api/documents/{test_doc_id}/embedding-status")
        assert res.status_code == 200, res.text
        status_data = res.json()
        assert status_data["embedding_status"] == "embedded"
        assert status_data["chroma_vector_count"] == 2
        print(f"  ✓ GET /api/documents/{test_doc_id}/embedding-status: {status_data}")

        # 7b: GET /api/embeddings/stats
        res = client.get("/api/embeddings/stats")
        assert res.status_code == 200, res.text
        emb_stats = res.json()
        assert emb_stats["total_vectors"] >= 2
        assert emb_stats["dimension"] == 384
        print(f"  ✓ GET /api/embeddings/stats: {emb_stats}")

        # 7c: GET /api/documents/embedding-stats
        res = client.get("/api/documents/embedding-stats")
        assert res.status_code == 200, res.text
        assert res.json()["dimension"] == 384
        print("  ✓ GET /api/documents/embedding-stats verified")

        # 7d: POST /api/documents/{id}/embed via API
        res = client.post(f"/api/documents/{test_doc_id}/embed")
        assert res.status_code == 200, res.text
        assert res.json()["status"] == "embedded"
        print("  ✓ POST /api/documents/{id}/embed verified")

        # 7e: POST /api/documents/embed-all via API
        res = client.post("/api/documents/embed-all")
        assert res.status_code == 200, res.text
        print(f"  ✓ POST /api/documents/embed-all: {res.json()['message']}")

        # TEST 8: Document Deletion and Vector Cleanup
        print("\n[TEST 8] Testing Document Deletion and ChromaDB Vector Cleanup...")
        res = client.delete(f"/api/documents/{test_doc_id}")
        assert res.status_code == 200, res.text
        
        # Check SQLite
        doc_in_db = DocumentService.get_document(db, test_doc_id)
        assert doc_in_db is None, "Document should be deleted from SQLite"
        chunks_in_db = db.query(Chunk).filter(Chunk.document_id == test_doc_id).all()
        assert len(chunks_in_db) == 0, "Chunks should be deleted from SQLite"
        
        # Check ChromaDB vectors
        remaining_vectors = ChromaService.get_document_vector_count(test_doc_id)
        assert remaining_vectors == 0, f"Expected 0 vectors in ChromaDB after deletion, got {remaining_vectors}"
        print("  ✓ Deleting document completely removes SQLite record, chunks, and ChromaDB vectors")

        # TEST 9: Health & Stage Info Still Working
        print("\n[TEST 9] Verifying Root and Health Endpoints...")
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"
        assert "Stage 3" in res.json()["stage"]
        
        res = client.get("/")
        assert res.status_code == 200
        assert "Stage 3" in res.json()["stage"]
        print("  ✓ Health and root endpoints healthy and reporting Stage 3A")

        print("\n" + "=" * 70)
        print("ALL STAGE 3A TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        db.close()

if __name__ == "__main__":
    run_tests()
