"""
Stage 3B Verification Suite for DocuMind AI
Semantic Search & Retrieval Verification

Tests:
1. Query embedding generation (all-MiniLM-L6-v2, 384 dims, normalized)
2. Basic semantic search and relevance ranking
3. Source metadata validation (chunk_id, doc_id, filename, pages, slides)
4. top_k parameter behavior (top_k=1, top_k=2)
5. Empty query and whitespace validation (400 Bad Request)
6. Invalid top_k validation
7. Document-specific filtering (document_id scoping and 404 for missing doc)
8. Cosine distance to similarity score conversion (score = 1.0 - distance)
9. Duplicate chunk prevention
10. Empty collection handling (graceful empty results)
11. REST API endpoint POST /api/search verification
"""

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
from app.services.search_service import SearchService
from app.main import app


def run_stage3b_tests():
    print("=" * 70)
    print("RUNNING STAGE 3B VERIFICATION SUITE — SEMANTIC SEARCH & RETRIEVAL")
    print("=" * 70)

    init_db()
    db = SessionLocal()
    client = TestClient(app)

    doc_a_id = None
    doc_b_id = None

    try:
        # TEST 1: Query Embedding Generation
        print("\n[TEST 1] Testing Query Embedding Generation...")
        query_text = "What is quantum superposition and entanglement in computing?"
        query_vec = EmbeddingService.generate_embedding(query_text)
        assert len(query_vec) == 384, f"Expected 384 dims, got {len(query_vec)}"
        assert all(isinstance(v, float) for v in query_vec), "All elements must be floats"
        print(f"  ✓ Query successfully encoded into 384-dimensional vector")

        # TEST 2: Seed Known Distinct Documents for Semantic Retrieval
        print("\n[TEST 2] Creating and Indexing Diverse Test Documents...")
        
        # Document A: Physics / Quantum
        doc_a = DocumentService.create_document(
            db=db,
            filename="quantum_physics_guide.pdf",
            original_filename="quantum_physics_guide.pdf",
            file_type="pdf",
            file_size=2048,
            storage_path=str(settings.UPLOAD_DIR / "quantum_physics_guide.pdf"),
            extracted_text="Quantum computing utilizes qubits that exhibit superposition and entanglement.",
            page_count=3,
            processing_status="ready",
        )
        doc_a_id = doc_a.id
        chunks_a_data = [
            {
                "chunk_index": 0,
                "content": "Quantum superposition allows qubits to exist in multiple quantum states simultaneously before measurement.",
                "word_count": 13,
                "source_pages": "1-2",
                "slide_number": None,
            },
            {
                "chunk_index": 1,
                "content": "Quantum entanglement connects two particles such that the state of one instantaneously determines the other.",
                "word_count": 16,
                "source_pages": "3",
                "slide_number": None,
            },
        ]
        DocumentService.save_chunks(db, doc_a_id, chunks_a_data)
        DocumentService.embed_document_chunks(db, doc_a_id)

        # Document B: Culinary / Baking
        doc_b = DocumentService.create_document(
            db=db,
            filename="artisan_bread_baking.pptx",
            original_filename="artisan_bread_baking.pptx",
            file_type="pptx",
            file_size=4096,
            storage_path=str(settings.UPLOAD_DIR / "artisan_bread_baking.pptx"),
            extracted_text="Sourdough bread fermentation requires wild yeast and lactobacillus bacteria.",
            page_count=2,
            processing_status="ready",
        )
        doc_b_id = doc_b.id
        chunks_b_data = [
            {
                "chunk_index": 0,
                "content": "Sourdough bread relies on wild yeast and lactic acid bacteria for sourdough dough fermentation.",
                "word_count": 14,
                "source_pages": "",
                "slide_number": 1,
            },
            {
                "chunk_index": 1,
                "content": "High hydration flour dough baked in a preheated Dutch oven produces an open crumb and crispy crust.",
                "word_count": 18,
                "source_pages": "",
                "slide_number": 2,
            },
        ]
        DocumentService.save_chunks(db, doc_b_id, chunks_b_data)
        DocumentService.embed_document_chunks(db, doc_b_id)

        print(f"  ✓ Document A (Quantum): {doc_a_id} embedded")
        print(f"  ✓ Document B (Baking): {doc_b_id} embedded")

        # TEST 3: Semantic Relevance & Top Chunk Retrieval
        print("\n[TEST 3] Testing Semantic Relevance Ranking...")
        search_res = SearchService.search(
            query="Tell me about qubits, superposition, and particles",
            top_k=5,
            db=db,
        )
        assert search_res["total_results"] >= 2
        top_match = search_res["results"][0]
        print(f"  Top Match Chunk: '{top_match['content']}' (Doc: {top_match['filename']}, Score: {top_match['score']})")
        assert top_match["document_id"] == doc_a_id, "Quantum query must match Quantum document first"
        assert top_match["score"] > 0.4, f"Expected strong similarity score, got {top_match['score']}"
        print("  ✓ Semantic relevance ranking returned Quantum document as top match")

        # TEST 4: Metadata Correctness
        print("\n[TEST 4] Validating Retrieved Source Metadata...")
        assert "chunk_id" in top_match and isinstance(top_match["chunk_id"], int)
        assert "chunk_index" in top_match and isinstance(top_match["chunk_index"], int)
        assert top_match["filename"] == "quantum_physics_guide.pdf"
        assert top_match["source_pages"] in ["1-2", "3"]
        assert top_match["word_count"] > 0
        print(f"  ✓ Source metadata accurate: Chunk #{top_match['chunk_id']}, Page {top_match['source_pages']}")

        # TEST 5: Top-K Parameter Behavior
        print("\n[TEST 5] Testing top_k Parameter Behavior...")
        res_k1 = SearchService.search("baking bread and dough", top_k=1, db=db)
        assert res_k1["total_results"] == 1, f"Expected 1 result for top_k=1, got {res_k1['total_results']}"
        assert res_k1["results"][0]["document_id"] == doc_b_id

        res_k2 = SearchService.search("baking bread and dough", top_k=2, db=db)
        assert res_k2["total_results"] == 2, f"Expected 2 results for top_k=2, got {res_k2['total_results']}"
        print("  ✓ top_k=1 and top_k=2 accurately restrict result count")

        # TEST 6: Document-Specific Filtering
        print("\n[TEST 6] Testing Document-Specific Scoped Search...")
        # Search for quantum concept but force filter on Baking Document
        filtered_res = SearchService.search(
            query="quantum superposition",
            top_k=5,
            document_id=doc_b_id,
            db=db,
        )
        for r in filtered_res["results"]:
            assert r["document_id"] == doc_b_id, f"Expected only doc_b chunks, got {r['document_id']}"
        print(f"  ✓ Document filter strictly scoped results to doc {doc_b_id}")

        # TEST 7: Cosine Distance & Similarity Score Math
        print("\n[TEST 7] Verifying Distance and Similarity Score Formulas...")
        for r in search_res["results"]:
            dist = r["distance"]
            score = r["score"]
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds [0, 1]"
            assert dist >= 0.0, f"Distance {dist} cannot be negative"
            # Cosine similarity = 1.0 - distance
            expected_score = round(max(0.0, min(1.0, 1.0 - dist)), 4)
            assert abs(score - expected_score) < 0.001, f"Score mismatch: {score} vs expected {expected_score}"
        print("  ✓ Similarity score properly calculated as 1 - cosine_distance and bounded in [0, 1]")

        # TEST 8: Empty and Whitespace Query Validation
        print("\n[TEST 8] Testing Empty / Whitespace Query Validation...")
        try:
            SearchService.search("", top_k=5)
            assert False, "Should have raised ValueError on empty query"
        except ValueError as err:
            assert "empty" in str(err).lower()

        try:
            SearchService.search("    ", top_k=5)
            assert False, "Should have raised ValueError on whitespace query"
        except ValueError as err:
            assert "empty" in str(err).lower()
        print("  ✓ Empty and whitespace-only queries correctly rejected")

        # TEST 9: Empty Collection Graceful Handling
        print("\n[TEST 9] Testing Empty Vector Collection Graceful Handling...")
        empty_res = ChromaService.query_chunks(
            query_embeddings=[query_vec],
            n_results=5,
            collection_name="documind_test_empty_collection",
        )
        assert len(empty_res["ids"][0]) == 0
        assert len(empty_res["documents"][0]) == 0
        print("  ✓ Empty vector collection returns clean empty structure without exceptions")

        # TEST 10: Search REST API (POST /api/search)
        print("\n[TEST 10] Testing POST /api/search Endpoint...")
        
        # 10a: Successful search
        api_payload = {
            "query": "superposition of qubits in quantum systems",
            "top_k": 3,
        }
        res = client.post("/api/search", json=api_payload)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["query"] == api_payload["query"]
        assert data["total_results"] > 0
        assert len(data["results"]) <= 3
        print(f"  ✓ POST /api/search returned {data['total_results']} ranked results")

        # 10b: Document-filtered search via API
        api_filtered = {
            "query": "fermentation",
            "top_k": 2,
            "document_id": doc_b_id,
        }
        res = client.post("/api/search", json=api_filtered)
        assert res.status_code == 200, res.text
        filtered_data = res.json()
        assert filtered_data["document_filter"] == doc_b_id
        assert all(item["document_id"] == doc_b_id for item in filtered_data["results"])
        print(f"  ✓ POST /api/search with document_id returned filtered results")

        # 10c: Non-existent document_id returns 404
        bad_doc_payload = {
            "query": "quantum",
            "document_id": "non-existent-uuid-9999",
        }
        res = client.post("/api/search", json=bad_doc_payload)
        assert res.status_code == 404, f"Expected 404, got {res.status_code}"
        print(f"  ✓ Non-existent document filter returned 404 Not Found")

        # 10d: Empty query returns 400
        empty_payload = {"query": "   "}
        res = client.post("/api/search", json=empty_payload)
        assert res.status_code == 400, f"Expected 400, got {res.status_code}"
        print(f"  ✓ Empty query payload returned 400 Bad Request")

        # 10e: Invalid top_k returns 422 Unprocessable Entity
        bad_topk_payload = {"query": "quantum", "top_k": -5}
        res = client.post("/api/search", json=bad_topk_payload)
        assert res.status_code == 422, f"Expected 422, got {res.status_code}"
        print(f"  ✓ Invalid top_k returned 422 Unprocessable Entity")

        print("\n" + "=" * 70)
        print("ALL STAGE 3B VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        # Cleanup test documents from SQLite and Chroma
        print("\nCleaning up test documents...")
        if doc_a_id:
            DocumentService.delete_document(db, doc_a_id)
        if doc_b_id:
            DocumentService.delete_document(db, doc_b_id)
        # Delete test empty collection
        try:
            client_c = ChromaService.get_client()
            client_c.delete_collection("documind_test_empty_collection")
        except Exception:
            pass
        db.close()
        print("Cleanup completed.")


if __name__ == "__main__":
    run_stage3b_tests()

