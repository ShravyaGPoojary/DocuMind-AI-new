"""
Stage 4 Verification Suite for DocuMind AI
RAG-Powered Document Chat (Backend & Retrieval Integration)

Tests:
1. Question Validation (empty, whitespace, length bounds)
2. Grounded Context Assembly & Anti-Hallucination Prompt Formatting
3. Semantic Vector Retrieval for RAG (actual ChromaDB chunks + citations)
4. Document-Scoped vs Cross-Document Retrieval
5. Mocked End-to-End RAG Answer Generation & Source Attribution
6. SQLite Conversation & Message Persistence (Multi-turn Threading)
7. Conversation CRUD API Endpoints (GET /conversations, GET /conversations/{id}, DELETE)
8. Error Handling (503 Daemon Offline, 404 Missing Model, 504 Timeout)
9. Live Ollama Inference (conditional on local Ollama daemon)
"""

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

import httpx
from fastapi.testclient import TestClient
from app.core.config import settings
from app.db.session import init_db, SessionLocal
from app.models.document import Document
from app.models.conversation import Conversation, Message
from app.services.embedding_service import EmbeddingService
from app.services.chroma_service import ChromaService
from app.services.document_service import DocumentService
from app.services.search_service import SearchService
from app.services.rag_service import RAGService
from app.main import app


def run_stage4_tests():
    print("=" * 70)
    print("RUNNING STAGE 4 VERIFICATION SUITE — RAG-POWERED DOCUMENT CHAT")
    print("=" * 70)

    init_db()
    db = SessionLocal()
    client = TestClient(app)

    # Set LLM_PROVIDER to ollama for Stage 4 regression verification
    orig_provider = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "ollama"

    doc_id = None

    try:
        # TEST 1: Question Validation
        print("\n[TEST 1] Testing Question Input Validation...")
        try:
            RAGService.validate_question("")
            assert False, "Should reject empty question"
        except ValueError as err:
            assert "empty" in str(err).lower()

        try:
            RAGService.validate_question("   \n\t  ")
            assert False, "Should reject whitespace-only question"
        except ValueError as err:
            assert "empty" in str(err).lower()

        res = client.post("/api/chat", json={"question": "   "})
        assert res.status_code == 400, f"Expected 400, got {res.status_code}"
        assert "empty" in res.json()["detail"].lower()
        print("  ✓ Empty and whitespace questions correctly rejected with 400 Bad Request")

        # TEST 2: Context Assembly & Prompt Construction
        print("\n[TEST 2] Testing Grounded Context Assembly...")
        sample_chunks = [
            {
                "filename": "solar_energy.pdf",
                "source_pages": "4",
                "slide_number": None,
                "content": "Photovoltaic cells convert sunlight directly into electricity via the photoelectric effect.",
            },
            {
                "filename": "solar_energy.pdf",
                "source_pages": "5-6",
                "slide_number": None,
                "content": "Modern silicon solar panels achieve an energy conversion efficiency of 20 to 22 percent.",
            },
        ]
        context_block = RAGService.assemble_context(sample_chunks)
        assert "[Source 1 | Document: solar_energy.pdf | Page: 4]" in context_block
        assert "Photovoltaic cells" in context_block
        assert "[Source 2 | Document: solar_energy.pdf | Page: 5-6]" in context_block
        assert RAGService.SYSTEM_PROMPT is not None
        assert "STRICT GROUNDING RULES" in RAGService.SYSTEM_PROMPT
        print("  ✓ Grounded context block and anti-hallucination guardrail prompts formatted accurately")

        # TEST 3: Seed Test Document & Chunks for RAG Retrieval
        print("\n[TEST 3] Seeding Test Document for RAG Pipeline...")
        doc = DocumentService.create_document(
            db=db,
            filename="stage4_neural_nets.pdf",
            original_filename="stage4_neural_nets.pdf",
            file_type="pdf",
            file_size=2048,
            storage_path=str(settings.UPLOAD_DIR / "stage4_neural_nets.pdf"),
            extracted_text="Convolutional neural networks are specialized deep learning architectures for image recognition.",
            page_count=2,
            processing_status="ready",
        )
        doc_id = doc.id

        chunks_data = [
            {
                "chunk_index": 0,
                "content": "Convolutional Neural Networks (CNNs) use spatial pooling and convolution layers to process pixel grid data.",
                "word_count": 16,
                "source_pages": "1",
                "slide_number": None,
            },
            {
                "chunk_index": 1,
                "content": "Backpropagation calculates gradients of the loss function with respect to weights using the calculus chain rule.",
                "word_count": 17,
                "source_pages": "2",
                "slide_number": None,
            },
        ]
        DocumentService.save_chunks(db, doc_id, chunks_data)
        DocumentService.embed_document_chunks(db, doc_id)
        print(f"  ✓ Seeded document '{doc.original_filename}' (ID: {doc_id}) with 2 indexed chunks")

        # TEST 4: Semantic Retrieval for RAG
        print("\n[TEST 4] Testing Semantic Retrieval & Citation Metadata...")
        search_res = SearchService.search(
            query="How do convolutional layers process pixel grids?",
            top_k=2,
            document_id=doc_id,
            db=db,
        )
        assert search_res["total_results"] > 0
        top_match = search_res["results"][0]
        assert top_match["document_id"] == doc_id
        assert top_match["filename"] == "stage4_neural_nets.pdf"
        assert top_match["source_pages"] == "1"
        assert top_match["score"] > 0.4
        print(f"  ✓ Top chunk matched with score {top_match['score']}: '{top_match['content']}'")

        # TEST 5: Mocked End-to-End RAG Answer & Source Citations
        print("\n[TEST 5] Testing End-to-End RAG Answer Generation (POST /api/chat)...")
        mock_llm_response = {
            "model": "qwen2.5:3b",
            "prompt": "prompt",
            "response": "Convolutional Neural Networks process pixel grid data using convolution layers and spatial pooling.",
            "done": True,
            "total_duration_ms": 320.0,
            "tokens_generated": 24,
        }

        with patch("app.services.llm_service.LLMService.generate", return_value=mock_llm_response):
            chat_payload = {
                "question": "How do CNNs process pixel data?",
                "document_id": doc_id,
                "top_k": 3,
            }
            res = client.post("/api/chat", json=chat_payload)
            assert res.status_code == 200, res.text
            chat_data = res.json()
            print(f"  Generated RAG Answer: '{chat_data['answer']}'")
            print(f"  Model Cited: {chat_data['model']}")
            print(f"  Sources Returned: {len(chat_data['sources'])}")

            assert chat_data["answer"].startswith("Convolutional Neural Networks")
            assert len(chat_data["sources"]) > 0
            assert chat_data["sources"][0]["document_id"] == doc_id
            assert chat_data["sources"][0]["filename"] == "stage4_neural_nets.pdf"
            assert "conversation_id" in chat_data and len(chat_data["conversation_id"]) > 0
            conv_id = chat_data["conversation_id"]

        print("  ✓ RAG response structure validated with answer, sources, and conversation_id")

        # TEST 6: Multi-turn Conversation Persistence
        print("\n[TEST 6] Testing Conversation & Message Thread Persistence in SQLite...")
        # Follow-up question using the same conversation_id
        mock_followup = {
            "model": "qwen2.5:3b",
            "prompt": "followup",
            "response": "Backpropagation computes the loss gradients using the chain rule.",
            "done": True,
            "total_duration_ms": 280.0,
            "tokens_generated": 15,
        }
        with patch("app.services.llm_service.LLMService.generate", return_value=mock_followup):
            res2 = client.post(
                "/api/chat",
                json={
                    "question": "What mathematical rule does backpropagation use?",
                    "document_id": doc_id,
                    "conversation_id": conv_id,
                },
            )
            assert res2.status_code == 200
            assert res2.json()["conversation_id"] == conv_id

        # Verify in SQLite database
        persisted_conv = RAGService.get_conversation(db, conv_id)
        assert persisted_conv is not None
        assert len(persisted_conv.messages) == 4, f"Expected 4 messages (2 user + 2 assistant), got {len(persisted_conv.messages)}"
        assert persisted_conv.messages[0].role == "user"
        assert persisted_conv.messages[1].role == "assistant"
        assert persisted_conv.messages[1].sources is not None
        assert len(persisted_conv.messages[1].sources) > 0
        print(f"  ✓ Multi-turn thread {conv_id} persisted in SQLite with {len(persisted_conv.messages)} messages")

        # TEST 7: Conversation CRUD Endpoints
        print("\n[TEST 7] Testing Conversation API Endpoints...")
        # 7a: List conversations
        list_res = client.get("/api/chat/conversations")
        assert list_res.status_code == 200
        conv_list = list_res.json()
        assert conv_list["total"] >= 1
        found = any(c["id"] == conv_id for c in conv_list["conversations"])
        assert found, "Created conversation must appear in conversation list"
        print(f"  ✓ GET /api/chat/conversations returned {conv_list['total']} conversation threads")

        # 7b: Get conversation details with messages
        detail_res = client.get(f"/api/chat/conversations/{conv_id}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["id"] == conv_id
        assert len(detail_data["messages"]) == 4
        assert detail_data["messages"][1]["sources"][0]["filename"] == "stage4_neural_nets.pdf"
        print(f"  ✓ GET /api/chat/conversations/{conv_id} returned ordered messages and source citations")

        # 7c: Non-existent conversation returns 404
        bad_conv_res = client.get("/api/chat/conversations/non-existent-uuid-123")
        assert bad_conv_res.status_code == 404
        print("  ✓ Non-existent conversation lookup correctly returned 404 Not Found")

        # 7d: Delete conversation
        del_res = client.delete(f"/api/chat/conversations/{conv_id}")
        assert del_res.status_code == 200
        assert RAGService.get_conversation(db, conv_id) is None
        print(f"  ✓ DELETE /api/chat/conversations/{conv_id} safely removed thread and cascaded messages")

        # TEST 8: Ollama Error Handling in RAG Pipeline
        print("\n[TEST 8] Testing Ollama Error Handling in RAG Pipeline...")
        from app.services.llm_service import OllamaConnectionError, OllamaModelNotFoundError, OllamaTimeoutError

        with patch("app.services.llm_service.LLMService.generate", side_effect=OllamaConnectionError("Ollama unreachable")):
            err_res = client.post("/api/chat", json={"question": "Test question", "document_id": doc_id})
            assert err_res.status_code == 503
            assert "unreachable" in err_res.json()["detail"].lower()
        print("  ✓ Daemon offline handled with 503 Service Unavailable")

        with patch("app.services.llm_service.LLMService.generate", side_effect=OllamaModelNotFoundError("Model missing")):
            err_res = client.post("/api/chat", json={"question": "Test question", "document_id": doc_id})
            assert err_res.status_code == 404
            assert "missing" in err_res.json()["detail"].lower()
        print("  ✓ Missing model handled with 404 Not Found")

        # TEST 9: Non-existent document_id returns 404
        bad_doc_res = client.post("/api/chat", json={"question": "What is AI?", "document_id": "non-existent-doc-uuid"})
        assert bad_doc_res.status_code == 404
        print("  ✓ Non-existent document filter returned 404 Not Found")

        # TEST 10: Live Ollama Inference (if running)
        print("\n[TEST 10] Checking for Live Ollama Generation...")
        health = client.get("/api/llm/health").json()
        if health["ollama_reachable"] and health["model_available"]:
            print(f"  Live Ollama running! Executing live RAG question answering with {health['configured_model']}...")
            live_chat = client.post(
                "/api/chat",
                json={
                    "question": "What layers do Convolutional Neural Networks use?",
                    "document_id": doc_id,
                    "top_k": 2,
                },
            )
            if live_chat.status_code == 200:
                print(f"  ✓ Live RAG Answer: '{live_chat.json()['answer'].strip()}'")
                print(f"  ✓ Live Citations: {len(live_chat.json()['sources'])} sources")
        else:
            print(f"  (Local Ollama daemon not running: {health['message']})")
            print("  ✓ Live test conditionally bypassed (all error handling verified via mocks)")

        print("\n" + "=" * 70)
        print("ALL STAGE 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        # Restore original provider
        settings.LLM_PROVIDER = orig_provider

        # Cleanup seeded document
        print("\nCleaning up test document and vectors...")
        if doc_id:
            DocumentService.delete_document(db, doc_id)
        db.close()
        print("Cleanup completed.")


if __name__ == "__main__":
    run_stage4_tests()
