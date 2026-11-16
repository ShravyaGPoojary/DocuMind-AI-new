"""
test_stage6_multidoc.py - Comprehensive Verification Suite for Stage 6 Multi-Document Q&A

Verifies:
1. Document IDs filter normalization & deduplication
2. Database migration & ORM persistence (Conversation.document_ids)
3. ChromaDB multi-document $in filtering (strict exclusion of non-selected docs)
4. Balanced Fair Representation retrieval algorithm (preventing document monopoly)
5. Structured context assembly with document boundaries
6. Multi-Document RAG Synthesis via FastAPI endpoint (/api/chat) with Groq
7. Grounded source citation mapping across documents (filename, page/slide, score)
8. Insufficient evidence response handling
9. Multi-document conversation history & persistence (/api/chat/conversations)
10. Search endpoint contract (/api/search) with document_ids
11. Backward compatibility regression checks (single-doc chat, summarization schemas)
"""

import sys
import os
import uuid
import json
import math
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app.main import app
from app.db.session import SessionLocal, init_db
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.conversation import Conversation, Message
from app.services.chroma_service import ChromaService
from app.services.search_service import SearchService
from app.services.rag_service import RAGService
from app.services.document_service import DocumentService
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.search import SearchRequest, SearchResponse


def deterministic_embedding(text: str) -> List[float]:
    """Generates a deterministic 384-dim normalized vector for fast, offline ChromaDB testing."""
    vec = [0.0] * 384
    text_lower = text.lower()
    if "solar" in text_lower or "photovoltaic" in text_lower or "alpha" in text_lower:
        vec[0] += 0.8
        vec[1] += 0.3
    if "wind" in text_lower or "turbine" in text_lower or "beta" in text_lower:
        vec[2] += 0.8
        vec[3] += 0.3
    if "nuclear" in text_lower or "gamma" in text_lower:
        vec[4] += 0.8
        vec[5] += 0.3
    if "energy" in text_lower or "megawatt" in text_lower or "expenditure" in text_lower or "efficiency" in text_lower:
        vec[0] += 0.5
        vec[2] += 0.5
    # Default non-zero component to prevent zero vector
    vec[10] += 0.1
    # Normalize L2
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def mock_generate_embeddings(texts: List[str]) -> List[List[float]]:
    return [deterministic_embedding(t) for t in texts]


def mock_generate_embedding(text: str) -> List[float]:
    return deterministic_embedding(text)


class Stage6TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.db = SessionLocal()
        self.client = TestClient(app)
        self.created_doc_ids = []
        self.created_conv_ids = []

        from app.db.session import get_db
        from app.api.deps import get_current_user
        from app.models.user import User
        self.admin = self.db.query(User).filter(User.email == "admin@documind.ai").first()

        def override_get_db():
            try:
                yield self.db
            finally:
                pass
        app.dependency_overrides[get_db] = override_get_db
        if self.admin:
            app.dependency_overrides[get_current_user] = lambda: self.admin

    def assert_true(self, condition: bool, msg: str):
        if condition:
            print(f"  [PASS] {msg}", flush=True)
            self.passed += 1
        else:
            print(f"  [FAIL] {msg}", flush=True)
            self.failed += 1
            raise AssertionError(msg)

    def cleanup(self):
        try:
            for cid in self.created_conv_ids:
                conv = self.db.query(Conversation).filter(Conversation.id == cid).first()
                if conv:
                    self.db.delete(conv)

            for did in self.created_doc_ids:
                self.db.query(Chunk).filter(Chunk.document_id == did).delete()
                doc = self.db.query(Document).filter(Document.id == did).first()
                if doc:
                    self.db.delete(doc)
                try:
                    ChromaService.delete_document_embeddings(did)
                except Exception:
                    pass

            self.db.commit()
            print("  [CLEANUP] Test resources cleanly purged.", flush=True)
        except Exception as e:
            print(f"  [CLEANUP WARN] Error during cleanup: {e}", flush=True)
        finally:
            app.dependency_overrides.clear()
            self.db.close()

    def run_all(self):
        print("=" * 70, flush=True)
        print("DocuMind AI - Stage 6 Multi-Document Q&A Comprehensive Test Suite", flush=True)
        print("=" * 70, flush=True)

        init_db()

        try:
            self.test_1_filter_normalization()
            self.test_2_conversation_model_persistence()

            with patch("app.services.embedding_service.EmbeddingService.generate_embeddings", side_effect=mock_generate_embeddings), \
                 patch("app.services.embedding_service.EmbeddingService.generate_embedding", side_effect=mock_generate_embedding), \
                 patch("app.services.search_service.EmbeddingService.generate_embedding", side_effect=mock_generate_embedding), \
                 patch("app.services.document_service.EmbeddingService.generate_embeddings", side_effect=mock_generate_embeddings):
                self.test_3_chromadb_multi_doc_filtering_and_fair_retrieval()
                self.test_4_rag_context_assembly()
                self.test_5_rag_multidoc_api_synthesis_and_citations()
                self.test_6_insufficient_evidence_synthesis()
                self.test_7_search_endpoint_contract()
                self.test_8_conversation_persistence_endpoints()
                self.test_9_backward_compatibility()
        finally:
            self.cleanup()

        print("=" * 70, flush=True)
        print(f"Test Suite Summary: {self.passed} PASSED | {self.failed} FAILED", flush=True)
        print("=" * 70, flush=True)
        if self.failed > 0:
            sys.exit(1)

    def test_1_filter_normalization(self):
        print("\n--- TEST 1: Document IDs Filter Normalization ---", flush=True)
        doc_a = str(uuid.uuid4())
        doc_b = str(uuid.uuid4())

        # Test deduplication & whitespace stripping
        raw_list = [f"  {doc_a}  ", doc_b, doc_a, "", None, "   "]
        normalized = list(dict.fromkeys(str(d).strip() for d in raw_list if d and str(d).strip()))
        self.assert_true(len(normalized) == 2, "Normalized list length is exactly 2")
        self.assert_true(normalized == [doc_a, doc_b], "Order preserved and duplicates removed")

        # Test single string handling
        norm_single = list(dict.fromkeys([doc_a.strip()]))
        self.assert_true(norm_single == [doc_a], "Single string normalized to 1-item list")

    def test_2_conversation_model_persistence(self):
        print("\n--- TEST 2: Conversation Model document_ids JSON Property Persistence ---", flush=True)
        doc_1 = str(uuid.uuid4())
        doc_2 = str(uuid.uuid4())
        conv_id = f"test-conv-{uuid.uuid4().hex[:8]}"
        self.created_conv_ids.append(conv_id)

        conv = Conversation(
            id=conv_id,
            title="Multi-Doc Strategy Thread",
            document_id=None,
            user_id=self.admin.id if self.admin else None,
        )
        conv.document_ids = [doc_1, doc_2]
        self.db.add(conv)
        self.db.commit()

        # Query back from SQLite
        loaded = self.db.query(Conversation).filter(Conversation.id == conv_id).first()
        self.assert_true(loaded is not None, "Conversation persisted in SQLite")
        self.assert_true(isinstance(loaded.document_ids, list), "document_ids returns list")
        self.assert_true(len(loaded.document_ids) == 2, "document_ids has length 2")
        self.assert_true(doc_1 in loaded.document_ids and doc_2 in loaded.document_ids, "Contains both doc UUIDs")
        self.assert_true(loaded.document_id is None, "Multi-doc conversation sets foreign key document_id to None")

        # Verify underlying JSON column
        self.assert_true(loaded.document_ids_json is not None, "document_ids_json column populated")
        parsed = json.loads(loaded.document_ids_json)
        self.assert_true(parsed == [doc_1, doc_2], "JSON serialization matches perfectly")

    def test_3_chromadb_multi_doc_filtering_and_fair_retrieval(self):
        print("\n--- TEST 3: ChromaDB Multi-Doc $in Query & Fair Representation Algorithm ---", flush=True)
        doc1_id = f"doc-alpha-{uuid.uuid4().hex[:6]}"
        doc2_id = f"doc-beta-{uuid.uuid4().hex[:6]}"
        doc3_unselected = f"doc-gamma-{uuid.uuid4().hex[:6]}"

        self.created_doc_ids.extend([doc1_id, doc2_id, doc3_unselected])

        # Create Documents in SQLite
        doc1 = Document(
            id=doc1_id,
            filename="Alpha_Solar_Plan.pdf",
            original_filename="Alpha_Solar_Plan.pdf",
            file_type="PDF",
            file_size=2048,
            processing_status="ready",
            chunk_count=2,
            user_id=self.admin.id if self.admin else None,
        )
        doc2 = Document(
            id=doc2_id,
            filename="Beta_Wind_Analysis.pdf",
            original_filename="Beta_Wind_Analysis.pdf",
            file_type="PDF",
            file_size=2048,
            processing_status="ready",
            chunk_count=2,
            user_id=self.admin.id if self.admin else None,
        )
        doc3 = Document(
            id=doc3_unselected,
            filename="Gamma_Unselected.pdf",
            original_filename="Gamma_Unselected.pdf",
            file_type="PDF",
            file_size=2048,
            processing_status="ready",
            chunk_count=1,
            user_id=self.admin.id if self.admin else None,
        )
        self.db.add_all([doc1, doc2, doc3])
        self.db.flush()

        # Create Chunks in SQLite
        chunks = [
            # Doc 1 Chunks
            Chunk(
                document_id=doc1_id,
                chunk_index=0,
                content="Solar photovoltaics generated 450 megawatts in Phase 1 with 98% operational efficiency.",
                word_count=12,
                source_pages="1",
            ),
            Chunk(
                document_id=doc1_id,
                chunk_index=1,
                content="The capital expenditure for solar inverters totaled $1.2 million in FY2025.",
                word_count=11,
                source_pages="2",
            ),
            # Doc 2 Chunks
            Chunk(
                document_id=doc2_id,
                chunk_index=0,
                content="Offshore wind turbine installation delivered 620 megawatts with offshore cabling.",
                word_count=10,
                source_pages="3",
            ),
            Chunk(
                document_id=doc2_id,
                chunk_index=1,
                content="Wind maintenance expenditure totaled $2.4 million across North Sea platforms.",
                word_count=10,
                source_pages="4",
            ),
            # Doc 3 (Unselected) Chunks
            Chunk(
                document_id=doc3_unselected,
                chunk_index=0,
                content="Nuclear fission reactor output delivered 900 megawatts in high temperature tests.",
                word_count=11,
                source_pages="1",
            ),
        ]
        self.db.add_all(chunks)
        self.db.commit()

        DocumentService.embed_document_chunks(self.db, doc1_id)
        DocumentService.embed_document_chunks(self.db, doc2_id)
        DocumentService.embed_document_chunks(self.db, doc3_unselected)

        # Execute Multi-Document Search targeting Doc 1 and Doc 2 only
        search_res = SearchService.search(
            query="energy generation megawatts expenditure",
            top_k=4,
            document_ids=[doc1_id, doc2_id],
            db=self.db,
        )

        results = search_res.get("results", [])
        self.assert_true(len(results) > 0, "Multi-document search returned results")

        # Verify strict exclusion of unselected document
        unselected_hits = [r for r in results if r["document_id"] == doc3_unselected]
        self.assert_true(len(unselected_hits) == 0, "Unselected document (Gamma) strictly excluded")

        # Verify Balanced Representation: Both doc1 and doc2 must have representation
        returned_doc_ids = set(r["document_id"] for r in results)
        self.assert_true(doc1_id in returned_doc_ids, "Alpha Solar represented in top-k results")
        self.assert_true(doc2_id in returned_doc_ids, "Beta Wind represented in top-k results")

        # Verify metadata integrity
        first_match = results[0]
        self.assert_true("filename" in first_match and bool(first_match["filename"]), "Filename present in citation")
        self.assert_true("source_pages" in first_match, "source_pages present in citation")
        self.assert_true("score" in first_match and isinstance(first_match["score"], float), "Similarity score present")

    def test_4_rag_context_assembly(self):
        print("\n--- TEST 4: Cross-Document Context Assembly & Formatting ---", flush=True)
        mock_sources = [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-a",
                "filename": "Q1_Financials.pdf",
                "source_pages": "5",
                "slide_number": None,
                "chunk_index": 2,
                "score": 0.88,
                "content": "Gross margin increased to 42.5% driven by SaaS enterprise subscriptions.",
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-b",
                "filename": "Q2_Financials.pdf",
                "source_pages": "3",
                "slide_number": None,
                "chunk_index": 0,
                "score": 0.85,
                "content": "Gross margin contracted to 39.8% due to higher server infrastructure costs.",
            },
        ]

        context_block = RAGService.assemble_context(mock_sources)
        self.assert_true("Q1_Financials.pdf" in context_block, "First document filename in assembled context")
        self.assert_true("Q2_Financials.pdf" in context_block, "Second document filename in assembled context")
        self.assert_true("Page: 5" in context_block, "Page 5 cited in context")
        self.assert_true("Page: 3" in context_block, "Page 3 cited in context")
        self.assert_true("42.5%" in context_block and "39.8%" in context_block, "Both factual points included")

    def test_5_rag_multidoc_api_synthesis_and_citations(self):
        print("\n--- TEST 5: End-to-End Multi-Document RAG API Synthesis (/api/chat) with Groq ---", flush=True)
        doc1_id = self.created_doc_ids[0]
        doc2_id = self.created_doc_ids[1]

        conv_id = f"test-multidoc-conv-{uuid.uuid4().hex[:6]}"
        self.created_conv_ids.append(conv_id)

        # Mock Groq AsyncGroq completion
        mock_choice = MagicMock()
        mock_choice.message.content = "Based on Alpha_Solar_Plan.pdf and Beta_Wind_Analysis.pdf, solar photovoltaics delivered 450 megawatts with $1.2 million expenditure, whereas offshore wind generated 620 megawatts with $2.4 million maintenance expenditure."
        mock_choice.message.reasoning = None

        mock_usage = MagicMock()
        mock_usage.completion_tokens = 45
        mock_usage.total_time = 0.25

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_completion.model = "openai/gpt-oss-120b"

        with patch("app.services.groq_service.AsyncGroq") as MockAsyncGroq:
            mock_client_instance = AsyncMock()
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
            MockAsyncGroq.return_value = mock_client_instance

            resp = self.client.post(
                "/api/chat",
                json={
                    "question": "Compare the energy generation in megawatts and capital expenditure between solar and wind.",
                    "document_ids": [doc1_id, doc2_id],
                    "top_k": 4,
                    "conversation_id": conv_id,
                },
            )

        self.assert_true(resp.status_code == 200, f"/api/chat returned 200 OK (got {resp.status_code})")
        data = resp.json()
        self.assert_true("answer" in data and len(data["answer"]) > 20, "Answer generated and non-empty")
        self.assert_true("sources" in data and len(data["sources"]) >= 2, "Multiple sources cited across documents")
        self.assert_true(data.get("conversation_id") == conv_id, "Conversation ID matches")
        self.assert_true(data.get("model") == "openai/gpt-oss-120b", "Default Groq model reported")
        self.assert_true("document_ids" in data and len(data["document_ids"]) == 2, "document_ids returned in API response")

        # Verify persistent conversation
        saved_conv = self.db.query(Conversation).filter(Conversation.id == conv_id).first()
        self.assert_true(saved_conv is not None, "Conversation persisted in DB")
        self.assert_true(len(saved_conv.messages) == 2, "Both user and assistant messages saved")
        self.assert_true(len(saved_conv.document_ids) == 2, "Conversation holds 2 document IDs")

        assistant_msg = [m for m in saved_conv.messages if m.role == "assistant"][0]
        self.assert_true(len(assistant_msg.sources) >= 2, "Assistant message persists source citations")

    def test_6_insufficient_evidence_synthesis(self):
        print("\n--- TEST 6: Multi-Document RAG Insufficient Evidence Handling ---", flush=True)
        doc1_id = self.created_doc_ids[0]
        doc2_id = self.created_doc_ids[1]

        mock_insufficient_choice = MagicMock()
        mock_insufficient_choice.message.content = "Based on the provided document context excerpts, there is no information or mention regarding the secret recipe for Belgian chocolate waffles."
        mock_insufficient_choice.message.reasoning = None

        mock_insufficient_usage = MagicMock()
        mock_insufficient_usage.completion_tokens = 25
        mock_insufficient_usage.total_time = 0.15

        mock_insufficient_completion = MagicMock()
        mock_insufficient_completion.choices = [mock_insufficient_choice]
        mock_insufficient_completion.usage = mock_insufficient_usage
        mock_insufficient_completion.model = "openai/gpt-oss-120b"

        with patch("app.services.groq_service.AsyncGroq") as MockAsyncGroq:
            mock_client_instance = AsyncMock()
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_insufficient_completion)
            MockAsyncGroq.return_value = mock_client_instance

            resp = self.client.post(
                "/api/chat",
                json={
                    "question": "What is the secret recipe for Belgian chocolate waffles?",
                    "document_ids": [doc1_id, doc2_id],
                    "top_k": 4,
                },
            )

        self.assert_true(resp.status_code == 200, "Insufficient evidence query returned 200 OK")
        ans = resp.json()["answer"].lower()
        self.assert_true(
            any(phrase in ans for phrase in ["not mentioned", "does not contain", "cannot find", "no information", "not provided", "not available"]),
            "Grounded response admits lack of document evidence without hallucinating",
        )

    def test_7_search_endpoint_contract(self):
        print("\n--- TEST 7: Search Endpoint Contract (/api/search) with document_ids ---", flush=True)
        doc1_id = self.created_doc_ids[0]
        doc2_id = self.created_doc_ids[1]

        resp = self.client.post(
            "/api/search",
            json={
                "query": "solar wind megawatts",
                "top_k": 3,
                "document_ids": [doc1_id, doc2_id],
            },
        )
        self.assert_true(resp.status_code == 200, "/api/search returned 200 OK")
        data = resp.json()
        self.assert_true(data["total_results"] > 0, "Search results found")
        self.assert_true("document_ids_filter" in data and len(data["document_ids_filter"]) == 2, "document_ids_filter in search response")

    def test_8_conversation_persistence_endpoints(self):
        print("\n--- TEST 8: Conversation History Endpoints (/api/chat/conversations) ---", flush=True)
        conv_id = self.created_conv_ids[1]  # Thread created in test 5 with doc1_id and doc2_id
        doc1_id = self.created_doc_ids[0]
        doc2_id = self.created_doc_ids[1]

        # 1. Get conversation detail
        resp = self.client.get(f"/api/chat/conversations/{conv_id}")
        self.assert_true(resp.status_code == 200, "GET /api/chat/conversations/{id} returned 200")
        detail = resp.json()
        self.assert_true(detail["id"] == conv_id, "Conversation detail ID matches")
        self.assert_true(detail.get("document_ids") == [doc1_id, doc2_id], "Conversation detail returns document_ids")
        self.assert_true(len(detail["messages"]) == 2, "Conversation detail has 2 messages")

        # 2. List conversations
        list_resp = self.client.get("/api/chat/conversations")
        self.assert_true(list_resp.status_code == 200, "GET /api/chat/conversations returned 200")
        summaries = list_resp.json()["conversations"]
        target_summary = next((c for c in summaries if c["id"] == conv_id), None)
        self.assert_true(target_summary is not None, "Target conversation found in summary list")
        self.assert_true(target_summary.get("document_ids") == [doc1_id, doc2_id], "Summary item includes document_ids")

    def test_9_backward_compatibility(self):
        print("\n--- TEST 9: Backward Compatibility & Regression Checks ---", flush=True)
        doc1_id = self.created_doc_ids[0]

        # 1. Single document_id request still works cleanly
        single_res = SearchService.search(
            query="solar efficiency",
            top_k=2,
            document_id=doc1_id,
            db=self.db,
        )
        self.assert_true(single_res["total_results"] > 0, "Single document_id search works backwards-compatible")
        for r in single_res["results"]:
            self.assert_true(r["document_id"] == doc1_id, "Single doc search strictly isolates to that document")

        # 2. Entire library search (all docs) works backwards-compatible
        all_res = SearchService.search(
            query="megawatts",
            top_k=4,
            document_id=None,
            document_ids=None,
            db=self.db,
        )
        self.assert_true(all_res["total_results"] > 0, "Entire library search works with no filter")

        # 3. Verify Summarization schemas import without conflict
        from app.schemas.summarization import SummarizationRequest, SummarizationResponse
        sum_req = SummarizationRequest(summary_type="brief")
        self.assert_true(sum_req.summary_type == "brief", "SummarizationRequest unaffected")


if __name__ == "__main__":
    runner = Stage6TestRunner()
    runner.run_all()
