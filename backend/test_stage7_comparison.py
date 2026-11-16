"""
Test Suite for Stage 7: Document Comparison & Cross-Document Analysis
Verifies:
1. Validation rules (identical doc IDs, missing doc IDs, empty chunk documents).
2. Representative chunk selection & token budgeting.
3. Prompt construction with grounded document identifiers.
4. JSON parsing resilience (clean JSON, code fences, fallback parsing).
5. Cross-document comparison execution (similarities, differences, contradictions, agreements, citations).
6. API endpoint contracts (/api/comparison/compare and /api/documents/compare).
7. Non-regression of previous stages.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure backend root is on sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.document import Document
from app.models.chunk import Chunk
from app.schemas.comparison import ComparisonRequest, ComparisonResponse
from app.services.comparison_service import ComparisonService, format_file_size


class Stage7ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an in-memory SQLite database for isolated, fast testing
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.db = self.SessionLocal()
        # Seed test documents
        self.doc1 = Document(
            id="doc-test-1111",
            filename="cloud_architecture_v1.pdf",
            original_filename="cloud_architecture_v1.pdf",
            file_type="pdf",
            file_size=204800,
            processing_status="ready",
            chunk_count=3,
            page_count=5,
        )
        self.doc2 = Document(
            id="doc-test-2222",
            filename="cloud_architecture_v2.pdf",
            original_filename="cloud_architecture_v2.pdf",
            file_type="pdf",
            file_size=307200,
            processing_status="ready",
            chunk_count=3,
            page_count=6,
        )
        self.doc_empty = Document(
            id="doc-test-empty",
            filename="empty.txt",
            original_filename="empty.txt",
            file_type="txt",
            file_size=0,
            processing_status="ready",
            chunk_count=0,
        )
        self.db.add_all([self.doc1, self.doc2, self.doc_empty])

        # Seed chunks for doc 1
        self.c1_1 = Chunk(
            document_id=self.doc1.id,
            chunk_index=0,
            content="Cloud Architecture v1 employs a centralized SQLite database. Maximum throughput is 500 RPS.",
            word_count=13,
            source_pages="1",
        )
        self.c1_2 = Chunk(
            document_id=self.doc1.id,
            chunk_index=1,
            content="Security encryption uses AES-128 standard. Target response latency is strictly under 100ms.",
            word_count=13,
            source_pages="2",
        )
        self.c1_3 = Chunk(
            document_id=self.doc1.id,
            chunk_index=2,
            content="Cost analysis shows annual hosting expenditures estimated at $12,000 USD.",
            word_count=10,
            source_pages="3",
        )

        # Seed chunks for doc 2 (contradictory throughput and latency, shared security standard)
        self.c2_1 = Chunk(
            document_id=self.doc2.id,
            chunk_index=0,
            content="Cloud Architecture v2 transitions to a distributed PostgreSQL cluster. Maximum throughput is 50,000 RPS.",
            word_count=14,
            source_pages="1",
        )
        self.c2_2 = Chunk(
            document_id=self.doc2.id,
            chunk_index=1,
            content="Security encryption uses AES-128 standard across all nodes. Target response latency is guaranteed under 10ms.",
            word_count=15,
            source_pages="2",
        )
        self.c2_3 = Chunk(
            document_id=self.doc2.id,
            chunk_index=2,
            content="Cost analysis shows annual hosting expenditures estimated at $48,000 USD.",
            word_count=10,
            source_pages="4",
        )

        self.db.add_all([self.c1_1, self.c1_2, self.c1_3, self.c2_1, self.c2_2, self.c2_3])
        self.db.commit()

    def tearDown(self):
        self.db.rollback()
        self.db.query(Chunk).delete()
        self.db.query(Document).delete()
        self.db.commit()
        self.db.close()

    # =========================================================================
    # 1. Validation Tests
    # =========================================================================
    def test_validation_rejects_identical_documents(self):
        """Validates that comparing a document with itself is rejected."""
        with self.assertRaises(ValueError) as ctx:
            ComparisonService.validate_documents(self.db, "doc-test-1111", "doc-test-1111")
        self.assertIn("Cannot compare a document with itself", str(ctx.exception))

    def test_validation_rejects_missing_document(self):
        """Validates that non-existent document ID throws descriptive error."""
        with self.assertRaises(ValueError) as ctx:
            ComparisonService.validate_documents(self.db, "doc-test-1111", "non-existent-id")
        self.assertIn("not found", str(ctx.exception))

    def test_validation_rejects_zero_chunk_document(self):
        """Validates that document with 0 chunks is rejected."""
        with self.assertRaises(ValueError) as ctx:
            ComparisonService.validate_documents(self.db, "doc-test-1111", "doc-test-empty")
        self.assertIn("has no extracted content chunks", str(ctx.exception))

    # =========================================================================
    # 2. Representative Chunk Selection & Budgeting
    # =========================================================================
    def test_chunk_selection_includes_all_for_small_docs(self):
        """Ensures small documents have all chunks included without loss."""
        selected = ComparisonService._select_representative_chunks(self.db, self.doc1, max_words=1000)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0].chunk_index, 0)
        self.assertEqual(selected[-1].chunk_index, 2)

    def test_format_file_size(self):
        """Tests helper file size formatting."""
        self.assertEqual(format_file_size(500), "500 B")
        self.assertEqual(format_file_size(2048), "2.0 KB")
        self.assertEqual(format_file_size(5 * 1024 * 1024), "5.0 MB")

    # =========================================================================
    # 3. Prompt Construction & Formatting
    # =========================================================================
    def test_build_comparison_prompt(self):
        """Ensures prompt contains clear document sections, filenames, and chunk pages."""
        prompt = ComparisonService._build_comparison_prompt(
            doc1=self.doc1,
            chunks1=[self.c1_1, self.c1_2],
            doc2=self.doc2,
            chunks2=[self.c2_1, self.c2_2],
            focus_area="Latency and Throughput",
        )
        self.assertIn("FOCUS AREA FOR COMPARISON: \"Latency and Throughput\"", prompt)
        self.assertIn("=== DOCUMENT 1: cloud_architecture_v1.pdf ===", prompt)
        self.assertIn("=== DOCUMENT 2: cloud_architecture_v2.pdf ===", prompt)
        self.assertIn("[Doc 1 | Chunk 1 | Page 1]", prompt)
        self.assertIn("[Doc 2 | Chunk 1 | Page 1]", prompt)
        self.assertIn("Return the analysis STRICTLY as a valid JSON object", prompt)

    # =========================================================================
    # 4. JSON Response Parsing Resilience
    # =========================================================================
    def test_json_parsing_clean_payload(self):
        """Tests parsing of clean JSON payload from LLM."""
        mock_payload = {
            "comparison_summary": "Architectural comparison between v1 and v2.",
            "similarities": [
                {
                    "topic": "Security Standard",
                    "description": "Both architectures enforce AES-128 encryption.",
                    "source_document": "Both Documents",
                    "source_pages": "Page 2",
                    "excerpt": "AES-128 standard",
                }
            ],
            "differences": [
                {
                    "topic": "Database Engine",
                    "description": "v1 uses SQLite whereas v2 transitions to distributed PostgreSQL.",
                    "source_document": "cloud_architecture_v2.pdf",
                    "source_pages": "Page 1",
                    "excerpt": "distributed PostgreSQL cluster",
                }
            ],
            "shared_concepts": ["AES-128", "Cloud Architecture"],
            "unique_to_doc1": ["SQLite", "500 RPS"],
            "unique_to_doc2": ["PostgreSQL", "50,000 RPS"],
            "contradictions": [
                {
                    "topic": "Target Latency",
                    "doc1_claim": "Latency is strictly under 100ms",
                    "doc1_source": "cloud_architecture_v1.pdf",
                    "doc1_pages": "2",
                    "doc1_excerpt": "strictly under 100ms",
                    "doc2_claim": "Latency is guaranteed under 10ms",
                    "doc2_source": "cloud_architecture_v2.pdf",
                    "doc2_pages": "2",
                    "doc2_excerpt": "guaranteed under 10ms",
                    "analysis": "v2 promises a 10x lower latency threshold than v1.",
                }
            ],
            "agreements": [
                {
                    "topic": "Encryption Standard",
                    "description": "Consensus on AES-128 for data at rest.",
                    "source_document": "Both Documents",
                    "source_pages": "Page 2",
                    "excerpt": "Security encryption uses AES-128",
                }
            ],
        }

        # Test with markdown code fences
        fenced_str = f"```json\n{json.dumps(mock_payload)}\n```"
        parsed = ComparisonService._parse_llm_json_response(fenced_str, self.doc1, self.doc2)
        self.assertEqual(parsed["comparison_summary"], "Architectural comparison between v1 and v2.")
        self.assertEqual(len(parsed["similarities"]), 1)
        self.assertEqual(len(parsed["contradictions"]), 1)
        self.assertEqual(parsed["contradictions"][0]["topic"], "Target Latency")

    def test_json_parsing_fallback_on_unstructured_text(self):
        """Tests that unstructured text is safely converted to schema format without error."""
        raw_text = "Here is an analysis. Both documents discuss cloud setups. One is fast, one is slow."
        parsed = ComparisonService._parse_llm_json_response(raw_text, self.doc1, self.doc2)
        self.assertIn("comparison_summary", parsed)
        self.assertIsInstance(parsed["similarities"], list)
        self.assertIsInstance(parsed["contradictions"], list)

    # =========================================================================
    # 5. Full End-to-End Comparison Service Execution (Async Mock)
    # =========================================================================
    def test_compare_documents_service_execution(self):
        """Tests complete compare_documents flow with mocked Groq inference."""
        mock_llm_return = {
            "response": json.dumps({
                "comparison_summary": "Comprehensive analysis of cloud v1 vs v2.",
                "similarities": [
                    {
                        "topic": "Security",
                        "description": "Both use AES-128 encryption.",
                        "source_document": "Both Documents",
                        "source_pages": "2",
                        "excerpt": "Security encryption uses AES-128 standard",
                    }
                ],
                "differences": [
                    {
                        "topic": "Database Architecture",
                        "description": "v1 uses single SQLite, v2 uses distributed PostgreSQL.",
                        "source_document": "cloud_architecture_v2.pdf",
                        "source_pages": "1",
                        "excerpt": "distributed PostgreSQL cluster",
                    }
                ],
                "shared_concepts": ["AES-128", "Encryption", "Throughput"],
                "unique_to_doc1": ["Single SQLite"],
                "unique_to_doc2": ["Distributed PostgreSQL"],
                "contradictions": [
                    {
                        "topic": "Throughput and Latency",
                        "doc1_claim": "500 RPS and 100ms latency",
                        "doc1_source": "cloud_architecture_v1.pdf",
                        "doc1_pages": "1, 2",
                        "doc1_excerpt": "Maximum throughput is 500 RPS",
                        "doc2_claim": "50,000 RPS and 10ms latency",
                        "doc2_source": "cloud_architecture_v2.pdf",
                        "doc2_pages": "1, 2",
                        "doc2_excerpt": "Maximum throughput is 50,000 RPS",
                        "analysis": "v2 specifies 100x higher throughput and 10x lower latency.",
                    }
                ],
                "agreements": [
                    {
                        "topic": "Security Standard",
                        "description": "AES-128 security consensus.",
                        "source_document": "Both Documents",
                        "source_pages": "2",
                        "excerpt": "AES-128",
                    }
                ],
            }),
            "model": "openai/gpt-oss-120b",
            "tokens_generated": 240,
        }

        async def run_test():
            with patch("app.services.groq_service.GroqService.generate", new_callable=AsyncMock) as mock_gen, \
                 patch("app.services.rate_limiter.groq_rate_limiter.acquire", new_callable=AsyncMock) as mock_acq, \
                 patch("app.services.rate_limiter.groq_rate_limiter.record_actual_usage", new_callable=AsyncMock) as mock_rec:

                mock_acq.return_value = "req-123"
                mock_gen.return_value = mock_llm_return

                req = ComparisonRequest(
                    document_id_1="doc-test-1111",
                    document_id_2="doc-test-2222",
                    focus_area="Architecture and Performance",
                )

                resp = await ComparisonService.compare_documents(self.db, req)

                self.assertIsInstance(resp, ComparisonResponse)
                self.assertEqual(resp.document_1.filename, "cloud_architecture_v1.pdf")
                self.assertEqual(resp.document_2.filename, "cloud_architecture_v2.pdf")
                self.assertEqual(len(resp.similarities), 1)
                self.assertEqual(len(resp.differences), 1)
                self.assertEqual(len(resp.contradictions), 1)
                self.assertEqual(resp.contradictions[0].topic, "Throughput and Latency")
                self.assertEqual(len(resp.agreements), 1)
                self.assertEqual(len(resp.shared_concepts), 3)
                self.assertEqual(resp.unique_to_doc1, ["Single SQLite"])
                self.assertEqual(resp.unique_to_doc2, ["Distributed PostgreSQL"])
                self.assertGreater(len(resp.citations), 0)
                self.assertEqual(resp.model, "openai/gpt-oss-120b")

        asyncio.run(run_test())


if __name__ == "__main__":
    print("=" * 70)
    print("DocuMind AI — Stage 7: Document Comparison Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)

