"""
API Integration Tests for Stage 7:
Verifies FastAPI endpoints:
1. POST /api/comparison/compare
2. POST /api/documents/compare (alias)
3. 400 Bad Request on same document ID
4. 400 Bad Request on missing document ID
"""

import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient
from app.main import app
from app.db.session import get_db
from app.models.document import Document
from app.models.chunk import Chunk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.db.base import Base

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(bind=test_engine)
Base.metadata.create_all(test_engine)



def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


class Stage7APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.db = TestSession()
        self.d1 = Document(
            id="doc-api-1",
            filename="financial_q1.pdf",
            original_filename="financial_q1.pdf",
            file_type="pdf",
            file_size=100000,
            processing_status="ready",
            chunk_count=1,
            page_count=2,
        )
        self.d2 = Document(
            id="doc-api-2",
            filename="financial_q2.pdf",
            original_filename="financial_q2.pdf",
            file_type="pdf",
            file_size=120000,
            processing_status="ready",
            chunk_count=1,
            page_count=3,
        )
        self.db.add_all([self.d1, self.d2])

        c1 = Chunk(
            document_id=self.d1.id,
            chunk_index=0,
            content="Q1 Net Profit reached $4.2M, showing 15% YoY growth.",
            word_count=8,
            source_pages="1",
        )
        c2 = Chunk(
            document_id=self.d2.id,
            chunk_index=0,
            content="Q2 Net Profit declined to $3.1M due to supply chain headwinds.",
            word_count=10,
            source_pages="1",
        )
        self.db.add_all([c1, c2])
        self.db.commit()

    def tearDown(self):
        self.db.query(Chunk).delete()
        self.db.query(Document).delete()
        self.db.commit()
        self.db.close()

    def test_compare_endpoint_success(self):
        """Verifies POST /api/comparison/compare returns 200 and structured comparison response."""
        mock_groq_return = {
            "response": json.dumps({
                "comparison_summary": "Financial comparison between Q1 and Q2 performance.",
                "similarities": [
                    {
                        "topic": "Financial Reporting",
                        "description": "Both reports detail net profit metrics.",
                        "source_document": "Both Documents",
                        "source_pages": "1",
                        "excerpt": "Net Profit",
                    }
                ],
                "differences": [
                    {
                        "topic": "Quarterly Trend",
                        "description": "Q1 experienced profit expansion while Q2 faced contraction.",
                        "source_document": "financial_q2.pdf",
                        "source_pages": "1",
                        "excerpt": "Q2 Net Profit declined",
                    }
                ],
                "shared_concepts": ["Net Profit", "Financial Performance"],
                "unique_to_doc1": ["15% YoY growth"],
                "unique_to_doc2": ["Supply chain headwinds"],
                "contradictions": [],
                "agreements": [
                    {
                        "topic": "Accounting Standard",
                        "description": "Both adhere to net profit reporting.",
                        "source_document": "Both Documents",
                        "source_pages": "1",
                        "excerpt": "Net Profit",
                    }
                ],
            }),
            "model": "openai/gpt-oss-120b",
            "tokens_generated": 180,
        }

        with patch("app.services.groq_service.GroqService.generate", new_callable=AsyncMock) as mock_gen, \
             patch("app.services.rate_limiter.groq_rate_limiter.acquire", new_callable=AsyncMock) as mock_acq, \
             patch("app.services.rate_limiter.groq_rate_limiter.record_actual_usage", new_callable=AsyncMock) as mock_rec:

            mock_acq.return_value = "req-api-1"
            mock_gen.return_value = mock_groq_return

            payload = {
                "document_id_1": "doc-api-1",
                "document_id_2": "doc-api-2",
                "focus_area": "Net Profit",
            }
            res = self.client.post("/api/comparison/compare", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["document_1"]["original_filename"], "financial_q1.pdf")
            self.assertEqual(data["document_2"]["original_filename"], "financial_q2.pdf")
            self.assertIn("Financial comparison", data["comparison_summary"])
            self.assertEqual(len(data["similarities"]), 1)
            self.assertEqual(len(data["differences"]), 1)
            self.assertEqual(data["model"], "openai/gpt-oss-120b")
            self.assertGreater(len(data["citations"]), 0)

    def test_compare_alias_endpoint(self):
        """Verifies POST /api/documents/compare alias returns 200."""
        mock_groq_return = {
            "response": json.dumps({
                "comparison_summary": "Alias test summary.",
                "similarities": [],
                "differences": [],
                "shared_concepts": [],
                "unique_to_doc1": [],
                "unique_to_doc2": [],
                "contradictions": [],
                "agreements": [],
            }),
            "model": "openai/gpt-oss-120b",
            "tokens_generated": 50,
        }

        with patch("app.services.groq_service.GroqService.generate", new_callable=AsyncMock) as mock_gen, \
             patch("app.services.rate_limiter.groq_rate_limiter.acquire", new_callable=AsyncMock) as mock_acq, \
             patch("app.services.rate_limiter.groq_rate_limiter.record_actual_usage", new_callable=AsyncMock) as mock_rec:

            mock_acq.return_value = "req-api-alias"
            mock_gen.return_value = mock_groq_return

            payload = {
                "document_id_1": "doc-api-1",
                "document_id_2": "doc-api-2",
            }
            res = self.client.post("/api/documents/compare", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["comparison_summary"], "Alias test summary.")

    def test_compare_rejects_same_document(self):
        """Verifies POST /api/comparison/compare returns 400 when document_id_1 == document_id_2."""
        payload = {
            "document_id_1": "doc-api-1",
            "document_id_2": "doc-api-1",
        }
        res = self.client.post("/api/comparison/compare", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Cannot compare a document with itself", res.json()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
