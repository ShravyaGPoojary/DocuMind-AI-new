"""
DocuMind AI — Stage 8 Comprehensive Test Suite
Verifies:
1. User Registration with Argon2 hashing and duplicate rejection.
2. User Login with Argon2 verification and JWT generation.
3. Server-side Google ID token authentication (valid & invalid).
4. Protected API authentication (401 on missing token).
5. User Data Isolation: User A cannot access User B's documents, conversations, or search chunks.
6. Advanced Document Intelligence: Information extraction, insights, question generation, and contextual reasoning.
7. Backward compatibility and non-regression.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.models.user import User
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.conversation import Conversation, Message
from app.core.security import hash_password, verify_password, create_access_token, decode_access_token

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


class Stage8ComprehensiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.db = TestSession()
        # Clean state
        self.db.query(Message).delete()
        self.db.query(Conversation).delete()
        self.db.query(Chunk).delete()
        self.db.query(Document).delete()
        self.db.query(User).delete()
        self.db.commit()

        # Create User A
        self.user_a = User(
            id="user-aaa-111",
            email="alice@documind.ai",
            full_name="Alice Anderson",
            password_hash=hash_password("AlicePassword123"),
            is_active=True,
        )
        # Create User B
        self.user_b = User(
            id="user-bbb-222",
            email="bob@documind.ai",
            full_name="Bob Builder",
            password_hash=hash_password("BobPassword123"),
            is_active=True,
        )
        self.db.add_all([self.user_a, self.user_b])
        self.db.commit()

        # Generate JWT tokens for testing
        self.token_a = create_access_token({"sub": self.user_a.id, "email": self.user_a.email})
        self.token_b = create_access_token({"sub": self.user_b.id, "email": self.user_b.email})

        self.headers_a = {"Authorization": f"Bearer {self.token_a}"}
        self.headers_b = {"Authorization": f"Bearer {self.token_b}"}

        # Create Document for User A
        self.doc_a = Document(
            id="doc-alice-001",
            user_id=self.user_a.id,
            filename="alice_secret_roadmap.pdf",
            original_filename="alice_secret_roadmap.pdf",
            file_type="pdf",
            file_size=102400,
            processing_status="ready",
            chunk_count=2,
            page_count=3,
        )
        # Create Document for User B
        self.doc_b = Document(
            id="doc-bob-002",
            user_id=self.user_b.id,
            filename="bob_public_spec.pdf",
            original_filename="bob_public_spec.pdf",
            file_type="pdf",
            file_size=204800,
            processing_status="ready",
            chunk_count=1,
            page_count=2,
        )
        self.db.add_all([self.doc_a, self.doc_b])
        self.db.commit()

        self.chunk_a1 = Chunk(
            document_id=self.doc_a.id,
            chunk_index=0,
            content="Project Apollo launch date is Q3 2026. Target latency is 25ms.",
            word_count=10,
            source_pages="1",
        )
        self.chunk_a2 = Chunk(
            document_id=self.doc_a.id,
            chunk_index=1,
            content="Alice's confidential budget is $500,000 USD for infrastructure.",
            word_count=8,
            source_pages="2",
        )
        self.chunk_b1 = Chunk(
            document_id=self.doc_b.id,
            chunk_index=0,
            content="Bob's public specification for solar grid monitoring with 99.9% uptime.",
            word_count=10,
            source_pages="1",
        )
        # Create Conversation for User A
        self.conv_a = Conversation(
            id="conv-alice-001",
            user_id=self.user_a.id,
            title="Alice's Private Discussion",
            document_id=self.doc_a.id,
        )

        self.db.add_all([self.chunk_a1, self.chunk_a2, self.chunk_b1, self.conv_a])
        self.db.commit()

    def tearDown(self):
        try:
            self.db.rollback()
            self.db.query(Message).delete()
            self.db.query(Conversation).delete()
            self.db.query(Chunk).delete()
            self.db.query(Document).delete()
            self.db.query(User).delete()
            self.db.commit()
        except Exception:
            pass
        finally:
            self.db.close()


    # =========================================================================
    # PART 1: Authentication Tests
    # =========================================================================
    def test_01_argon2_hashing_and_verification(self):
        """Verifies Argon2 generates valid hashes and verifies correctly."""
        raw_pw = "SuperSecurePassword@2026"
        hashed = hash_password(raw_pw)
        self.assertTrue(hashed.startswith("$argon2id$"))
        self.assertTrue(verify_password(raw_pw, hashed))
        self.assertFalse(verify_password("WrongPassword", hashed))

    def test_02_register_success_and_duplicate_rejection(self):
        """Verifies user registration works and rejects duplicate emails."""
        # 1. Successful registration
        payload = {
            "full_name": "Charlie Chaplin",
            "email": "charlie@documind.ai",
            "password": "CharliePassword123",
        }
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["email"], "charlie@documind.ai")

        # 2. Duplicate registration rejected
        res_dup = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res_dup.status_code, 400)
        self.assertIn("already exists", res_dup.json()["detail"])

    def test_03_login_success_and_invalid_credentials(self):
        """Verifies login succeeds with valid password and fails with invalid."""
        # Valid login
        res = self.client.post("/api/auth/login", json={"email": "alice@documind.ai", "password": "AlicePassword123"})
        self.assertEqual(res.status_code, 200)
        token = res.json()["access_token"]
        payload = decode_access_token(token)
        self.assertEqual(payload["sub"], self.user_a.id)

        # Invalid password
        res_bad = self.client.post("/api/auth/login", json={"email": "alice@documind.ai", "password": "WrongPassword"})
        self.assertEqual(res_bad.status_code, 401)

        # Non-existent user
        res_none = self.client.post("/api/auth/login", json={"email": "ghost@documind.ai", "password": "password"})
        self.assertEqual(res_none.status_code, 401)

    def test_04_google_auth_server_verification(self):
        """Verifies server-side Google OAuth token verification and account creation."""
        mock_google_id_info = {
            "sub": "google-uid-998877",
            "email": "google_user@gmail.com",
            "name": "Google User",
            "picture": "https://lh3.googleusercontent.com/photo.jpg",
        }

        # Valid Google token
        with patch("google.oauth2.id_token.verify_oauth2_token", return_value=mock_google_id_info):
            res = self.client.post("/api/auth/google", json={"credential": "mock-google-id-token-valid"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("access_token", data)
            self.assertEqual(data["user"]["email"], "google_user@gmail.com")

        # Invalid Google token
        with patch("google.oauth2.id_token.verify_oauth2_token", side_effect=ValueError("Token expired")):
            res_err = self.client.post("/api/auth/google", json={"credential": "mock-google-id-token-bad"})
            self.assertEqual(res_err.status_code, 401)

    def test_05_protected_route_rejects_unauthenticated(self):
        """Verifies protected endpoints return 401 when no token is supplied."""
        res = self.client.get("/api/documents")
        self.assertEqual(res.status_code, 401)

        res_chat = self.client.post("/api/chat", json={"question": "Hello"})
        self.assertEqual(res_chat.status_code, 401)

        res_me = self.client.get("/api/auth/me")
        self.assertEqual(res_me.status_code, 401)

    # =========================================================================
    # PART 2: User Data Isolation & Authorization Tests
    # =========================================================================
    def test_06_documents_list_isolation(self):
        """Verifies User A only sees their own documents, User B only sees theirs."""
        res_a = self.client.get("/api/documents", headers=self.headers_a)
        self.assertEqual(res_a.status_code, 200)
        docs_a = res_a.json()["documents"]
        self.assertEqual(len(docs_a), 1)
        self.assertEqual(docs_a[0]["id"], self.doc_a.id)

        res_b = self.client.get("/api/documents", headers=self.headers_b)
        self.assertEqual(res_b.status_code, 200)
        docs_b = res_b.json()["documents"]
        self.assertEqual(len(docs_b), 1)
        self.assertEqual(docs_b[0]["id"], self.doc_b.id)

    def test_07_cross_user_document_access_forbidden(self):
        """Verifies User B cannot read or delete User A's document."""
        # Bob tries to read Alice's doc details -> 403 Forbidden
        res_read = self.client.get(f"/api/documents/{self.doc_a.id}", headers=self.headers_b)
        self.assertEqual(res_read.status_code, 403)
        self.assertIn("do not own this document", res_read.json()["detail"])

        # Bob tries to delete Alice's doc -> 403 Forbidden
        res_del = self.client.delete(f"/api/documents/{self.doc_a.id}", headers=self.headers_b)
        self.assertEqual(res_del.status_code, 403)

        # Bob tries to get chunks of Alice's doc -> 403 Forbidden
        res_chunks = self.client.get(f"/api/documents/{self.doc_a.id}/chunks", headers=self.headers_b)
        self.assertEqual(res_chunks.status_code, 403)

    def test_08_cross_user_conversation_access_forbidden(self):
        """Verifies User B cannot read or delete User A's conversation thread."""
        res_get = self.client.get(f"/api/chat/conversations/{self.conv_a.id}", headers=self.headers_b)
        self.assertEqual(res_get.status_code, 403)

        res_del = self.client.delete(f"/api/chat/conversations/{self.conv_a.id}", headers=self.headers_b)
        self.assertEqual(res_del.status_code, 403)

    def test_09_semantic_search_cross_user_isolation(self):
        """Verifies ChromaDB semantic search prevents searching other users' documents."""
        # Bob tries to explicitly search Alice's doc -> 403 Forbidden
        payload = {"query": "Apollo budget", "document_id": self.doc_a.id}
        res = self.client.post("/api/search", json=payload, headers=self.headers_b)
        self.assertEqual(res.status_code, 403)

    # =========================================================================
    # PART 3: Advanced Document Intelligence Tests
    # =========================================================================
    def test_10_information_extraction(self):
        """Verifies intelligent extraction endpoint returns structured entities, dates, definitions, and metrics."""
        mock_extraction = {
            "entities": [
                {"name": "Project Apollo", "category": "Product", "description": "Next-gen platform", "source_page": "1"}
            ],
            "dates_and_events": [
                {"date_str": "Q3 2026", "event_description": "Target launch timeframe", "source_page": "1"}
            ],
            "definitions": [
                {"term": "Target Latency", "definition": "Response threshold under 25ms", "source_page": "1", "context": "Target latency is 25ms."}
            ],
            "key_metrics": [
                {"metric_name": "Infrastructure Budget", "value": "$500,000", "context": "Annual hosting allocation", "source_page": "2"}
            ],
        }

        with patch("app.services.groq_service.GroqService.generate", new_callable=AsyncMock) as mock_gen, \
             patch("app.services.rate_limiter.groq_rate_limiter.acquire", new_callable=AsyncMock) as mock_acq, \
             patch("app.services.rate_limiter.groq_rate_limiter.record_actual_usage", new_callable=AsyncMock) as mock_rec:

            mock_acq.return_value = "req-intel-1"
            mock_gen.return_value = {"response": json.dumps(mock_extraction), "tokens_generated": 150}

            res = self.client.post(f"/api/intelligence/extract/{self.doc_a.id}", headers=self.headers_a)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(len(data["entities"]), 1)
            self.assertEqual(data["entities"][0]["name"], "Project Apollo")
            self.assertEqual(len(data["dates_and_events"]), 1)
            self.assertEqual(len(data["definitions"]), 1)
            self.assertEqual(len(data["key_metrics"]), 1)

    def test_11_document_insights(self):
        """Verifies insights generation returns executive summary, roadmap, findings, and risks."""
        mock_insights = {
            "executive_summary": "Alice's roadmap focuses on low latency and infrastructure scaling.",
            "key_topics": ["Project Apollo", "Cloud Architecture"],
            "major_findings": ["25ms target latency requires edge nodes"],
            "risks_and_challenges": ["Budget constraints under $500,000"],
            "structural_roadmap": [
                {"section_name": "Phase 1: Architecture", "overview": "Initial core layout", "key_takeaway": "Centralized cluster"}
            ],
            "strategic_takeaways": ["Prioritize Apollo launch in Q3"],
        }

        with patch("app.services.groq_service.GroqService.generate", new_callable=AsyncMock) as mock_gen, \
             patch("app.services.rate_limiter.groq_rate_limiter.acquire", new_callable=AsyncMock) as mock_acq, \
             patch("app.services.rate_limiter.groq_rate_limiter.record_actual_usage", new_callable=AsyncMock) as mock_rec:

            mock_acq.return_value = "req-intel-2"
            mock_gen.return_value = {"response": json.dumps(mock_insights), "tokens_generated": 180}

            res = self.client.post(f"/api/intelligence/insights/{self.doc_a.id}", headers=self.headers_a)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("Alice's roadmap", data["executive_summary"])
            self.assertEqual(len(data["key_topics"]), 2)
            self.assertEqual(len(data["structural_roadmap"]), 1)

    def test_12_question_generation(self):
        """Verifies question generation generates knowledge-check Q&A with grounded answers."""
        mock_questions = {
            "questions": [
                {
                    "question": "When is Project Apollo scheduled to launch?",
                    "answer": "Project Apollo is scheduled to launch in Q3 2026.",
                    "question_type": "factual",
                    "difficulty": "easy",
                    "source_pages": "Page 1",
                    "evidence_excerpt": "Project Apollo launch date is Q3 2026.",
                }
            ]
        }

        with patch("app.services.groq_service.GroqService.generate", new_callable=AsyncMock) as mock_gen, \
             patch("app.services.rate_limiter.groq_rate_limiter.acquire", new_callable=AsyncMock) as mock_acq, \
             patch("app.services.rate_limiter.groq_rate_limiter.record_actual_usage", new_callable=AsyncMock) as mock_rec:

            mock_acq.return_value = "req-intel-3"
            mock_gen.return_value = {"response": json.dumps(mock_questions), "tokens_generated": 120}

            res = self.client.post(f"/api/intelligence/generate-questions/{self.doc_a.id}", headers=self.headers_a)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(len(data["questions"]), 1)
            self.assertEqual(data["questions"][0]["question"], "When is Project Apollo scheduled to launch?")
            self.assertIn("Q3 2026", data["questions"][0]["answer"])

    def test_13_contextual_analysis(self):
        """Verifies contextual analysis evaluates inquiry with multi-hop evidence reasoning."""
        mock_contextual = {
            "thesis_assessment": "The document supports low latency feasibility within the stated budget.",
            "supporting_evidence": [
                {"claim": "25ms latency is targeted", "source_pages": "1", "excerpt": "Target latency is 25ms", "significance": "Core KPI"}
            ],
            "counter_evidence_or_limitations": [
                {"claim": "Budget cap of $500k may limit redundant clusters", "source_pages": "2", "excerpt": "$500,000 USD", "significance": "Cost constraint"}
            ],
            "synthesis_and_implications": "Latency and budget requirements are balanced for Q3 launch.",
        }

        with patch("app.services.groq_service.GroqService.generate", new_callable=AsyncMock) as mock_gen, \
             patch("app.services.rate_limiter.groq_rate_limiter.acquire", new_callable=AsyncMock) as mock_acq, \
             patch("app.services.rate_limiter.groq_rate_limiter.record_actual_usage", new_callable=AsyncMock) as mock_rec, \
             patch("app.services.embedding_service.EmbeddingService.generate_embedding", return_value=[0.05] * 384), \
             patch("app.services.embedding_service.EmbeddingService.generate_embeddings", return_value=[[0.05] * 384, [0.05] * 384]):

            mock_acq.return_value = "req-intel-4"
            mock_gen.return_value = {"response": json.dumps(mock_contextual), "tokens_generated": 200}


            payload = {"inquiry": "Does the budget permit the target latency?"}
            res = self.client.post(f"/api/intelligence/contextual-analysis/{self.doc_a.id}", json=payload, headers=self.headers_a)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("low latency feasibility", data["thesis_assessment"])
            self.assertEqual(len(data["supporting_evidence"]), 1)
            self.assertEqual(len(data["counter_evidence_or_limitations"]), 1)


if __name__ == "__main__":
    print("=" * 70)
    print("DocuMind AI — Stage 8 Security, Auth & Intelligence Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)
