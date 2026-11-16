import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.chunk import Chunk
from app.schemas.intelligence import (
    ExtractionResponse,
    ExtractionEntityItem,
    ExtractionDateItem,
    ExtractionDefinitionItem,
    ExtractionMetricItem,
    DocumentInsightsResponse,
    StructuralSectionItem,
    QuestionGenerationResponse,
    QuestionItem,
    ContextualAnalysisResponse,
    EvidencePoint,
)
from app.services.token_estimator import TokenEstimator
from app.services.rate_limiter import groq_rate_limiter
from app.services.groq_service import GroqService

logger = logging.getLogger(__name__)


class IntelligenceService:
    """
    Advanced Document Intelligence Service powered by Groq GPT-OSS 120B.
    Provides:
    1. Intelligent Information Extraction (entities, dates, definitions, key metrics).
    2. Deep Document Insights (structural roadmap, major findings, risk factors).
    3. Document-based Knowledge-Check Question Generation with grounded answers.
    4. Multi-hop Contextual Analysis across document sections.
    """

    @classmethod
    def _get_document_chunks(cls, db: Session, document_id: str, max_words: int = 2400) -> List[Chunk]:
        """
        Retrieves representative chunks from a document, ensuring beginning, middle,
        and conclusion sections are included within token limits.
        """
        all_chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )
        if not all_chunks:
            return []

        total_words = sum(c.word_count or len(c.content.split()) for c in all_chunks)
        if total_words <= max_words:
            return all_chunks

        # Representative sampling: include first chunk, evenly spaced chunks, and last chunk
        selected = [all_chunks[0]]
        current_words = all_chunks[0].word_count or len(all_chunks[0].content.split())

        if len(all_chunks) > 2:
            step = max(1, len(all_chunks) // 6)
            for idx in range(1, len(all_chunks) - 1, step):
                chunk = all_chunks[idx]
                w = chunk.word_count or len(chunk.content.split())
                if current_words + w <= max_words:
                    selected.append(chunk)
                    current_words += w

        if all_chunks[-1] not in selected:
            selected.append(all_chunks[-1])

        selected.sort(key=lambda c: c.chunk_index)
        return selected

    @classmethod
    def _format_chunks_text(cls, chunks: List[Chunk]) -> str:
        """Formats chunks with clear section and page boundaries."""
        lines = []
        for c in chunks:
            loc = f"Page {c.source_pages}" if c.source_pages else (f"Slide {c.slide_number}" if c.slide_number else f"Section {c.chunk_index + 1}")
            lines.append(f"\n--- [Chunk #{c.chunk_index + 1} | {loc}] ---")
            lines.append(c.content.strip())
        return "\n".join(lines)

    @classmethod
    def _parse_json(cls, raw_text: str) -> Dict[str, Any]:
        """Safely extracts and parses JSON payload from LLM completion."""
        cleaned = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            brace_match = re.search(r"\{[\s\S]*\}", cleaned)
            if brace_match:
                try:
                    return json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    pass
        logger.warning("Could not cleanly parse JSON from LLM; attempting fallback structure.")
        return {}

    # -------------------------------------------------------------------------
    # 1. Information Extraction
    # -------------------------------------------------------------------------
    @classmethod
    async def extract_information(cls, db: Session, document_id: str) -> ExtractionResponse:
        """
        Extracts structured entities, dates, technical definitions, and numerical metrics from document.
        """
        start_time = time.perf_counter()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document with ID '{document_id}' not found.")

        chunks = cls._get_document_chunks(db, document_id, max_words=2200)
        chunks_text = cls._format_chunks_text(chunks)

        system_prompt = (
            "You are DocuMind AI, an advanced document information extraction engine.\n"
            "Extract structured intelligence from the document text strictly grounded in factual content.\n"
            "Do NOT invent entities, dates, or metrics.\n"
            "Return STRICTLY a JSON object with this exact structure:\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "...", "category": "Organization|Technology|Product|Person|Location|Standard", "description": "...", "source_page": "..."}\n'
            "  ],\n"
            '  "dates_and_events": [\n'
            '    {"date_str": "...", "event_description": "...", "source_page": "..."}\n'
            "  ],\n"
            '  "definitions": [\n'
            '    {"term": "...", "definition": "...", "source_page": "...", "context": "..."}\n'
            "  ],\n"
            '  "key_metrics": [\n'
            '    {"metric_name": "...", "value": "...", "context": "...", "source_page": "..."}\n'
            "  ]\n"
            "}"
        )

        prompt = f"DOCUMENT: {doc.original_filename}\n{chunks_text}\n\nExtract all significant entities, dates, technical definitions, and quantitative metrics."

        est_tokens = TokenEstimator.estimate_tokens(prompt) + 1500
        rate_id = await groq_rate_limiter.acquire(est_tokens)

        try:
            res = await GroqService.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=2048,
                system_prompt=system_prompt,
                model=settings.GROQ_MODEL,
            )
            raw = res.get("response", "")
            actual_tokens = res.get("tokens_generated", 0) + TokenEstimator.estimate_tokens(prompt)
            await groq_rate_limiter.record_actual_usage(rate_id, actual_tokens)
        except Exception as e:
            logger.error("Information extraction failed: %s", e)
            raise e

        parsed = cls._parse_json(raw)

        entities = [
            ExtractionEntityItem(**e) for e in parsed.get("entities", [])
            if isinstance(e, dict) and e.get("name")
        ]
        dates = [
            ExtractionDateItem(**d) for d in parsed.get("dates_and_events", [])
            if isinstance(d, dict) and d.get("date_str")
        ]
        defs = [
            ExtractionDefinitionItem(**item) for item in parsed.get("definitions", [])
            if isinstance(item, dict) and item.get("term")
        ]
        metrics = [
            ExtractionMetricItem(**m) for m in parsed.get("key_metrics", [])
            if isinstance(m, dict) and m.get("metric_name")
        ]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return ExtractionResponse(
            document_id=doc.id,
            document_name=doc.original_filename,
            entities=entities,
            dates_and_events=dates,
            definitions=defs,
            key_metrics=metrics,
            model=settings.GROQ_MODEL,
            processing_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # 2. Document Insights
    # -------------------------------------------------------------------------
    @classmethod
    async def generate_insights(cls, db: Session, document_id: str) -> DocumentInsightsResponse:
        """
        Generates comprehensive strategic insights, major findings, risks, and structural roadmap.
        """
        start_time = time.perf_counter()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document with ID '{document_id}' not found.")

        chunks = cls._get_document_chunks(db, document_id, max_words=2200)
        chunks_text = cls._format_chunks_text(chunks)

        system_prompt = (
            "You are DocuMind AI, an elite document strategy and analytical insight engine.\n"
            "Analyze the provided document text to derive evidence-grounded insights.\n"
            "Return STRICTLY a JSON object with this exact structure:\n"
            "{\n"
            '  "executive_summary": "Comprehensive analytical summary of objectives and results",\n'
            '  "key_topics": ["Topic 1", "Topic 2", ...],\n'
            '  "major_findings": ["Key finding 1 with specifics", "Key finding 2...", ...],\n'
            '  "risks_and_challenges": ["Identified challenge or risk 1", ...],\n'
            '  "structural_roadmap": [\n'
            '    {"section_name": "...", "overview": "...", "key_takeaway": "..."}\n'
            "  ],\n"
            '  "strategic_takeaways": ["Takeaway 1", "Takeaway 2", ...]\n'
            "}"
        )

        prompt = f"DOCUMENT: {doc.original_filename}\n{chunks_text}\n\nGenerate deep document insights, key topics, findings, risks, roadmap, and takeaways."

        est_tokens = TokenEstimator.estimate_tokens(prompt) + 1500
        rate_id = await groq_rate_limiter.acquire(est_tokens)

        try:
            res = await GroqService.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=2048,
                system_prompt=system_prompt,
                model=settings.GROQ_MODEL,
            )
            raw = res.get("response", "")
            actual_tokens = res.get("tokens_generated", 0) + TokenEstimator.estimate_tokens(prompt)
            await groq_rate_limiter.record_actual_usage(rate_id, actual_tokens)
        except Exception as e:
            logger.error("Insights generation failed: %s", e)
            raise e

        parsed = cls._parse_json(raw)

        roadmap = [
            StructuralSectionItem(**item) for item in parsed.get("structural_roadmap", [])
            if isinstance(item, dict) and item.get("section_name")
        ]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return DocumentInsightsResponse(
            document_id=doc.id,
            document_name=doc.original_filename,
            executive_summary=parsed.get("executive_summary", "Analytical synthesis completed."),
            key_topics=[str(t) for t in parsed.get("key_topics", [])],
            major_findings=[str(f) for f in parsed.get("major_findings", [])],
            risks_and_challenges=[str(r) for r in parsed.get("risks_and_challenges", [])],
            structural_roadmap=roadmap,
            strategic_takeaways=[str(st) for st in parsed.get("strategic_takeaways", [])],
            model=settings.GROQ_MODEL,
            processing_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # 3. Question Generation (Knowledge-Check)
    # -------------------------------------------------------------------------
    @classmethod
    async def generate_questions(
        cls,
        db: Session,
        document_id: str,
        num_questions: int = 5,
        difficulty: str = "mixed",
    ) -> QuestionGenerationResponse:
        """
        Generates revision and knowledge-check questions with grounded answers and source page references.
        """
        start_time = time.perf_counter()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document with ID '{document_id}' not found.")

        chunks = cls._get_document_chunks(db, document_id, max_words=2200)
        chunks_text = cls._format_chunks_text(chunks)

        system_prompt = (
            "You are DocuMind AI, an expert educational and revision question generator.\n"
            f"Generate exactly {num_questions} high-value questions and answers based STRICTLY on the document text.\n"
            "Each answer MUST be factually supported by the document with clear reasoning.\n"
            "Return STRICTLY a JSON object with this exact structure:\n"
            "{\n"
            '  "questions": [\n'
            "    {\n"
            '      "question": "Clear question text",\n'
            '      "answer": "Thorough, completely grounded answer explaining the concept or facts",\n'
            '      "question_type": "conceptual|analytical|factual",\n'
            '      "difficulty": "easy|medium|hard",\n'
            '      "source_pages": "e.g. Page 2",\n'
            '      "evidence_excerpt": "Direct supporting sentence or phrase from the document"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        prompt = (
            f"DOCUMENT: {doc.original_filename}\n{chunks_text}\n\n"
            f"Generate {num_questions} questions (Difficulty: {difficulty}) testing comprehension, analysis, and key facts."
        )

        est_tokens = TokenEstimator.estimate_tokens(prompt) + 1500
        rate_id = await groq_rate_limiter.acquire(est_tokens)

        try:
            res = await GroqService.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2048,
                system_prompt=system_prompt,
                model=settings.GROQ_MODEL,
            )
            raw = res.get("response", "")
            actual_tokens = res.get("tokens_generated", 0) + TokenEstimator.estimate_tokens(prompt)
            await groq_rate_limiter.record_actual_usage(rate_id, actual_tokens)
        except Exception as e:
            logger.error("Question generation failed: %s", e)
            raise e

        parsed = cls._parse_json(raw)

        questions = []
        for idx, q in enumerate(parsed.get("questions", [])):
            if isinstance(q, dict) and q.get("question") and q.get("answer"):
                questions.append(
                    QuestionItem(
                        id=f"q_{idx + 1}_{uuid.uuid4().hex[:6]}",
                        question=q["question"],
                        answer=q["answer"],
                        question_type=q.get("question_type", "conceptual"),
                        difficulty=q.get("difficulty", "medium"),
                        source_pages=q.get("source_pages"),
                        evidence_excerpt=q.get("evidence_excerpt"),
                    )
                )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return QuestionGenerationResponse(
            document_id=doc.id,
            document_name=doc.original_filename,
            questions=questions,
            model=settings.GROQ_MODEL,
            processing_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # 4. Contextual Analysis
    # -------------------------------------------------------------------------
    @classmethod
    async def perform_contextual_analysis(
        cls,
        db: Session,
        document_id: str,
        inquiry: str,
    ) -> ContextualAnalysisResponse:
        """
        Performs multi-hop evidence reasoning on a specific question, theme, or proposition across document sections.
        """
        start_time = time.perf_counter()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document with ID '{document_id}' not found.")

        # Prioritize chunks matching the inquiry via semantic search
        from app.services.search_service import SearchService
        try:
            search_res = SearchService.search(query=inquiry, top_k=5, document_id=doc.id, db=db)
            matched_indices = {r["chunk_index"] for r in search_res.get("results", [])}
        except Exception:
            matched_indices = set()

        all_chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )

        selected_chunks = [c for c in all_chunks if c.chunk_index in matched_indices]
        if all_chunks and all_chunks[0] not in selected_chunks:
            selected_chunks.insert(0, all_chunks[0])
        if all_chunks and all_chunks[-1] not in selected_chunks:
            selected_chunks.append(all_chunks[-1])

        # If few chunks matched, pick representative chunks
        if len(selected_chunks) < 4:
            selected_chunks = cls._get_document_chunks(db, document_id, max_words=2200)

        selected_chunks.sort(key=lambda c: c.chunk_index)
        chunks_text = cls._format_chunks_text(selected_chunks)

        system_prompt = (
            "You are DocuMind AI, an elite contextual reasoning and cross-section evidence engine.\n"
            "Analyze the user inquiry across the document sections with rigorous multi-hop reasoning.\n"
            "Return STRICTLY a JSON object with this exact structure:\n"
            "{\n"
            '  "thesis_assessment": "Direct evidence-based verdict addressing the inquiry",\n'
            '  "supporting_evidence": [\n'
            '    {"claim": "...", "source_pages": "...", "excerpt": "...", "significance": "..."}\n'
            "  ],\n"
            '  "counter_evidence_or_limitations": [\n'
            '    {"claim": "...", "source_pages": "...", "excerpt": "...", "significance": "..."}\n'
            "  ],\n"
            '  "synthesis_and_implications": "Comprehensive synthesis evaluating trade-offs, nuances, and conclusions"\n'
            "}"
        )

        prompt = (
            f"DOCUMENT: {doc.original_filename}\n{chunks_text}\n\n"
            f"INQUIRY / HYPOTHESIS: \"{inquiry.strip()}\"\n"
            "Perform comprehensive contextual reasoning examining supporting points, counter-evidence or limitations, and synthesize the conclusion."
        )

        est_tokens = TokenEstimator.estimate_tokens(prompt) + 1500
        rate_id = await groq_rate_limiter.acquire(est_tokens)

        try:
            res = await GroqService.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=2048,
                system_prompt=system_prompt,
                model=settings.GROQ_MODEL,
            )
            raw = res.get("response", "")
            actual_tokens = res.get("tokens_generated", 0) + TokenEstimator.estimate_tokens(prompt)
            await groq_rate_limiter.record_actual_usage(rate_id, actual_tokens)
        except Exception as e:
            logger.error("Contextual analysis failed: %s", e)
            raise e

        parsed = cls._parse_json(raw)

        supporting = [
            EvidencePoint(**item) for item in parsed.get("supporting_evidence", [])
            if isinstance(item, dict) and item.get("claim")
        ]
        counter = [
            EvidencePoint(**item) for item in parsed.get("counter_evidence_or_limitations", [])
            if isinstance(item, dict) and item.get("claim")
        ]

        citations = [
            {
                "chunk_id": c.id,
                "chunk_index": c.chunk_index,
                "source_pages": c.source_pages,
                "slide_number": c.slide_number,
                "excerpt": c.content[:200].strip() + "...",
            }
            for c in selected_chunks
        ]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return ContextualAnalysisResponse(
            document_id=doc.id,
            document_name=doc.original_filename,
            inquiry=inquiry.strip(),
            thesis_assessment=parsed.get("thesis_assessment", "Contextual reasoning analysis completed."),
            supporting_evidence=supporting,
            counter_evidence_or_limitations=counter,
            synthesis_and_implications=parsed.get("synthesis_and_implications", "Evidence analyzed across sections."),
            citations=citations,
            model=settings.GROQ_MODEL,
            processing_time_ms=elapsed_ms,
        )

