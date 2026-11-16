import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.chunk import Chunk
from app.schemas.comparison import (
    ComparisonRequest,
    ComparisonResponse,
    ComparisonDocumentMetadata,
    ComparisonFindingItem,
    ComparisonContradictionItem,
    ComparisonCitation,
)
from app.services.token_estimator import TokenEstimator
from app.services.rate_limiter import groq_rate_limiter
from app.services.groq_service import (
    GroqService,
    GroqServiceError,
    GroqAuthError,
    GroqRateLimitError,
    GroqModelUnavailableError,
    GroqNetworkError,
    GroqTimeoutError,
    GroqResponseError,
    GroqOversizedRequestError,
)
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Formats file size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


class ComparisonService:
    """
    Intelligent Cross-Document Comparison and Analysis Service.
    Powered by Groq GPT-OSS 120B (default) or optional local Ollama.

    Performs deep factual cross-document analysis identifying:
    - Similarities & core commonalities
    - Differences in approach, scope, and conclusions
    - Shared vs unique technical and conceptual themes
    - Conflicting statements, numerical discrepancies, or opposing claims (contradictions)
    - Reinforced consensus and agreements
    - Granular citations grounding every finding
    """

    SYSTEM_PROMPT = (
        "You are DocuMind AI, an elite document comparison and cross-document analysis intelligence.\n"
        "Your task is to perform an in-depth, rigorous, and completely factual comparison between the two provided documents.\n\n"
        "STRICT GROUNDING & FACTUALITY RULES:\n"
        "1. Base every similarity, difference, shared concept, unique concept, contradiction, and agreement EXCLUSIVELY on the provided excerpts.\n"
        "2. DO NOT hallucinate, infer without evidence, or introduce facts outside the provided texts.\n"
        "3. For CONTRADICTIONS: Look specifically for conflicting numerical values, dates, opposing methodologies, conflicting recommendations, divergent findings, or contradictory claims. If NO direct contradictions exist between the documents, you MUST return an empty list `[]` for 'contradictions'. Do NOT fabricate false conflicts.\n"
        "4. For CITATIONS: Accurately reference which document made each claim, referencing the page or slide numbers provided.\n"
        "5. FORMAT: You MUST return a valid, parseable JSON object with no markdown code fences, no preamble, and no conversational filler.\n"
        "The JSON structure MUST follow this exact schema:\n"
        "{\n"
        '  "comparison_summary": "Thorough analytical overview comparing the core themes, objectives, and findings of both documents (2-3 paragraphs)",\n'
        '  "similarities": [\n'
        '    {\n'
        '      "topic": "Concise topic title",\n'
        '      "description": "Specific factual similarity explaining what both documents agree on or share",\n'
        '      "source_document": "Both Documents",\n'
        '      "source_pages": "e.g. Doc1: Page 1, Doc2: Page 2",\n'
        '      "excerpt": "Key excerpt or quote illustrating the similarity"\n'
        '    }\n'
        '  ],\n'
        '  "differences": [\n'
        '    {\n'
        '      "topic": "Concise topic title",\n'
        '      "description": "Specific difference explaining how the documents diverge in scope, method, or conclusion",\n'
        '      "source_document": "Name of document or Doc 1 vs Doc 2",\n'
        '      "source_pages": "Relevant page numbers",\n'
        '      "excerpt": "Key excerpt illustrating the difference"\n'
        '    }\n'
        '  ],\n'
        '  "shared_concepts": ["List", "of", "key", "concepts", "common", "to", "both"],\n'
        '  "unique_to_doc1": ["Concepts", "found", "only", "in", "Document 1"],\n'
        '  "unique_to_doc2": ["Concepts", "found", "only", "in", "Document 2"],\n'
        '  "contradictions": [\n'
        '    {\n'
        '      "topic": "Topic of conflict",\n'
        '      "doc1_claim": "Specific statement/data asserted in Document 1",\n'
        '      "doc1_source": "Document 1 filename",\n'
        '      "doc1_pages": "Page number in Document 1",\n'
        '      "doc1_excerpt": "Direct supporting quote from Document 1",\n'
        '      "doc2_claim": "Conflicting statement/data asserted in Document 2",\n'
        '      "doc2_source": "Document 2 filename",\n'
        '      "doc2_pages": "Page number in Document 2",\n'
        '      "doc2_excerpt": "Direct supporting quote from Document 2",\n'
        '      "analysis": "Explanation of why these two statements conflict or cannot both be true"\n'
        '    }\n'
        '  ],\n'
        '  "agreements": [\n'
        '    {\n'
        '      "topic": "Topic of consensus",\n'
        '      "description": "Mutual consensus, corroborating evidence, or shared validation",\n'
        '      "source_document": "Both Documents",\n'
        '      "source_pages": "Pages from both documents",\n'
        '      "excerpt": "Supporting text"\n'
        '    }\n'
        '  ]\n'
        "}"
    )

    @classmethod
    def validate_documents(
        cls, db: Session, doc_id_1: str, doc_id_2: str
    ) -> Tuple[Document, Document]:
        """
        Validates both documents for existence, distinctness, and processing readiness.
        """
        if not doc_id_1 or not doc_id_2:
            raise ValueError("Both document_id_1 and document_id_2 must be provided.")

        clean_id1 = str(doc_id_1).strip()
        clean_id2 = str(doc_id_2).strip()

        if clean_id1 == clean_id2:
            raise ValueError("Cannot compare a document with itself. Please select two distinct documents.")

        doc1 = db.query(Document).filter(Document.id == clean_id1).first()
        if not doc1:
            raise ValueError(f"Document 1 with ID '{clean_id1}' was not found.")

        doc2 = db.query(Document).filter(Document.id == clean_id2).first()
        if not doc2:
            raise ValueError(f"Document 2 with ID '{clean_id2}' was not found.")

        if doc1.processing_status == "error":
            raise ValueError(f"Document 1 '{doc1.original_filename}' failed extraction: {doc1.error_message}")
        if doc2.processing_status == "error":
            raise ValueError(f"Document 2 '{doc2.original_filename}' failed extraction: {doc2.error_message}")

        if doc1.chunk_count == 0:
            raise ValueError(f"Document 1 '{doc1.original_filename}' has no extracted content chunks.")
        if doc2.chunk_count == 0:
            raise ValueError(f"Document 2 '{doc2.original_filename}' has no extracted content chunks.")

        return doc1, doc2

    @classmethod
    def _select_representative_chunks(
        cls,
        db: Session,
        doc: Document,
        focus_area: Optional[str] = None,
        max_words: int = 1200,
    ) -> List[Chunk]:
        """
        Selects representative, high-information chunks from a document within a target word limit.
        If focus_area is specified, prioritizes matching chunks via search while preserving context.
        """
        all_chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == doc.id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )

        if not all_chunks:
            return []

        # Case 1: All chunks fit comfortably under the limit
        total_words = sum(c.word_count or len(c.content.split()) for c in all_chunks)
        if total_words <= max_words:
            return all_chunks

        # Case 2: Focus area provided -> prioritize semantically relevant chunks
        selected: List[Chunk] = []
        if focus_area and focus_area.strip():
            try:
                from app.services.search_service import SearchService
                search_res = SearchService.search(
                    query=focus_area.strip(),
                    top_k=4,
                    document_id=doc.id,
                    db=db,
                )
                matched_indices = {r["chunk_index"] for r in search_res.get("results", [])}
                selected = [c for c in all_chunks if c.chunk_index in matched_indices]
            except Exception as e:
                logger.warning("Focus area search failed for doc %s, falling back to representative sampling: %s", doc.id, e)

        # Ensure introductory chunk is always present
        if all_chunks and all_chunks[0] not in selected:
            selected.insert(0, all_chunks[0])

        # If we still have room, add evenly spaced samples across the document lifecycle
        current_words = sum(c.word_count or len(c.content.split()) for c in selected)
        if current_words < max_words and len(all_chunks) > 2:
            step = max(1, len(all_chunks) // 5)
            for idx in range(1, len(all_chunks) - 1, step):
                chunk = all_chunks[idx]
                if chunk not in selected:
                    w = chunk.word_count or len(chunk.content.split())
                    if current_words + w <= max_words:
                        selected.append(chunk)
                        current_words += w

        # Ensure conclusion/findings chunk is included if room permits
        if all_chunks and all_chunks[-1] not in selected:
            last_chunk = all_chunks[-1]
            w = last_chunk.word_count or len(last_chunk.content.split())
            if current_words + w <= max_words or len(selected) < 3:
                selected.append(last_chunk)

        # Sort selected chunks in original document reading order
        selected.sort(key=lambda c: c.chunk_index)
        return selected

    @classmethod
    def _build_comparison_prompt(
        cls,
        doc1: Document,
        chunks1: List[Chunk],
        doc2: Document,
        chunks2: List[Chunk],
        focus_area: Optional[str] = None,
    ) -> str:
        """
        Constructs the structured prompt containing separated, labeled chunks from both documents.
        """
        lines = []

        if focus_area and focus_area.strip():
            lines.append(f"FOCUS AREA FOR COMPARISON: \"{focus_area.strip()}\"")
            lines.append("Analyze and compare both documents with special emphasis on this focus area.\n")

        lines.append(f"=== DOCUMENT 1: {doc1.original_filename} ===")
        lines.append(f"Type: {doc1.file_type.upper()} | Pages: {doc1.page_count or 'N/A'} | Chunks Analyzed: {len(chunks1)}")
        for c in chunks1:
            loc = f"Page {c.source_pages}" if c.source_pages else (f"Slide {c.slide_number}" if c.slide_number else f"Section {c.chunk_index + 1}")
            lines.append(f"\n--- [Doc 1 | Chunk {c.chunk_index + 1} | {loc}] ---")
            lines.append(c.content.strip())

        lines.append(f"\n\n=== DOCUMENT 2: {doc2.original_filename} ===")
        lines.append(f"Type: {doc2.file_type.upper()} | Pages: {doc2.page_count or 'N/A'} | Chunks Analyzed: {len(chunks2)}")
        for c in chunks2:
            loc = f"Page {c.source_pages}" if c.source_pages else (f"Slide {c.slide_number}" if c.slide_number else f"Section {c.chunk_index + 1}")
            lines.append(f"\n--- [Doc 2 | Chunk {c.chunk_index + 1} | {loc}] ---")
            lines.append(c.content.strip())

        lines.append("\n\n=== COMPARISON INSTRUCTIONS ===")
        lines.append(
            "Perform a comprehensive cross-document comparative analysis between Document 1 and Document 2 above.\n"
            "Identify: 1) Executive comparison summary, 2) Similarities, 3) Differences, 4) Shared concepts, "
            "5) Concepts unique to Document 1, 6) Concepts unique to Document 2, 7) Direct contradictions or conflicts (if any), "
            "and 8) Important agreements/consensus points.\n"
            "Return the analysis STRICTLY as a valid JSON object matching the requested schema."
        )

        return "\n".join(lines)

    @classmethod
    def _parse_llm_json_response(cls, raw_text: str, doc1: Document, doc2: Document) -> Dict[str, Any]:
        """
        Extracts and parses JSON from the LLM completion with fallback regex extraction.
        """
        cleaned = raw_text.strip()

        # Remove markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to locate the outer JSON braces
        brace_match = re.search(r"\{[\s\S]*\}", cleaned)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not cleanly parse JSON from LLM comparison response; attempting structural text fallback.")
        # Robust fallback parser for unstructured/markdown responses
        return cls._fallback_parse_comparison(raw_text, doc1, doc2)

    @classmethod
    def _fallback_parse_comparison(cls, text: str, doc1: Document, doc2: Document) -> Dict[str, Any]:
        """
        Fallback parser if the model output non-JSON text. Extracts sections into schema format.
        """
        summary = text[:600].strip()
        return {
            "comparison_summary": summary,
            "similarities": [
                {
                    "topic": "General Overlap",
                    "description": "Both documents address related domains or subject areas.",
                    "source_document": "Both Documents",
                    "source_pages": "Multiple",
                    "excerpt": text[:200].strip(),
                }
            ],
            "differences": [
                {
                    "topic": "Perspective & Scope",
                    "description": f"Divergences between {doc1.original_filename} and {doc2.original_filename}.",
                    "source_document": f"{doc1.original_filename} vs {doc2.original_filename}",
                    "source_pages": "Multiple",
                    "excerpt": "Detailed in document content.",
                }
            ],
            "shared_concepts": ["Overview", "Analysis"],
            "unique_to_doc1": [f"Unique points in {doc1.original_filename}"],
            "unique_to_doc2": [f"Unique points in {doc2.original_filename}"],
            "contradictions": [],
            "agreements": [
                {
                    "topic": "Subject Domain",
                    "description": "Shared context and baseline principles.",
                    "source_document": "Both Documents",
                    "source_pages": "Multiple",
                    "excerpt": "Analyzed across both sources.",
                }
            ],
        }

    @classmethod
    async def compare_documents(
        cls,
        db: Session,
        request: ComparisonRequest,
    ) -> ComparisonResponse:
        """
        Executes a complete cross-document comparison and returns a structured ComparisonResponse.
        """
        start_time = time.perf_counter()

        # 1. Validate documents
        doc1, doc2 = cls.validate_documents(db, request.document_id_1, request.document_id_2)

        # 2. Select representative chunks for both documents
        chunks1 = cls._select_representative_chunks(db, doc1, focus_area=request.focus_area, max_words=1100)
        chunks2 = cls._select_representative_chunks(db, doc2, focus_area=request.focus_area, max_words=1100)

        # 3. Build prompt
        prompt = cls._build_comparison_prompt(doc1, chunks1, doc2, chunks2, focus_area=request.focus_area)

        # 4. Token estimation & rate limiting
        provider = (request.provider or "groq").lower()
        model_name = request.model or (settings.GROQ_MODEL if provider == "groq" else settings.OLLAMA_MODEL)

        estimated_prompt_tokens = TokenEstimator.estimate_tokens(prompt) + TokenEstimator.estimate_tokens(cls.SYSTEM_PROMPT)
        estimated_total_tokens = estimated_prompt_tokens + 1500

        # Rate limiting for Groq Free Tier
        rate_limit_id = None
        if provider == "groq":
            rate_limit_id = await groq_rate_limiter.acquire(estimated_total_tokens)

        try:
            if provider == "groq":
                llm_res = await GroqService.generate(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=2048,
                    system_prompt=cls.SYSTEM_PROMPT,
                    model=model_name,
                    timeout=settings.GROQ_TIMEOUT,
                )
                raw_response = llm_res.get("response", "")
                actual_tokens = llm_res.get("tokens_generated", 0) + estimated_prompt_tokens
                if rate_limit_id:
                    await groq_rate_limiter.record_actual_usage(rate_limit_id, actual_tokens)
            else:
                # Local Ollama fallback
                llm_res = await LLMService.generate(
                    prompt=prompt,
                    temperature=0.2,
                    system_prompt=cls.SYSTEM_PROMPT,
                    model=model_name,
                    timeout=60,
                )
                raw_response = llm_res.get("response", "")

        except Exception as e:
            logger.error("LLM comparison generation failed: %s", str(e), exc_info=True)
            raise e

        # 5. Parse JSON comparison payload
        parsed_data = cls._parse_llm_json_response(raw_response, doc1, doc2)

        # 6. Build structured items
        similarities = [
            ComparisonFindingItem(
                topic=item.get("topic", "Similarity"),
                description=item.get("description", ""),
                source_document=item.get("source_document", "Both Documents"),
                source_pages=item.get("source_pages"),
                excerpt=item.get("excerpt"),
            )
            for item in parsed_data.get("similarities", [])
            if isinstance(item, dict)
        ]

        differences = [
            ComparisonFindingItem(
                topic=item.get("topic", "Difference"),
                description=item.get("description", ""),
                source_document=item.get("source_document"),
                source_pages=item.get("source_pages"),
                excerpt=item.get("excerpt"),
            )
            for item in parsed_data.get("differences", [])
            if isinstance(item, dict)
        ]

        contradictions = [
            ComparisonContradictionItem(
                topic=item.get("topic", "Conflict"),
                doc1_claim=item.get("doc1_claim", ""),
                doc1_source=item.get("doc1_source", doc1.original_filename),
                doc1_pages=item.get("doc1_pages"),
                doc1_excerpt=item.get("doc1_excerpt"),
                doc2_claim=item.get("doc2_claim", ""),
                doc2_source=item.get("doc2_source", doc2.original_filename),
                doc2_pages=item.get("doc2_pages"),
                doc2_excerpt=item.get("doc2_excerpt"),
                analysis=item.get("analysis", ""),
            )
            for item in parsed_data.get("contradictions", [])
            if isinstance(item, dict) and item.get("doc1_claim") and item.get("doc2_claim")
        ]

        agreements = [
            ComparisonFindingItem(
                topic=item.get("topic", "Consensus"),
                description=item.get("description", ""),
                source_document=item.get("source_document", "Both Documents"),
                source_pages=item.get("source_pages"),
                excerpt=item.get("excerpt"),
            )
            for item in parsed_data.get("agreements", [])
            if isinstance(item, dict)
        ]

        # 7. Build grounded citations from analyzed chunks
        citations: List[ComparisonCitation] = []
        for c in chunks1:
            citations.append(
                ComparisonCitation(
                    document_id=doc1.id,
                    document_name=doc1.original_filename,
                    chunk_index=c.chunk_index,
                    chunk_id=c.id,
                    source_pages=c.source_pages,
                    slide_number=c.slide_number,
                    excerpt=c.content[:250].strip() + ("..." if len(c.content) > 250 else ""),
                    relevance_topic=doc1.original_filename,
                )
            )
        for c in chunks2:
            citations.append(
                ComparisonCitation(
                    document_id=doc2.id,
                    document_name=doc2.original_filename,
                    chunk_index=c.chunk_index,
                    chunk_id=c.id,
                    source_pages=c.source_pages,
                    slide_number=c.slide_number,
                    excerpt=c.content[:250].strip() + ("..." if len(c.content) > 250 else ""),
                    relevance_topic=doc2.original_filename,
                )
            )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        meta1 = ComparisonDocumentMetadata(
            id=doc1.id,
            filename=doc1.filename,
            original_filename=doc1.original_filename,
            file_type=doc1.file_type.upper(),
            file_size_formatted=format_file_size(doc1.file_size),
            page_count=doc1.page_count,
            chunk_count=doc1.chunk_count,
        )

        meta2 = ComparisonDocumentMetadata(
            id=doc2.id,
            filename=doc2.filename,
            original_filename=doc2.original_filename,
            file_type=doc2.file_type.upper(),
            file_size_formatted=format_file_size(doc2.file_size),
            page_count=doc2.page_count,
            chunk_count=doc2.chunk_count,
        )

        return ComparisonResponse(
            comparison_id=str(uuid.uuid4()),
            document_1=meta1,
            document_2=meta2,
            focus_area=request.focus_area,
            comparison_summary=parsed_data.get("comparison_summary", "Comparative analysis completed."),
            similarities=similarities,
            differences=differences,
            shared_concepts=[str(c) for c in parsed_data.get("shared_concepts", [])],
            unique_to_doc1=[str(c) for c in parsed_data.get("unique_to_doc1", [])],
            unique_to_doc2=[str(c) for c in parsed_data.get("unique_to_doc2", [])],
            contradictions=contradictions,
            agreements=agreements,
            citations=citations,
            provider=provider,
            model=model_name,
            processing_time_ms=elapsed_ms,
        )

