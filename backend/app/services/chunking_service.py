from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from app.services.text_cleaner import TextCleaner


@dataclass
class ChunkItem:
    """Represents a generated text chunk with word-level metadata."""
    chunk_index: int
    content: str
    word_count: int
    source_pages: Optional[str] = None
    slide_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "content": self.content,
            "word_count": self.word_count,
            "source_pages": self.source_pages,
            "slide_number": self.slide_number,
        }


class ChunkingService:
    """
    Modular word-based document chunking service.

    Implements a sliding-window chunking strategy based on word counts.
    Defaults:
        - chunk_size: 500 words
        - chunk_overlap: 100 words

    Note: This service provides word-based chunking with approximate sizing
    and deliberately does not claim exact LLM/tokenizer token counts.
    """

    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 100

    @classmethod
    def chunk_text(
        cls,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        source_pages: Optional[str] = None,
        slide_number: Optional[int] = None,
    ) -> List[ChunkItem]:
        """
        Splits cleaned text into overlapping chunks using word boundaries.

        Args:
            text: The cleaned source text to chunk.
            chunk_size: Target number of words per chunk (default 500).
            chunk_overlap: Number of overlapping words between consecutive chunks (default 100).
            source_pages: Optional page reference (e.g., '1', '1-2').
            slide_number: Optional slide reference for presentations.

        Returns:
            List of ChunkItem objects containing chunk index, content, and word count.
        """
        if not text or not text.strip():
            return []

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        words = text.split()
        if not words:
            return []

        # If document is smaller than or equal to one chunk size, return single chunk
        if len(words) <= chunk_size:
            return [
                ChunkItem(
                    chunk_index=0,
                    content=" ".join(words),
                    word_count=len(words),
                    source_pages=source_pages,
                    slide_number=slide_number,
                )
            ]

        step = chunk_size - chunk_overlap
        chunks: List[ChunkItem] = []
        chunk_idx = 0
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_content = " ".join(chunk_words)

            chunks.append(
                ChunkItem(
                    chunk_index=chunk_idx,
                    content=chunk_content,
                    word_count=len(chunk_words),
                    source_pages=source_pages,
                    slide_number=slide_number,
                )
            )
            chunk_idx += 1

            if end >= len(words):
                break

            start += step

        return chunks

    @classmethod
    def chunk_pages_or_slides(
        cls,
        pages_text: List[str],
        is_presentation: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[ChunkItem]:
        """
        Chunks multi-page or multi-slide text while preserving actual source page/slide metadata.
        Words from each page are tracked to accurately label chunk boundaries.
        """
        if not pages_text:
            return []

        all_words: List[str] = []
        word_markers: List[int] = []

        for idx, page_raw in enumerate(pages_text, start=1):
            cleaned = TextCleaner.clean(page_raw)
            p_words = cleaned.split()
            if p_words:
                all_words.extend(p_words)
                word_markers.extend([idx] * len(p_words))

        if not all_words:
            return []

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        # Single chunk case
        if len(all_words) <= chunk_size:
            min_m = min(word_markers)
            max_m = max(word_markers)
            if is_presentation:
                src = f"Slide {min_m}" if min_m == max_m else f"Slides {min_m}-{max_m}"
                slide_num = min_m
            else:
                src = f"{min_m}" if min_m == max_m else f"{min_m}-{max_m}"
                slide_num = None

            return [
                ChunkItem(
                    chunk_index=0,
                    content=" ".join(all_words),
                    word_count=len(all_words),
                    source_pages=src,
                    slide_number=slide_num,
                )
            ]

        # Sliding window
        step = chunk_size - chunk_overlap
        chunks: List[ChunkItem] = []
        chunk_idx = 0
        start = 0

        while start < len(all_words):
            end = min(start + chunk_size, len(all_words))
            chunk_words = all_words[start:end]
            chunk_markers = word_markers[start:end]

            min_m = min(chunk_markers)
            max_m = max(chunk_markers)

            if is_presentation:
                src = f"Slide {min_m}" if min_m == max_m else f"Slides {min_m}-{max_m}"
                slide_num = min_m
            else:
                src = f"{min_m}" if min_m == max_m else f"{min_m}-{max_m}"
                slide_num = None

            chunks.append(
                ChunkItem(
                    chunk_index=chunk_idx,
                    content=" ".join(chunk_words),
                    word_count=len(chunk_words),
                    source_pages=src,
                    slide_number=slide_num,
                )
            )
            chunk_idx += 1

            if end >= len(all_words):
                break

            start += step

        return chunks


# Convenience helpers
def chunk_text(
    text: str,
    chunk_size: int = ChunkingService.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = ChunkingService.DEFAULT_CHUNK_OVERLAP,
    source_pages: Optional[str] = None,
    slide_number: Optional[int] = None,
) -> List[ChunkItem]:
    """Convenience helper for ChunkingService.chunk_text()."""
    return ChunkingService.chunk_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_pages=source_pages,
        slide_number=slide_number,
    )
