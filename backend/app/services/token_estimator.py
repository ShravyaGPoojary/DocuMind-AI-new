"""
Token Estimation and Budget Management Service for DocuMind AI
Provides conservative token estimation, completion reservations, and oversized text splitting
for Groq Free Tier LLM inference (target: ~6,000 tokens request budget).

Estimation Strategy:
Since the full offline tokenizer vocabulary for openai/gpt-oss-120b is not stored locally in this
environment, this service implements an empirical conservative token estimation model:
- English words: 1 word ~ 1.35 tokens.
- Character fallback: len(text) / 3.6 tokens.
- Punctuation, symbols, numbers, and code syntax: automatically weighted.
- Message envelopes: ChatML / OpenAI roles (<|im_start|>, <|im_end|>) require 4 overhead tokens per message.
- A configurable safety margin (default: 500 tokens) is always reserved below the budget.

Limitations:
- Character and word heuristics may slightly overestimate tokens for dense continuous prose,
  or underestimate for highly complex mathematical equations / dense code sequences.
  The 500-token safety margin and automatic HTTP 413 dynamic splitting handle any edge cases.
"""

import logging
import re
from typing import List, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)


class TokenEstimator:
    """
    Token-aware budgeting and text estimation for LLM requests.
    Enforces Groq request token limits (~3,000–3,500 tokens budget) and reserves completion tokens.
    """

    DEFAULT_TARGET_BUDGET = getattr(settings, "SUMMARIZATION_TARGET_TOKENS", 3200)
    DEFAULT_SAFETY_MARGIN = getattr(settings, "SUMMARIZATION_TOKEN_SAFETY_MARGIN", 400)
    CHAT_MESSAGE_OVERHEAD = 4  # Overhead tokens per ChatML message envelope

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """
        Conservatively estimates the token count of a given text string.
        Uses dual word-and-character heuristic to avoid underestimating tokenization.
        """
        if not text:
            return 0
        cleaned = text.strip()
        if not cleaned:
            return 0

        words = len(cleaned.split())
        chars = len(cleaned)

        # Word-based estimate (1 word ~ 1.35 tokens) vs Character-based estimate (1 token ~ 3.6 chars)
        word_est = int(words * 1.35)
        char_est = int(chars / 3.6)

        # High-punctuation or symbol-heavy penalty
        symbols = len(re.findall(r"[\r\n\t\{\}\[\]\(\)<>=+\-*\/\\%$#@!^&|;:_~`]", cleaned))
        symbol_penalty = int(symbols * 0.35)

        estimated = max(word_est, char_est) + symbol_penalty + 4
        return max(1, estimated)

    @classmethod
    def estimate_prompt_tokens(
        cls,
        system_prompt: str,
        user_prompt: str,
    ) -> int:
        """
        Estimates total input tokens for a complete chat completion prompt,
        including system prompt, user prompt, and ChatML envelope overhead.
        """
        sys_tokens = cls.estimate_tokens(system_prompt) + cls.CHAT_MESSAGE_OVERHEAD if system_prompt else 0
        user_tokens = cls.estimate_tokens(user_prompt) + cls.CHAT_MESSAGE_OVERHEAD
        return sys_tokens + user_tokens + 3  # +3 for prime/reply tokens

    @classmethod
    def get_available_input_budget(
        cls,
        reserved_completion: int,
        target_budget: int = DEFAULT_TARGET_BUDGET,
        safety_margin: int = DEFAULT_SAFETY_MARGIN,
    ) -> int:
        """
        Calculates the maximum allowed input prompt tokens, reserving space for
        the model completion output and a safety margin.
        """
        available = target_budget - reserved_completion - safety_margin
        return max(500, available)

    @classmethod
    def split_oversized_text(
        cls,
        text: str,
        max_tokens: int,
    ) -> List[str]:
        """
        Recursively splits text exceeding max_tokens into smaller sequential segments,
        preserving natural paragraph, sentence, and word boundaries.
        """
        if cls.estimate_tokens(text) <= max_tokens:
            return [text]

        # 1. Try splitting on paragraph breaks (\n\n)
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            segments: List[str] = []
            curr_segment: List[str] = []
            curr_tokens = 0

            for p in paragraphs:
                p_tokens = cls.estimate_tokens(p)
                if p_tokens > max_tokens:
                    # Paragraph itself is oversized: recursively split it
                    if curr_segment:
                        segments.append("\n\n".join(curr_segment))
                        curr_segment = []
                        curr_tokens = 0
                    sub_parts = cls.split_oversized_text(p, max_tokens)
                    segments.extend(sub_parts)
                elif curr_tokens + p_tokens + 4 <= max_tokens:
                    curr_segment.append(p)
                    curr_tokens += p_tokens + 4
                else:
                    if curr_segment:
                        segments.append("\n\n".join(curr_segment))
                    curr_segment = [p]
                    curr_tokens = p_tokens

            if curr_segment:
                segments.append("\n\n".join(curr_segment))
            return segments

        # 2. Try splitting on line breaks (\n)
        lines = text.split("\n")
        if len(lines) > 1:
            segments = []
            curr_segment = []
            curr_tokens = 0

            for line in lines:
                l_tokens = cls.estimate_tokens(line)
                if l_tokens > max_tokens:
                    if curr_segment:
                        segments.append("\n".join(curr_segment))
                        curr_segment = []
                        curr_tokens = 0
                    segments.extend(cls.split_oversized_text(line, max_tokens))
                elif curr_tokens + l_tokens + 2 <= max_tokens:
                    curr_segment.append(line)
                    curr_tokens += l_tokens + 2
                else:
                    if curr_segment:
                        segments.append("\n".join(curr_segment))
                    curr_segment = [line]
                    curr_tokens = l_tokens

            if curr_segment:
                segments.append("\n".join(curr_segment))
            return segments

        # 3. Fallback: split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            segments = []
            curr_segment = []
            curr_tokens = 0

            for s in sentences:
                s_tokens = cls.estimate_tokens(s)
                if s_tokens > max_tokens:
                    if curr_segment:
                        segments.append(" ".join(curr_segment))
                        curr_segment = []
                        curr_tokens = 0
                    # Hard word-based split if sentence exceeds limit
                    words = s.split()
                    half = len(words) // 2
                    part1 = " ".join(words[:half])
                    part2 = " ".join(words[half:])
                    segments.extend(cls.split_oversized_text(part1, max_tokens))
                    segments.extend(cls.split_oversized_text(part2, max_tokens))
                elif curr_tokens + s_tokens + 2 <= max_tokens:
                    curr_segment.append(s)
                    curr_tokens += s_tokens + 2
                else:
                    if curr_segment:
                        segments.append(" ".join(curr_segment))
                    curr_segment = [s]
                    curr_tokens = s_tokens

            if curr_segment:
                segments.append(" ".join(curr_segment))
            return segments

        # 4. Final emergency fallback: word split
        words = text.split()
        if len(words) <= 1:
            return [text]
        mid = len(words) // 2
        return (
            cls.split_oversized_text(" ".join(words[:mid]), max_tokens)
            + cls.split_oversized_text(" ".join(words[mid:]), max_tokens)
        )

