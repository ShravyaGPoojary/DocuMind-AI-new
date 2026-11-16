import re
import unicodedata


class TextCleaner:
    """
    Reusable text sanitization and normalization service for raw extracted document text.
    Handles Unicode normalization, control character removal, whitespace collapsing,
    and multi-line standardisation.
    """

    @staticmethod
    def strip_control_characters(text: str) -> str:
        """
        Removes non-printable and control characters (ASCII < 32 and DEL),
        while preserving standard whitespace: \n (newline), \t (tab), \r (carriage return).
        """
        if not text:
            return ""
        # Keep newline (0x0A), tab (0x09), carriage return (0x0D)
        return "".join(
            ch for ch in text
            if ch in ("\n", "\r", "\t") or (unicodedata.category(ch)[0] != "C" and ord(ch) >= 32)
        )

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalizes Unicode characters into standard NFKC representation.
        Converts ligatures, smart quotes, dashes, and composite characters to canonical forms.
        """
        if not text:
            return ""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalizes line endings to LF (\n), collapses multiple consecutive horizontal
        spaces/tabs to a single space, and reduces 3+ consecutive newlines to 2.
        """
        if not text:
            return ""
        # Standardize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace non-breaking spaces and unusual space variants with regular space
        text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)

        # Collapse horizontal whitespace within each line
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @classmethod
    def clean(cls, raw_text: str) -> str:
        """
        Applies the complete text cleaning pipeline:
        1. Unicode normalization (NFKC)
        2. Control character removal
        3. Whitespace & newline normalization
        """
        if not raw_text:
            return ""

        text = cls.normalize_unicode(raw_text)
        text = cls.strip_control_characters(text)
        text = cls.normalize_whitespace(text)
        return text


# Module-level convenience functions
def clean_text(raw_text: str) -> str:
    """Convenience helper for TextCleaner.clean()."""
    return TextCleaner.clean(raw_text)

