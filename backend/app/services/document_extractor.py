from pathlib import Path
from typing import Dict, Any, Optional, List, Union


class ExtractionError(Exception):
    """Raised when document extraction fails."""
    pass


class UnsupportedFormatError(ExtractionError):
    """Raised when the specified file format is not supported."""
    pass


class DocumentExtractor:
    """
    Modular document text extraction foundation supporting PDF, DOCX, PPTX,
    TXT, MD, and CSV files.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv"}

    @classmethod
    def extract(
        cls,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extracts raw text and metadata from a local document file.

        Args:
            file_path: Path to the target document.
            file_type: Optional explicit file extension / type (e.g. 'pdf', '.pdf').

        Returns:
            Dictionary containing:
                - raw_text: Extracted plain text content.
                - page_count: Total pages (or slides for PPTX), or None for flat text.
                - pages_text: List of strings for each page/slide if applicable.
                - file_type: Standardized lowercase extension (e.g. 'pdf').
                - metadata: Additional extraction metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found at: {file_path}")

        # Determine extension
        ext = file_type if file_type else path.suffix.lower()
        if not ext.startswith("."):
            ext = f".{ext.lower()}"

        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported format '{ext}'. Supported formats: {sorted(cls.SUPPORTED_EXTENSIONS)}"
            )

        clean_ext = ext.lstrip(".")

        if ext in {".txt", ".md", ".csv"}:
            return cls._extract_plain_text(path, clean_ext)
        elif ext == ".pdf":
            return cls._extract_pdf(path, clean_ext)
        elif ext == ".docx":
            return cls._extract_docx(path, clean_ext)
        elif ext == ".pptx":
            return cls._extract_pptx(path, clean_ext)
        else:
            raise UnsupportedFormatError(f"Handler not configured for {ext}")

    @staticmethod
    def _extract_plain_text(path: Path, file_type: str) -> Dict[str, Any]:
        """Extracts text from plain text formats (.txt, .md, .csv) with encoding fallback."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        content = None
        for enc in encodings:
            try:
                content = path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise ExtractionError(f"Could not decode file {path.name} with standard text encodings.")

        return {
            "raw_text": content,
            "page_count": 1,
            "pages_text": [content],
            "file_type": file_type,
            "metadata": {"encoding_used": enc},
        }

    @staticmethod
    def _extract_pdf(path: Path, file_type: str) -> Dict[str, Any]:
        """Extracts text and page count from PDF using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ExtractionError("PyMuPDF (fitz) is required for PDF extraction but not installed.")

        try:
            doc = fitz.open(str(path))
            pages_text: List[str] = []
            for page in doc:
                pages_text.append(page.get_text() or "")
            doc.close()

            full_text = "\n\n".join(pages_text)
            return {
                "raw_text": full_text,
                "page_count": len(pages_text),
                "pages_text": pages_text,
                "file_type": file_type,
                "metadata": {"tool": "pymupdf"},
            }
        except Exception as e:
            raise ExtractionError(f"Failed to extract PDF contents: {e}") from e

    @staticmethod
    def _extract_docx(path: Path, file_type: str) -> Dict[str, Any]:
        """Extracts text from DOCX files using python-docx."""
        try:
            import docx
        except ImportError:
            raise ExtractionError("python-docx is required for DOCX extraction but not installed.")

        try:
            doc = docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text]
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)

            full_text = "\n\n".join(paragraphs)
            return {
                "raw_text": full_text,
                "page_count": None,  # Word documents do not have native page counts in docx model
                "pages_text": paragraphs,
                "file_type": file_type,
                "metadata": {"tool": "python-docx", "paragraph_count": len(paragraphs)},
            }
        except Exception as e:
            raise ExtractionError(f"Failed to extract DOCX contents: {e}") from e

    @staticmethod
    def _extract_pptx(path: Path, file_type: str) -> Dict[str, Any]:
        """Extracts text from PPTX files using python-pptx."""
        try:
            from pptx import Presentation
        except ImportError:
            raise ExtractionError("python-pptx is required for PPTX extraction but not installed.")

        try:
            prs = Presentation(str(path))
            slides_text: List[str] = []

            for slide in prs.slides:
                slide_content: List[str] = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_content.append(shape.text.strip())
                slides_text.append("\n".join(slide_content))

            full_text = "\n\n".join(slides_text)
            return {
                "raw_text": full_text,
                "page_count": len(slides_text),
                "pages_text": slides_text,
                "file_type": file_type,
                "metadata": {"tool": "python-pptx", "slide_count": len(slides_text)},
            }
        except Exception as e:
            raise ExtractionError(f"Failed to extract PPTX contents: {e}") from e

