from fastapi import APIRouter
from app.core.config import settings
from app.schemas.info import ProjectInfoResponse, FeatureItem, SupportedFormat

router = APIRouter(tags=["Project Info"])


@router.get("/", response_model=ProjectInfoResponse, summary="Project Information")
@router.get("/info", response_model=ProjectInfoResponse, summary="Detailed Project Information")
async def get_project_info():
    """
    Returns foundational metadata, supported document formats, architectural blueprint,
    and planned RAG milestones for DocuMind AI.
    """
    return ProjectInfoResponse(
        name=settings.PROJECT_NAME,
        tagline="Intelligent Multi-Format Document Understanding Platform powered by RAG",
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        stage=settings.STAGE,
        architecture="FastAPI + React 18 + Vite + Modular CSS",
        supported_formats=[
            SupportedFormat(
                extension=".pdf",
                name="Portable Document Format",
                mime_type="application/pdf",
                max_size_mb=25,
                description="Financial reports, academic papers, legal filings, and scanned PDFs.",
            ),
            SupportedFormat(
                extension=".docx",
                name="Microsoft Word Document",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                max_size_mb=25,
                description="Technical specifications, company policies, manuals, and proposals.",
            ),
            SupportedFormat(
                extension=".pptx",
                name="Microsoft PowerPoint Presentation",
                mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                max_size_mb=25,
                description="Pitch decks, lecture slides, diagrams, and business presentations.",
            ),
            SupportedFormat(
                extension=".txt",
                name="Plain Text File",
                mime_type="text/plain",
                max_size_mb=10,
                description="Raw text transcripts, system logs, notes, and documentation.",
            ),
            SupportedFormat(
                extension=".md",
                name="Markdown Document",
                mime_type="text/markdown",
                max_size_mb=10,
                description="Developer documentation, READMEs, technical runbooks, and notes.",
            ),
            SupportedFormat(
                extension=".csv",
                name="Comma-Separated Values",
                mime_type="text/csv",
                max_size_mb=20,
                description="Tabular datasets, product catalogs, and tabular financial summaries.",
            ),
        ],
        features=[
            FeatureItem(
                id="doc_ingestion",
                title="Multi-Format Ingestion",
                description="Drag-and-drop ingestion of PDF, DOCX, TXT, MD, and CSV files.",
                status="UI Foundation Complete",
                stage="Stage 1",
            ),
            FeatureItem(
                id="rag_chat",
                title="Conversational Document AI",
                description="Interactive conversational interface ready for vector retrieval & source grounding.",
                status="UI Foundation Complete",
                stage="Stage 1 (LLM/RAG in Stage 2)",
            ),
            FeatureItem(
                id="summarization",
                title="Smart Summarization",
                description="Executive TL;DR and comprehensive section-by-section breakdown.",
                status="UI Foundation Complete",
                stage="Stage 1 (Pipeline in Stage 2)",
            ),
            FeatureItem(
                id="multi_doc_qa",
                title="Multi-Document Synthesis",
                description="Cross-document question answering with side-by-side source attribution.",
                status="UI Foundation Complete",
                stage="Stage 1 (Synthesis in Stage 2)",
            ),
        ],
        roadmap=[
            "Stage 1: Complete UI & Backend Foundation (Current)",
            "Stage 2: Document Processing, Chunking & Vector DB Ingestion (ChromaDB / LangChain / LlamaIndex)",
            "Stage 3: Hybrid Retrieval & Citation Grounding (Dense + BM25 Sparse Search)",
            "Stage 4: LLM Integration (OpenAI / Claude / Gemini / Ollama) with Multi-Doc Synthesis",
            "Stage 5: Production Hardening, Auth, Rate Limiting & Enterprise Export",
        ],
    )
