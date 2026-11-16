# DocuMind AI – AI-Powered Document Intelligence Platform

DocuMind AI is an AI-powered document intelligence platform that enables users to upload, analyze, summarize, compare, and interact with documents using Retrieval-Augmented Generation (RAG).

It transforms static documents into an interactive knowledge source through semantic search, grounded AI responses, document insights, and cross-document analysis.

## Key Features

### Document Management & Processing
- Upload PDF, DOCX, TXT, and PPTX documents.
- Extract, clean, and chunk document content.
- Generate local embeddings and store them in ChromaDB.
- Manage uploaded documents through a centralized document library.

### AI Document Chat
- Ask natural-language questions about uploaded documents.
- Retrieve relevant document passages using semantic search.
- Generate context-aware answers using RAG.
- Display source citations with document names, page numbers, and relevant excerpts.

### Document Summarization
- Generate concise and detailed document summaries.
- Summarize lengthy documents using token-aware processing.
- Export summaries in Markdown format.

### Multi-Document Q&A
- Query multiple documents simultaneously.
- Retrieve relevant information across selected documents.
- Generate grounded answers with document-specific citations.
- Maintain document context across conversation threads.

### Document Comparison
- Compare two documents based on similarities and differences.
- Identify shared concepts and document-specific information.
- Highlight potential contradictions and disagreements.
- Generate comparative analysis with supporting evidence.

### Advanced Document Intelligence
- Extract important entities, dates, milestones, definitions, and metrics.
- Generate executive insights, findings, and key takeaways.
- Create document-based questions and answers.
- Perform contextual analysis using supporting and counter-evidence.

### Authentication & Security
- Email/password authentication.
- Optional Google sign-in.
- Protected routes and user-specific document access.
- Backend-enforced document ownership and access control.

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | React.js, Vite, JavaScript, CSS |
| Backend | Python, FastAPI |
| Database | SQLite, SQLAlchemy |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers – all-MiniLM-L6-v2 |
| LLM | Groq API – GPT-OSS 120B |
| Local LLM Option | Ollama – Qwen2.5 3B |
| Authentication | JWT, Argon2, Google Identity Services |
| Document Processing | PyPDF, python-docx, document extraction libraries |

## Architecture

```text
                  React Frontend
                        |
                     FastAPI
                        |
        +---------------+---------------+
        |               |               |
     SQLite DB       ChromaDB        Groq API
        |               |               |
   Metadata &       Embeddings      AI Responses
   Ownership        & Retrieval      & Analysis
        |
   Local File System
   Original Documents
```

## RAG Workflow

1. Upload a document.
2. Extract and clean its textual content.
3. Split content into smaller chunks.
4. Generate embeddings locally using Sentence Transformers.
5. Store document chunks and embeddings in ChromaDB.
6. Convert user questions into embeddings.
7. Retrieve relevant document passages through semantic search.
8. Provide retrieved context to the language model.
9. Generate grounded answers with source citations.

## Project Structure

```text
DocuMind-AI-new/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── styles/
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── db/
│   │   ├── models/
│   │   ├── services/
│   │   └── main.py
│   ├── requirements.txt
│   └── .env.example
│
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- Git
- Groq API key (Free Tier, subject to provider limits)
- Optional: Ollama for local LLM inference

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ShravyaGPoojary/DocuMind-AI-new.git
cd DocuMind-AI-new
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file inside the backend directory using `.env.example` as a reference.

Configure the required environment variables, including your Groq API key and authentication secrets.

Start the backend:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend API: http://127.0.0.1:8000

Swagger Documentation: http://127.0.0.1:8000/docs

### 3. Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

## Configuration

- **Groq:** Cloud-based LLM inference using the Groq API Free Tier, subject to provider limits.
- **Ollama:** Optional local inference for supported models.
- **Embeddings:** Generated locally using Sentence Transformers.
- **Storage:** Documents, metadata, and vector data are stored locally.

## Privacy & Data Handling

- Uploaded documents are stored locally by the application.
- Embeddings are generated locally.
- When using Groq, retrieved document passages are sent to the external API for inference.
- Ollama provides an alternative for local LLM inference.
- Authentication and document ownership checks restrict access to user-specific resources.

## API Documentation

FastAPI provides interactive API documentation at:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Development Status

Implemented capabilities include:

- Document upload and processing
- Local embedding generation
- Semantic search
- RAG-based document chat
- Document summarization
- Multi-document Q&A
- Document comparison
- Advanced document intelligence
- Authentication and document ownership

## Future Improvements

- Document processing performance optimization
- Improved retrieval evaluation
- Automated regression testing
- Enhanced citation verification
- Additional document intelligence capabilities

