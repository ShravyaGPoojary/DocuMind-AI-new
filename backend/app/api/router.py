from fastapi import APIRouter
from app.api.routes import health, info, auth, documents, embeddings, search, llm, chat, summarization, comparison, intelligence

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(info.router)
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(embeddings.router)
api_router.include_router(search.router)
api_router.include_router(llm.router)
api_router.include_router(chat.router)
api_router.include_router(summarization.router)
api_router.include_router(comparison.router)
api_router.include_router(intelligence.router)



