from fastapi import APIRouter
from app.schemas.document import EmbeddingStatsResponse
from app.services.chroma_service import ChromaService
from app.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.get("/stats", response_model=EmbeddingStatsResponse, summary="Get Embedding & Vector DB Stats")
async def get_embedding_stats():
    """
    Returns statistics regarding the ChromaDB vector collection and SentenceTransformer model.
    """
    stats = ChromaService.get_stats()
    model_info = EmbeddingService.get_model_info()
    return EmbeddingStatsResponse(
        collection_name=stats["collection_name"],
        total_vectors=stats["total_vectors"],
        persist_directory=stats["persist_directory"],
        embedding_model=stats["embedding_model"],
        dimension=model_info["dimension"],
        distance_metric=stats["distance_metric"],
        status=stats["status"],
    )

