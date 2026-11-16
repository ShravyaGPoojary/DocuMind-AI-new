import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User
from app.api.deps import get_current_user
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic Document Search",
    status_code=status.HTTP_200_OK,
)
async def semantic_search(
    request: SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Performs semantic vector search across document chunks using dense embeddings,
    strictly isolated to documents owned by the authenticated user.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace only.",
        )

    try:
        results = SearchService.search(
            query=request.query,
            top_k=request.top_k or 5,
            document_id=request.document_id,
            document_ids=request.document_ids,
            db=db,
            user_id=current_user.id,
        )
        return SearchResponse(**results)
    except ValueError as val_err:
        err_msg = str(val_err)
        if "forbidden" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=err_msg,
            )
        if "not found" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=err_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        )

    except Exception as err:
        logger.error("Search execution failed: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during semantic search: {str(err)}",
        )

