import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.api.deps import get_current_user, verify_document_ownership
from app.schemas.comparison import ComparisonRequest, ComparisonResponse
from app.services.comparison_service import ComparisonService

from app.services.groq_service import (
    GroqAuthError,
    GroqRateLimitError,
    GroqModelUnavailableError,
    GroqNetworkError,
    GroqTimeoutError,
    GroqResponseError,
    GroqServiceError,
)
from app.services.llm_service import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comparison", tags=["Document Comparison"])


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare Two Documents & Cross-Document Analysis",
    status_code=status.HTTP_200_OK,
)
async def compare_documents(
    request: ComparisonRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ComparisonResponse:
    """
    Analyzes and compares two distinct documents to generate:
    - Overall comparison summary
    - Key similarities and overlapping arguments
    - Structural and qualitative differences
    - Shared vs. unique thematic concepts
    - Direct contradictions, opposing claims, or numerical conflicts
    - Important agreements and corroborating evidence
    - Granular citations grounding each finding
    """
    # Enforce document ownership
    verify_document_ownership(db, request.document_id_1, current_user.id)
    verify_document_ownership(db, request.document_id_2, current_user.id)

    try:
        response = await ComparisonService.compare_documents(db=db, request=request)
        return response


    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )

    except GroqAuthError as gae:
        logger.error("Groq authentication failure during comparison: %s", gae)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Groq API key is invalid or unauthorized. Please verify GROQ_API_KEY in backend/.env.",
        )

    except GroqRateLimitError as grle:
        logger.warning("Groq rate limit exceeded during comparison: %s", grle)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Groq rate limit exceeded. Please wait a moment before running another comparison.",
        )

    except GroqTimeoutError as gte:
        logger.error("Groq API timed out during comparison: %s", gte)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Document comparison request timed out. Try specifying a narrower focus area or reducing document scope.",
        )

    except GroqServiceError as gse:
        logger.error("Groq inference error during comparison: %s", gse)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq LLM service error: {str(gse)}",
        )

    except (OllamaConnectionError, OllamaModelNotFoundError, OllamaTimeoutError, OllamaServiceError) as ose:
        logger.error("Ollama error during comparison: %s", ose)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Local LLM service error: {str(ose)}",
        )

    except Exception as e:
        logger.error("Unexpected error comparing documents: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during document comparison: {str(e)}",
        )

