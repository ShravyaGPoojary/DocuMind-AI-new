import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.api.deps import get_current_user, verify_document_ownership
from app.schemas.intelligence import (
    ExtractionResponse,
    DocumentInsightsResponse,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    ContextualAnalysisRequest,
    ContextualAnalysisResponse,
)
from app.services.intelligence_service import IntelligenceService
from app.services.groq_service import (
    GroqAuthError,
    GroqRateLimitError,
    GroqTimeoutError,
    GroqServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intelligence", tags=["Advanced Document Intelligence"])


@router.post(
    "/extract/{document_id}",
    response_model=ExtractionResponse,
    summary="Intelligent Information Extraction",
    status_code=status.HTTP_200_OK,
)
async def extract_information(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Extracts structured entities, key dates, technical definitions, and quantitative metrics
    from a document owned by the authenticated user.
    """
    verify_document_ownership(db, document_id, current_user.id)
    try:
        return await IntelligenceService.extract_information(db=db, document_id=document_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except GroqRateLimitError as rle:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(rle))
    except GroqTimeoutError:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Groq extraction request timed out.")
    except Exception as e:
        logger.error("Extraction failed for doc %s: %s", document_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Information extraction failed: {str(e)}")


@router.post(
    "/insights/{document_id}",
    response_model=DocumentInsightsResponse,
    summary="Generate Deep Document Insights",
    status_code=status.HTTP_200_OK,
)
async def generate_insights(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Derives strategic insights, structural roadmap, major findings, and risk factors
    from a document owned by the authenticated user.
    """
    verify_document_ownership(db, document_id, current_user.id)
    try:
        return await IntelligenceService.generate_insights(db=db, document_id=document_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except GroqRateLimitError as rle:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(rle))
    except GroqTimeoutError:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Groq insights request timed out.")
    except Exception as e:
        logger.error("Insights failed for doc %s: %s", document_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Insights generation failed: {str(e)}")


@router.post(
    "/generate-questions/{document_id}",
    response_model=QuestionGenerationResponse,
    summary="Generate Document Knowledge-Check Questions",
    status_code=status.HTTP_200_OK,
)
async def generate_questions(
    document_id: str,
    request: Optional[QuestionGenerationRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generates educational, revision, or comprehension questions with answers grounded in the document.
    """
    verify_document_ownership(db, document_id, current_user.id)
    req = request or QuestionGenerationRequest()
    try:
        return await IntelligenceService.generate_questions(
            db=db,
            document_id=document_id,
            num_questions=req.num_questions,
            difficulty=req.difficulty or "mixed",
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except GroqRateLimitError as rle:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(rle))
    except GroqTimeoutError:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Groq question generation timed out.")
    except Exception as e:
        logger.error("Question generation failed for doc %s: %s", document_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Question generation failed: {str(e)}")


@router.post(
    "/contextual-analysis/{document_id}",
    response_model=ContextualAnalysisResponse,
    summary="Multi-Hop Contextual Analysis",
    status_code=status.HTTP_200_OK,
)
async def contextual_analysis(
    document_id: str,
    request: ContextualAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Performs multi-hop evidence reasoning on a specific question, theme, or proposition across document sections.
    """
    verify_document_ownership(db, document_id, current_user.id)
    try:
        return await IntelligenceService.perform_contextual_analysis(
            db=db,
            document_id=document_id,
            inquiry=request.inquiry,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except GroqRateLimitError as rle:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(rle))
    except GroqTimeoutError:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Groq contextual analysis timed out.")
    except Exception as e:
        logger.error("Contextual analysis failed for doc %s: %s", document_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Contextual analysis failed: {str(e)}")

