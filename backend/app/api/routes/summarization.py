import asyncio
import logging
from typing import Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.api.deps import get_current_user, verify_document_ownership
from app.models.summarization import SummarizationJob
from app.schemas.summarization import (

    SummarizationRequest,
    SummarizationResponse,
    SummarizationJobResponse,
    SummaryTypesResponse,
    SummaryTypeItem,
)
from app.services.summarization_service import SummarizationService
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

router = APIRouter(prefix="/summarization", tags=["Document Summarization"])


@router.get(
    "/types",
    response_model=SummaryTypesResponse,
    summary="List Supported Summary Types",
    status_code=status.HTTP_200_OK,
)
def get_summary_types():
    """
    Returns the catalog of supported summary formats:
    - Brief Summary
    - Detailed Analytical Summary
    - Key Points & Actionable Insights
    - Executive Briefing
    """
    types_data = SummarizationService.get_supported_types()
    items = [SummaryTypeItem(**t) for t in types_data]
    return SummaryTypesResponse(types=items)


@router.get(
    "/jobs/{job_id}",
    response_model=SummarizationJobResponse,
    summary="Get Status and Progress of a Summarization Job",
    status_code=status.HTTP_200_OK,
)
def get_summarization_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves the real-time progress, status, batch metrics, and parsed results
    for an asynchronous or background summarization job, enforcing document ownership.
    """
    job = db.query(SummarizationJob).filter(SummarizationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Summarization job '{job_id}' not found.")
    verify_document_ownership(db, job.document_id, current_user.id)

    try:
        status_data = SummarizationService.get_job_status(db, job_id)
        return SummarizationJobResponse(**status_data)

    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as err:
        logger.error("Error retrieving job status '%s': %s", job_id, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query summarization job status: {str(err)}",
        )


@router.post(
    "/jobs/{job_id}/resume",
    response_model=SummarizationJobResponse,
    summary="Resume Interrupted or Failed Summarization Job",
    status_code=status.HTTP_200_OK,
)
async def resume_summarization_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Resumes a paused, rate-limited, or failed summarization job, enforcing document ownership.
    Picks up immediately from already-completed batch digests stored in SQLite,
    avoiding duplicate API calls and saving tokens.
    """
    job = db.query(SummarizationJob).filter(SummarizationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Summarization job '{job_id}' not found.")
    verify_document_ownership(db, job.document_id, current_user.id)

    try:
        resumed_job = SummarizationService.resume_job(db, job_id)
        status_data = SummarizationService.get_job_status(db, resumed_job.id)
        return SummarizationJobResponse(**status_data)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as err:
        logger.error("Error resuming job '%s': %s", job_id, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume summarization job: {str(err)}",
        )


@router.post(
    "/{document_id}",
    response_model=Union[SummarizationResponse, SummarizationJobResponse],
    summary="Generate Intelligent Document Summary",
    status_code=status.HTTP_200_OK,
)
async def summarize_document(
    document_id: str,
    request: Optional[SummarizationRequest] = None,
    background: Optional[bool] = Query(None, description="Run in background with job tracking"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generates a grounded, high-fidelity summary for an uploaded document using Groq GPT-OSS 120B,
    enforcing document ownership.
    - If background=True (via body or query param), returns immediately with a job ID for polling.
    - If background=False (default), executes synchronously and returns the complete summary.
    - Preserves sequential document chunk ordering.
    - Dynamically applies single-pass direct or hierarchical map-reduce summarization.
    """
    verify_document_ownership(db, document_id, current_user.id)
    req = request or SummarizationRequest()
    is_background = background if background is not None else bool(req.background)


    if is_background:
        try:
            job = SummarizationService.create_job(
                db=db,
                document_id=document_id,
                summary_type=req.summary_type,
                provider=req.provider,
                model=req.model,
            )
            # Dispatch background worker
            asyncio.create_task(SummarizationService.run_background_job(job.id))
            status_data = SummarizationService.get_job_status(db, job.id)
            return SummarizationJobResponse(**status_data)
        except ValueError as val_err:
            err_msg = str(val_err)
            if "not found" in err_msg.lower():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
        except Exception as err:
            logger.error("Error initializing background summarization job: %s", err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to dispatch background summarization: {str(err)}",
            )

    # Synchronous execution
    try:
        result = await SummarizationService.summarize_document(
            db=db,
            document_id=document_id,
            summary_type=req.summary_type,
            provider=req.provider,
            model=req.model,
        )
        return SummarizationResponse(**result)

    except ValueError as val_err:
        err_msg = str(val_err)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)

    except GroqAuthError as auth_err:
        logger.warning("Summarization: Groq authentication error: %s", auth_err)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(auth_err),
        )

    except GroqRateLimitError as rate_err:
        logger.warning("Summarization: Groq rate limit: %s", rate_err)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(rate_err),
        )

    except GroqModelUnavailableError as model_err:
        logger.warning("Summarization: Groq model error: %s", model_err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(model_err),
        )

    except GroqTimeoutError as timeout_err:
        logger.warning("Summarization: Groq timeout: %s", timeout_err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(timeout_err),
        )

    except GroqNetworkError as net_err:
        logger.warning("Summarization: Groq network error: %s", net_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(net_err),
        )

    except GroqResponseError as resp_err:
        logger.error("Summarization: Groq empty response: %s", resp_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(resp_err),
        )

    except GroqServiceError as serv_err:
        logger.error("Summarization: Groq service error: %s", serv_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(serv_err),
        )

    except OllamaConnectionError as conn_err:
        logger.warning("Summarization: Ollama connection error: %s", conn_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(conn_err),
        )

    except OllamaModelNotFoundError as model_err:
        logger.warning("Summarization: Ollama model error: %s", model_err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(model_err),
        )

    except OllamaTimeoutError as timeout_err:
        logger.warning("Summarization: Ollama timeout error: %s", timeout_err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(timeout_err),
        )

    except OllamaServiceError as serv_err:
        logger.error("Summarization: Ollama service error: %s", serv_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(serv_err),
        )

    except Exception as err:
        logger.error("Unexpected error during document summarization: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during document summarization: {str(err)}",
        )
