import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.core.config import settings
from app.schemas.llm import (
    LLMGenerateRequest,
    LLMGenerateResponse,
    LLMHealthResponse,
    LLMModelsResponse,
    LLMModelItem,
)
from app.services.llm_service import (
    LLMService,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaServiceError,
)
from app.services.groq_service import (
    GroqService,
    GroqAuthError,
    GroqRateLimitError,
    GroqModelUnavailableError,
    GroqNetworkError,
    GroqTimeoutError,
    GroqResponseError,
    GroqServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM Services"])


@router.post(
    "/generate",
    response_model=LLMGenerateResponse,
    summary="Generate Text via Default or Specified LLM Provider",
    status_code=status.HTTP_200_OK,
)
async def generate_text(request: LLMGenerateRequest):
    """
    Submits a prompt to the configured LLM provider (default: Groq GPT-OSS 120B, optional: Ollama Qwen2.5 3B).
    Returns the generated completion text and token metrics.
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty or whitespace only.",
        )

    target_provider = (request.provider or settings.LLM_PROVIDER or "groq").strip().lower()

    try:
        if target_provider == "groq":
            result = await GroqService.generate(
                prompt=request.prompt,
                temperature=request.temperature if request.temperature is not None else 0.2,
                max_tokens=request.max_tokens or 1024,
                system_prompt=request.system_prompt,
                model=request.model,
            )
        elif target_provider == "ollama":
            result = await LLMService.generate(
                prompt=request.prompt,
                temperature=request.temperature if request.temperature is not None else 0.3,
                max_tokens=request.max_tokens or 512,
                system_prompt=request.system_prompt,
                model=request.model,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported LLM provider '{target_provider}'. Choose 'groq' or 'ollama'.",
            )

        return LLMGenerateResponse(**result)

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except GroqAuthError as auth_err:
        logger.warning("Groq authentication error: %s", auth_err)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(auth_err),
        )
    except GroqRateLimitError as rate_err:
        logger.warning("Groq rate limit: %s", rate_err)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(rate_err),
        )
    except GroqModelUnavailableError as model_err:
        logger.warning("Groq model error: %s", model_err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(model_err),
        )
    except GroqTimeoutError as timeout_err:
        logger.warning("Groq timeout: %s", timeout_err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(timeout_err),
        )
    except GroqNetworkError as net_err:
        logger.warning("Groq network error: %s", net_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(net_err),
        )
    except GroqResponseError as resp_err:
        logger.error("Groq response error: %s", resp_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(resp_err),
        )
    except GroqServiceError as serv_err:
        logger.error("Groq service error: %s", serv_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(serv_err),
        )
    except OllamaConnectionError as conn_err:
        logger.warning("Ollama connection error: %s", conn_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(conn_err),
        )
    except OllamaModelNotFoundError as model_err:
        logger.warning("Ollama model not found: %s", model_err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(model_err),
        )
    except OllamaTimeoutError as timeout_err:
        logger.warning("Ollama timeout error: %s", timeout_err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(timeout_err),
        )
    except OllamaServiceError as serv_err:
        logger.error("Ollama service error: %s", serv_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(serv_err),
        )
    except Exception as err:
        logger.error("Unexpected error during LLM generation: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal LLM processing error: {str(err)}",
        )


@router.get(
    "/health",
    response_model=LLMHealthResponse,
    summary="Check Default and Configured LLM Health",
    status_code=status.HTTP_200_OK,
)
async def get_llm_health():
    """
    Checks the status of the configured primary LLM provider (Groq Cloud) and local Ollama daemon.
    Always returns HTTP 200 with clear diagnostics for dashboard and monitoring.
    """
    default_provider = (settings.LLM_PROVIDER or "groq").strip().lower()

    if default_provider == "groq":
        health_data = await GroqService.check_health()
        ollama_reachable = False
        try:
            ollama_health = await LLMService.check_health()
            ollama_reachable = ollama_health.get("ollama_reachable", False)
        except Exception:
            pass

        return LLMHealthResponse(
            status=health_data["status"],
            provider="groq",
            configured_model=health_data["configured_model"],
            model_available=health_data["model_available"],
            available_models=health_data["available_models"],
            groq_ready=health_data["groq_reachable"] and health_data["model_available"],
            ollama_reachable=ollama_reachable,
            error_message=health_data.get("error_message"),
            message=health_data["message"],
        )
    else:
        health_data = await LLMService.check_health()
        return LLMHealthResponse(
            status=health_data["status"],
            provider="ollama",
            configured_model=health_data["configured_model"],
            model_available=health_data["model_available"],
            available_models=health_data["available_models"],
            groq_ready=bool(settings.GROQ_API_KEY),
            ollama_reachable=health_data["ollama_reachable"],
            error_message=health_data.get("error_message"),
            message=health_data["message"],
        )


@router.get(
    "/models",
    response_model=LLMModelsResponse,
    summary="List Supported Models from Selected or Default Provider",
    status_code=status.HTTP_200_OK,
)
async def list_llm_models(
    provider: Optional[str] = Query(None, description="Optional provider filter ('groq' or 'ollama')")
):
    """
    Queries model catalog from Groq or local Ollama daemon based on parameter or default settings.
    """
    target_provider = (provider or settings.LLM_PROVIDER or "groq").strip().lower()

    if target_provider == "groq":
        try:
            models_data = await GroqService.list_models()
            items = [
                LLMModelItem(
                    name=m["id"],
                    modified_at=str(m.get("created")),
                    digest=m.get("owned_by"),
                )
                for m in models_data
            ]
            return LLMModelsResponse(total=len(items), models=items)
        except GroqAuthError as auth_err:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(auth_err))
        except GroqRateLimitError as rate_err:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(rate_err))
        except GroqTimeoutError as timeout_err:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(timeout_err))
        except GroqNetworkError as net_err:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(net_err))
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to query Groq models: {str(err)}",
            )
    elif target_provider == "ollama":
        try:
            models_data = await LLMService.list_models()
            items = [LLMModelItem(**m) for m in models_data]
            return LLMModelsResponse(total=len(items), models=items)
        except OllamaConnectionError as conn_err:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(conn_err))
        except OllamaTimeoutError as timeout_err:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(timeout_err))
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to retrieve model list from Ollama: {str(err)}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider '{target_provider}'. Choose 'groq' or 'ollama'.",
        )


