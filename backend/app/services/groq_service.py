import logging
import re
import time
from typing import Any, Dict, List, Optional
from groq import (
    AsyncGroq,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    NotFoundError,
    APITimeoutError,
    APIConnectionError,
    BadRequestError,
    APIError,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqServiceError(Exception):
    """Base exception for Groq LLM integration errors."""
    pass


class GroqAuthError(GroqServiceError):
    """Raised when Groq API key is invalid, missing, or unauthorized (401/403)."""
    pass


class GroqRateLimitError(GroqServiceError):
    """Raised when Groq API rate limits or TPM/RPM/TPD are exceeded (429)."""
    def __init__(self, message: str, retry_after: Optional[float] = None, is_daily_limit: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.is_daily_limit = is_daily_limit



class GroqModelUnavailableError(GroqServiceError):
    """Raised when the requested Groq model is missing or unsupported (404/400)."""
    pass


class GroqNetworkError(GroqServiceError):
    """Raised when unable to establish connection to Groq API servers."""
    pass


class GroqTimeoutError(GroqServiceError):
    """Raised when Groq API call exceeds request timeout threshold."""
    pass


class GroqResponseError(GroqServiceError):
    """Raised when Groq returns an empty or unparseable response payload."""
    pass


class GroqOversizedRequestError(GroqServiceError):
    """Raised when request payload or token count exceeds model context/request limits (HTTP 413)."""
    pass



class GroqService:
    """
    Asynchronous service managing interactions with Groq Cloud Inference API.
    Provides fast, enterprise-grade LLM inference via AsyncGroq with OpenAI-compatible API.
    """

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Returns the configured Groq API key from environment settings."""
        return settings.GROQ_API_KEY

    @classmethod
    def get_configured_model(cls) -> str:
        """Returns the configured default Groq model identifier."""
        return settings.GROQ_MODEL

    @classmethod
    def get_client(cls, timeout: Optional[float] = None) -> AsyncGroq:
        """
        Instantiates and returns an AsyncGroq client configured with the API key.

        Raises:
            GroqAuthError: If GROQ_API_KEY is not set or empty.
        """
        api_key = cls.get_api_key()
        if not api_key or not api_key.strip():
            raise GroqAuthError(
                "Groq API key is not configured. "
                "Please set GROQ_API_KEY in backend/.env to enable Groq answer generation."
            )
        call_timeout = float(timeout or settings.GROQ_TIMEOUT)
        return AsyncGroq(
            api_key=api_key.strip(),
            timeout=call_timeout,
        )

    @classmethod
    async def check_health(cls) -> Dict[str, Any]:
        """
        Verifies Groq API connectivity and checks whether the configured model is available.

        Returns:
            Dict containing status ('ok', 'degraded', 'unavailable'), reachability,
            configured model name, model availability, and installed models list.
        """
        configured_model = cls.get_configured_model()
        api_key = cls.get_api_key()

        if not api_key or not api_key.strip():
            return {
                "status": "unavailable",
                "groq_reachable": False,
                "configured_model": configured_model,
                "model_available": False,
                "available_models": [],
                "error_message": "GROQ_API_KEY is not configured in backend/.env.",
                "message": "Groq is unavailable: API key is not configured.",
            }

        try:
            client = cls.get_client(timeout=5.0)
            models_page = await client.models.list()
            available_models = [m.id for m in models_page.data if hasattr(m, "id")]

            model_available = configured_model in available_models
            if not model_available:
                # Flexible match in case of minor tag differences
                model_available = any(
                    m == configured_model or configured_model in m
                    for m in available_models
                )

            status = "ok" if model_available else "degraded"
            summary = (
                f"Groq API is operational with model '{configured_model}' ready."
                if model_available
                else f"Groq API reachable, but configured model '{configured_model}' was not found in supported list."
            )

            return {
                "status": status,
                "groq_reachable": True,
                "configured_model": configured_model,
                "model_available": model_available,
                "available_models": available_models,
                "error_message": None if model_available else f"Model '{configured_model}' not found in Groq model list.",
                "message": summary,
            }

        except (AuthenticationError, PermissionDeniedError):
            return {
                "status": "unavailable",
                "groq_reachable": False,
                "configured_model": configured_model,
                "model_available": False,
                "available_models": [],
                "error_message": "Invalid or unauthorized Groq API key.",
                "message": "Groq authentication failed. Please verify GROQ_API_KEY in backend/.env.",
            }
        except RateLimitError:
            return {
                "status": "degraded",
                "groq_reachable": True,
                "configured_model": configured_model,
                "model_available": False,
                "available_models": [],
                "error_message": "Groq API rate limit exceeded.",
                "message": "Groq rate limit reached. Please wait before retrying.",
            }
        except APITimeoutError:
            return {
                "status": "unavailable",
                "groq_reachable": False,
                "configured_model": configured_model,
                "model_available": False,
                "available_models": [],
                "error_message": "Connection to Groq API timed out.",
                "message": "Groq API health check timed out.",
            }
        except APIConnectionError:
            return {
                "status": "unavailable",
                "groq_reachable": False,
                "configured_model": configured_model,
                "model_available": False,
                "available_models": [],
                "error_message": "Unable to connect to Groq API servers.",
                "message": "Failed to reach Groq API. Please check internet connectivity.",
            }
        except Exception as e:
            return {
                "status": "unavailable",
                "groq_reachable": False,
                "configured_model": configured_model,
                "model_available": False,
                "available_models": [],
                "error_message": f"Unexpected error probing Groq: {str(e)}",
                "message": "Groq API health check encountered an unexpected error.",
            }

    @classmethod
    async def list_models(cls) -> List[Dict[str, Any]]:
        """
        Retrieves the catalog of supported models from Groq.

        Returns:
            List of model metadata dictionaries.
        """
        client = cls.get_client(timeout=10.0)
        try:
            models_page = await client.models.list()
            return [
                {
                    "id": m.id,
                    "created": getattr(m, "created", None),
                    "owned_by": getattr(m, "owned_by", "groq"),
                    "active": getattr(m, "active", True),
                }
                for m in models_page.data
                if hasattr(m, "id")
            ]
        except (AuthenticationError, PermissionDeniedError):
            raise GroqAuthError("Invalid or unauthorized Groq API key.")
        except RateLimitError:
            raise GroqRateLimitError("Groq API rate limit reached.")
        except APITimeoutError:
            raise GroqTimeoutError("Timed out while retrieving Groq model catalog.")
        except APIConnectionError:
            raise GroqNetworkError("Cannot reach Groq API. Please verify network connectivity.")
        except Exception as e:
            raise GroqServiceError(f"Failed to query Groq model list: {str(e)}")

    @classmethod
    async def generate(
        cls,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 1024,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously generates text completion using Groq chat completion API.

        Args:
            prompt: User or RAG prompt text.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate (default: 1024).
            system_prompt: Optional system instruction for grounding / persona.
            model: Optional model override (defaults to settings.GROQ_MODEL).
            timeout: Optional per-request timeout in seconds.

        Returns:
            Dict containing prompt, response, model, done, total_duration_ms, and tokens_generated.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace only.")

        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0.")

        if max_tokens is not None and (max_tokens < 1 or max_tokens > 8192):
            raise ValueError("max_tokens must be an integer between 1 and 8192.")

        target_model = (model or cls.get_configured_model()).strip()
        call_timeout = float(timeout or settings.GROQ_TIMEOUT)
        client = cls.get_client(timeout=call_timeout)

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        start_time = time.perf_counter()

        try:
            chat_completion = await client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            if not chat_completion.choices:
                raise GroqResponseError("Groq API returned an empty choices list.")

            choice = chat_completion.choices[0]
            message_obj = choice.message
            response_content = (message_obj.content or "").strip()

            # Handle reasoning models where content may be in reasoning field
            if not response_content and hasattr(message_obj, "reasoning") and message_obj.reasoning:
                response_content = message_obj.reasoning.strip()

            if not response_content:
                raise GroqResponseError("Groq returned an empty response text.")

            usage = getattr(chat_completion, "usage", None)
            tokens_generated = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None

            # Prefer Groq's reported total_time if present in usage
            if usage and hasattr(usage, "total_time") and usage.total_time is not None:
                duration_ms = round(usage.total_time * 1000, 2)
            else:
                duration_ms = elapsed_ms

            actual_model = getattr(chat_completion, "model", target_model)

            return {
                "prompt": prompt,
                "response": response_content,
                "model": actual_model,
                "provider": "groq",
                "done": True,
                "total_duration_ms": duration_ms,
                "tokens_generated": tokens_generated,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
            }

        except AuthenticationError:
            logger.error("Groq authentication failure (401/403).")
            raise GroqAuthError(
                "Invalid or unauthorized Groq API key. Please check GROQ_API_KEY in backend/.env."
            )
        except PermissionDeniedError:
            logger.error("Groq permission denied (403).")
            raise GroqAuthError(
                "Groq API access denied. Ensure your API key has permissions for this model."
            )
        except RateLimitError as rl_err:
            err_text = str(rl_err).lower()
            logger.warning("Groq rate limit exceeded: %s", rl_err)
            is_daily = any(k in err_text for k in ["tokens per day", "tpd", "daily limit", "requests per day", "rpd", "day"])
            retry_after_sec = None
            if hasattr(rl_err, "response") and rl_err.response is not None:
                headers = getattr(rl_err.response, "headers", {})
                ra = headers.get("retry-after") or headers.get("Retry-After")
                if ra:
                    try:
                        retry_after_sec = float(ra)
                    except (ValueError, TypeError):
                        pass
            if retry_after_sec is None:
                match = re.search(r"try again in ([\d\.]+)s", str(rl_err))
                if match:
                    try:
                        retry_after_sec = float(match.group(1))
                    except (ValueError, TypeError):
                        pass
                else:
                    match_m = re.search(r"try again in ([\d\.]+)m", str(rl_err))
                    if match_m:
                        try:
                            retry_after_sec = float(match_m.group(1)) * 60.0
                        except (ValueError, TypeError):
                            pass

            if is_daily:
                retry_msg = f"Groq daily quota reached ({str(rl_err)}). Free Tier daily limit reset is required."
            elif retry_after_sec:
                retry_msg = f"Groq API rate limit reached. Please wait {round(retry_after_sec, 1)}s before retrying."
            else:
                retry_msg = "Groq API rate limit reached. Please wait a moment before sending another request."

            raise GroqRateLimitError(retry_msg, retry_after=retry_after_sec, is_daily_limit=is_daily)
        except NotFoundError as nf_err:
            logger.warning("Groq model not found: %s", nf_err)
            raise GroqModelUnavailableError(
                f"Groq model '{target_model}' was not found or is currently unsupported. "
                f"Please verify GROQ_MODEL in backend/.env."
            )
        except BadRequestError as br_err:
            err_str = str(br_err).lower()
            if "model" in err_str:
                raise GroqModelUnavailableError(
                    f"Groq rejected model '{target_model}': {str(br_err)}"
                )
            if any(term in err_str for term in ["too large", "context length", "maximum context", "reduce your prompt", "413"]):
                logger.warning("Groq prompt oversized (HTTP 413 / context exceeded): %s", br_err)
                raise GroqOversizedRequestError(f"Groq request payload too large: {str(br_err)}")
            raise GroqServiceError(f"Groq request error: {str(br_err)}")
        except APIError as api_err:
            status_code = getattr(api_err, "status_code", None)
            err_msg = str(api_err).lower()
            if status_code == 413 or "413" in err_msg or "too large" in err_msg:
                logger.warning("Groq HTTP 413 payload too large: %s", api_err)
                raise GroqOversizedRequestError(f"Groq HTTP 413 request too large: {str(api_err)}")
            raise GroqServiceError(f"Groq API error: {str(api_err)}")
        except APITimeoutError:
            logger.warning("Groq generation timed out after %s seconds.", call_timeout)
            raise GroqTimeoutError(
                f"Groq API request timed out after {call_timeout} seconds. "
                "Consider asking a more specific question."
            )
        except APIConnectionError as conn_err:
            logger.error("Groq connection error: %s", conn_err)
            raise GroqNetworkError(
                "Failed to connect to Groq API. Please check your network connection."
            )
        except (GroqServiceError, ValueError):
            raise
        except Exception as e:
            logger.error("Unexpected error during Groq generation: %s", e)
            raise GroqServiceError(f"Unexpected error communicating with Groq: {str(e)}")

