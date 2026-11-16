import logging
from typing import Any, Dict, List, Optional
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaServiceError(Exception):
    """Base exception for Ollama LLM integration errors."""
    pass


class OllamaConnectionError(OllamaServiceError):
    """Raised when Ollama HTTP daemon is unreachable."""
    pass


class OllamaModelNotFoundError(OllamaServiceError):
    """Raised when the specified model is not downloaded in Ollama."""
    pass


class OllamaTimeoutError(OllamaServiceError):
    """Raised when an Ollama inference request times out."""
    pass


def format_bytes(size_bytes: Optional[int]) -> str:
    """Formats bytes into human-readable MB or GB string."""
    if not size_bytes or size_bytes <= 0:
        return "0 MB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class LLMService:
    """
    Asynchronous service managing interactions with a locally running Ollama LLM daemon.
    Communicates via Ollama's local HTTP API (default: http://localhost:11434).
    """

    @classmethod
    def get_base_url(cls) -> str:
        """Returns the configured Ollama base URL without trailing slash."""
        return settings.OLLAMA_BASE_URL.rstrip("/")

    @classmethod
    def get_configured_model(cls) -> str:
        """Returns the configured default LLM model name."""
        return settings.OLLAMA_MODEL

    @classmethod
    async def check_health(cls) -> Dict[str, Any]:
        """
        Checks connectivity to Ollama daemon and whether the configured model is ready.

        Returns:
            Dict containing status ('ok', 'degraded', 'unavailable'), reachability,
            configured model name, model availability, and installed models list.
        """
        base_url = cls.get_base_url()
        configured_model = cls.get_configured_model()
        available_models: List[str] = []
        is_reachable = False
        model_available = False
        error_msg: Optional[str] = None

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{base_url}/api/tags")
                if res.status_code == 200:
                    is_reachable = True
                    data = res.json()
                    models_list = data.get("models", [])
                    available_models = [m.get("name", "") for m in models_list if m.get("name")]

                    # Check if configured model or variant is installed
                    # Handles tags like 'qwen2.5:3b' or 'qwen2.5:3b-instruct' or 'qwen2.5:latest'
                    for m_name in available_models:
                        if (
                            m_name == configured_model
                            or m_name.startswith(f"{configured_model}:")
                            or configured_model.startswith(m_name.split(":")[0])
                        ):
                            model_available = True
                            break
                else:
                    error_msg = f"Ollama daemon returned HTTP {res.status_code}"
        except httpx.ConnectError:
            error_msg = (
                f"Ollama is not running at {base_url}. "
                "Ensure Ollama is installed and started ('ollama serve')."
            )
        except httpx.TimeoutException:
            error_msg = f"Connection to Ollama at {base_url} timed out."
        except Exception as e:
            error_msg = f"Error probing Ollama: {str(e)}"

        if is_reachable and model_available:
            status = "ok"
            summary = f"Ollama is operational with model '{configured_model}' ready."
        elif is_reachable and not model_available:
            status = "degraded"
            summary = (
                f"Ollama is running, but model '{configured_model}' is not downloaded. "
                f"Run: ollama pull {configured_model}"
            )
        else:
            status = "unavailable"
            summary = error_msg or f"Ollama is unavailable at {base_url}."

        return {
            "status": status,
            "ollama_reachable": is_reachable,
            "configured_model": configured_model,
            "model_available": model_available,
            "available_models": available_models,
            "error_message": error_msg,
            "message": summary,
        }

    @classmethod
    async def list_models(cls) -> List[Dict[str, Any]]:
        """
        Retrieves the list of installed models from Ollama's /api/tags endpoint.

        Returns:
            List of model metadata dictionaries.
        """
        base_url = cls.get_base_url()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{base_url}/api/tags")
                if res.status_code != 200:
                    raise OllamaServiceError(f"Failed to query Ollama models: HTTP {res.status_code}")
                data = res.json()
                raw_models = data.get("models", [])
                formatted = []
                for m in raw_models:
                    size = m.get("size", 0)
                    formatted.append({
                        "name": m.get("name", ""),
                        "modified_at": m.get("modified_at"),
                        "size_bytes": size,
                        "size_formatted": format_bytes(size),
                        "digest": m.get("digest"),
                    })
                return formatted
        except httpx.ConnectError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {base_url}. Please ensure Ollama is running."
            )
        except httpx.TimeoutException:
            raise OllamaTimeoutError(f"Timed out connecting to Ollama at {base_url}.")
        except OllamaServiceError:
            raise
        except Exception as e:
            raise OllamaServiceError(f"Unexpected error retrieving models from Ollama: {str(e)}")

    @classmethod
    async def generate(
        cls,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = 512,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Sends an asynchronous text completion request to Ollama's /api/generate endpoint.

        Args:
            prompt: User or RAG prompt text.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate (num_predict in Ollama).
            system_prompt: Optional system instruction.
            model: Optional model override.
            timeout: Optional timeout in seconds (defaults to settings.OLLAMA_TIMEOUT).

        Returns:
            Dict containing prompt, response text, model name, done, and performance metrics.
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace only.")

        # Validate temperature
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0.")

        # Validate max_tokens
        if max_tokens is not None and (max_tokens < 1 or max_tokens > 8192):
            raise ValueError("max_tokens must be an integer between 1 and 8192.")

        target_model = (model or cls.get_configured_model()).strip()
        base_url = cls.get_base_url()
        call_timeout = float(timeout or settings.OLLAMA_TIMEOUT)

        # Prepare Ollama /api/generate payload
        options: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt.strip(),
            "stream": False,
            "options": options,
        }
        if system_prompt and system_prompt.strip():
            payload["system"] = system_prompt.strip()

        try:
            async with httpx.AsyncClient(timeout=call_timeout) as client:
                res = await client.post(f"{base_url}/api/generate", json=payload)

                if res.status_code == 404:
                    err_detail = "Model not found."
                    try:
                        err_detail = res.json().get("error", err_detail)
                    except Exception:
                        pass
                    raise OllamaModelNotFoundError(
                        f"Model '{target_model}' not found in Ollama. "
                        f"Please pull it first by running: ollama pull {target_model} (Detail: {err_detail})"
                    )

                if res.status_code != 200:
                    err_msg = f"Ollama HTTP {res.status_code}"
                    try:
                        err_msg += f": {res.json().get('error', res.text)}"
                    except Exception:
                        err_msg += f": {res.text}"
                    raise OllamaServiceError(err_msg)

                data = res.json()
                response_text = data.get("response", "")
                total_duration_ns = data.get("total_duration")
                total_duration_ms = (
                    round(total_duration_ns / 1_000_000, 2)
                    if total_duration_ns is not None
                    else None
                )
                tokens_generated = data.get("eval_count")

                return {
                    "prompt": prompt,
                    "response": response_text,
                    "model": data.get("model", target_model),
                    "done": data.get("done", True),
                    "total_duration_ms": total_duration_ms,
                    "tokens_generated": tokens_generated,
                }

        except httpx.ConnectError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                "Ensure Ollama is installed and running on your system."
            )
        except httpx.TimeoutException:
            raise OllamaTimeoutError(
                f"Ollama generation timed out after {call_timeout} seconds. "
                "Consider reducing max_tokens or choosing a smaller model."
            )
        except (OllamaServiceError, ValueError):
            raise
        except Exception as e:
            logger.error("Unexpected error during Ollama generation: %s", e)
            raise OllamaServiceError(f"Failed to communicate with Ollama: {str(e)}")

