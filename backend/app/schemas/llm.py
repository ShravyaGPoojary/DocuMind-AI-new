from typing import List, Optional
from pydantic import BaseModel, Field


class LLMGenerateRequest(BaseModel):
    """Payload for requesting text completion from the default or specified LLM provider."""

    prompt: str = Field(..., min_length=1, description="Input prompt text for LLM generation")
    temperature: Optional[float] = Field(
        0.3, ge=0.0, le=2.0, description="Sampling temperature (lower is more deterministic)"
    )
    max_tokens: Optional[int] = Field(
        512, ge=1, le=8192, description="Maximum number of tokens to generate"
    )
    system_prompt: Optional[str] = Field(
        None, description="Optional system instruction for guiding model demeanor"
    )
    model: Optional[str] = Field(
        None, description="Optional model name override"
    )
    provider: Optional[str] = Field(
        None, description="Optional LLM provider ('groq' or 'ollama'). Defaults to configured LLM_PROVIDER."
    )


class LLMGenerateResponse(BaseModel):
    """Structured response returned by LLM text generation."""

    prompt: str = Field(..., description="The original input prompt")
    response: str = Field(..., description="The generated text response from the LLM")
    model: str = Field(..., description="The actual model name used for generation")
    provider: Optional[str] = Field("groq", description="The LLM provider used for generation ('groq' or 'ollama')")
    done: bool = Field(True, description="Whether generation is complete")
    total_duration_ms: Optional[float] = Field(None, description="Total generation duration in milliseconds")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated in completion")


class LLMHealthResponse(BaseModel):
    """Status report of LLM provider services and model availability."""

    status: str = Field(..., description="'ok', 'degraded', or 'unavailable'")
    provider: str = Field("groq", description="Current default LLM provider ('groq' or 'ollama')")
    ollama_reachable: bool = Field(False, description="True if local Ollama HTTP API is responding")
    groq_ready: bool = Field(False, description="True if Groq API key is configured and operational")
    configured_model: str = Field(..., description="Model configured in DocuMind AI settings")
    model_available: bool = Field(..., description="True if configured model is ready/supported")
    available_models: List[str] = Field(default_factory=list, description="Names of available models")
    error_message: Optional[str] = Field(None, description="Diagnostic error detail if unavailable")
    message: str = Field(..., description="Human-readable status summary")


class LLMModelItem(BaseModel):
    """Metadata describing a single installed model in Ollama."""

    name: str = Field(..., description="Model tag name, e.g. 'qwen2.5:3b'")
    modified_at: Optional[str] = Field(None, description="Last modification timestamp")
    size_bytes: Optional[int] = Field(None, description="Model file size in bytes")
    size_formatted: Optional[str] = Field(None, description="Human-readable file size (MB/GB)")
    digest: Optional[str] = Field(None, description="Model checksum hash")


class LLMModelsResponse(BaseModel):
    """List of all available models reported by local Ollama."""

    total: int = Field(..., description="Number of installed models")
    models: List[LLMModelItem] = Field(default_factory=list, description="List of installed model metadata")

