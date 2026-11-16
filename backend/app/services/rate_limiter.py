"""
Token-Aware Rolling Rate Limiter for Groq Free Tier Inference
Enforces rolling 8,000 Tokens Per Minute (TPM) and 30 Requests Per Minute (RPM) limits.
Tracks both pre-flight estimated tokens and post-flight actual tokens returned by Groq.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class RequestUsageRecord:
    """Represents a recorded LLM request in the rolling 60-second window."""

    def __init__(self, request_id: str, timestamp: float, tokens: int):
        self.request_id = request_id
        self.timestamp = timestamp
        self.tokens = tokens


class TokenAwareRateLimiter:
    """
    Sliding 60-second window rate limiter for Groq Cloud Inference API.
    Maintains rolling TPM/RPM compliance and throttles proactively before requests are submitted.
    """

    def __init__(
        self,
        tpm_limit: int = settings.GROQ_TPM_LIMIT,  # 8000
        rpm_limit: int = settings.GROQ_RPM_LIMIT,  # 30
        window_seconds: float = 60.0,
    ):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.window_seconds = window_seconds
        self._history: List[RequestUsageRecord] = []
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        """Removes records that fall outside the rolling window."""
        cutoff = now - self.window_seconds
        self._history = [r for r in self._history if r.timestamp >= cutoff]

    async def get_rolling_metrics(self) -> Dict[str, Any]:
        """Returns current rolling metrics within the sliding window."""
        async with self._lock:
            now = time.time()
            self._prune(now)
            current_tpm = sum(r.tokens for r in self._history)
            current_rpm = len(self._history)
            return {
                "current_tpm": current_tpm,
                "tpm_limit": self.tpm_limit,
                "available_tpm": max(0, self.tpm_limit - current_tpm),
                "current_rpm": current_rpm,
                "rpm_limit": self.rpm_limit,
                "available_rpm": max(0, self.rpm_limit - current_rpm),
            }

    async def acquire(
        self,
        estimated_tokens: int,
        on_wait: Optional[Callable[[float, str], Any]] = None,
    ) -> str:
        """
        Acquires token and request quota in the rolling window.
        If estimated request tokens exceed available rolling TPM or RPM, waits asynchronously
        until sufficient quota expires from the sliding window.

        Returns:
            Unique request_id string to update with actual usage upon completion.
        """
        request_id = str(uuid.uuid4())

        while True:
            async with self._lock:
                now = time.time()
                self._prune(now)

                current_tpm = sum(r.tokens for r in self._history)
                current_rpm = len(self._history)

                # Check if request fits in budget
                tpm_ok = (current_tpm + estimated_tokens) <= self.tpm_limit
                rpm_ok = (current_rpm + 1) <= self.rpm_limit

                # Allow at least 1 request if history is completely empty even if est_tokens > tpm_limit
                if (tpm_ok and rpm_ok) or len(self._history) == 0:
                    # Allocate tentative estimated tokens
                    record = RequestUsageRecord(
                        request_id=request_id,
                        timestamp=now,
                        tokens=estimated_tokens,
                    )
                    self._history.append(record)
                    return request_id

                # Calculate required sleep time until enough entries expire to free needed quota
                needed_tpm = (current_tpm + estimated_tokens) - self.tpm_limit
                needed_rpm = (current_rpm + 1) - self.rpm_limit

                tokens_freed = 0
                reqs_freed = 0
                target_record = self._history[0]

                for r in self._history:
                    tokens_freed += r.tokens
                    reqs_freed += 1
                    target_record = r
                    if (needed_tpm <= 0 or tokens_freed >= needed_tpm) and (needed_rpm <= 0 or reqs_freed >= needed_rpm):
                        break

                wait_sec = max(0.5, (target_record.timestamp + self.window_seconds) - now + 0.2)
                reason = "TPM quota" if not tpm_ok else "RPM quota"
                msg = (
                    f"Waiting {round(wait_sec, 1)}s for rolling {reason} to refresh "
                    f"({current_tpm}/{self.tpm_limit} tokens, {current_rpm}/{self.rpm_limit} reqs)..."
                )
                logger.info(msg)

            # Release lock before sleeping
            if on_wait:
                try:
                    res = on_wait(wait_sec, msg)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as err:
                    logger.debug("on_wait callback error: %s", err)

            await asyncio.sleep(wait_sec)

    async def record_actual_usage(self, request_id: str, actual_tokens: int) -> None:
        """
        Updates the usage record with the exact tokens reported by Groq's usage.total_tokens.
        Passing actual_tokens=0 zeroes out the token reservation (e.g. if the request failed).
        """
        async with self._lock:
            now = time.time()
            self._prune(now)
            for r in self._history:
                if r.request_id == request_id:
                    old_tokens = r.tokens
                    r.tokens = max(0, actual_tokens)
                    logger.debug(
                        "RateLimiter updated request %s: estimated=%d -> actual=%d",
                        request_id,
                        old_tokens,
                        r.tokens,
                    )
                    break


# Global singleton instance for the application
groq_rate_limiter = TokenAwareRateLimiter()
