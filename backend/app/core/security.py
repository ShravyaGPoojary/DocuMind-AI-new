"""
DocuMind AI — Security and Cryptographic Utilities
Implements:
1. Secure password hashing and verification using Argon2.
2. Cryptographic JWT access token encoding and decoding.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

from app.core.config import settings

# Initialize Argon2 Password Hasher with OWASP-recommended parameters
hasher = PasswordHasher(
    time_cost=3,        # 3 iterations
    memory_cost=65536,  # 64 MiB
    parallelism=4,      # 4 lanes/threads
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    """Hashes a plain-text password using Argon2."""
    if not password or not password.strip():
        raise ValueError("Password cannot be empty or whitespace only.")
    return hasher.hash(password.strip())


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against an Argon2 hash string."""
    if not plain_password or not hashed_password:
        return False
    try:
        return hasher.verify(hashed_password, plain_password.strip())
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Encodes a JWT access token containing subject (user_id) and expiration claims.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=getattr(settings, "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))

    to_encode.update({
        "exp": expire,
        "iat": now,
        "iss": "documind-ai",
    })

    secret_key = getattr(settings, "JWT_SECRET_KEY", "documind-ai-super-secret-key-change-in-production-stage8-2026")
    algorithm = getattr(settings, "JWT_ALGORITHM", "HS256")

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and cryptographically validates a JWT access token.
    Raises jwt.PyJWTError on invalid or expired token.
    """
    secret_key = getattr(settings, "JWT_SECRET_KEY", "documind-ai-super-secret-key-change-in-production-stage8-2026")
    algorithm = getattr(settings, "JWT_ALGORITHM", "HS256")

    payload = jwt.decode(
        token,
        secret_key,
        algorithms=[algorithm],
        issuer="documind-ai",
        options={"require": ["exp", "iat", "sub"]},
    )
    return payload

