import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from app.core.config import settings
from app.core.security import hash_password, verify_password, create_access_token
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import (
    UserRegisterRequest,
    UserLoginRequest,
    GoogleAuthRequest,
    UserResponse,
    TokenResponse,
    MessageResponse,
)
from app.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication & User Management"])


def serialize_user(user: User) -> UserResponse:
    """Helper to convert User model to UserResponse schema."""
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        created_at=user.created_at.strftime("%b %d, %Y") if user.created_at else "",
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User Account",
)
def register(
    request: UserRegisterRequest,
    db: Session = Depends(get_db),
):
    """
    Registers a new user account with full name, email, and password.
    Enforces email uniqueness, Argon2 password hashing, and returns a JWT access token.
    """
    clean_email = request.email.strip().lower()
    existing = db.query(User).filter(User.email == clean_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists. Please log in.",
        )

    hashed_pw = hash_password(request.password)
    new_user = User(
        full_name=request.full_name.strip(),
        email=clean_email,
        password_hash=hashed_pw,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(data={"sub": new_user.id, "email": new_user.email})
    return TokenResponse(access_token=token, token_type="bearer", user=serialize_user(new_user))


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login with Email & Password",
)
def login(
    request: UserLoginRequest,
    db: Session = Depends(get_db),
):
    """
    Authenticates a user with email and password via Argon2 verification.
    Returns a cryptographic JWT access token on success.
    """
    clean_email = request.email.strip().lower()
    user = db.query(User).filter(User.email == clean_email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account was registered using Google Sign-In. Please sign in with Google.",
        )

    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This user account has been deactivated.",
        )

    token = create_access_token(data={"sub": user.id, "email": user.email})
    return TokenResponse(access_token=token, token_type="bearer", user=serialize_user(user))


@router.post(
    "/google",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Sign in or Register with Google ID Token",
)
def google_auth(
    request: GoogleAuthRequest,
    db: Session = Depends(get_db),
):
    """
    Verifies a Google OAuth ID token server-side using google.oauth2.id_token.
    Validates token signature, issuer, audience, and expiration.
    Creates or links local user account safely.
    """
    client_id = settings.GOOGLE_CLIENT_ID
    request_transport = google_requests.Request()

    try:
        # Server-side cryptographic token verification
        # If client_id is set, it validates audience against it.
        # If client_id is None, it verifies Google signature and certs.
        id_info = id_token.verify_oauth2_token(
            request.credential,
            request_transport,
            audience=client_id if client_id else None,
        )
    except ValueError as e:
        logger.warning("Google ID token validation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token. Authentication failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Unexpected error validating Google token: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google token verification failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    google_sub = id_info.get("sub")
    email = id_info.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account does not provide a valid email address.",
        )

    clean_email = email.strip().lower()
    full_name = id_info.get("name") or id_info.get("given_name") or clean_email.split("@")[0]
    picture = id_info.get("picture")

    # Check if user already exists with this email or google_id
    user = db.query(User).filter((User.email == clean_email) | (User.google_id == google_sub)).first()

    if user:
        # Link google_id if not already linked
        if not user.google_id:
            user.google_id = google_sub
        if picture and not user.avatar_url:
            user.avatar_url = picture
        db.commit()
        db.refresh(user)
    else:
        # First-time Google user: create new local account
        user = User(
            full_name=full_name,
            email=clean_email,
            google_id=google_sub,
            avatar_url=picture,
            password_hash=None,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(data={"sub": user.id, "email": user.email})
    return TokenResponse(access_token=token, token_type="bearer", user=serialize_user(user))


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Authenticated User Profile",
)
def get_me(current_user: User = Depends(get_current_user)):
    """Returns the authenticated user's profile details."""
    return serialize_user(current_user)


@router.post(
    "/logout",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="User Logout",
)
def logout(current_user: User = Depends(get_current_user)):
    """Acknowledges user logout."""
    return MessageResponse(message=f"User {current_user.email} successfully logged out.")

