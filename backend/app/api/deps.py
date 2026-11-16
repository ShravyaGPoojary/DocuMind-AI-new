import logging
from typing import List, Optional
import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User
from app.models.document import Document
from app.models.conversation import Conversation

logger = logging.getLogger(__name__)

security_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency that enforces valid JWT Bearer token authentication.
    Resolves the authenticated user from the database.
    """
    if not auth_credentials or not auth_credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_credentials.credentials
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token payload.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        logger.warning("JWT verification error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account associated with this token no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return user


def get_optional_current_user(
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Optional authentication dependency. Returns User if valid token is provided, else None.
    """
    if not auth_credentials or not auth_credentials.credentials:
        return None
    try:
        payload = decode_access_token(auth_credentials.credentials)
        user_id = payload.get("sub")
        if user_id:
            return db.query(User).filter(User.id == user_id, User.is_active == True).first()
    except Exception:
        pass
    return None


def verify_document_ownership(
    db: Session,
    document_id: str,
    user_id: str,
) -> Document:
    """
    Strict ownership verification: ensures a document exists and belongs to user_id.
    Raises 404 if not found, 403 if belonging to a different user.
    """
    clean_id = str(document_id).strip()
    doc = db.query(Document).filter(Document.id == clean_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{clean_id}' was not found.",
        )
    if doc.user_id and doc.user_id != user_id:
        logger.warning("Unauthorized document access attempt: user %s tried accessing doc %s owned by %s", user_id, clean_id, doc.user_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not own this document.",
        )
    return doc


def verify_documents_ownership(
    db: Session,
    document_ids: List[str],
    user_id: str,
) -> List[Document]:
    """
    Verifies that a collection of document IDs all belong to the user.
    """
    if not document_ids:
        return []
    docs = []
    for d_id in document_ids:
        docs.append(verify_document_ownership(db, d_id, user_id))
    return docs


def verify_conversation_ownership(
    db: Session,
    conversation_id: str,
    user_id: str,
) -> Conversation:
    """
    Strict ownership verification: ensures a conversation exists and belongs to user_id.
    """
    clean_id = str(conversation_id).strip()
    conv = db.query(Conversation).filter(Conversation.id == clean_id).first()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID '{clean_id}' was not found.",
        )
    if conv.user_id and conv.user_id != user_id:
        logger.warning("Unauthorized conversation access attempt: user %s tried accessing conversation %s owned by %s", user_id, clean_id, conv.user_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not own this conversation.",
        )
    return conv

