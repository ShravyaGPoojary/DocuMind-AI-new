import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.models.conversation import Conversation
from app.api.deps import get_current_user, verify_conversation_ownership, verify_document_ownership
from app.schemas.chat import (

    ChatRequest,
    ChatResponse,
    ChatSourceItem,
    MessageSchema,
    ConversationSummarySchema,
    ConversationDetailResponse,
    ConversationListResponse,
)
from app.services.rag_service import RAGService
from app.services.llm_service import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaServiceError,
)
from app.services.groq_service import (
    GroqAuthError,
    GroqRateLimitError,
    GroqModelUnavailableError,
    GroqNetworkError,
    GroqTimeoutError,
    GroqResponseError,
    GroqServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["RAG Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="RAG Document Chat Question",
    status_code=status.HTTP_200_OK,
)
async def chat_with_documents(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    RAG Question-Answering Endpoint, strictly enforcing user data isolation:
    1. Validates ownership of conversation and target documents.
    2. Retrieves top-k semantically relevant chunks from ChromaDB for the user question.
    3. Prompts the selected LLM (Groq default, Ollama optional) to synthesize an accurate answer.
    4. Persists the conversation thread and message history under the authenticated user.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace only.",
        )

    # Validate conversation ownership if continuing an existing thread
    if request.conversation_id:
        existing_conv = db.query(Conversation).filter(Conversation.id == request.conversation_id.strip()).first()
        if existing_conv and existing_conv.user_id and existing_conv.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You do not own this conversation.",
            )

    # Validate target document ownership
    if request.document_ids:
        for d_id in request.document_ids:
            verify_document_ownership(db, d_id, current_user.id)
    elif request.document_id and request.document_id.lower() != "all":
        verify_document_ownership(db, request.document_id, current_user.id)

    try:
        result = await RAGService.ask_rag(
            db=db,
            question=request.question,
            document_id=request.document_id,
            document_ids=request.document_ids,
            top_k=request.top_k or 5,
            conversation_id=request.conversation_id,
            provider=request.provider,
            model=request.model,
            user_id=current_user.id,
        )
        return ChatResponse(**result)


    except ValueError as val_err:
        err_msg = str(val_err)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)

    except GroqAuthError as auth_err:
        logger.warning("RAG Chat: Groq authentication error: %s", auth_err)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(auth_err),
        )

    except GroqRateLimitError as rate_err:
        logger.warning("RAG Chat: Groq rate limit: %s", rate_err)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(rate_err),
        )

    except GroqModelUnavailableError as model_err:
        logger.warning("RAG Chat: Groq model error: %s", model_err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(model_err),
        )

    except GroqTimeoutError as timeout_err:
        logger.warning("RAG Chat: Groq timeout: %s", timeout_err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(timeout_err),
        )

    except GroqNetworkError as net_err:
        logger.warning("RAG Chat: Groq network error: %s", net_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(net_err),
        )

    except GroqResponseError as resp_err:
        logger.error("RAG Chat: Groq empty response: %s", resp_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(resp_err),
        )

    except GroqServiceError as serv_err:
        logger.error("RAG Chat: Groq service error: %s", serv_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(serv_err),
        )

    except OllamaConnectionError as conn_err:
        logger.warning("RAG Chat: Ollama connection error: %s", conn_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(conn_err),
        )

    except OllamaModelNotFoundError as model_err:
        logger.warning("RAG Chat: Ollama model error: %s", model_err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(model_err),
        )

    except OllamaTimeoutError as timeout_err:
        logger.warning("RAG Chat: Ollama timeout: %s", timeout_err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(timeout_err),
        )

    except OllamaServiceError as serv_err:
        logger.error("RAG Chat: Ollama service error: %s", serv_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(serv_err),
        )

    except Exception as err:
        logger.error("Unexpected error in RAG chat endpoint: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during document question answering: {str(err)}",
        )


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List Saved Conversations",
    status_code=status.HTTP_200_OK,
)
def list_conversations(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieves all saved conversation threads owned by the authenticated user."""
    convs = RAGService.list_conversations(db, skip=skip, limit=limit, user_id=current_user.id)
    summaries = [
        ConversationSummarySchema(
            id=c.id,
            title=c.title,
            document_id=c.document_id,
            document_ids=c.document_ids,
            created_at=c.created_at.strftime("%b %d, %Y %H:%M") if c.created_at else "",
            updated_at=c.updated_at.strftime("%b %d, %Y %H:%M") if c.updated_at else "",
            message_count=len(c.messages),
        )
        for c in convs
    ]
    return ConversationListResponse(total=len(summaries), conversations=summaries)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Get Conversation Thread with Messages",
    status_code=status.HTTP_200_OK,
)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieves a specific conversation thread owned by the authenticated user."""
    conv = verify_conversation_ownership(db, conversation_id, current_user.id)

    messages = []
    for m in conv.messages:
        sources_list = [ChatSourceItem(**s) for s in m.sources] if m.sources else []
        messages.append(
            MessageSchema(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                sources=sources_list,
                created_at=m.created_at.strftime("%I:%M %p") if m.created_at else "",
            )
        )

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        document_id=conv.document_id,
        document_ids=conv.document_ids,
        messages=messages,
    )


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete Conversation Thread",
    status_code=status.HTTP_200_OK,
)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Deletes a conversation thread owned by the authenticated user."""
    verify_conversation_ownership(db, conversation_id, current_user.id)
    deleted = RAGService.delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID '{conversation_id}' not found.",
        )
    return {"message": "Conversation deleted successfully.", "id": conversation_id}


