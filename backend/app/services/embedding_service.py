import logging
from typing import Any, Dict, List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton service managing Sentence Transformers embedding model.
    Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
    """

    _model: Optional[Any] = None
    _model_name: str = settings.EMBEDDING_MODEL_NAME

    @classmethod
    def get_model(cls) -> Any:
        """Lazily load and return the SentenceTransformer model instance."""
        if cls._model is None:
            logger.info("Loading SentenceTransformer model '%s'...", cls._model_name)
            from sentence_transformers import SentenceTransformer

            try:
                cls._model = SentenceTransformer(cls._model_name, local_files_only=True)
            except Exception:
                cls._model = SentenceTransformer(cls._model_name)
            logger.info("Model '%s' loaded successfully.", cls._model_name)
        return cls._model

    @classmethod
    def generate_embeddings(cls, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of text strings.

        Args:
            texts: List of strings to encode.

        Returns:
            List of float lists, each having length equal to model dimension (384).
        """
        if not texts:
            return []

        model = cls.get_model()
        # SentenceTransformer.encode returns numpy ndarray
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    @classmethod
    def generate_embedding(cls, text: str) -> List[float]:
        """Generates embedding for a single string."""
        results = cls.generate_embeddings([text])
        return results[0] if results else []

    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Returns model metadata including name, vector dimension, and max sequence length."""
        model = cls.get_model()
        if hasattr(model, "get_embedding_dimension"):
            dimension = model.get_embedding_dimension()
        else:
            dimension = model.get_sentence_embedding_dimension()
        max_seq_length = getattr(model, "max_seq_length", 256)
        device = str(getattr(model, "device", "cpu"))
        return {
            "model_name": cls._model_name,
            "dimension": dimension,
            "max_seq_length": max_seq_length,
            "device": device,
        }
