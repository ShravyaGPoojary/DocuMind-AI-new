from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Import all models here so that Base.metadata has them registered
try:
    from app.models.user import User                  # noqa: E402, F401
    from app.models.document import Document          # noqa: E402, F401
    from app.models.chunk import Chunk                # noqa: E402, F401
    from app.models.conversation import Conversation, Message  # noqa: E402, F401
    from app.models.summarization import SummarizationJob, SummarizationBatch  # noqa: E402, F401
except ImportError:
    pass

