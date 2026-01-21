from typing import Any
from config_loader import EMBEDDING_PROVIDER, EMBEDDING_MODEL


def get_embedding_model() -> Any:
    """Get the configured embedding model."""
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    elif EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")
