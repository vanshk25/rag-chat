from abc import ABC, abstractmethod
from typing import List, Any


class BaseVectorDB(ABC):
    """Abstract base class for vector database implementations."""

    @abstractmethod
    def set_embedding(self, embedding_function: Any) -> None:
        """Set the embedding function for the vector store."""
        pass

    @abstractmethod
    def get_or_create_collection(self, collection_name: str) -> Any:
        """Get or create a collection in the vector store."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Any], collection_name: str) -> None:
        """Add documents to the specified collection."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> List[Any]:
        """Retrieve relevant documents from the specified collection."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector store."""
        pass
