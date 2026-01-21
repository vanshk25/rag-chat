from typing import List, Any, Dict
import chromadb
from langchain_chroma import Chroma
from base_vectordb import BaseVectorDB


class ChromaVectorDB(BaseVectorDB):
    """ChromaDB implementation of the vector database."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        tenant: str = chromadb.DEFAULT_TENANT,
        database: str = chromadb.DEFAULT_DATABASE,
    ) -> None:
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Path to persist the database.
            tenant: Tenant name for multi-tenancy.
            database: Database name.
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            tenant=tenant,
            database=database,
        )
        self.embedding_function = None
        self._vectorstores: Dict[str, Chroma] = {}

    def set_embedding(self, embedding_function: Any) -> None:
        """Set the embedding function for the vector store."""
        self.embedding_function = embedding_function

    def get_or_create_collection(self, collection_name: str) -> Any:
        """Get or create a collection and its associated vectorstore."""
        self.client.get_or_create_collection(name=collection_name)

        if collection_name not in self._vectorstores:
            if self.embedding_function is None:
                raise ValueError("Embedding function not set. Call set_embedding() first.")

            self._vectorstores[collection_name] = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding_function,
            )

        return self._vectorstores[collection_name]

    def add_documents(self, documents: List[Any], collection_name: str) -> None:
        """
        Add documents to the specified collection.

        Args:
            documents: List of LangChain Document objects.
            collection_name: Name of the collection to add documents to.
        """
        vectorstore = self.get_or_create_collection(collection_name)
        vectorstore.add_documents(documents)

    def retrieve(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> List[Any]:
        """
        Retrieve relevant documents from the specified collection.

        Args:
            query: Search query string.
            collection_name: Name of the collection to search.
            k: Number of documents to retrieve.

        Returns:
            List of relevant documents.
        """
        vectorstore = self.get_or_create_collection(collection_name)
        return vectorstore.similarity_search(query, k=k)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector store."""
        self.client.delete_collection(name=collection_name)
        if collection_name in self._vectorstores:
            del self._vectorstores[collection_name]

    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        return [col.name for col in self.client.list_collections()]
