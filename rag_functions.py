from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from base_vectordb import BaseVectorDB
from config_loader import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_K


def load_documents_from_directory(
    directory_path: str,
    glob_pattern: str = "**/*.*",
    show_progress: bool = True,
) -> List[Document]:
    """
    Load documents from a directory.

    Args:
        directory_path: Path to the directory containing documents.
        glob_pattern: Pattern to match files.
        show_progress: Whether to show loading progress.

    Returns:
        List of loaded documents.
    """
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        show_progress=show_progress,
    )
    return loader.load()


def load_text_file(file_path: str) -> List[Document]:
    """Load a single text file."""
    loader = TextLoader(file_path)
    return loader.load()


def load_pdf_file(file_path: str) -> List[Document]:
    """Load a single PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks using config settings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def ingest(
    vector_db: BaseVectorDB,
    collection_name: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Ingest documents into a vector database collection.

    Args:
        vector_db: Vector database instance.
        collection_name: Name of the collection to ingest into.
        source: Path to file or directory to ingest.
        metadata: Optional metadata to add to all documents.

    Returns:
        Number of document chunks ingested.
    """
    source_path = Path(source)

    # Load documents based on source type
    if source_path.is_dir():
        documents = load_documents_from_directory(str(source_path))
    elif source_path.suffix.lower() == ".pdf":
        documents = load_pdf_file(str(source_path))
    else:
        documents = load_text_file(str(source_path))

    # Split documents
    chunks = split_documents(documents)

    # Add metadata if provided
    if metadata:
        for chunk in chunks:
            chunk.metadata.update(metadata)

    # Add to vector database
    vector_db.add_documents(chunks, collection_name)

    return len(chunks)


def retrieve(
    vector_db: BaseVectorDB,
    query: str,
    collection_name: str,
    k: Optional[int] = None,
) -> List[Document]:
    """
    Retrieve relevant documents from a vector database collection.

    Args:
        vector_db: Vector database instance.
        query: Search query string.
        collection_name: Name of the collection to search.
        k: Number of documents to retrieve (uses config default if not provided).

    Returns:
        List of relevant documents.
    """
    if k is None:
        k = DEFAULT_K
    return vector_db.retrieve(query, collection_name, k=k)
