from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List
import shutil
import os
from pathlib import Path

from dotenv import load_dotenv

from chroma_vectordb import ChromaVectorDB
from openai_llm import OpenAICompatibleLLM
from embeddings import get_embedding_model
from rag_functions import ingest, retrieve
from config_loader import (
    CHROMA_DB_PATH,
    UPLOAD_DIR,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

load_dotenv()

app = FastAPI(title="RAG API", version="1.0.0")

# Initialize vector database and embeddings
db = ChromaVectorDB(persist_directory=CHROMA_DB_PATH)
db.set_embedding(get_embedding_model())

# Initialize LLM
llm = OpenAICompatibleLLM(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

UPLOAD_PATH = Path(UPLOAD_DIR)
UPLOAD_PATH.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    collection_name: str



class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatResponse)
def chat_with_llm(request: ChatRequest):
    """Chat directly with the LLM (no retrieval)."""
    try:
        answer = llm.generate(prompt=request.prompt)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app on port 54161, use:
# uvicorn main:app --reload --port 54161


@app.post("/ingest")
async def ingest_file(
    collection_name: str,
    file: UploadFile = File(...),
):
    """Upload and ingest a file into a collection."""
    try:
        file_path = UPLOAD_PATH / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        num_chunks = ingest(
            vector_db=db,
            collection_name=collection_name,
            source=str(file_path),
            metadata={
                "source": str(file_path),
                "file_name": file.filename,
            },
        )

        os.remove(file_path)

        return {"message": "File ingested successfully", "chunks": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-multiple")
async def ingest_multiple_files(
    collection_name: str,
    files: List[UploadFile] = File(...),
):
    """Upload and ingest multiple files into a collection in a single request."""
    try:
        total_chunks = 0
        file_summaries = []

        for file in files:
            file_path = UPLOAD_PATH / file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            num_chunks = ingest(
                vector_db=db,
                collection_name=collection_name,
                source=str(file_path),
                metadata={
                    "source": str(file_path),
                    "file_name": file.filename,
                },
            )

            os.remove(file_path)

            total_chunks += num_chunks
            file_summaries.append({"filename": file.filename, "chunks": num_chunks})

        return {
            "message": "Files ingested successfully",
            "total_chunks": total_chunks,
            "files": file_summaries,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Query documents and get an LLM-generated response."""
    try:
        results = retrieve(
            vector_db=db,
            query=request.query,
            collection_name=request.collection_name,
        )

        context = [doc.page_content for doc in results]
        answer = llm.generate_with_context(query=request.query, context=context)

        sources = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]

        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
def list_collections():
    """List all available collections."""
    return {"collections": db.list_collections()}


@app.get("/collections/{collection_name}/documents")
def list_documents_in_collection(collection_name: str):
    """List document ids and names for a specific collection."""
    try:
        raw_documents = db.list_documents(collection_name)

        documents: List[Dict[str, Any]] = []
        for doc in raw_documents:
            doc_id = doc.get("id")
            metadata = doc.get("metadata") or {}
            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("filename")
            )

            name = Path(source).name if source else doc_id

            documents.append({"id": doc_id, "name": name})

        return {"collection": collection_name, "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    """Delete a collection."""
    try:
        db.delete_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}/documents/{document_id}")
def delete_document_from_collection(collection_name: str, document_id: str):
    """Delete a single document from a collection by its id."""
    try:
        db.delete_document(collection_name, document_id)
        return {
            "message": f"Document '{document_id}' deleted from collection '{collection_name}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))