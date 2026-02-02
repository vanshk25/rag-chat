from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from uuid import uuid4
import shutil
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_postgres import PostgresChatMessageHistory
import psycopg

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
tags_metadata = [
    {
        "name": "Ingestion",
        "description": "Upload and ingest documents into vector collections.",
    },
    {
        "name": "Query",
        "description": "RAG query endpoints using retrieval + LLM + memory.",
    },
    {
        "name": "Collections",
        "description": "Manage and inspect document collections.",
    },
    {
        "name": "Sessions",
        "description": "Inspect conversation sessions and their histories.",
    },
    {
        "name": "Debug",
        "description": "Debug and testing endpoints.",
    },
]

app = FastAPI(title="Baker Triangle", version="1.0.0", openapi_tags=tags_metadata)

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


# Initialize Postgres chat history connection (optional)
CHAT_HISTORY_DB_URL = os.getenv("POSTGRES_CHAT_HISTORY_URL")
CHAT_HISTORY_TABLE = os.getenv("POSTGRES_CHAT_HISTORY_TABLE", "chat_history")
CHAT_HISTORY_CONN = None

if CHAT_HISTORY_DB_URL:
    try:
        CHAT_HISTORY_CONN = psycopg.connect(CHAT_HISTORY_DB_URL)
        PostgresChatMessageHistory.create_tables(CHAT_HISTORY_CONN, CHAT_HISTORY_TABLE)
        print("[chat_history] Connected to Postgres and ensured table '%s'" % CHAT_HISTORY_TABLE)
    except Exception as e:
        CHAT_HISTORY_CONN = None
        print("[chat_history] Failed to connect to Postgres: %s" % e)
else:
    print("[chat_history] POSTGRES_CHAT_HISTORY_URL not set; chat history is disabled")


class QueryRequest(BaseModel):
    query: str
    collection_name: str
    session_id: Optional[str] = None



class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str

class QueryResponse(BaseModel):
    answer: str
    session_id: str

@app.post("/chat", response_model=ChatResponse, tags=["Debug"])
def chat_with_llm(request: ChatRequest):
    """Chat directly with the LLM (no retrieval)."""
    try:
        answer = llm.generate(prompt=request.prompt)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app on port 54161, use:
# uvicorn main:app --reload --port 54161


@app.post("/ingest", tags=["Ingestion"])
async def ingest_file(
    collection_name: str,
    file: UploadFile = File(...),
):
    """Upload and ingest a file into a collection."""
    try:
        document_id = str(uuid4())
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
                "document_id": document_id,
            },
        )

        os.remove(file_path)

        return {
            "message": "File ingested successfully",
            "chunks": num_chunks,
            "document_id": document_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-multiple", tags=["Ingestion"])
async def ingest_multiple_files(
    collection_name: str,
    files: List[UploadFile] = File(...),
):
    """Upload and ingest multiple files into a collection in a single request."""
    try:
        total_chunks = 0
        file_summaries = []

        for file in files:
            document_id = str(uuid4())
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
                    "document_id": document_id,
                },
            )

            os.remove(file_path)

            total_chunks += num_chunks
            file_summaries.append(
                {
                    "filename": file.filename,
                    "chunks": num_chunks,
                    "document_id": document_id,
                }
            )

        return {
            "message": "Files ingested successfully",
            "total_chunks": total_chunks,
            "files": file_summaries,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query_documents(
    query: str,
    collection_name: str,
    session_id: Optional[str] = None,
):
    """Query documents and get an LLM-generated response.

    Parameters are exposed as separate fields in the docs UI.
    """
    try:
        # Resolve or create session id
        session_id = session_id or str(uuid4())

        # Initialize Postgres-backed chat history if configured
        history = None
        if CHAT_HISTORY_CONN is not None:
            history = PostgresChatMessageHistory(
                CHAT_HISTORY_TABLE,
                session_id,
                sync_connection=CHAT_HISTORY_CONN,
            )

        results = retrieve(
            vector_db=db,
            query=query,
            collection_name=collection_name,
        )

        # Build context from retrieved docs
        context = [doc.page_content for doc in results]

        # Prepend chat history as additional context if available
        if history is not None:
            messages = history.messages
            recent = messages[-10:]
            if recent:
                history_lines = []
                for m in recent:
                    role = "user" if m.type == "human" else "assistant" if m.type == "ai" else m.type
                    history_lines.append(f"{role}: {m.content}")
                history_text = "Chat history:\n" + "\n".join(history_lines)
                context.insert(0, history_text)

        answer = llm.generate_with_context(query=query, context=context)

        # Persist this turn in history
        if history is not None:
            history.add_user_message(query)
            history.add_ai_message(answer)

        return QueryResponse(answer=answer, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections", tags=["Collections"])
def list_collections():
    """List all available collections."""
    return {"collections": db.list_collections()}


@app.get("/collections/{collection_name}/documents", tags=["Collections"])
def list_documents_in_collection(collection_name: str):
    """List unique documents (grouped over chunks) in a collection."""
    try:
        raw_documents = db.list_documents(collection_name)

        grouped: Dict[str, Dict[str, Any]] = {}

        for doc in raw_documents:
            metadata = doc.get("metadata") or {}
            logical_id = metadata.get("document_id")

            # Fallback for older data without document_id: group by file name
            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("filename")
            )
            name = Path(source).name if source else logical_id or doc.get("id")

            if logical_id is None:
                logical_id = name

            if logical_id not in grouped:
                grouped[logical_id] = {"id": logical_id, "name": name}

        documents = list(grouped.values())

        return {"collection": collection_name, "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}", tags=["Collections"])
def delete_collection(collection_name: str):
    """Delete a collection."""
    try:
        db.delete_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/history", tags=["Sessions"])
def get_session_history(session_id: str):
    """Return the chat history for a given session id."""
    if CHAT_HISTORY_CONN is None:
        raise HTTPException(
            status_code=500,
            detail="Chat history storage is not configured.",
        )

    try:
        history = PostgresChatMessageHistory(
            CHAT_HISTORY_TABLE,
            session_id,
            sync_connection=CHAT_HISTORY_CONN,
        )

        messages = []
        for m in history.messages:
            if m.type in ("human", "user"):
                role = "user"
            elif m.type in ("ai", "assistant"):
                role = "assistant"
            else:
                role = m.type

            messages.append({"role": role, "content": m.content})

        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}/documents/{document_id}", tags=["Collections"])
def delete_document_from_collection(collection_name: str, document_id: str):
    """Delete a single document from a collection by its id."""
    try:
        db.delete_document(collection_name, document_id)
        return {
            "message": f"Document '{document_id}' deleted from collection '{collection_name}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))