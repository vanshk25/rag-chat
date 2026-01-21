# RAG Template

## Running the FastAPI Server

To start the FastAPI server on port 54161, run the following command from the project directory:

```
uvicorn main:app --host 0.0.0.0 --port 54161
```

This will launch the API at http://localhost:54161/.

## Requirements
- Python 3.8+
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Project Structure
- `main.py`: FastAPI application entry point
- `config.yaml`: Configuration file
- `requirements.txt`: Python dependencies

## API Endpoints
- `/ingest`: Upload and ingest a file
- `/query`: Query documents
- `/collections`: List collections
- `/collections/{collection_name}`: Delete a collection
