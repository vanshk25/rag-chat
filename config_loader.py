import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


config = load_config()

# Chunking settings
CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]

# Retrieval settings
DEFAULT_K = config["retrieval"]["default_k"]

# Storage settings
CHROMA_DB_PATH = config["storage"]["chroma_db_path"]
UPLOAD_DIR = config["storage"]["upload_dir"]

# LLM settings
LLM_MODEL = config["llm"]["model"]
LLM_BASE_URL = config["llm"]["base_url"]
LLM_API_KEY = config["llm"]["api_key"]
LLM_TEMPERATURE = config["llm"]["temperature"]
LLM_MAX_TOKENS = config["llm"]["max_tokens"]

# Embedding settings
EMBEDDING_PROVIDER = config["embedding"]["provider"]
EMBEDDING_MODEL = config["embedding"]["model"]
