"""
Config for RAG agent
"""
import os

class Config:
    # Embedding model name for LlamaIndex
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
    # LLM (OpenAI or local)
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

    # Persistent index path for LlamaIndex
    INDEX_PATH = os.environ.get("INDEX_PATH", "./data/index")
    # Directory with documents/corpus
    CORPUS_PATH = os.environ.get("CORPUS_PATH", "./data/corpus")
    # Maximum number of retrieved results
    TOP_K = int(os.environ.get("TOP_K", 4))
    # Other parameters can be added as needed
