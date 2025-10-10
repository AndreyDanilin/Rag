"""
Configuration for RAG system
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Models
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    
    # Chunking parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Search parameters
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Data paths
    DATA_DIR = "data"
    CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
    VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db")
    
    # Vector database settings
    COLLECTION_NAME = "arxiv_papers"
