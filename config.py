"""
Конфигурация для RAG-системы
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Модели
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    
    # Параметры чанкинга
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Параметры поиска
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Пути к данным
    DATA_DIR = "data"
    CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
    VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db")
    
    # Настройки векторной базы
    COLLECTION_NAME = "arxiv_papers"
