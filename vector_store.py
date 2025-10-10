"""
Модуль для работы с векторным хранилищем
"""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Класс для работы с векторным хранилищем ChromaDB"""
    
    def __init__(self, 
                 collection_name: str = "arxiv_papers",
                 persist_directory: str = "data/vector_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Создаем директорию для хранения
        os.makedirs(persist_directory, exist_ok=True)
        
        # Инициализируем ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Загружаем модель эмбеддингов
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Получаем или создаем коллекцию
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Получает существующую коллекцию или создает новую"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Найдена существующая коллекция: {self.collection_name}")
            return collection
        except Exception:
            logger.info(f"Создание новой коллекции: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Open RAG Benchmark papers collection"}
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Добавляет документы в векторное хранилище
        """
        if not documents:
            logger.warning("Нет документов для добавления")
            return
        
        # Подготавливаем данные для ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{len(ids)}")
            ids.append(chunk_id)
            texts.append(doc.page_content)
            
            # Очищаем метаданные для ChromaDB (убираем сложные типы)
            clean_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            
            metadatas.append(clean_metadata)
        
        # Добавляем в коллекцию
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Добавлено {len(documents)} документов в коллекцию")
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Выполняет поиск по векторному хранилищу
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Преобразуем результаты в удобный формат
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    search_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'id': results['ids'][0][i]
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о коллекции
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}")
            return {}
    
    def delete_collection(self) -> None:
        """
        Удаляет коллекцию
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Коллекция {self.collection_name} удалена")
        except Exception as e:
            logger.error(f"Ошибка при удалении коллекции: {e}")
    
    def reset_collection(self) -> None:
        """
        Сбрасывает коллекцию (удаляет и создает заново)
        """
        self.delete_collection()
        self.collection = self._get_or_create_collection()
        logger.info(f"Коллекция {self.collection_name} сброшена")
    
    def get_documents_by_paper_id(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Получает все документы по ID статьи
        """
        try:
            results = self.collection.get(
                where={"paper_id": paper_id}
            )
            
            documents = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    documents.append({
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'id': results['ids'][i]
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка при получении документов по paper_id: {e}")
            return []
    
    def get_documents_by_content_type(self, content_type: str) -> List[Dict[str, Any]]:
        """
        Получает документы по типу контента
        """
        try:
            results = self.collection.get(
                where={"content_type": content_type}
            )
            
            documents = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    documents.append({
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'id': results['ids'][i]
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка при получении документов по content_type: {e}")
            return []
