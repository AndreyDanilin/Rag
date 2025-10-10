"""
Основной класс RAG-системы
"""
import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import logging

from config import Config
from data_loader import OpenRAGDataLoader
from document_processor import DocumentProcessor
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class RAGSystem:
    """Основной класс RAG-системы"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Инициализируем компоненты
        self.data_loader = OpenRAGDataLoader(self.config.DATA_DIR)
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(
            collection_name=self.config.COLLECTION_NAME,
            persist_directory=self.config.VECTOR_DB_PATH,
            embedding_model=self.config.EMBEDDING_MODEL
        )
        
        # Инициализируем языковую модель
        if self.config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=self.config.LLM_MODEL,
                api_key=self.config.OPENAI_API_KEY,
                temperature=0.1
            )
        else:
            logger.warning("OpenAI API ключ не найден. Используйте локальную модель.")
            self.llm = None
        
        # Создаем промпт для генерации ответов
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Создает шаблон промпта для генерации ответов"""
        template = """Ты - помощник по научным статьям. Используй предоставленный контекст для ответа на вопрос.

Контекст:
{context}

Вопрос: {question}

Инструкции:
1. Отвечай на русском языке
2. Используй только информацию из предоставленного контекста
3. Если в контексте нет ответа на вопрос, скажи об этом
4. Структурируй ответ четко и понятно
5. При необходимости ссылайся на конкретные части контекста

Ответ:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def initialize_system(self, download_data: bool = False) -> None:
        """
        Инициализирует RAG-систему
        """
        logger.info("Инициализация RAG-системы...")
        
        # Загружаем данные
        if download_data:
            self.data_loader.download_dataset()
        
        # Загружаем корпус документов
        papers = self.data_loader.load_corpus()
        
        if not papers:
            logger.warning("Корпус документов пуст")
            return
        
        # Обрабатываем документы
        documents = self.document_processor.process_corpus(papers)
        
        if not documents:
            logger.warning("Не удалось создать документы")
            return
        
        # Добавляем документы в векторное хранилище
        self.vector_store.add_documents(documents)
        
        # Выводим статистику
        stats = self.document_processor.get_document_stats(documents)
        collection_info = self.vector_store.get_collection_info()
        
        logger.info(f"RAG-система инициализирована:")
        logger.info(f"- Обработано статей: {stats['papers_count']}")
        logger.info(f"- Создано чанков: {stats['total_chunks']}")
        logger.info(f"- Типы контента: {stats['content_types']}")
        logger.info(f"- Документов в векторной БД: {collection_info.get('document_count', 0)}")
    
    def search_documents(self, 
                        query: str, 
                        n_results: int = None,
                        content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Выполняет поиск релевантных документов
        """
        n_results = n_results or self.config.TOP_K_RESULTS
        
        # Подготавливаем фильтры
        where_filter = None
        if content_type:
            where_filter = {"content_type": content_type}
        
        # Выполняем поиск
        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            where=where_filter
        )
        
        logger.info(f"Найдено {len(results)} релевантных документов для запроса: '{query}'")
        return results
    
    def generate_answer(self, 
                       query: str, 
                       context_documents: List[Dict[str, Any]] = None,
                       n_results: int = None) -> Dict[str, Any]:
        """
        Генерирует ответ на основе запроса и контекста
        """
        if not self.llm:
            return {
                "answer": "Языковая модель не настроена. Пожалуйста, настройте OpenAI API ключ.",
                "context": [],
                "error": "LLM not configured"
            }
        
        # Получаем контекстные документы
        if context_documents is None:
            context_documents = self.search_documents(query, n_results)
        
        if not context_documents:
            return {
                "answer": "Не найдено релевантных документов для ответа на ваш вопрос.",
                "context": [],
                "error": "No relevant documents found"
            }
        
        # Формируем контекст
        context = self._format_context(context_documents)
        
        try:
            # Создаем цепочку для генерации ответа
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            # Генерируем ответ
            answer = chain.invoke({"context": context, "question": query})
            
            return {
                "answer": answer,
                "context": context_documents,
                "query": query,
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return {
                "answer": f"Произошла ошибка при генерации ответа: {str(e)}",
                "context": context_documents,
                "error": str(e)
            }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Форматирует контекстные документы для промпта
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc['content']
            metadata = doc['metadata']
            
            # Добавляем метаданные о документе
            paper_title = metadata.get('title', 'Неизвестная статья')
            section_id = metadata.get('section_id', 'N/A')
            content_type = metadata.get('content_type', 'text')
            
            context_part = f"[Документ {i}]\n"
            context_part += f"Статья: {paper_title}\n"
            context_part += f"Секция: {section_id}\n"
            context_part += f"Тип контента: {content_type}\n"
            context_part += f"Содержание:\n{content}\n"
            
            context_parts.append(context_part)
        
        return "\n" + "="*50 + "\n".join(context_parts)
    
    def ask_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Основной метод для задавания вопросов RAG-системе
        """
        logger.info(f"Обработка вопроса: '{question}'")
        
        # Генерируем ответ
        result = self.generate_answer(question, **kwargs)
        
        # Добавляем дополнительную информацию
        result["system_info"] = {
            "collection_name": self.config.COLLECTION_NAME,
            "embedding_model": self.config.EMBEDDING_MODEL,
            "llm_model": self.config.LLM_MODEL
        }
        
        return result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику системы
        """
        collection_info = self.vector_store.get_collection_info()
        
        return {
            "vector_store": collection_info,
            "config": {
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "top_k_results": self.config.TOP_K_RESULTS,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "llm_model": self.config.LLM_MODEL
            },
            "llm_configured": self.llm is not None
        }
    
    def reset_system(self) -> None:
        """
        Сбрасывает систему (удаляет векторное хранилище)
        """
        logger.info("Сброс RAG-системы...")
        self.vector_store.reset_collection()
        logger.info("RAG-система сброшена")
