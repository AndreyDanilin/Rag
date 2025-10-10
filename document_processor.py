"""
Модуль для обработки документов и создания чанков
"""
from typing import List, Dict, Any, Tuple
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Класс для обработки документов и создания чанков"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_paper(self, paper: Dict[str, Any]) -> List[Document]:
        """
        Обрабатывает одну статью и создает чанки
        """
        documents = []
        
        # Создаем базовую информацию о документе
        metadata = {
            "paper_id": paper["id"],
            "title": paper["title"],
            "authors": ", ".join(paper["authors"]),
            "categories": ", ".join(paper["categories"]),
            "abstract": paper["abstract"],
            "published": paper["published"],
            "updated": paper["updated"]
        }
        
        # Обрабатываем каждую секцию
        for section_idx, section in enumerate(paper["sections"]):
            section_metadata = metadata.copy()
            section_metadata["section_id"] = section_idx
            
            # Обрабатываем текст секции
            if section["text"]:
                text_chunks = self._process_text_section(
                    section["text"], 
                    section_metadata,
                    section_idx
                )
                documents.extend(text_chunks)
            
            # Обрабатываем таблицы
            if section["tables"]:
                table_chunks = self._process_tables(
                    section["tables"],
                    section_metadata,
                    section_idx
                )
                documents.extend(table_chunks)
            
            # Обрабатываем изображения
            if section["images"]:
                image_chunks = self._process_images(
                    section["images"],
                    section_metadata,
                    section_idx
                )
                documents.extend(image_chunks)
        
        return documents
    
    def _process_text_section(self, text: str, metadata: Dict, section_idx: int) -> List[Document]:
        """Обрабатывает текстовую секцию"""
        if not text.strip():
            return []
        
        # Разбиваем текст на чанки
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "content_type": "text",
                "chunk_id": f"{metadata['paper_id']}_section_{section_idx}_chunk_{chunk_idx}",
                "chunk_index": chunk_idx
            })
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def _process_tables(self, tables: Dict[str, str], metadata: Dict, section_idx: int) -> List[Document]:
        """Обрабатывает таблицы"""
        documents = []
        
        for table_id, table_content in tables.items():
            if not table_content.strip():
                continue
            
            # Создаем описание таблицы
            table_text = f"Таблица {table_id}:\n{table_content}"
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "content_type": "table",
                "table_id": table_id,
                "chunk_id": f"{metadata['paper_id']}_section_{section_idx}_table_{table_id}"
            })
            
            documents.append(Document(
                page_content=table_text,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def _process_images(self, images: Dict[str, str], metadata: Dict, section_idx: int) -> List[Document]:
        """Обрабатывает изображения"""
        documents = []
        
        for image_id, image_content in images.items():
            if not image_content.strip():
                continue
            
            # Создаем описание изображения
            image_text = f"Изображение {image_id}: [Изображение в формате base64]"
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "content_type": "image",
                "image_id": image_id,
                "chunk_id": f"{metadata['paper_id']}_section_{section_idx}_image_{image_id}",
                "image_base64": image_content  # Сохраняем base64 для дальнейшего использования
            })
            
            documents.append(Document(
                page_content=image_text,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def process_corpus(self, papers: List[Dict[str, Any]]) -> List[Document]:
        """
        Обрабатывает весь корпус документов
        """
        all_documents = []
        
        for paper in papers:
            try:
                paper_documents = self.process_paper(paper)
                all_documents.extend(paper_documents)
                logger.info(f"Обработана статья {paper['id']}: {len(paper_documents)} чанков")
            except Exception as e:
                logger.error(f"Ошибка при обработке статьи {paper.get('id', 'unknown')}: {e}")
        
        logger.info(f"Всего создано {len(all_documents)} чанков из {len(papers)} статей")
        return all_documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Возвращает статистику по документам
        """
        stats = {
            "total_chunks": len(documents),
            "content_types": {},
            "papers_count": len(set(doc.metadata.get("paper_id", "") for doc in documents)),
            "avg_chunk_length": 0
        }
        
        total_length = 0
        for doc in documents:
            content_type = doc.metadata.get("content_type", "unknown")
            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
            total_length += len(doc.page_content)
        
        if documents:
            stats["avg_chunk_length"] = total_length / len(documents)
        
        return stats
