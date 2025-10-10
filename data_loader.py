"""
Модуль для загрузки и обработки данных из Open RAG Benchmark
"""
import json
import os
import requests
from typing import List, Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRAGDataLoader:
    """Класс для загрузки данных из Open RAG Benchmark датасета"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.corpus_dir = self.data_dir / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, huggingface_url: str = None):
        """
        Загружает датасет с Hugging Face или из локального источника
        """
        if huggingface_url:
            logger.info(f"Загрузка датасета с {huggingface_url}")
            # Здесь можно добавить код для загрузки с Hugging Face
            pass
        else:
            logger.info("Используем локальные данные или создаем примеры")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Создает примеры данных для демонстрации"""
        sample_papers = [
            {
                "id": "sample_001",
                "title": "Introduction to Machine Learning",
                "authors": ["John Doe", "Jane Smith"],
                "categories": ["cs.LG", "cs.AI"],
                "abstract": "This paper provides an introduction to machine learning concepts and applications.",
                "published": "2024-01-01",
                "updated": "2024-01-01",
                "sections": [
                    {
                        "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. The main types of machine learning include supervised learning, unsupervised learning, and reinforcement learning.",
                        "tables": {},
                        "images": {}
                    },
                    {
                        "text": "Supervised learning involves training a model on labeled data. Common algorithms include linear regression, decision trees, and neural networks. The goal is to learn a mapping from inputs to outputs.",
                        "tables": {},
                        "images": {}
                    }
                ]
            },
            {
                "id": "sample_002", 
                "title": "Deep Learning Fundamentals",
                "authors": ["Alice Johnson", "Bob Wilson"],
                "categories": ["cs.LG", "cs.NE"],
                "abstract": "This paper covers the fundamentals of deep learning and neural networks.",
                "published": "2024-01-15",
                "updated": "2024-01-15",
                "sections": [
                    {
                        "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks can learn complex patterns in data through backpropagation.",
                        "tables": {},
                        "images": {}
                    },
                    {
                        "text": "Convolutional Neural Networks (CNNs) are particularly effective for image recognition tasks. They use convolutional layers to detect features in images.",
                        "tables": {},
                        "images": {}
                    }
                ]
            }
        ]
        
        # Сохраняем примеры документов
        for paper in sample_papers:
            paper_path = self.corpus_dir / f"{paper['id']}.json"
            with open(paper_path, 'w', encoding='utf-8') as f:
                json.dump(paper, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Создано {len(sample_papers)} примеров документов")
    
    def load_corpus(self) -> List[Dict[str, Any]]:
        """
        Загружает все документы из корпуса
        """
        papers = []
        
        if not self.corpus_dir.exists():
            logger.warning("Папка корпуса не найдена, создаем примеры данных")
            self._create_sample_data()
        
        for paper_file in self.corpus_dir.glob("*.json"):
            try:
                with open(paper_file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    papers.append(paper)
            except Exception as e:
                logger.error(f"Ошибка при загрузке {paper_file}: {e}")
        
        logger.info(f"Загружено {len(papers)} документов")
        return papers
    
    def load_queries(self, queries_path: str = None) -> Dict[str, Any]:
        """
        Загружает запросы из файла queries.json
        """
        if queries_path and os.path.exists(queries_path):
            with open(queries_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Создаем примеры запросов
            return {
                "query_001": {
                    "query": "Что такое машинное обучение?",
                    "type": "abstractive",
                    "source": "text"
                },
                "query_002": {
                    "query": "Какие типы нейронных сетей используются для распознавания изображений?",
                    "type": "extractive", 
                    "source": "text"
                }
            }
    
    def load_qrels(self, qrels_path: str = None) -> Dict[str, Any]:
        """
        Загружает релевантность запросов к документам
        """
        if qrels_path and os.path.exists(qrels_path):
            with open(qrels_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Создаем примеры релевантности
            return {
                "query_001": {
                    "doc_id": "sample_001",
                    "section_id": 0
                },
                "query_002": {
                    "doc_id": "sample_002", 
                    "section_id": 1
                }
            }
