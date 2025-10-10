"""
Module for loading and processing data from Open RAG Benchmark
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
    """Class for loading data from Open RAG Benchmark dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.corpus_dir = self.data_dir / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, huggingface_url: str = None):
        """
        Downloads dataset from Hugging Face or local source
        """
        if huggingface_url:
            logger.info(f"Downloading dataset from {huggingface_url}")
            # Code for downloading from Hugging Face can be added here
            pass
        else:
            logger.info("Using local data or creating examples")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Creates sample data for demonstration"""
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
        
        # Save sample documents
        for paper in sample_papers:
            paper_path = self.corpus_dir / f"{paper['id']}.json"
            with open(paper_path, 'w', encoding='utf-8') as f:
                json.dump(paper, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created {len(sample_papers)} sample documents")
    
    def load_corpus(self) -> List[Dict[str, Any]]:
        """
        Loads all documents from corpus
        """
        papers = []
        
        if not self.corpus_dir.exists():
            logger.warning("Corpus directory not found, creating sample data")
            self._create_sample_data()
        
        for paper_file in self.corpus_dir.glob("*.json"):
            try:
                with open(paper_file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    papers.append(paper)
            except Exception as e:
                logger.error(f"Error loading {paper_file}: {e}")
        
        logger.info(f"Loaded {len(papers)} documents")
        return papers
    
    def load_queries(self, queries_path: str = None) -> Dict[str, Any]:
        """
        Loads queries from queries.json file
        """
        if queries_path and os.path.exists(queries_path):
            with open(queries_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create sample queries
            return {
                "query_001": {
                    "query": "What is machine learning?",
                    "type": "abstractive",
                    "source": "text"
                },
                "query_002": {
                    "query": "What types of neural networks are used for image recognition?",
                    "type": "extractive", 
                    "source": "text"
                }
            }
    
    def load_qrels(self, qrels_path: str = None) -> Dict[str, Any]:
        """
        Loads query-document relevance
        """
        if qrels_path and os.path.exists(qrels_path):
            with open(qrels_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create sample relevance
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
