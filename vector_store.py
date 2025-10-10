"""
Module for working with vector storage
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
    """Class for working with ChromaDB vector storage"""
    
    def __init__(self, 
                 collection_name: str = "arxiv_papers",
                 persist_directory: str = "data/vector_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Create storage directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Gets existing collection or creates new one"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Found existing collection: {self.collection_name}")
            return collection
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Open RAG Benchmark papers collection"}
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds documents to vector storage
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{len(ids)}")
            ids.append(chunk_id)
            texts.append(doc.page_content)
            
            # Clean metadata for ChromaDB (remove complex types)
            clean_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            
            metadatas.append(clean_metadata)
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to collection")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Performs search in vector storage
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Convert results to convenient format
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
            logger.error(f"Search error: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Returns collection information
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
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_collection(self) -> None:
        """
        Deletes collection
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def reset_collection(self) -> None:
        """
        Resets collection (deletes and creates new one)
        """
        self.delete_collection()
        self.collection = self._get_or_create_collection()
        logger.info(f"Collection {self.collection_name} reset")
    
    def get_documents_by_paper_id(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Gets all documents by paper ID
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
            logger.error(f"Error getting documents by paper_id: {e}")
            return []
    
    def get_documents_by_content_type(self, content_type: str) -> List[Dict[str, Any]]:
        """
        Gets documents by content type
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
            logger.error(f"Error getting documents by content_type: {e}")
            return []
