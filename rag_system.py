"""
Main RAG system class
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
    """Main RAG system class"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize components
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
        
        # Initialize language model
        if self.config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=self.config.LLM_MODEL,
                api_key=self.config.OPENAI_API_KEY,
                temperature=0.1
            )
        else:
            logger.warning("OpenAI API key not found. Use local model.")
            self.llm = None
        
        # Create prompt template for answer generation
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Creates prompt template for answer generation"""
        template = """You are a scientific paper assistant. Use the provided context to answer the question.

Context:
{context}

Question: {question}

Instructions:
1. Answer in English
2. Use only information from the provided context
3. If there is no answer in the context, say so
4. Structure the answer clearly and understandably
5. Reference specific parts of the context when necessary

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def initialize_system(self, download_data: bool = False) -> None:
        """
        Initializes RAG system
        """
        logger.info("Initializing RAG system...")
        
        # Load data
        if download_data:
            self.data_loader.download_dataset()
        
        # Load document corpus
        papers = self.data_loader.load_corpus()
        
        if not papers:
            logger.warning("Document corpus is empty")
            return
        
        # Process documents
        documents = self.document_processor.process_corpus(papers)
        
        if not documents:
            logger.warning("Failed to create documents")
            return
        
        # Add documents to vector storage
        self.vector_store.add_documents(documents)
        
        # Output statistics
        stats = self.document_processor.get_document_stats(documents)
        collection_info = self.vector_store.get_collection_info()
        
        logger.info(f"RAG system initialized:")
        logger.info(f"- Processed papers: {stats['papers_count']}")
        logger.info(f"- Created chunks: {stats['total_chunks']}")
        logger.info(f"- Content types: {stats['content_types']}")
        logger.info(f"- Documents in vector DB: {collection_info.get('document_count', 0)}")
    
    def search_documents(self, 
                        query: str, 
                        n_results: int = None,
                        content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs search for relevant documents
        """
        n_results = n_results or self.config.TOP_K_RESULTS
        
        # Prepare filters
        where_filter = None
        if content_type:
            where_filter = {"content_type": content_type}
        
        # Perform search
        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            where=where_filter
        )
        
        logger.info(f"Found {len(results)} relevant documents for query: '{query}'")
        return results
    
    def generate_answer(self, 
                       query: str, 
                       context_documents: List[Dict[str, Any]] = None,
                       n_results: int = None) -> Dict[str, Any]:
        """
        Generates answer based on query and context
        """
        if not self.llm:
            return {
                "answer": "Language model not configured. Please set up OpenAI API key.",
                "context": [],
                "error": "LLM not configured"
            }
        
        # Get context documents
        if context_documents is None:
            context_documents = self.search_documents(query, n_results)
        
        if not context_documents:
            return {
                "answer": "No relevant documents found to answer your question.",
                "context": [],
                "error": "No relevant documents found"
            }
        
        # Format context
        context = self._format_context(context_documents)
        
        try:
            # Create chain for answer generation
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = chain.invoke({"context": context, "question": query})
            
            return {
                "answer": answer,
                "context": context_documents,
                "query": query,
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"An error occurred while generating answer: {str(e)}",
                "context": context_documents,
                "error": str(e)
            }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Formats context documents for prompt
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc['content']
            metadata = doc['metadata']
            
            # Add document metadata
            paper_title = metadata.get('title', 'Unknown paper')
            section_id = metadata.get('section_id', 'N/A')
            content_type = metadata.get('content_type', 'text')
            
            context_part = f"[Document {i}]\n"
            context_part += f"Paper: {paper_title}\n"
            context_part += f"Section: {section_id}\n"
            context_part += f"Content type: {content_type}\n"
            context_part += f"Content:\n{content}\n"
            
            context_parts.append(context_part)
        
        return "\n" + "="*50 + "\n".join(context_parts)
    
    def ask_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Main method for asking questions to RAG system
        """
        logger.info(f"Processing question: '{question}'")
        
        # Generate answer
        result = self.generate_answer(question, **kwargs)
        
        # Add additional information
        result["system_info"] = {
            "collection_name": self.config.COLLECTION_NAME,
            "embedding_model": self.config.EMBEDDING_MODEL,
            "llm_model": self.config.LLM_MODEL
        }
        
        return result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Returns system statistics
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
        Resets system (deletes vector storage)
        """
        logger.info("Resetting RAG system...")
        self.vector_store.reset_collection()
        logger.info("RAG system reset")
