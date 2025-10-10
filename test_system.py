"""
Script for testing RAG system
"""
import os
import logging
from dotenv import load_dotenv

from rag_system import RAGSystem
from config import Config

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_system():
    """Tests main RAG system functions"""
    
    print("🧪 Testing RAG system")
    print("=" * 40)
    
    # Initialization
    config = Config()
    rag_system = RAGSystem(config)
    
    # Initialize with data
    print("📚 Initializing system...")
    rag_system.initialize_system()
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "What types of neural networks are used for images?",
        "Explain the concept of deep learning"
    ]
    
    print("\n🔍 Testing document search...")
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        # Test search
        search_results = rag_system.search_documents(question, n_results=3)
        print(f"Found documents: {len(search_results)}")
        
        if search_results:
            for i, doc in enumerate(search_results, 1):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Unknown paper')
                content_type = metadata.get('content_type', 'text')
                print(f"  {i}. {title} ({content_type})")
    
    # Test answer generation (if LLM is configured)
    if rag_system.llm:
        print("\n💬 Testing answer generation...")
        for question in test_questions[:2]:  # Test only first 2 questions
            print(f"\nQuestion: {question}")
            result = rag_system.generate_answer(question, n_results=2)
            
            if result.get('error'):
                print(f"Error: {result['error']}")
            else:
                print(f"Answer: {result['answer'][:200]}...")
    else:
        print("\n⚠️ LLM not configured, skipping answer generation test")
    
    # Statistics
    print("\n📊 System statistics:")
    stats = rag_system.get_system_stats()
    print(f"- Documents in DB: {stats['vector_store'].get('document_count', 0)}")
    print(f"- Embedding model: {stats['config']['embedding_model']}")
    print(f"- LLM configured: {stats['llm_configured']}")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    test_rag_system()
