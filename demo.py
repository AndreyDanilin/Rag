"""
Demonstration script for RAG system
"""
import os
import time
from dotenv import load_dotenv

from rag_system import RAGSystem
from config import Config

# Load environment variables
load_dotenv()

def print_separator(title=""):
    """Prints separator with title"""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)

def demo_rag_system():
    """Demonstration of RAG system capabilities"""
    
    print_separator("🚀 RAG SYSTEM DEMONSTRATION")
    print("Search and answer generation system based on Open RAG Benchmark")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print_separator("⚠️ WARNING")
        print("OpenAI API key not found!")
        print("For full demonstration, create .env file with your API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("\nContinuing demonstration without answer generation...")
    
    # System initialization
    print_separator("📚 SYSTEM INITIALIZATION")
    config = Config()
    rag_system = RAGSystem(config)
    
    print("Loading and processing data...")
    start_time = time.time()
    rag_system.initialize_system()
    init_time = time.time() - start_time
    
    print(f"✅ System initialized in {init_time:.2f} seconds")
    
    # Statistics
    stats = rag_system.get_system_stats()
    print(f"📊 Documents in database: {stats['vector_store'].get('document_count', 0)}")
    print(f"🤖 Embedding model: {stats['config']['embedding_model']}")
    print(f"🧠 LLM model: {stats['config']['llm_model']}")
    
    # Search demonstration
    print_separator("🔍 SEARCH DEMONSTRATION")
    
    demo_queries = [
        "machine learning",
        "neural networks",
        "deep learning",
        "image recognition"
    ]
    
    for query in demo_queries:
        print(f"\n🔎 Search: '{query}'")
        results = rag_system.search_documents(query, n_results=3)
        
        if results:
            print(f"   Found {len(results)} relevant documents:")
            for i, doc in enumerate(results, 1):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Unknown paper')
                content_type = metadata.get('content_type', 'text')
                distance = doc.get('distance', 0)
                similarity = 1 - distance if distance else 0
                
                print(f"   {i}. {title}")
                print(f"      Type: {content_type}, Similarity: {similarity:.3f}")
        else:
            print("   No documents found")
    
    # Answer generation demonstration
    if rag_system.llm:
        print_separator("💬 ANSWER GENERATION DEMONSTRATION")
        
        demo_questions = [
            "What is machine learning?",
            "What types of neural networks are used for working with images?"
        ]
        
        for question in demo_questions:
            print(f"\n❓ Question: {question}")
            print("🤔 Generating answer...")
            
            start_time = time.time()
            result = rag_system.generate_answer(question, n_results=3)
            gen_time = time.time() - start_time
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                print(f"💡 Answer ({gen_time:.2f}s):")
                print(f"   {result['answer']}")
                print(f"   📄 Sources used: {len(result.get('context', []))}")
    else:
        print_separator("⚠️ ANSWER GENERATION DISABLED")
        print("To demonstrate answer generation, configure OpenAI API key")
    
    # Filtering demonstration
    print_separator("🎯 FILTERING DEMONSTRATION")
    
    print("🔍 Search only in text documents:")
    text_results = rag_system.search_documents("machine learning", content_type="text")
    print(f"   Found text documents: {len(text_results)}")
    
    print("🔍 Search only in tables:")
    table_results = rag_system.search_documents("statistics", content_type="table")
    print(f"   Found tables: {len(table_results)}")
    
    print("🔍 Search only in images:")
    image_results = rag_system.search_documents("diagram", content_type="image")
    print(f"   Found images: {len(image_results)}")
    
    # Conclusion
    print_separator("🎉 DEMONSTRATION COMPLETED")
    print("RAG system is working successfully!")
    print("\nFor interactive use:")
    print("  • Console mode: python main.py")
    print("  • Web interface: streamlit run app.py")
    print("  • Testing: python test_system.py")

if __name__ == "__main__":
    demo_rag_system()
