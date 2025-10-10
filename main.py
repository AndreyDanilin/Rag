"""
Main file for running RAG system
"""
import os
import logging
from dotenv import load_dotenv

from rag_system import RAGSystem
from config import Config

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for RAG system demonstration"""
    
    print("🚀 Starting RAG system on Open RAG Benchmark")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Check for API key
    if not config.OPENAI_API_KEY:
        print("⚠️  OpenAI API key not found!")
        print("Create .env file and add: OPENAI_API_KEY=your_api_key")
        print("Or set environment variable OPENAI_API_KEY")
        return
    
    # Initialize RAG system
    print("📚 Initializing RAG system...")
    rag_system = RAGSystem(config)
    
    # Initialize with data
    print("📥 Loading and processing data...")
    rag_system.initialize_system()
    
    # Output statistics
    stats = rag_system.get_system_stats()
    print(f"\n📊 System statistics:")
    print(f"- Documents in vector DB: {stats['vector_store'].get('document_count', 0)}")
    print(f"- Embedding model: {stats['config']['embedding_model']}")
    print(f"- LLM model: {stats['config']['llm_model']}")
    
    # Interactive mode
    print("\n💬 Interactive mode (enter 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'выход']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🔍 Searching for relevant documents...")
            
            # Get answer
            result = rag_system.ask_question(question)
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                continue
            
            # Output answer
            print(f"\n💡 Answer:")
            print(result['answer'])
            
            # Output sources
            if result.get('context'):
                print(f"\n📄 Sources used: {len(result['context'])}")
                for i, doc in enumerate(result['context'][:3], 1):  # Show first 3
                    metadata = doc.get('metadata', {})
                    title = metadata.get('title', 'Unknown paper')
                    content_type = metadata.get('content_type', 'text')
                    print(f"  {i}. {title} ({content_type})")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
