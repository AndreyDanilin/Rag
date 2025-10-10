"""
Streamlit interface for RAG system
"""
import streamlit as st
import logging
from typing import Dict, Any
import json

from rag_system import RAGSystem
from config import Config

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page setup
st.set_page_config(
    page_title="RAG System on Open RAG Benchmark",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def initialize_rag_system():
    """Initializes RAG system with caching"""
    config = Config()
    rag_system = RAGSystem(config)
    
    # Check if there's already data in vector storage
    collection_info = rag_system.vector_store.get_collection_info()
    
    if collection_info.get('document_count', 0) == 0:
        # Initialize system with data
        with st.spinner("Initializing RAG system..."):
            rag_system.initialize_system()
    
    return rag_system

def display_document_info(doc: Dict[str, Any]):
    """Displays document information"""
    metadata = doc.get('metadata', {})
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Paper:** {metadata.get('title', 'Unknown')}")
        st.write(f"**Authors:** {metadata.get('authors', 'Unknown')}")
        st.write(f"**Section:** {metadata.get('section_id', 'N/A')}")
        st.write(f"**Content type:** {metadata.get('content_type', 'text')}")
    
    with col2:
        distance = doc.get('distance')
        if distance is not None:
            similarity = 1 - distance  # Convert distance to similarity
            st.metric("Similarity", f"{similarity:.3f}")

def main():
    """Main application function"""
    
    # Header
    st.title("📚 RAG System on Open RAG Benchmark")
    st.markdown("Search and answer generation system based on scientific papers")
    
    # System initialization
    try:
        rag_system = initialize_rag_system()
    except Exception as e:
        st.error(f"System initialization error: {e}")
        return
    
    # Sidebar with settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Number of results
        n_results = st.slider(
            "Number of search results",
            min_value=1,
            max_value=10,
            value=5
        )
        
        # Content type filter
        content_type = st.selectbox(
            "Content type",
            ["All", "text", "table", "image"],
            index=0
        )
        
        if content_type == "All":
            content_type = None
        
        # System statistics
        st.header("📊 Statistics")
        stats = rag_system.get_system_stats()
        
        st.metric("Documents in DB", stats['vector_store'].get('document_count', 0))
        st.metric("Embedding model", stats['config']['embedding_model'].split('/')[-1])
        st.metric("LLM model", stats['config']['llm_model'])
        
        # Reset button
        if st.button("🔄 Reset system"):
            with st.spinner("Resetting system..."):
                rag_system.reset_system()
                st.success("System reset!")
                st.rerun()
    
    # Main area
    tab1, tab2, tab3 = st.tabs(["💬 Questions", "🔍 Search", "📈 Analysis"])
    
    with tab1:
        st.header("Ask a question to the system")
        
        # Question input field
        question = st.text_area(
            "Your question:",
            placeholder="For example: What is machine learning?",
            height=100
        )
        
        # Submit button
        if st.button("🚀 Get answer", type="primary"):
            if question.strip():
                with st.spinner("Processing question..."):
                    try:
                        # Get answer
                        result = rag_system.ask_question(
                            question,
                            n_results=n_results,
                            content_type=content_type
                        )
                        
                        # Display answer
                        st.subheader("💡 Answer:")
                        st.write(result['answer'])
                        
                        # Display context
                        if result.get('context'):
                            st.subheader("📄 Sources used:")
                            
                            for i, doc in enumerate(result['context'], 1):
                                with st.expander(f"Source {i}"):
                                    display_document_info(doc)
                                    st.write("**Content:**")
                                    st.write(doc['content'])
                        
                        # Additional information
                        with st.expander("ℹ️ Additional information"):
                            st.json({
                                "Context length": result.get('context_length', 0),
                                "Number of sources": len(result.get('context', [])),
                                "Model": result.get('system_info', {}).get('llm_model', 'N/A')
                            })
                    
                    except Exception as e:
                        st.error(f"Error processing question: {e}")
            else:
                st.warning("Please enter a question")
    
    with tab2:
        st.header("Document search")
        
        # Search query
        search_query = st.text_input(
            "Search query:",
            placeholder="Enter keywords to search"
        )
        
        if st.button("🔍 Find documents"):
            if search_query.strip():
                with st.spinner("Searching documents..."):
                    try:
                        # Perform search
                        results = rag_system.search_documents(
                            search_query,
                            n_results=n_results,
                            content_type=content_type
                        )
                        
                        if results:
                            st.subheader(f"Found {len(results)} documents:")
                            
                            for i, doc in enumerate(results, 1):
                                with st.expander(f"Document {i}"):
                                    display_document_info(doc)
                                    st.write("**Content:**")
                                    st.write(doc['content'])
                        else:
                            st.warning("No documents found")
                    
                    except Exception as e:
                        st.error(f"Search error: {e}")
            else:
                st.warning("Please enter a search query")
    
    with tab3:
        st.header("System analysis")
        
        # Statistics by content type
        st.subheader("📊 Statistics by content type")
        
        try:
            text_docs = rag_system.vector_store.get_documents_by_content_type("text")
            table_docs = rag_system.vector_store.get_documents_by_content_type("table")
            image_docs = rag_system.vector_store.get_documents_by_content_type("image")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Text chunks", len(text_docs))
            
            with col2:
                st.metric("Tables", len(table_docs))
            
            with col3:
                st.metric("Images", len(image_docs))
            
            # Detailed statistics
            st.subheader("🔍 Detailed information")
            
            stats = rag_system.get_system_stats()
            st.json(stats)
            
        except Exception as e:
            st.error(f"Error getting statistics: {e}")

if __name__ == "__main__":
    main()
