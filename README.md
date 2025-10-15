# RAG System on Open RAG Benchmark

This project implements a Retrieval-Augmented Generation (RAG) system based on the [Open RAG Benchmark](https://github.com/vectara/open-rag-bench) dataset from Vectara.

## 🚀 Features

- **Multimodal Processing**: Working with text, tables, and images from scientific papers
- **Vector Search**: Using ChromaDB for efficient retrieval of relevant documents
- **Answer Generation**: Integration with OpenAI GPT for creating high-quality answers
- **Web Interface**: Convenient Streamlit interface for system interaction
- **Flexible Configuration**: Configurable chunking, search, and model parameters

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- 4GB+ RAM (for working with embeddings)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Rag
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
```bash
# Copy the configuration example
cp env_example.txt .env

# Edit the .env file and add your OpenAI API key
OPENAI_API_KEY=your_api_key_here
```

## 🎯 Usage

### Command Line Interface

```bash
python main.py
```

### Web Interface

```bash
streamlit run app.py
```

Then open your browser at: http://localhost:8501

## 📁 Project Structure

```
├── config.py              # System configuration
├── data_loader.py         # Data loading from Open RAG Benchmark
├── document_processor.py  # Document processing and chunking
├── vector_store.py        # ChromaDB vector store operations
├── rag_system.py          # Main RAG system class
├── main.py               # Console interface
├── app.py                # Streamlit web interface
├── requirements.txt      # Python dependencies
└── README.md            # Documentation
```

## ⚙️ Configuration

Main parameters can be configured in `config.py`:

- `CHUNK_SIZE`: Chunk size for document splitting (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RESULTS`: Number of search results (default: 5)
- `EMBEDDING_MODEL`: Model for creating embeddings
- `LLM_MODEL`: Model for answer generation

## 🔧 System Components

### 1. Data Loading (`data_loader.py`)
- Loads documents from Open RAG Benchmark dataset
- Supports both real data and examples for demonstration
- Processes JSON format with multimodal content

### 2. Document Processing (`document_processor.py`)
- Splits documents into chunks with configurable parameters
- Processes text, tables, and images separately
- Preserves metadata for each chunk

### 3. Vector Store (`vector_store.py`)
- Uses ChromaDB for vector storage and search
- Supports filtering by content type
- Automatically creates embeddings using SentenceTransformers

### 4. RAG System (`rag_system.py`)
- Combines all components into a unified system
- Implements relevant document search
- Generates answers based on found context

## 📊 Usage Examples

### Searching for machine learning information
```python
from rag_system import RAGSystem
from config import Config

# Initialization
config = Config()
rag = RAGSystem(config)
rag.initialize_system()

# Ask a question
result = rag.ask_question("What is machine learning?")
print(result['answer'])
```

### Searching only in tables
```python
# Search with content type filter
results = rag.search_documents(
    "statistical data", 
    content_type="table"
)
```

## 🎨 Web Interface

The Streamlit interface provides:

- **"Questions" Tab**: Interactive chat with the system
- **"Search" Tab**: Document search without answer generation
- **"Analysis" Tab**: System statistics and analysis
- **Sidebar**: Settings and statistics

## 🔍 Dataset Features

Open RAG Benchmark includes:
- **1000 PDF papers** from arXiv
- **3045 question-answer pairs**
- **Multimodal content**: text, tables, images
- **Various query types**: abstractive and extractive

## 🚧 Limitations

- Requires OpenAI API key for answer generation
- Initial setup may take time (loading embedding model)
- Vector database size grows with document count


## 📄 License

This project uses MIT license.

## 🙏 Acknowledgments

- [Vectara](https://github.com/vectara/open-rag-bench) for creating Open RAG Benchmark dataset
- [LangChain](https://github.com/langchain-ai/langchain) for RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Streamlit](https://github.com/streamlit/streamlit) for web interface
