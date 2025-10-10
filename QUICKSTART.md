# 🚀 RAG System Quick Start

## Installation and launch in 5 minutes

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key (optional)
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 3. Run demonstration
```bash
python demo.py
```

### 4. Interactive mode
```bash
# Console interface
python main.py

# Web interface
streamlit run app.py
```

## 🎯 What you'll see

### Demonstration (`python demo.py`)
- ✅ System initialization with sample data
- 🔍 Document search
- 💬 Answer generation (if API key is configured)
- 🎯 Content type filtering
- 📊 System statistics

### Console mode (`python main.py`)
- Interactive chat with system
- Relevant document search
- Answer generation for questions

### Web interface (`streamlit run app.py`)
- Convenient graphical interface
- Search settings
- Result analysis
- System statistics

## 📝 Example questions

- "What is machine learning?"
- "What types of neural networks are used for images?"
- "Explain the concept of deep learning"
- "How do convolutional neural networks work?"

## ⚙️ Configuration

Main parameters in `config.py`:
- `CHUNK_SIZE`: chunk size (default: 1000)
- `TOP_K_RESULTS`: number of search results (default: 5)
- `EMBEDDING_MODEL`: embedding model

## 🔧 Troubleshooting

### Error "OpenAI API key not found"
- Create `.env` file with your API key
- Or set environment variable: `export OPENAI_API_KEY=your_key`

### Slow initialization
- First run downloads embedding model (~100MB)
- Subsequent runs will be faster

### Insufficient memory
- Reduce `CHUNK_SIZE` in `config.py`
- Use smaller embedding model

## 📚 Additional

- [Full documentation](README.md)
- [System testing](test_system.py)
- [Production setup](README.md#configuration)
