# RAG Agent with LlamaIndex & LangChain

A modern Retrieval Augmented Generation (RAG) agent using LlamaIndex for vector storage and retrieval, and LangChain for agent orchestration and tool support.

## Features
- Fast retrieval with LlamaIndex and OpenAI embeddings
- Agent-style answers powered by LangChain with tool calling
- Pluggable LLMs (OpenAI or your own)
- Easy CLI and testable interface
- Extensible with additional tools (e.g. search_web)

## Usage

### 1. Install requirements (Python 3.9+)
```bash
pip install llama-index langchain-openai openai pytest
```

### 2. Prepare your document corpus
- Place your `.txt` or `.md` files in `./data/corpus/`

### 3. Configure your API keys and options
Edit (or copy) `env_example.txt` to `.env`, add your OpenAI API key.

### 4. Build or rebuild the index
The agent auto-builds on first query. To force rebuild, run in CLI:
```bash
python main.py
# then type: rebuild
```

### 5. Ask questions!
Run the CLI:
```bash
python main.py
```

### 6. Run tests
```bash
pytest
```

## Docker

You can run this RAG agent fully containerized:

```bash
docker build -t rag-agent .
docker run -it --rm \
  -e OPENAI_API_KEY=sk-... \
  -v $PWD/data/corpus:/app/data/corpus \
  rag-agent
```
- Place your documents in `./data/corpus` (will be mounted from host).
- To rebuild index or use commands, interact with the CLI in docker: `docker exec -it ... bash` or pass `CMD ["python", "main.py"]` as override.
- Run tests in container:
```bash
docker run --rm -e OPENAI_API_KEY=sk-... rag-agent pytest
```

## Agent Tools
- **rag_search**: Retrieves relevant context from your documents via LlamaIndex.
- **search_web**: (Stub, for demonstration) A placeholder for a tool to extend your agent (web calls, APIs, etc).

## Example Queries
- What is machine learning?
- Summarize the contents of [document_name].
- Which document explains neural networks?

## Extending
- Add new tools to the agent (see `rag_agent.py`, add more LangChain Tools)
- Swap LLM in config (OpenAI, or other if supported)
- Automate corpus ingestion (or integrate with pipelines)

---
MIT License. 2025
