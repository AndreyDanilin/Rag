"""
RAG agent with LlamaIndex storage and LangChain agent logic
"""
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from core.config import Config
import os

class RAGAgent:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL,
            api_key=self.config.OPENAI_API_KEY,
            temperature=0.1
        )
        self.embedding = OpenAIEmbedding(model=self.config.EMBEDDING_MODEL)
        self.index = self._load_or_build_index()

        self.rag_tool = Tool(
            name="rag_search",
            func=self._context_search,
            description="Useful for answering questions about documents. Input should be a user question."
        )
        self.search_web_tool = Tool(
            name="search_web",
            func=lambda q: "(This would query the web; not implemented)",
            description="Useful for open web lookups. Input should be a user question."
        )
        self.agent = AgentExecutor.from_agent_and_tools(
            agent=create_openai_functions_agent(self.llm, [self.rag_tool, self.search_web_tool]),
            tools=[self.rag_tool, self.search_web_tool],
            verbose=True
        )

    def _load_or_build_index(self):
        index_path = self.config.INDEX_PATH
        # Try load
        if os.path.exists(index_path):
            return load_index_from_storage(StorageContext.from_defaults(persist_dir=index_path))
        # Build new
        docs = SimpleDirectoryReader(self.config.CORPUS_PATH).load_data()
        index = VectorStoreIndex.from_documents(docs, embed_model=self.embedding)
        index.storage_context.persist(index_path)
        return index

    def rebuild_index(self):
        """Force rebuild index from current corpus"""
        docs = SimpleDirectoryReader(self.config.CORPUS_PATH).load_data()
        index = VectorStoreIndex.from_documents(docs, embed_model=self.embedding)
        index.storage_context.persist(self.config.INDEX_PATH)
        self.index = index

    def _context_search(self, query: str, top_k: int=None) -> str:
        retriever = self.index.as_retriever(similarity_top_k=top_k or self.config.TOP_K)
        nodes = retriever.retrieve(query)
        if not nodes:
            return "No documents found."
        return "\n".join(n.get_content() for n in nodes)

    def answer(self, question: str) -> str:
        # LangChain agent call with given tools (RAG + web-search)
        resp = self.agent.invoke({"input": question})
        return resp["output"]  # final answer
