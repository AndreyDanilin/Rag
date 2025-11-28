"""
Main file for running RAG system
"""
import os
import logging
from dotenv import load_dotenv

from rag_system import RAGSystem
from config import Config
from core.rag_agent import RAGAgent

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    agent = RAGAgent()
    print("RAG Agent CLI. Type your question.")
    print("Type 'rebuild' to rebuild index, 'exit' or 'quit' to quit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if question.lower() == "rebuild":
            print("Rebuilding index...")
            agent.rebuild_index()
            print("Index rebuilt!")
            continue
        if not question:
            continue
        print("\nAnswer:")
        print(agent.answer(question))

if __name__ == "__main__":
    main()
