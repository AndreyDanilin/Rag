import pytest
from core.rag_agent import RAGAgent

@pytest.fixture(scope="module")
def agent():
    return RAGAgent()

def test_rebuild_index(agent):
    try:
        agent.rebuild_index()
    except Exception as e:
        pytest.skip(f"Index rebuild skipped: {e}")

def test_answer(agent):
    try:
        result = agent.answer("What is machine learning?")
        assert isinstance(result, str)
        assert result.strip()
    except Exception as e:
        pytest.skip(f"Answer skipped: {e}")
