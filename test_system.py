"""
Скрипт для тестирования RAG-системы
"""
import os
import logging
from dotenv import load_dotenv

from rag_system import RAGSystem
from config import Config

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_system():
    """Тестирует основные функции RAG-системы"""
    
    print("🧪 Тестирование RAG-системы")
    print("=" * 40)
    
    # Инициализация
    config = Config()
    rag_system = RAGSystem(config)
    
    # Инициализация с данными
    print("📚 Инициализация системы...")
    rag_system.initialize_system()
    
    # Тестовые вопросы
    test_questions = [
        "Что такое машинное обучение?",
        "Какие типы нейронных сетей используются для изображений?",
        "Объясни концепцию глубокого обучения"
    ]
    
    print("\n🔍 Тестирование поиска документов...")
    for question in test_questions:
        print(f"\nВопрос: {question}")
        
        # Тест поиска
        search_results = rag_system.search_documents(question, n_results=3)
        print(f"Найдено документов: {len(search_results)}")
        
        if search_results:
            for i, doc in enumerate(search_results, 1):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Неизвестная статья')
                content_type = metadata.get('content_type', 'text')
                print(f"  {i}. {title} ({content_type})")
    
    # Тест генерации ответов (если настроен LLM)
    if rag_system.llm:
        print("\n💬 Тестирование генерации ответов...")
        for question in test_questions[:2]:  # Тестируем только первые 2 вопроса
            print(f"\nВопрос: {question}")
            result = rag_system.generate_answer(question, n_results=2)
            
            if result.get('error'):
                print(f"Ошибка: {result['error']}")
            else:
                print(f"Ответ: {result['answer'][:200]}...")
    else:
        print("\n⚠️ LLM не настроен, пропускаем тест генерации ответов")
    
    # Статистика
    print("\n📊 Статистика системы:")
    stats = rag_system.get_system_stats()
    print(f"- Документов в БД: {stats['vector_store'].get('document_count', 0)}")
    print(f"- Модель эмбеддингов: {stats['config']['embedding_model']}")
    print(f"- LLM настроен: {stats['llm_configured']}")
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    test_rag_system()
