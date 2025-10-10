"""
Основной файл для запуска RAG-системы
"""
import os
import logging
from dotenv import load_dotenv

from rag_system import RAGSystem
from config import Config

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Основная функция для демонстрации RAG-системы"""
    
    print("🚀 Запуск RAG-системы на Open RAG Benchmark")
    print("=" * 50)
    
    # Инициализация конфигурации
    config = Config()
    
    # Проверяем наличие API ключа
    if not config.OPENAI_API_KEY:
        print("⚠️  OpenAI API ключ не найден!")
        print("Создайте файл .env и добавьте: OPENAI_API_KEY=your_api_key")
        print("Или установите переменную окружения OPENAI_API_KEY")
        return
    
    # Инициализация RAG-системы
    print("📚 Инициализация RAG-системы...")
    rag_system = RAGSystem(config)
    
    # Инициализация с данными
    print("📥 Загрузка и обработка данных...")
    rag_system.initialize_system()
    
    # Выводим статистику
    stats = rag_system.get_system_stats()
    print(f"\n📊 Статистика системы:")
    print(f"- Документов в векторной БД: {stats['vector_store'].get('document_count', 0)}")
    print(f"- Модель эмбеддингов: {stats['config']['embedding_model']}")
    print(f"- LLM модель: {stats['config']['llm_model']}")
    
    # Интерактивный режим
    print("\n💬 Интерактивный режим (введите 'quit' для выхода)")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Ваш вопрос: ").strip()
            
            if question.lower() in ['quit', 'exit', 'выход']:
                print("👋 До свидания!")
                break
            
            if not question:
                continue
            
            print("🔍 Поиск релевантных документов...")
            
            # Получаем ответ
            result = rag_system.ask_question(question)
            
            if result.get('error'):
                print(f"❌ Ошибка: {result['error']}")
                continue
            
            # Выводим ответ
            print(f"\n💡 Ответ:")
            print(result['answer'])
            
            # Выводим источники
            if result.get('context'):
                print(f"\n📄 Использовано источников: {len(result['context'])}")
                for i, doc in enumerate(result['context'][:3], 1):  # Показываем первые 3
                    metadata = doc.get('metadata', {})
                    title = metadata.get('title', 'Неизвестная статья')
                    content_type = metadata.get('content_type', 'text')
                    print(f"  {i}. {title} ({content_type})")
        
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
