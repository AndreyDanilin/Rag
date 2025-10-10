"""
Демонстрационный скрипт для RAG-системы
"""
import os
import time
from dotenv import load_dotenv

from rag_system import RAGSystem
from config import Config

# Загружаем переменные окружения
load_dotenv()

def print_separator(title=""):
    """Печатает разделитель с заголовком"""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)

def demo_rag_system():
    """Демонстрация возможностей RAG-системы"""
    
    print_separator("🚀 ДЕМОНСТРАЦИЯ RAG-СИСТЕМЫ")
    print("Система поиска и генерации ответов на основе Open RAG Benchmark")
    
    # Проверка API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print_separator("⚠️ ВНИМАНИЕ")
        print("OpenAI API ключ не найден!")
        print("Для полной демонстрации создайте файл .env с вашим API ключом:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("\nПродолжаем демонстрацию без генерации ответов...")
    
    # Инициализация системы
    print_separator("📚 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ")
    config = Config()
    rag_system = RAGSystem(config)
    
    print("Загрузка и обработка данных...")
    start_time = time.time()
    rag_system.initialize_system()
    init_time = time.time() - start_time
    
    print(f"✅ Система инициализирована за {init_time:.2f} секунд")
    
    # Статистика
    stats = rag_system.get_system_stats()
    print(f"📊 Документов в базе: {stats['vector_store'].get('document_count', 0)}")
    print(f"🤖 Модель эмбеддингов: {stats['config']['embedding_model']}")
    print(f"🧠 LLM модель: {stats['config']['llm_model']}")
    
    # Демонстрация поиска
    print_separator("🔍 ДЕМОНСТРАЦИЯ ПОИСКА")
    
    demo_queries = [
        "машинное обучение",
        "нейронные сети",
        "глубокое обучение",
        "распознавание изображений"
    ]
    
    for query in demo_queries:
        print(f"\n🔎 Поиск: '{query}'")
        results = rag_system.search_documents(query, n_results=3)
        
        if results:
            print(f"   Найдено {len(results)} релевантных документов:")
            for i, doc in enumerate(results, 1):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Неизвестная статья')
                content_type = metadata.get('content_type', 'text')
                distance = doc.get('distance', 0)
                similarity = 1 - distance if distance else 0
                
                print(f"   {i}. {title}")
                print(f"      Тип: {content_type}, Схожесть: {similarity:.3f}")
        else:
            print("   Документы не найдены")
    
    # Демонстрация генерации ответов
    if rag_system.llm:
        print_separator("💬 ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ОТВЕТОВ")
        
        demo_questions = [
            "Что такое машинное обучение?",
            "Какие типы нейронных сетей используются для работы с изображениями?"
        ]
        
        for question in demo_questions:
            print(f"\n❓ Вопрос: {question}")
            print("🤔 Генерация ответа...")
            
            start_time = time.time()
            result = rag_system.generate_answer(question, n_results=3)
            gen_time = time.time() - start_time
            
            if result.get('error'):
                print(f"❌ Ошибка: {result['error']}")
            else:
                print(f"💡 Ответ ({gen_time:.2f}с):")
                print(f"   {result['answer']}")
                print(f"   📄 Использовано источников: {len(result.get('context', []))}")
    else:
        print_separator("⚠️ ГЕНЕРАЦИЯ ОТВЕТОВ ОТКЛЮЧЕНА")
        print("Для демонстрации генерации ответов настройте OpenAI API ключ")
    
    # Демонстрация фильтрации
    print_separator("🎯 ДЕМОНСТРАЦИЯ ФИЛЬТРАЦИИ")
    
    print("🔍 Поиск только в текстовых документах:")
    text_results = rag_system.search_documents("машинное обучение", content_type="text")
    print(f"   Найдено текстовых документов: {len(text_results)}")
    
    print("🔍 Поиск только в таблицах:")
    table_results = rag_system.search_documents("статистика", content_type="table")
    print(f"   Найдено таблиц: {len(table_results)}")
    
    print("🔍 Поиск только в изображениях:")
    image_results = rag_system.search_documents("диаграмма", content_type="image")
    print(f"   Найдено изображений: {len(image_results)}")
    
    # Заключение
    print_separator("🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("RAG-система успешно работает!")
    print("\nДля интерактивного использования:")
    print("  • Консольный режим: python main.py")
    print("  • Веб-интерфейс: streamlit run app.py")
    print("  • Тестирование: python test_system.py")

if __name__ == "__main__":
    demo_rag_system()
