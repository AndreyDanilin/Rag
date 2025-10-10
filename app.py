"""
Streamlit интерфейс для RAG-системы
"""
import streamlit as st
import logging
from typing import Dict, Any
import json

from rag_system import RAGSystem
from config import Config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка страницы
st.set_page_config(
    page_title="RAG-система на Open RAG Benchmark",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def initialize_rag_system():
    """Инициализирует RAG-систему с кэшированием"""
    config = Config()
    rag_system = RAGSystem(config)
    
    # Проверяем, есть ли уже данные в векторном хранилище
    collection_info = rag_system.vector_store.get_collection_info()
    
    if collection_info.get('document_count', 0) == 0:
        # Инициализируем систему с данными
        with st.spinner("Инициализация RAG-системы..."):
            rag_system.initialize_system()
    
    return rag_system

def display_document_info(doc: Dict[str, Any]):
    """Отображает информацию о документе"""
    metadata = doc.get('metadata', {})
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Статья:** {metadata.get('title', 'Неизвестная')}")
        st.write(f"**Авторы:** {metadata.get('authors', 'Неизвестны')}")
        st.write(f"**Секция:** {metadata.get('section_id', 'N/A')}")
        st.write(f"**Тип контента:** {metadata.get('content_type', 'text')}")
    
    with col2:
        distance = doc.get('distance')
        if distance is not None:
            similarity = 1 - distance  # Преобразуем расстояние в схожесть
            st.metric("Схожесть", f"{similarity:.3f}")

def main():
    """Основная функция приложения"""
    
    # Заголовок
    st.title("📚 RAG-система на Open RAG Benchmark")
    st.markdown("Система поиска и генерации ответов на основе научных статей")
    
    # Инициализация системы
    try:
        rag_system = initialize_rag_system()
    except Exception as e:
        st.error(f"Ошибка инициализации системы: {e}")
        return
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Количество результатов
        n_results = st.slider(
            "Количество результатов поиска",
            min_value=1,
            max_value=10,
            value=5
        )
        
        # Фильтр по типу контента
        content_type = st.selectbox(
            "Тип контента",
            ["Все", "text", "table", "image"],
            index=0
        )
        
        if content_type == "Все":
            content_type = None
        
        # Статистика системы
        st.header("📊 Статистика")
        stats = rag_system.get_system_stats()
        
        st.metric("Документов в БД", stats['vector_store'].get('document_count', 0))
        st.metric("Модель эмбеддингов", stats['config']['embedding_model'].split('/')[-1])
        st.metric("LLM модель", stats['config']['llm_model'])
        
        # Кнопка сброса
        if st.button("🔄 Сбросить систему"):
            with st.spinner("Сброс системы..."):
                rag_system.reset_system()
                st.success("Система сброшена!")
                st.rerun()
    
    # Основная область
    tab1, tab2, tab3 = st.tabs(["💬 Вопросы", "🔍 Поиск", "📈 Анализ"])
    
    with tab1:
        st.header("Задайте вопрос системе")
        
        # Поле ввода вопроса
        question = st.text_area(
            "Ваш вопрос:",
            placeholder="Например: Что такое машинное обучение?",
            height=100
        )
        
        # Кнопка отправки
        if st.button("🚀 Получить ответ", type="primary"):
            if question.strip():
                with st.spinner("Обработка вопроса..."):
                    try:
                        # Получаем ответ
                        result = rag_system.ask_question(
                            question,
                            n_results=n_results,
                            content_type=content_type
                        )
                        
                        # Отображаем ответ
                        st.subheader("💡 Ответ:")
                        st.write(result['answer'])
                        
                        # Отображаем контекст
                        if result.get('context'):
                            st.subheader("📄 Использованные источники:")
                            
                            for i, doc in enumerate(result['context'], 1):
                                with st.expander(f"Источник {i}"):
                                    display_document_info(doc)
                                    st.write("**Содержание:**")
                                    st.write(doc['content'])
                        
                        # Дополнительная информация
                        with st.expander("ℹ️ Дополнительная информация"):
                            st.json({
                                "Длина контекста": result.get('context_length', 0),
                                "Количество источников": len(result.get('context', [])),
                                "Модель": result.get('system_info', {}).get('llm_model', 'N/A')
                            })
                    
                    except Exception as e:
                        st.error(f"Ошибка при обработке вопроса: {e}")
            else:
                st.warning("Пожалуйста, введите вопрос")
    
    with tab2:
        st.header("Поиск по документам")
        
        # Поисковый запрос
        search_query = st.text_input(
            "Поисковый запрос:",
            placeholder="Введите ключевые слова для поиска"
        )
        
        if st.button("🔍 Найти документы"):
            if search_query.strip():
                with st.spinner("Поиск документов..."):
                    try:
                        # Выполняем поиск
                        results = rag_system.search_documents(
                            search_query,
                            n_results=n_results,
                            content_type=content_type
                        )
                        
                        if results:
                            st.subheader(f"Найдено {len(results)} документов:")
                            
                            for i, doc in enumerate(results, 1):
                                with st.expander(f"Документ {i}"):
                                    display_document_info(doc)
                                    st.write("**Содержание:**")
                                    st.write(doc['content'])
                        else:
                            st.warning("Документы не найдены")
                    
                    except Exception as e:
                        st.error(f"Ошибка при поиске: {e}")
            else:
                st.warning("Пожалуйста, введите поисковый запрос")
    
    with tab3:
        st.header("Анализ системы")
        
        # Статистика по типам контента
        st.subheader("📊 Статистика по типам контента")
        
        try:
            text_docs = rag_system.vector_store.get_documents_by_content_type("text")
            table_docs = rag_system.vector_store.get_documents_by_content_type("table")
            image_docs = rag_system.vector_store.get_documents_by_content_type("image")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Текстовые чанки", len(text_docs))
            
            with col2:
                st.metric("Таблицы", len(table_docs))
            
            with col3:
                st.metric("Изображения", len(image_docs))
            
            # Детальная статистика
            st.subheader("🔍 Детальная информация")
            
            stats = rag_system.get_system_stats()
            st.json(stats)
            
        except Exception as e:
            st.error(f"Ошибка при получении статистики: {e}")

if __name__ == "__main__":
    main()
