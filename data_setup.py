"""
Скрипт для настройки данных из Open RAG Benchmark
"""
import os
import json
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class OpenRAGDataSetup:
    """Класс для настройки данных из Open RAG Benchmark"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.corpus_dir = self.data_dir / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_huggingface(self, dataset_name: str = "vectara/open-rag-bench"):
        """
        Загружает датасет с Hugging Face
        Требует установки: pip install datasets
        """
        try:
            from datasets import load_dataset
            
            print(f"📥 Загрузка датасета {dataset_name} с Hugging Face...")
            
            # Загружаем датасет
            dataset = load_dataset(dataset_name)
            
            # Сохраняем корпус документов
            if 'corpus' in dataset:
                corpus_data = dataset['corpus']
                print(f"📚 Сохранение {len(corpus_data)} документов...")
                
                for i, doc in enumerate(corpus_data):
                    doc_path = self.corpus_dir / f"{doc['id']}.json"
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        json.dump(doc, f, ensure_ascii=False, indent=2)
                    
                    if (i + 1) % 100 == 0:
                        print(f"   Обработано {i + 1} документов...")
            
            # Сохраняем запросы
            if 'queries' in dataset:
                queries_data = dataset['queries']
                queries_path = self.data_dir / "queries.json"
                with open(queries_path, 'w', encoding='utf-8') as f:
                    json.dump(queries_data, f, ensure_ascii=False, indent=2)
                print(f"💬 Сохранено {len(queries_data)} запросов")
            
            # Сохраняем релевантность
            if 'qrels' in dataset:
                qrels_data = dataset['qrels']
                qrels_path = self.data_dir / "qrels.json"
                with open(qrels_path, 'w', encoding='utf-8') as f:
                    json.dump(qrels_data, f, ensure_ascii=False, indent=2)
                print(f"🔗 Сохранено {len(qrels_data)} связей запрос-документ")
            
            print("✅ Датасет успешно загружен!")
            
        except ImportError:
            print("❌ Для загрузки с Hugging Face установите: pip install datasets")
        except Exception as e:
            print(f"❌ Ошибка при загрузке датасета: {e}")
    
    def download_sample_papers(self, num_papers: int = 10):
        """
        Загружает примеры статей из arXiv для демонстрации
        """
        print(f"📥 Загрузка {num_papers} примеров статей...")
        
        # Примеры ID статей из разных категорий arXiv
        sample_paper_ids = [
            "2301.00001",  # cs.AI
            "2301.00002",  # cs.LG
            "2301.00003",  # cs.CV
            "2301.00004",  # cs.CL
            "2301.00005",  # cs.IR
            "2301.00006",  # cs.NE
            "2301.00007",  # cs.RO
            "2301.00008",  # cs.SE
            "2301.00009",  # cs.DC
            "2301.00010",  # cs.DB
        ]
        
        for i, paper_id in enumerate(sample_paper_ids[:num_papers]):
            try:
                # Создаем пример статьи
                sample_paper = self._create_sample_paper(paper_id, i)
                
                # Сохраняем
                paper_path = self.corpus_dir / f"{paper_id}.json"
                with open(paper_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_paper, f, ensure_ascii=False, indent=2)
                
                print(f"   ✅ Создана статья {paper_id}")
                
            except Exception as e:
                print(f"   ❌ Ошибка при создании статьи {paper_id}: {e}")
        
        print(f"✅ Создано {num_papers} примеров статей")
    
    def _create_sample_paper(self, paper_id: str, index: int) -> dict:
        """Создает пример статьи"""
        
        topics = [
            "машинное обучение", "нейронные сети", "глубокое обучение",
            "обработка естественного языка", "компьютерное зрение",
            "робототехника", "искусственный интеллект", "анализ данных"
        ]
        
        categories = [
            ["cs.AI", "cs.LG"], ["cs.LG", "cs.NE"], ["cs.CV", "cs.LG"],
            ["cs.CL", "cs.AI"], ["cs.IR", "cs.LG"], ["cs.RO", "cs.AI"]
        ]
        
        topic = topics[index % len(topics)]
        category = categories[index % len(categories)]
        
        return {
            "id": paper_id,
            "title": f"Исследование {topic}: современные подходы и методы",
            "authors": [f"Исследователь {i+1}" for i in range(2 + index % 3)],
            "categories": category,
            "abstract": f"В данной работе представлены современные подходы к решению задач в области {topic}. Рассматриваются различные методы и алгоритмы, их преимущества и недостатки. Проведен сравнительный анализ эффективности предложенных решений.",
            "published": "2024-01-01",
            "updated": "2024-01-01",
            "sections": [
                {
                    "text": f"Введение в область {topic}. Данная область исследований является одной из наиболее динамично развивающихся в современной информатике. Основные задачи включают в себя разработку эффективных алгоритмов и методов для решения сложных вычислительных проблем.",
                    "tables": {},
                    "images": {}
                },
                {
                    "text": f"Методология исследования. В рамках данного исследования были применены следующие подходы: статистический анализ, машинное обучение, глубокие нейронные сети. Каждый из методов имеет свои особенности и область применения.",
                    "tables": {
                        "table_1": "| Метод | Точность | Время обучения |\n|-------|----------|----------------|\n| SVM | 85% | 10 мин |\n| Random Forest | 88% | 5 мин |\n| Neural Network | 92% | 30 мин |"
                    },
                    "images": {}
                },
                {
                    "text": f"Результаты и обсуждение. Экспериментальные результаты показывают высокую эффективность предложенных методов в области {topic}. Достигнуто улучшение показателей точности на 15% по сравнению с существующими решениями.",
                    "tables": {},
                    "images": {
                        "figure_1": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }
    
    def setup_data(self, use_huggingface: bool = False, num_samples: int = 10):
        """
        Основной метод для настройки данных
        """
        print("🚀 Настройка данных для RAG-системы")
        print("=" * 50)
        
        if use_huggingface:
            self.download_from_huggingface()
        else:
            self.download_sample_papers(num_samples)
        
        print(f"\n📁 Данные сохранены в: {self.data_dir}")
        print("✅ Настройка данных завершена!")

def main():
    """Основная функция для настройки данных"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Настройка данных для RAG-системы")
    parser.add_argument("--huggingface", action="store_true", 
                       help="Загрузить данные с Hugging Face")
    parser.add_argument("--samples", type=int, default=10,
                       help="Количество примеров статей (по умолчанию: 10)")
    
    args = parser.parse_args()
    
    setup = OpenRAGDataSetup()
    setup.setup_data(
        use_huggingface=args.huggingface,
        num_samples=args.samples
    )

if __name__ == "__main__":
    main()
