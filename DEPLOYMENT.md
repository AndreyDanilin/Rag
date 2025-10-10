# 🚀 Развертывание RAG-системы

## Локальное развертывание

### 1. Установка зависимостей
```bash
# Создание виртуального окружения
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# или
rag_env\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Настройка конфигурации
```bash
# Создание файла конфигурации
cp env_example.txt .env

# Редактирование .env файла
nano .env  # или любой другой редактор
```

### 3. Настройка данных
```bash
# Создание примеров данных (быстро)
python data_setup.py --samples 20

# Или загрузка полного датасета с Hugging Face
python data_setup.py --huggingface
```

### 4. Запуск системы
```bash
# Демонстрация
python demo.py

# Консольный режим
python main.py

# Веб-интерфейс
streamlit run app.py
```

## Docker развертывание

### 1. Создание Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создание директории для данных
RUN mkdir -p data/vector_db

# Настройка портов
EXPOSE 8501

# Запуск Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Создание docker-compose.yml
```yaml
version: '3.8'

services:
  rag-system:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
```

### 3. Запуск с Docker
```bash
# Сборка и запуск
docker-compose up --build

# Или только сборка
docker build -t rag-system .

# Запуск контейнера
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key rag-system
```

## Облачное развертывание

### Streamlit Cloud
1. Загрузите код в GitHub репозиторий
2. Подключите репозиторий к [Streamlit Cloud](https://streamlit.io/cloud)
3. Настройте переменные окружения в панели управления
4. Запустите приложение

### Heroku
```bash
# Установка Heroku CLI
# Создание Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Создание runtime.txt
echo "python-3.9.18" > runtime.txt

# Развертывание
heroku create your-rag-app
heroku config:set OPENAI_API_KEY=your_key
git push heroku main
```

### AWS EC2
```bash
# Подключение к EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Установка зависимостей
sudo apt update
sudo apt install python3-pip python3-venv

# Клонирование репозитория
git clone your-repo-url
cd your-repo

# Настройка и запуск
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Настройка systemd сервиса
sudo nano /etc/systemd/system/rag-system.service
```

Пример systemd сервиса:
```ini
[Unit]
Description=RAG System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/your-repo
Environment=PATH=/home/ubuntu/your-repo/venv/bin
ExecStart=/home/ubuntu/your-repo/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

## Масштабирование

### Горизонтальное масштабирование
- Используйте Redis для кэширования эмбеддингов
- Разделите векторную базу данных на несколько инстансов
- Используйте load balancer для распределения нагрузки

### Вертикальное масштабирование
- Увеличьте RAM для работы с большими моделями
- Используйте GPU для ускорения эмбеддингов
- Оптимизируйте размер чанков

## Мониторинг

### Логирование
```python
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/rag_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
```

### Метрики
- Время ответа системы
- Количество обработанных запросов
- Точность поиска
- Использование ресурсов

## Безопасность

### API ключи
- Никогда не коммитьте API ключи в репозиторий
- Используйте переменные окружения
- Регулярно ротируйте ключи

### Доступ к данным
- Ограничьте доступ к векторной базе данных
- Используйте HTTPS для веб-интерфейса
- Настройте аутентификацию при необходимости

## Резервное копирование

### Векторная база данных
```bash
# Создание резервной копии ChromaDB
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz data/vector_db/

# Восстановление
tar -xzf chroma_backup_20240101.tar.gz
```

### Конфигурация
```bash
# Резервное копирование конфигурации
cp -r data/ config_backup/
```

## Устранение неполадок

### Частые проблемы
1. **Ошибка памяти**: Уменьшите размер чанков или используйте меньшую модель
2. **Медленный поиск**: Оптимизируйте индексы или используйте более мощное железо
3. **Неточные ответы**: Настройте промпты или увеличьте количество контекстных документов

### Логи для диагностики
```bash
# Просмотр логов
tail -f logs/rag_system_*.log

# Проверка статуса сервиса
systemctl status rag-system
```
