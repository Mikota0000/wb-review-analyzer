# Оптимизированный Dockerfile для больших ML моделей с uv
FROM python:3.11-slim as builder

# Метаданные образа
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="Wildberries Review Analyzer API with Large ML Model"

# Установка uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget --fix-missing \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов конфигурации
COPY pyproject.toml .python-version ./

# Создание виртуального окружения и установка зависимостей
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка зависимостей с оптимизацией для ML
RUN uv pip install --no-cache -e .[production]

# Production stage
FROM python:3.11-slim as production

# Установка системных зависимостей для runtime
RUN apt-get update && apt-get install -y \
    curl \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копирование виртуального окружения из builder stage
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Создание пользователя для безопасности
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Создание рабочей директории
WORKDIR /app

# Копирование кода приложения (сначала код, чтобы не пересобирать при изменении модели)
COPY app/ ./app/

# Создание необходимых директорий
RUN mkdir -p logs temp models

# Копирование большой модели в отдельном слое (для оптимизации Docker layer cache)
COPY models/ ./models/

# Оптимизация для больших файлов: проверяем размер модели
RUN echo "Размер модели:" && du -sh models/ || true

# Меняем владельца файлов
RUN chown -R appuser:appuser /app

# Переключаемся на непривилегированного пользователя
USER appuser

# Открываем порт
EXPOSE 8000

# Настройки Python для оптимизации памяти
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app

# Настройки для ONNX Runtime
ENV OMP_NUM_THREADS=2
ENV ONNXRUNTIME_LOG_SEVERITY_LEVEL=3

# Health check с увеличенным временем для ML модели
HEALTHCHECK --interval=60s --timeout=60s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Команда запуска с оптимизацией памяти
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]