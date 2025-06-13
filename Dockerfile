# Оптимизированный Dockerfile для больших ML моделей
FROM python:3.11-slim as builder

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget --fix-missing \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt
COPY requirements.txt .

# Создаем виртуальное окружение и устанавливаем зависимости
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка зависимостей с оптимизацией для ML
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Производственный образ
FROM python:3.11-slim as production

# Метаданные образа
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="Wildberries Review Analyzer API with Large ML Model"

# Установка системных зависимостей для runtime
RUN apt-get update && apt-get install -y \
    curl \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копируем виртуальное окружение из builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Создаем пользователя для безопасности
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Создаем рабочую директорию
WORKDIR /app

# Копируем код приложения (сначала код, чтобы не пересобирать при изменении модели)
COPY app/ ./app/

# Создаем необходимые директории
RUN mkdir -p logs temp models

# Копируем большую модель в отдельном слое (для оптимизации Docker layer cache)
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

# Настройки для ONNX Runtime
ENV OMP_NUM_THREADS=2
ENV ONNXRUNTIME_LOG_SEVERITY_LEVEL=3

HEALTHCHECK --interval=60s --timeout=60s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Команда запуска с оптимизацией памяти
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]