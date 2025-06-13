from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Конфигурация приложения"""
    
    # API настройки
    app_name: str = "Wildberries Review Analyzer"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Модель настройки
    model_path: Path = Path("models/onnx_model_s/model.onnx")
    max_text_length: int = 512
    batch_size: int = 32
    
    # Аспекты для анализа
    aspects: list[str] = ["delivery", "quality", "price", "service"]
    
    # Кэширование
    cache_predictions: bool = True
    cache_ttl: int = 3600  # 1 час
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()