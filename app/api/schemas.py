from pydantic import BaseModel
from pydantic import Field
from typing import Dict
from enum import Enum


class SentimentType(str, Enum):
    """Типы тональности"""
    positive = "Позитивный"
    negative = "Негативный" 
    neutral = "Нейтральный"


class AspectSentiment(str, Enum):
    """Тональность по аспектам"""
    positive = "Позитивный"
    negative = "Негативный"
    neutral = "Нейтральный"


class ReviewRequest(BaseModel):
    """Запрос на анализ отзыва"""
    text: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Текст отзыва для анализа",
        example="Товар пришел быстро, но качество не очень. Упаковка помята."
    )


class AspectAnalysis(BaseModel):
    """Анализ по аспектам"""
    delivery: AspectSentiment = Field(description="Тональность по доставке")
    quality: AspectSentiment = Field(description="Тональность по качеству")
    price: AspectSentiment = Field(description="Тональность по цене")
    service: AspectSentiment = Field(description="Тональность по сервису")


class ReviewResponse(BaseModel):
    """Ответ анализа отзыва"""
    sentiment: SentimentType = Field(description="Общая тональность отзыва")
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Уверенность модели в предсказании"
    )
    key_aspects: AspectAnalysis = Field(description="Анализ по ключевым аспектам")
    summary: str = Field(description="Краткое резюме отзыва")


class HealthResponse(BaseModel):
    """Ответ проверки здоровья сервиса"""
    status: str = "ok"
    version: str
    model_loaded: bool