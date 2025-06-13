"""
API endpoints для анализа тональности отзывов
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from app.api.schemas import ReviewRequest, ReviewResponse, HealthResponse
from app.models.sentiment_model import sentiment_model
from app.models.aspect_analyzer import aspect_analyzer
from app.core.config import settings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем роутер
router = APIRouter()

@router.on_event("startup")
async def startup_event():
    """Инициализация при запуске API"""
    try:
        logger.info("Загрузка модели анализа тональности...")
        sentiment_model.load_model()
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        # Продолжаем работу, но отмечаем что модель не загружена

@router.post("/analyze", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest) -> ReviewResponse:
    """
    Анализ тональности отзыва
    
    Args:
        request: Запрос с текстом отзыва
        
    Returns:
        ReviewResponse: Результат анализа
    """
    try:
        # Проверяем, загружена ли модель
        if not sentiment_model.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Модель не загружена. Попробуйте позже."
            )
        
        # Анализ общей тональности
        logger.info(f"Анализ отзыва: {request.text[:50]}...")
        sentiment, confidence = sentiment_model.predict(request.text)
        
        # Анализ аспектов
        aspects = aspect_analyzer.analyze_aspects(request.text, sentiment)
        
        # Генерация резюме
        summary = aspect_analyzer.generate_summary(request.text, sentiment, aspects)
        
        # Формируем ответ
        response = ReviewResponse(
            sentiment=sentiment,
            confidence=round(confidence, 3),
            key_aspects=aspects,
            summary=summary
        )
        
        logger.info(f"Результат анализа: {sentiment} (уверенность: {confidence:.3f})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при анализе отзыва: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка обработки запроса: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Проверка состояния сервиса
    
    Returns:
        HealthResponse: Статус сервиса
    """
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        model_loaded=sentiment_model.is_loaded
    )

@router.get("/")
async def root() -> Dict[str, Any]:
    """
    Корневой endpoint с информацией об API
    
    Returns:
        Dict: Основная информация об API
    """
    return {
        "message": "Wildberries Review Analyzer API",
        "version": settings.app_version,
        "endpoints": {
            "analyze": "/api/v1/analyze - POST - Анализ отзыва",
            "health": "/api/v1/health - GET - Статус сервиса",
            "docs": "/docs - Swagger документация"
        },
        "example_request": {
            "url": "/api/v1/analyze",
            "method": "POST",
            "body": {
                "text": "Товар пришел быстро, но качество не очень. Упаковка помята."
            }
        }
    }

@router.post("/test")
async def test_model() -> Dict[str, Any]:
    """
    Тестовый endpoint для проверки модели
    Использует примеры из данных для быстрой проверки
    
    Returns:
        Dict: Результаты тестирования
    """
    test_reviews = [
        "5 попыток было испечь хлеб с этой муки, и один результат - пирог из глины.",
        "Товар отличный, быстро доставили, качество супер!",
        "Нормальный товар, ничего особенного."
    ]
    
    results = []
    
    for review in test_reviews:
        try:
            if sentiment_model.is_loaded:
                sentiment, confidence = sentiment_model.predict(review)
                aspects = aspect_analyzer.analyze_aspects(review, sentiment)
                summary = aspect_analyzer.generate_summary(review, sentiment, aspects)
                
                results.append({
                    "text": review,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 3),
                    "aspects": aspects,
                    "summary": summary
                })
            else:
                results.append({
                    "text": review,
                    "error": "Модель не загружена"
                })
        except Exception as e:
            results.append({
                "text": review,
                "error": str(e)
            })
    
    return {
        "model_loaded": sentiment_model.is_loaded,
        "test_results": results
    }
