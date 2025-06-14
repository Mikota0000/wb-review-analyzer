from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.api.endpoints import router
from app.models.sentiment_model import sentiment_model

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    try:
        logger.info("Инициализация приложения...")
        logger.info("Загрузка модели анализа тональности...")
        sentiment_model.load_model()
        logger.info("Модель успешно загружена")
        logger.info("Приложение готово к работе")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        
    
    yield  
    
    logger.info("Завершение работы приложения...")

def create_app() -> FastAPI:
    """Создание FastAPI приложения"""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="API для анализа тональности отзывов с Wildberries",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan  # Подключил lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Подключение роутов
    app.include_router(router, prefix="/api/v1")
    
    # Обработчик ошибок
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Общий обработчик ошибок"""
        logger.error(f"Необработанная ошибка: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Внутренняя ошибка сервера"}
        )
    
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )