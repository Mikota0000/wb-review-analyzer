"""
Модель для анализа тональности отзывов
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

from app.core.config import settings


class SentimentModel:
    """Модель анализа тональности"""
    
    def __init__(self, model_path: Path = None):
        self.model_path = model_path or Path(__file__).parent.parent.parent / "models" / "onnx_model_s"
        self.tokenizer = None
        self.model = None
        self.sentiment_labels = {
            0: "Негативный",
            1: "Нейтральный", 
            2: "Позитивный"
        }
        self.is_loaded = False
    
    def load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            print(f"Загрузка модели из {self.model_path}")
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Загружаем ONNX модель
            self.model = ORTModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            
            self.is_loaded = True
            print("Модель успешно загружена")
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise e
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Предсказание тональности для текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Tuple[str, float]: (sentiment, confidence)
        """
        if not self.is_loaded:
            self.load_model()
        
        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_text_length,
            padding=True
        )
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        sentiment = self.sentiment_labels[predicted_class]
        return sentiment, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Предсказание тональности для списка текстов
        
        Args:
            texts: Список текстов для анализа
            
        Returns:
            List[Tuple[str, float]]: Список (sentiment, confidence)
        """
        if not self.is_loaded:
            self.load_model()
        
        results = []
        
        # Обрабатываем батчами
        for i in range(0, len(texts), settings.batch_size):
            batch_texts = texts[i:i + settings.batch_size]
            
            # Токенизация батча
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=settings.max_text_length,
                padding=True
            )
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Обработка результатов батча
            for j in range(len(batch_texts)):
                predicted_class = torch.argmax(predictions[j], dim=-1).item()
                confidence = predictions[j][predicted_class].item()
                sentiment = self.sentiment_labels[predicted_class]
                results.append((sentiment, confidence))
        
        return results


# Глобальная инстанция модели
sentiment_model = SentimentModel()