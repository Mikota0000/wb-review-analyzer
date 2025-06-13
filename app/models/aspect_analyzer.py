"""
Анализатор аспектов отзывов
"""

import re
from typing import Dict
from app.api.schemas import AspectSentiment

class AspectAnalyzer:
    """Анализатор аспектов отзывов"""
    
    def __init__(self):
        # Ключевые слова для каждого аспекта
        self.aspect_keywords = {
            "delivery": {
                "positive": [
                    "быстро дошло", "быстрая доставка", "пришло быстро", 
                    "доставили вовремя", "курьер хороший", "упаковка хорошая",
                    "доставка супер", "быстро привезли"
                ],
                "negative": [
                    "долго шло", "медленная доставка", "опоздали", 
                    "не привезли", "вместо", "неправильный товар",
                    "упаковка помята", "коробка разорвана", "курьер грубый",
                    "не дошло", "потерялся", "задержка"
                ]
            },
            "quality": {
                "positive": [
                    "качество отличное", "хорошее качество", "качественный",
                    "материал хороший", "прочный", "крепкий", "добротный",
                    "качество супер", "отлично сделано"
                ],
                "negative": [
                    "плохое качество", "качество ужасное", "некачественный",
                    "материал плохой", "хрупкий", "сломался", "развалился",
                    "брак", "дефект", "прогорклый", "испорченный", "запах",
                    "вкус плохой", "не вкусно", "противный вкус"
                ]
            },
            "price": {
                "positive": [
                    "хорошая цена", "недорого", "дешево", "выгодно",
                    "за свои деньги", "цена приемлемая", "стоит своих денег"
                ],
                "negative": [
                    "дорого", "цена завышена", "переплата", "не стоит денег",
                    "цена кусается", "слишком дорого", "обман с ценой"
                ]
            },
            "service": {
                "positive": [
                    "сервис хороший", "обслуживание отличное", "поддержка помогла",
                    "консультация хорошая", "менеджер вежливый"
                ],
                "negative": [
                    "плохой сервис", "не помогли", "грубые", "не отвечают",
                    "поддержка не работает", "возврат денег", "не вернули деньги",
                    "жду возврата", "обман", "мошенники"
                ]
            }
        }
    
    def analyze_aspects(self, text: str, overall_sentiment: str) -> Dict[str, AspectSentiment]:
        """
        Анализ аспектов в тексте отзыва
        
        Args:
            text: Текст отзыва
            overall_sentiment: Общая тональность отзыва
            
        Returns:
            Dict с тональностью по каждому аспекту
        """
        text_lower = text.lower()
        results = {}
        
        for aspect in ["delivery", "quality", "price", "service"]:
            aspect_sentiment = self._analyze_single_aspect(text_lower, aspect, overall_sentiment)
            results[aspect] = aspect_sentiment
        
        return results
    
    def _analyze_single_aspect(self, text: str, aspect: str, overall_sentiment: str) -> AspectSentiment:
        """Анализ одного аспекта"""
        
        positive_score = 0
        negative_score = 0
        
        # Подсчитываем совпадения с ключевыми словами
        for keyword in self.aspect_keywords[aspect]["positive"]:
            if keyword in text:
                positive_score += 1
                
        for keyword in self.aspect_keywords[aspect]["negative"]:
            if keyword in text:
                negative_score += 1
        
        # Дополнительные правила для конкретных аспектов
        if aspect == "delivery":
            # Проверяем паттерн "вместо X пришло Y"
            if re.search(r"вместо.*приш[лёе]", text):
                negative_score += 2
            # Проверяем упоминание скорости
            if re.search(r"быстро|скор", text) and overall_sentiment == "Позитивный":
                positive_score += 1
                
        elif aspect == "quality":
            # Проверяем упоминание вкуса, запаха
            if re.search(r"вкус|запах", text):
                if overall_sentiment == "Негативный":
                    negative_score += 1
                else:
                    positive_score += 1
                    
        elif aspect == "service":
            # Проверяем упоминание возврата
            if re.search(r"возврат|верну", text):
                negative_score += 1
        
        # Определяем итоговую тональность аспекта
        if positive_score > negative_score:
            return AspectSentiment.positive
        elif negative_score > positive_score:
            return AspectSentiment.negative
        else:
            # Если нет явных индикаторов, используем общую тональность
            if overall_sentiment == "Позитивный":
                return AspectSentiment.neutral
            elif overall_sentiment == "Негативный":
                return AspectSentiment.neutral
            else:
                return AspectSentiment.neutral
    
    def generate_summary(self, text: str, sentiment: str, aspects: Dict[str, AspectSentiment]) -> str:
        """
        Генерация краткого резюме отзыва
        
        Args:
            text: Исходный текст отзыва
            sentiment: Общая тональность
            aspects: Анализ по аспектам
            
        Returns:
            Краткое резюме
        """
        text_lower = text.lower()
        summary_parts = []
        
        # Анализируем основные проблемы/достоинства
        if "вместо" in text_lower and "приш" in text_lower:
            summary_parts.append("Ошибка в заказе - доставлен неправильный товар")
            
        elif aspects["delivery"] == AspectSentiment.negative:
            if "долго" in text_lower:
                summary_parts.append("Проблемы со сроками доставки")
            else:
                summary_parts.append("Проблемы с доставкой")
                
        if aspects["quality"] == AspectSentiment.negative:
            if any(word in text_lower for word in ["вкус", "запах", "прогорк"]):
                summary_parts.append("Проблемы с качеством продукта")
            else:
                summary_parts.append("Низкое качество товара")
                
        if aspects["price"] == AspectSentiment.negative:
            summary_parts.append("Завышенная цена")
            
        if aspects["service"] == AspectSentiment.negative:
            if "возврат" in text_lower:
                summary_parts.append("Проблемы с возвратом средств")
            else:
                summary_parts.append("Плохое обслуживание")
        
        # Положительные моменты
        if sentiment == "Позитивный" and not summary_parts:
            if aspects["quality"] == AspectSentiment.positive:
                summary_parts.append("Качество товара устраивает")
            if aspects["delivery"] == AspectSentiment.positive:
                summary_parts.append("Быстрая доставка")
        
        # Если ничего не выявлено, делаем базовое резюме
        if not summary_parts:
            if sentiment == "Позитивный":
                return "Покупатель доволен товаром"
            elif sentiment == "Негативный":
                return "Покупатель недоволен покупкой"
            else:
                return "Нейтральный отзыв о товаре"
        
        return "; ".join(summary_parts)


# Глобальная инстанция анализатора
aspect_analyzer = AspectAnalyzer()