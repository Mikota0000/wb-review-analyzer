"""
Оптимизированное обучение модели анализа тональности с поддержкой GPU
"""

import csv
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from optimum.onnxruntime import ORTModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# "cointegrated/rubert-tiny2" - текущая, быстрая.
# "DeepPavlov/rubert-base-cased" - классический BERT

BASE_MODEL_NAME = "cointegrated/rubert-tiny2"

# Путь к файлу данных (относительно корневой директории проекта)
DATA_PATH = Path("C:/Users/mg821/Desktop/Учеба/python-programs/wildberries-review-analyzer/data/wildberries_reviews.csv")

# Основная директория для сохранения обученных моделей
MODELS_DIR = Path("C:/Users/mg821/Desktop/Учеба/python-programs/wildberries-review-analyzer/models")

# Дополнительные параметры обучения (можно менять)
NUM_TRAIN_EPOCHS = 3 # Увеличим число эпох, если позволяет время и ресурсы
PER_DEVICE_TRAIN_BATCH_SIZE = 16 # Увеличили batch size для GPU
GRADIENT_ACCUMULATION_STEPS = 1 # Установим 1, если batch_size уже достаточен
MAX_TEXT_LENGTH = 256 # Максимальная длина текста для токенизации

# --- КОНЕЦ КОНФИГУРАЦИИ ---


# Проверка доступности GPU
def check_gpu():
    """Проверка доступности CUDA GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU доступен: {gpu_name}")
        print(f"   Память GPU: {gpu_memory:.1f} GB")
        print(f"   CUDA версия: {torch.version.cuda}")
        return device
    else:
        print("❌ GPU недоступен, используется CPU")
        return torch.device("cpu")

class OptimizedWildberriesDataset(torch.utils.data.Dataset):
    """Оптимизированный Dataset для отзывов Wildberries"""
    
    def __init__(self, texts, labels, tokenizer, max_length=MAX_TEXT_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Предварительная очистка текста для ускорения
        # Обрезание слишком длинных текстов до max_length производится токенизатором
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length', # Лучше использовать 'longest' или 'max_length'
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_and_prepare_data(data_path: Path, sample_size=None):
    """Загрузка и подготовка данных из CSV-файла с оптимизацией"""
    print("Загрузка данных из CSV...")

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            try:
                # Предполагаем формат CSV: первая колонка - рейтинг, вторая - отзыв
                rating = int(row[0])
                review = row[1].strip('"')
                
                # Фильтруем слишком короткие или пустые отзывы
                if len(review.strip()) < 10: # Или даже 20-30, чтобы были осмысленные отзывы
                    continue
                data.append({
                    'rating': rating,
                    'review': review
                })
                # Ограничиваем размер для быстрого тестирования
                if sample_size and len(data) >= sample_size:
                    break
            except (IndexError, ValueError) as e:
                # Пропускаем строки с ошибками или неверным форматом
                print(f"Пропущена строка из-за ошибки формата: {row}. Ошибка: {e}")
                continue

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("Данные не были загружены. Проверьте формат CSV и путь к файлу.")

    print(f"Загружено {len(df)} отзывов")
    print("Распределение по оценкам:")
    print(df['rating'].value_counts().sort_index())

    # Создаем метки тональности
    def rating_to_sentiment(rating):
        if rating <= 2:
            return 0  # Негативный
        elif rating == 3:
            return 1  # Нейтральный
        else:
            return 2  # Позитивный

    df['sentiment'] = df['rating'].apply(rating_to_sentiment)
    print("\nРаспределение по тональности:")
    sentiment_names = {0: 'Негативный', 1: 'Нейтральный', 2: 'Позитивный'}
    for i, count in df['sentiment'].value_counts().sort_index().items():
        print(f"{sentiment_names[i]}: {count}")

    return df

def train_optimized_model(df: pd.DataFrame, model_output_dir: Path, base_model_name: str, device):
    """Оптимизированное обучение модели с GPU"""
    print(f"\nИнициализация модели: {base_model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=3,
        problem_type="single_label_classification"
    )
    
    # Перемещаем модель на GPU
    model.to(device)
    print(f"Модель перемещена на: {device}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'].values, 
        df['sentiment'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )
    
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Тестовая выборка: {len(X_test)}")
    
    # Создание датасетов
    train_dataset = OptimizedWildberriesDataset(X_train, y_train, tokenizer, max_length=MAX_TEXT_LENGTH)
    test_dataset = OptimizedWildberriesDataset(X_test, y_test, tokenizer, max_length=MAX_TEXT_LENGTH)
    
    # Оптимизированные параметры обучения для GPU
    training_args = TrainingArguments(
        output_dir=f'./results_{base_model_name.replace("/", "_")}', 
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100, 
        weight_decay=0.01,
        learning_rate=3e-5, 
        logging_dir='./logs',
        logging_steps=50, 
        eval_strategy="steps",
        eval_steps=200, 
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        fp16=torch.cuda.is_available(), 
        dataloader_num_workers=2, 
        remove_unused_columns=False,
        report_to=None, 
    )
    
    # Trainer с оптимизациями
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("\nНачало обучения...")
    print(f"Параметры: batch_size={training_args.per_device_train_batch_size}, "
              f"epochs={training_args.num_train_epochs}, fp16={training_args.fp16}")
    
    # Измеряем время обучения
    import time
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n⏱️ Время обучения: {training_time/60:.1f} минут")
    
    # Оценка модели
    print("\nОценка модели...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    sentiment_names = ['Негативный', 'Нейтральный', 'Позитивный']
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred, target_names=sentiment_names))
    
    # Сохранение модели PyTorch
    # Создаем уникальную папку для каждой обученной модели
    model_save_path = model_output_dir / base_model_name.replace("/", "_") 
    print(f"\nСохранение PyTorch модели в {model_save_path}...")
    model_save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_save_path)) 
    tokenizer.save_pretrained(str(model_save_path)) 
    
    return model, tokenizer, model_save_path 

def convert_to_onnx_optimized(model, tokenizer, pytorch_model_path: Path):
    """Оптимизированная конвертация в ONNX"""
    print("\nКонвертация в ONNX...")
    
    onnx_output_path = pytorch_model_path / "onnx_model" 
    onnx_output_path.mkdir(exist_ok=True)
    
    try:
        # Перемещаем модель на CPU для экспорта
        model.to('cpu')
        
        # Конвертация в ONNX
        # from_pretrained теперь принимает путь к сохраненной PyTorch модели
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            str(pytorch_model_path), # Путь к сохраненной PyTorch модели
            export=True # Обязательно указываем export=True
        )
        
        # Сохранение ONNX модели и токенизатора
        ort_model.save_pretrained(str(onnx_output_path))
        tokenizer.save_pretrained(str(onnx_output_path))
        
        print(f"ONNX модель сохранена в {onnx_output_path}")
        return onnx_output_path
        
    except Exception as e:
        print(f"Ошибка при конвертации в ONNX: {e}")
        return None

def main():
    """Основная функция с оптимизациями"""
    # Проверяем GPU
    device = check_gpu()
    
    # Проверяем наличие данных
    if not DATA_PATH.exists():
        print(f"Файл с данными не найден: {DATA_PATH}")
        print("Создайте файл data/wildberries_reviews.csv с вашими отзывами")
        print("или измените путь DATA_PATH в начале скрипта.")
        return
    
    # Загрузка данных
    df = load_and_prepare_data(DATA_PATH) # Все данные
    
    # Обучение модели
    model, tokenizer, pytorch_model_path = train_optimized_model(df, MODELS_DIR, BASE_MODEL_NAME, device)
    
    # Конвертация в ONNX
    # Передаем путь к сохраненной PyTorch модели
    onnx_path = convert_to_onnx_optimized(model, tokenizer, pytorch_model_path)
    
    if onnx_path:
        print(f"\n✅ Модель готова! ONNX файлы находятся в: {onnx_path}")
    else:
        print("\n⚠️ Модель обучена, но ONNX конвертация не удалась")

if __name__ == "__main__":
    main()