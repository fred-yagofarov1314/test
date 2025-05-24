# Техническая документация
## Система распознавания именованных сущностей на основе BERT с CRF

### 1. ВВЕДЕНИЕ

Настоящий документ содержит техническую документацию программной системы распознавания именованных сущностей (Named Entity Recognition, NER), разработанной на основе нейросетевых моделей BERT с дополнительными слоями CRF (Conditional Random Fields) и LSTM (Long Short-Term Memory). Система предназначена для обнаружения и классификации именованных сущностей в русскоязычных текстах.

### 2. НАЗНАЧЕНИЕ И ОБЛАСТЬ ПРИМЕНЕНИЯ

Система предназначена для решения задачи распознавания именованных сущностей в текстах на русском языке. Система поддерживает идентификацию 46 типов именованных сущностей, включая:
- Персональные данные (ФИО, СНИЛС, ИНН и др.)
- Контактную информацию (телефоны, email)
- Адресные данные (страна, регион, город и др.)
- Идентификаторы (серийные номера, IMEI и др.)
- Банковские данные (номера карт, счетов, БИК)
- Даты и временные метки

### 3. АРХИТЕКТУРА СИСТЕМЫ

#### 3.1 Общая структура системы

Система построена на модульной архитектуре и включает следующие основные компоненты:
- Модуль обучения моделей (`kaggle.py`)
- Модуль кастомных моделей (`custom_models.py`)
- Модуль кастомных тренеров (`custom_trainer.py`)
- Модуль утилит (`custom_utils.py`)
- Модуль генерации данных (состоит из `main_generate_conll.py` и `sentence_generator.py`)

#### 3.2 Используемые технологии

- PyTorch (версия 2.1.1 и выше)
- Transformers (библиотека Hugging Face)
- Accelerate (для ускорения обучения)
- Datasets (для работы с датасетами)
- BitsAndBytes (опционально, для 8-битной оптимизации)
- Seqeval (для оценки качества)
- Flash Attention 2 (опционально, для ускорения работы модели)

### 4. КОМПОНЕНТЫ СИСТЕМЫ

#### 4.1 Модели для распознавания именованных сущностей

В системе реализована одна основная архитектура модели:

1. **SpanBertLSTMForTokenClassification** - расширенная архитектура BERT с LSTM и CRF слоями
   - Включает двунаправленный LSTM слой
   - Реализует механизм внимания поверх LSTM
   - Включает слой CRF для улучшения классификации последовательностей
   - Использует трансформационный слой для улучшения представлений

#### 4.2 Тренеры моделей

1. **CustomTrainerWithCRF** - специализированный тренер для моделей с CRF
   - Адаптированные функции вычисления потерь для CRF
   - Специализированные методы для предсказания и декодирования последовательностей

#### 4.3 Утилиты и вспомогательные компоненты

- Функции оптимизации для GPU (Flash Attention, gradient checkpointing)
- Функции обработки данных в формате CoNLL
- Структуры для генерации BIO-меток
- Инициализация весов переходов CRF
- Оптимизированные загрузчики данных

### 5. ПРОЦЕСС ОБУЧЕНИЯ МОДЕЛЕЙ

#### 5.1 Подготовка данных

Данные должны быть представлены в формате CoNLL, где каждая строка содержит токен и его метку, разделенные пробелом. Предложения разделяются пустыми строками.

Пример данных в формате CoNLL:
```
Сергеев B-SURNAME
Андрей B-NAME
Николаевич B-PATRONYMIC
сегодня O
на O
улице O
сороколетия B-STREET
комсомола I-STREET
64 B-NUMBER
в O
квартире O
70 B-NUMBER
прорвало O
трубу O

```

#### 5.2 Процесс обучения

Обучение модели запускается через скрипт `kaggle.py` и включает следующие этапы:
1. Загрузка и токенизация данных
2. Инициализация модели и оптимизатора
3. Обучение с использованием CustomTrainerWithCRF
4. Оценка модели на тестовых примерах
5. Сохранение обученной модели

#### 5.3 Основные параметры конфигурации

Основные параметры конфигурации находятся в классе `MainConfig` в файле `kaggle.py`:
- `MODEL_PATH` - путь к базовой предобученной модели
- `OUTPUT_DIR` - директория для сохранения результатов
- `MODEL_CHOICE` - выбор архитектуры модели ('SpanBertCRF' или 'SpanBertLSTMCRF')
- `EPOCHS` - количество эпох обучения
- `TRAIN_BATCH_SIZE_PER_DEVICE` - размер батча для обучения
- `LEARNING_RATE` - скорость обучения
- `MAX_SEQ_LENGTH` - максимальная длина последовательности
- `OPTIMIZER_CHOICE` - выбор оптимизатора
- `USE_FLASH_ATTENTION` - использование Flash Attention для ускорения
- `GRADIENT_CHECKPOINTING` - использование gradient checkpointing для экономии памяти

### 6. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ

Система включает комплексную оценку качества распознавания именованных сущностей:
- Расчет метрик precision, recall, F1-score на уровне токенов и сущностей
- Формирование матрицы ошибок для анализа проблемных классов
- Адаптивная настройка весов классов на основе результатов оценки
- Периодическое тестирование на статических примерах для визуального контроля

### 7. ТРЕБОВАНИЯ К АППАРАТНОМУ И ПРОГРАММНОМУ ОБЕСПЕЧЕНИЮ

#### 7.1 Требования к аппаратному обеспечению
- GPU с поддержкой CUDA (рекомендуется NVIDIA T4 или выше)
- Оперативная память: не менее 16 ГБ
- Свободное место на диске: не менее 10 ГБ

#### 7.2 Требования к программному обеспечению
- Python 3.8 или выше
- CUDA 11.7 или выше
- PyTorch 2.0 или выше
- Transformers 4.28 или выше
- Accelerate 0.20 или выше
- Datasets 2.12 или выше
- Seqeval 1.2 или выше

### 8. ИНСТРУКЦИЯ ПО УСТАНОВКЕ И ЗАПУСКУ

#### 8.1 Установка необходимых библиотек

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate datasets seqeval wandb
pip install bitsandbytes  # Опционально, для 8-битной оптимизации
```

#### 8.2 Запуск обучения модели

```bash
python kaggle.py
```

#### 8.3 Использование обученной модели для предсказаний

```python
from custom_models import SpanBertLSTMForTokenClassification
from transformers import AutoTokenizer

# Загрузка модели и токенизатора
model = SpanBertLSTMForTokenClassification.from_pretrained("путь_к_сохраненной_модели")
tokenizer = AutoTokenizer.from_pretrained("путь_к_сохраненной_модели")

# Пример текста для анализа
text = "Сергеев Андрей Николаевич сегодня на улице сороколетия комсомола 64 в квартире 70 прорвало трубу"
words = text.split()

# Токенизация
encoded_input = tokenizer(
    words,
    is_split_into_words=True,
    add_special_tokens=True,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Получение предсказаний
with torch.no_grad():
    outputs = model(**encoded_input)
    logits = outputs['logits']
    predictions = model.decode(logits, encoded_input['attention_mask'])

# Обработка результатов
entities = []
word_ids = encoded_input.word_ids(batch_index=0)
previous_word_idx = None
current_entity = []
current_entity_type = None

for token_idx, pred_label_id in enumerate(predictions[0]):
    word_idx = word_ids[token_idx]
    if word_idx is None:
        continue
    
    if word_idx != previous_word_idx:
        if current_entity:
            entities.append((current_entity_type, " ".join(current_entity)))
            current_entity = []
        
        label = model.id2label.get(pred_label_id, "O")
        if label.startswith("B-"):
            current_entity = [words[word_idx]]
            current_entity_type = label[2:]
        elif label.startswith("I-") and not current_entity:
            current_entity = [words[word_idx]]
            current_entity_type = label[2:]
    
    previous_word_idx = word_idx

if current_entity:
    entities.append((current_entity_type, " ".join(current_entity)))

# Вывод результатов
print("Обнаруженные сущности:")
for entity_type, entity_text in entities:
    print(f"  - {entity_type}: {entity_text}")
```

### 9. ЗАКЛЮЧЕНИЕ

Представленная система распознавания именованных сущностей обеспечивает высокое качество обнаружения и классификации сущностей в русскоязычных текстах. Модульная архитектура и гибкие настройки позволяют адаптировать систему под различные задачи и условия применения.

Система включает передовые методы оптимизации для эффективного обучения на ограниченных вычислительных ресурсах и предоставляет богатый набор инструментов для тонкой настройки моделей и анализа их качества.

### ПРИЛОЖЕНИЕ А. СПИСОК ПОДДЕРЖИВАЕМЫХ ТИПОВ СУЩНОСТЕЙ

1. SURNAME - Фамилия
2. NAME - Имя
3. PATRONYMIC - Отчество
4. INDIVIDUAL_TAX_ID - ИНН физического лица
5. SNILS - СНИЛС
6. USERNAME - Имя пользователя
7. PASSWORD - Пароль
8. PIN - ПИН-код
9. TOKEN - Токен
10. EMAIL - Электронная почта
11. MOBILE_PHONE - Мобильный телефон
12. LANDLINE_PHONE - Стационарный телефон
13. INTERNATIONAL_PHONE - Международный телефон
14. COUNTRY - Страна
15. REGION - Регион
16. CITY - Город
17. DISTRICT - Район
18. LOCALITY - Населенный пункт
19. STREET - Улица
20. COORDINATES - Координаты
21. POSTAL_CODE - Почтовый индекс
22. ORGANIZATION - Организация
23. DEPARTMENT - Отдел
24. AUTHORITY - Орган власти
25. SUBDIVISION - Подразделение
26. OGRN - ОГРН
27. OGRNIP - ОГРНИП
28. LEGAL_TAX_ID - ИНН юридического лица
29. DEVICE_ID - ID устройства
30. SERIAL_NUMBER - Серийный номер
31. IMEI - IMEI
32. MAC - MAC-адрес
33. IBAN - IBAN
34. CARD_NUMBER - Номер карты
35. BIK - БИК
36. ACCOUNT - Номер счета
37. LICENSE_PLATE - Номер автомобиля
38. KPP - КПП
39. OMS_POLICY_NUMBER - Номер полиса ОМС
40. CARD_TYPE - Тип карты
41. NUMBER - Числовое значение
42. IPV4 - IPv4 адрес
43. IPV6 - IPv6 адрес
44. URL - URL
45. DATE_TIME - Дата и время
46. DATE_MONTH - Месяц
47. DATE_YEAR - Год
48. DATE_DAY - День 