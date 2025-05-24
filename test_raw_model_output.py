import torch
from transformers import AutoConfig, AutoTokenizer
import os
from custom_models import SpanBertLSTMForTokenClassification

# Конфигурация
MODEL_DIR = "model"
MAX_LENGTH = 164
BASE_MODEL_PATH = "DeepPavlov/rubert-base-cased"

# Глобальные переменные для модели
model = None
tokenizer = None
id2label = None
device = None

def load_model_and_tokenizer():
    """Загружает модель SpanBertLSTMForTokenClassification и токенизатор."""
    global model, tokenizer, id2label, device
    
    if not os.path.isdir(MODEL_DIR):
        print(f"Ошибка: Директория с моделью не найдена: {MODEL_DIR}")
        return False

    print(f"Загрузка модели и токенизатора из {MODEL_DIR}...")
    try:
        model_config = AutoConfig.from_pretrained(MODEL_DIR)
        
        # Проверяем, что в конфигурации указана правильная архитектура
        if model_config.architectures[0] != "SpanBertLSTMForTokenClassification":
            print(f"Предупреждение: Ожидалась архитектура SpanBertLSTMForTokenClassification, "
                  f"но найдена {model_config.architectures[0]}. Всё равно продолжаем.")
        
        # Создаем модель
        model = SpanBertLSTMForTokenClassification(
            base_model_path=BASE_MODEL_PATH,
            num_labels=model_config.num_labels,
            config=model_config,
            lstm_hidden_size=getattr(model_config, 'lstm_hidden_size', model_config.hidden_size // 2),
            dropout_rate=getattr(model_config, 'dropout_rate', 0.2)
        )

        # Загружаем веса модели
        state_dict_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.load_state_dict(torch.load(state_dict_path, map_location=map_location, weights_only=True))
            print(f"Веса модели успешно загружены из {state_dict_path}.")
        else:
            print(f"Предупреждение: Файл весов pytorch_model.bin не найден в {MODEL_DIR}.")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        id2label = {int(k): v for k, v in model_config.id2label.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Модель успешно загружена и переведена в режим оценки на устройстве: {device}.")
        return True
    except Exception as e:
        print(f"Критическая ошибка при загрузке модели: {e}")
        model = None
        return False

def get_model_predictions_for_sentence(raw_sentence_text):
    """Получает предсказания модели для одного предложения с правильной обработкой BIO меток."""
    global model, tokenizer, id2label, device
    
    if not raw_sentence_text.strip():
        return []

    # Токенизируем слова (более детально)
    raw_words = raw_sentence_text.split()
    
    # Кодируем входной текст через tokenizer с важным флагом is_split_into_words=True
    encoded = tokenizer(
        raw_words,
        is_split_into_words=True,  # Важно! Это подсказывает токенизатору, что входные данные уже разделены на слова
        return_tensors='pt',
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Получаем предсказания модели
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Получаем предсказания из CRF слоя модели
        decoded_paths_batch = model.decode(outputs['logits'], mask=attention_mask.bool())
        
        if decoded_paths_batch and decoded_paths_batch[0] is not None:
            subtoken_labels_from_crf = decoded_paths_batch[0]
            predicted_subtoken_label_ids = [-100] * MAX_LENGTH 
            active_subtoken_idx = 0
            
            for i in range(min(MAX_LENGTH, len(attention_mask[0]))):
                if attention_mask[0, i].item() == 1:  # Только для активных токенов в маске
                    if active_subtoken_idx < len(subtoken_labels_from_crf):
                        predicted_subtoken_label_ids[i] = subtoken_labels_from_crf[active_subtoken_idx]
                        active_subtoken_idx += 1
                    else:
                        # Если CRF вернул меньше меток, чем активных токенов
                        o_label_id = next((k_id for k_id, v_label in id2label.items() if v_label == 'O'), 0)
                        predicted_subtoken_label_ids[i] = o_label_id
        else:
            # Если CRF не вернул результаты, используем argmax
            print("ПРЕДУПРЕЖДЕНИЕ: CRF декодирование не вернуло результатов. Используется argmax.")
            argmax_preds = torch.argmax(outputs['logits'], dim=2)
            predicted_subtoken_label_ids = argmax_preds[0].tolist()

    # Новый улучшенный алгоритм для сопоставления субтокенов с оригинальными словами
    word_predictions = []
    word_ids = encoded.word_ids(batch_index=0)  # Получаем связь между токенами и словами
    
    for word_idx in range(len(raw_words)):
        word_subtoken_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
        
        if not word_subtoken_indices:
            continue  # Пропускаем, если слово не имеет субтокенов (странный случай)
        
        # Берем метку от первого субтокена для данного слова согласно BIO схеме
        first_subtoken_pos = word_subtoken_indices[0]
        
        if first_subtoken_pos >= len(predicted_subtoken_label_ids):
            # Защита от выхода за границы
            label_id = next((k_id for k_id, v_label in id2label.items() if v_label == 'O'), 0)
        else:
            label_id = predicted_subtoken_label_ids[first_subtoken_pos]
            if label_id == -100:
                label_id = next((k_id for k_id, v_label in id2label.items() if v_label == 'O'), 0)
        
        label_str = id2label.get(label_id, "O")
        
        # Проверка последовательности BIO тегов
        # Если текущее слово имеет I- метку, но до него не было соответствующей B- метки,
        # преобразуем I- в B-, чтобы сохранить правильную BIO схему
        if label_str.startswith("I-"):
            entity_type = label_str[2:]
            if word_idx == 0 or not word_predictions or not word_predictions[-1][1].endswith(entity_type):
                label_str = f"B-{entity_type}"
        
        word_predictions.append((raw_words[word_idx], label_str))
    
    return word_predictions


if __name__ == "__main__":
    if not load_model_and_tokenizer():
        print("Не удалось загрузить модель. Выход.")
        exit()

    print("Введите текст для распознавания сущностей. Для выхода введите 'exit', 'quit' или 'выход'.")
    print("------")

    try:
        counter = 1
        while True:
            # Получаем ввод от пользователя
            user_input = input(f"\nПример {counter}. Введите текст: ")
            
            # Проверяем на команду выхода
            if user_input.lower() in ["exit", "quit", "выход"]:
                print("Завершение работы.")
                break
                
            # Обрабатываем ввод
            if not user_input.strip():
                print("Введен пустой текст. Попробуйте ещё раз.")
                continue
                
            print(f"Текст: {user_input}")
            model_predictions = get_model_predictions_for_sentence(user_input)
            
            if not model_predictions:
                print("  Pred: [Нет предсказаний от модели]")
                counter += 1
                continue

            # Выводим предсказания
            pred_output_parts = [f"{token_text}/{pred_label}" for token_text, pred_label in model_predictions]
            pred_display = " ".join(pred_output_parts)
            
            max_display_len = 700 
            if len(pred_display) > max_display_len:
                pred_display = pred_display[:max_display_len] + "..."
                
            print(f"  Pred: {pred_display}")
            counter += 1
            
    except KeyboardInterrupt:
        print("\nРабота программы прервана пользователем.")
    
    print("--- Тестирование завершено ---") 