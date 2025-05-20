import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoConfig
import os
import re # Добавляем импорт re для токенизации
import random
import json # Добавляем для вывода
from custom_models import SpanBertLSTMForTokenClassification, SpanBertForTokenClassificationCRF
from natasha import MorphVocab, Doc, NewsEmbedding, NewsMorphTagger, Segmenter

# --- Глобальные переменные для Flask приложения и моделей ---
# NER модель
model_ner = None
tokenizer_ner = None
id2label_ner = None
device_ner = None
# Модель для исправления грамматики (SageFredT5)
model_corrector = None
tokenizer_corrector = None
device_corrector = None # Может быть тот же device_ner или другой
# Natasha
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# ---------------------------------------------------------

# --- Попытка импорта библиотек для валидации (скопировано из main_generate_conll.py) ---
try:
    from email_validator import validate_email, EmailNotValidError
    USE_EMAIL_VALIDATOR = True
except ImportError:
    print("Warning: Библиотека email-validator не найдена (`pip install email-validator`). Используется regex для проверки Email.")
    USE_EMAIL_VALIDATOR = False

try:
    import ipaddress
    USE_IPADDRESS = True
except ImportError:
    print("Warning: Библиотека ipaddress не найдена (стандартная с Python 3.3+). Используется regex для проверки IP.")
    USE_IPADDRESS = False
# ---------------------------------------------

# --- Функции для проверки контрольных сумм (скопировано из main_generate_conll.py) ---
def check_inn_checksum(inn_str):
    if not inn_str.isdigit():
        return False, "ИНН не только цифры"
    if len(inn_str) == 10:
        coeffs = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        check_sum = sum(int(digit) * coeff for digit, coeff in zip(inn_str[:-1], coeffs))
        if int(inn_str[-1]) == check_sum % 11 % 10:
            return True, "ОК (ИНН10)"
        else:
            return False, "Ошибка к.с. (ИНН10)"
    elif len(inn_str) == 12:
        coeffs1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        coeffs2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        check_sum1 = sum(int(digit) * coeff for digit, coeff in zip(inn_str[:-2], coeffs1))
        check_sum2 = sum(int(digit) * coeff for digit, coeff in zip(inn_str[:-1], coeffs2))
        if int(inn_str[-2]) == check_sum1 % 11 % 10 and int(inn_str[-1]) == check_sum2 % 11 % 10:
            return True, "ОК (ИНН12)"
        else:
            return False, "Ошибка к.с. (ИНН12)"
    else:
        return False, f"ИНН не из 10/12 цифр (длина {len(inn_str)})"

def check_luhn(card_number_str):
    if not card_number_str.isdigit():
        return False, "Не только цифры (Луна)"
    digits = [int(d) for d in card_number_str]
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled if doubled < 10 else doubled - 9
        else:
            checksum += digit
    if checksum % 10 == 0:
        return True, "ОК (Луна)"
    else:
        return False, "Ошибка к.с. (Луна)"

def check_snils_checksum(snils_str):
    # snils_str ожидается уже очищенным и проверенным на isdigit и длину в is_likely_snils
    coeffs = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    check_sum_val = sum(int(digit) * coeff for digit, coeff in zip(snils_str[:9], coeffs))
    
    check_digit = 0
    if check_sum_val < 100:
        check_digit = check_sum_val
    elif check_sum_val == 100 or check_sum_val == 101:
        check_digit = 0
    elif check_sum_val > 101:
        check_digit = check_sum_val % 101
        if check_digit == 100:
            check_digit = 0
            
    if check_digit == int(snils_str[9:]):
        return True, "ОК (СНИЛС к.с.)"
    else:
        return False, "Ошибка к.с. (СНИЛС)"
    
def check_ogrn_checksum(ogrn_str):
    # ogrn_str ожидается очищенным и проверенным на isdigit и длину в is_likely_ogrn
    num = int(ogrn_str[:-1])
    check_digit_calc = num % 11 % 10 
    if check_digit_calc == int(ogrn_str[-1]):
        return True, "ОК (ОГРН к.с.)"
    else:
        return False, "Ошибка к.с. (ОГРН)"
    
def check_ogrnip_checksum(ogrnip_str):
    # ogrnip_str ожидается очищенным и проверенным на isdigit и длину в is_likely_ogrnip
    num = int(ogrnip_str[:-1])
    check_digit_calc = num % 13 % 10
    if check_digit_calc == int(ogrnip_str[-1]):
        return True, "ОК (ОГРНИП к.с.)"
    else:
        return False, "Ошибка к.с. (ОГРНИП)"

def check_imei_luhn(imei_str):
    # imei_str ожидается очищенным и проверенным на isdigit и длину в is_likely_imei
    return check_luhn(imei_str) # check_luhn уже возвращает (bool, comment)

# --- Простые функции проверки формата (скопировано из main_generate_conll.py) ---
def is_likely_ipv4(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if USE_IPADDRESS:
        try:
            ipaddress.IPv4Address(text)
            return True, "ОК (IPv4 ipaddress)"
        except ipaddress.AddressValueError:
            return False, "Не IPv4 (ipaddress)"
    else: 
        if bool(re.fullmatch(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', text)):
            return True, "ОК (IPv4 regex)"
        else:
            return False, "Не IPv4 (regex)"

def is_likely_ipv6(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if USE_IPADDRESS:
        try:
            ipaddress.IPv6Address(text)
            return True, "ОК (IPv6 ipaddress)"
        except ipaddress.AddressValueError:
            return False, "Не IPv6 (ipaddress)"
    else: 
        parts = text.split(':')
        if 2 < len(parts) <= 8 and all(re.fullmatch(r'[0-9a-fA-F]{1,4}', part) or part == '' for part in parts):
            return True, "ОК (IPv6 regex)"
        else:
            return False, "Не IPv6 (regex)"

def is_likely_snils(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    cleaned_text = re.sub(r'[-\s]', '', text)
    if not cleaned_text.isdigit():
        return False, "СНИЛС не только цифры"
    if len(cleaned_text) != 11:
        return False, f"СНИЛС не 11 цифр (длина {len(cleaned_text)})"
    return check_snils_checksum(cleaned_text)

def is_likely_date_day(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if text.isdigit():
        try:
            day = int(text)
            if 1 <= day <= 31:
                return True, "ОК (День)"
            else:
                return False, f"Не день (диапазон {day})"
        except ValueError:
            return False, "Не день (не число)"
    return False, "Не день (не цифры)"

def is_likely_date_month(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    text_lower = text.lower()
    if text_lower.isdigit():
        try:
            month = int(text_lower)
            if 1 <= month <= 12:
                return True, "ОК (Месяц числом)"
            else:
                return False, f"Не месяц (число диапазон {month})"
        except ValueError:
             return False, "Не месяц (число не конверт.)"
    else:
        common_months_ru = [
            "январь", "февраль", "март", "апрель", "май", "июнь",
            "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря"
        ]
        if text_lower in common_months_ru:
            return True, f"ОК (Месяц словом: {text})"
        else:
            return False, f"Не месяц (слово не опознано: {text})"
    return False, "Не месяц (неясный формат)" # Эта строка не должна достигаться по идее

def is_likely_date_year(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if text.isdigit() and len(text) == 4:
        try:
            year = int(text)
            if 1900 <= year <= 2099: # Примерный диапазон
                return True, "ОК (Год)"
            else:
                return False, f"Не год (диапазон {year})"
        except ValueError:
            return False, "Не год (не число)"
    return False, "Не год (не 4 цифры)"

def is_likely_number(tokens_list_or_string):
    text: str  # Эта строка будет использоваться для всех проверок на число

    is_list = isinstance(tokens_list_or_string, list)
    # Сначала получаем необработанное строковое представление из входных данных
    raw_str_from_input = "".join(tokens_list_or_string) if is_list else tokens_list_or_string

    # Решаем, заменять ли запятую на точку, в зависимости от контекста
    if is_list and \
       len(tokens_list_or_string) > 0 and \
       tokens_list_or_string[-1] == ',' and \
       len(tokens_list_or_string) > 1 and \
       "".join(tokens_list_or_string[:-1]).strip() != "":
        # Случай: ['ЧАСТЬ_ЧИСЛА', ','] например ['5', ',']
        # Завершающая запятая является отдельным токеном и не должна преобразовываться в точку.
        # Регулярное выражение будет проверяться по строке с буквальной запятой, например, "5,".
        text = raw_str_from_input
    else:
        # Поведение по умолчанию:
        # - Для одиночных строковых вводов типа "3,14", преобразует в "3.14".
        # - Для списковых вводов типа ['3', ',', '14'], объединяет в "3,14", затем преобразует в "3.14".
        # - Для списковых вводов типа ['5'], объединяет в "5", замена не имеет эффекта.
        # - Для списковых вводов типа [','], объединяет в ",", преобразует в ".". (Обрабатывается последующими проверками)
        text = raw_str_from_input.replace(',', '.')
    
    # Улучшенный regex: требует хотя бы одну цифру, если есть точка, или просто цифры.
    # Не должен срабатывать на одиночную '.'
    # Примеры: 5, 5.0, .5, 5., -5, +5.0e-10
    # Не примеры: ., -. , +.
    regex_pattern = r'[-+]?((\d+\.\d*|\d*\.\d+|\d+)([eE][-+]?\d+)?)'
    
    # Дополнительная проверка, чтобы одиночная точка или точка со знаком не считались числом
    if text.strip() in [".", "+.", "-."]:
        return False, "Не число (одиночная точка)"

    if bool(re.fullmatch(regex_pattern, text)):
        return True, "ОК (Число)" 
    else:
        # Попытка определить, почему не совпало (очень упрощенно)
        if not any(char.isdigit() for char in text):
            return False, "Не число (нет цифр)"
        if text.count('.') > 1:
            return False, "Не число (много точек)"
        if 'e' in text.lower() and text.lower().count('e') > 1:
            return False, "Не число (много 'e')"
        return False, "Не число (общий формат)"

# Новая функция для проверки ДЕНЬ-МЕСЯЦ
def is_likely_day_month(tokens_list_or_string):
    # Используем " "join, т.к. regex ожидает пробел
    text = " ".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    # Regex: 1 или 2 цифры, пробел, затем слово (название месяца) или 1-2 цифры (номер месяца)
    # Пример: "29 марта", "29 03", "1 января"
    # Не слишком строгий, чтобы покрыть варианты, но может дать ложные срабатывания на других комбинациях.
    if bool(re.fullmatch(r'\d{1,2}\s+([a-zA-Zа-яА-ЯёЁ]+|\d{1,2})', text.strip())):
        # Дополнительная проверка: если месяц - слово, оно не должно быть слишком коротким (типа "г")
        parts = text.strip().split()
        if len(parts) == 2 and parts[1].isalpha() and len(parts[1]) < 2:
             return False, "Не День-Месяц (месяц-буква?)"
        return True, "ОК (День-Месяц)"
    else:
        return False, "Не День-Месяц (формат)"

def is_likely_postal_code(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if len(text) == 6 and text.isdigit():
        return True, "ОК (Почт. индекс)"
    elif not text.isdigit():
        return False, "Индекс не только цифры"
    else:
        return False, f"Индекс не 6 цифр (длина {len(text)})"

def is_likely_kpp(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if not text.isdigit():
        return False, "КПП не только цифры"
    if len(text) == 9:
        return True, "ОК (КПП)"
    else:
        return False, f"КПП не 9 цифр (длина {len(text)})"

def is_likely_inn_individual(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    # check_inn_checksum уже проверяет isdigit и длину
    return check_inn_checksum(text) 

def is_likely_inn_legal(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    return check_inn_checksum(text)

def is_likely_ogrn(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if not text.isdigit():
        return False, "ОГРН не только цифры"
    if len(text) != 13:
        return False, f"ОГРН не 13 цифр (длина {len(text)})"
    return check_ogrn_checksum(text)

def is_likely_ogrnip(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if not text.isdigit():
        return False, "ОГРНИП не только цифры"
    if len(text) != 15:
        return False, f"ОГРНИП не 15 цифр (длина {len(text)})"
    return check_ogrnip_checksum(text)

def is_likely_mac(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if bool(re.fullmatch(r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})', text)):
        return True, "ОК (MAC)"
    else:
        return False, "Не MAC (формат)"

def is_likely_email(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if USE_EMAIL_VALIDATOR:
        try:
            validate_email(text, check_deliverability=False)
            return True, "ОК (Email validator)"
        except EmailNotValidError as e:
            return False, f"Не Email ({e})"
    else: 
        if bool(re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)):
            return True, "ОК (Email regex)"
        else:
            return False, "Не Email (regex)"

def is_likely_mobile_phone(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    cleaned = re.sub(r'[+\s()\-]', '', text)
    if cleaned.startswith('8') and len(cleaned) == 11:
         cleaned = '7' + cleaned[1:]
    if cleaned.startswith('7') and len(cleaned) == 11 and cleaned.isdigit():
        return True, "ОК (Моб. телефон)"
    elif not cleaned.isdigit():
        return False, "Моб. телефон не только цифры (после очистки)"
    else:
        return False, "Не моб. телефон РФ (формат/длина)"

def is_likely_url(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if bool(re.fullmatch(r'(https?|ftp)://[\w-]+(\.[\w-]+)+([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?|www\.[\w-]+(\.[\w-]+)+([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', text, re.IGNORECASE)):
        return True, "ОК (URL)"
    else:
        return False, "Не URL (формат)"

def is_likely_card_number(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if not text.isdigit():
        return False, "Номер карты не только цифры"
    if not (13 <= len(text) <= 19):
        return False, f"Номер карты неверной длины ({len(text)})"
    return check_luhn(text) 

def is_likely_bik(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if not text.isdigit():
        return False, "БИК не только цифры"
    if len(text) == 9:
        return True, "ОК (БИК)"
    else:
        return False, f"БИК не 9 цифр (длина {len(text)})"

def is_likely_account(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if not text.isdigit():
        return False, "Счет не только цифры"
    if len(text) == 20:
        return True, "ОК (Счет РФ)"
    else:
        return False, f"Счет РФ не 20 цифр (длина {len(text)})"

def is_likely_iban(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if len(text) in [33, 34] and text.startswith('RU') and re.fullmatch(r'RU\d{2}[A-Z0-9]{1,30}', text):
        return True, "ОК (IBAN RU)"
    else:
        return False, "Не IBAN RU (формат)"

def is_likely_license_plate(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if bool(re.fullmatch(r'[АВЕКМНОРСТУХABEKMHOPCTYX]{1}\d{3}[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\d{2,3}|\d{4}[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\d{2,3}', text, re.IGNORECASE)):
        return True, "ОК (Номер РФ авто)"
    else:
        return False, "Не номер РФ авто (формат)"

def is_likely_phone_landline(tokens_list_or_string):
    # Используем " "join для сохранения пробелов, т.к. проверяем наличие скобок/дефисов
    text = " ".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    cleaned = re.sub(r'\D', '', text) 
    has_brackets = '(' in text and ')' in text
    has_hyphen = '-' in text
    if len(cleaned) > 5 and (has_brackets or has_hyphen):
        return True, "ОК (Тел. стац.)"
    else:
        return False, "Не тел. стац. (формат)"

def is_likely_phone_international(tokens_list_or_string):
    # Используем " "join
    text = " ".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    cleaned = re.sub(r'\D', '', text)
    if text.startswith('+') and len(cleaned) > 11 and ('(' in text or '-' in text or ' ' in text):
        return True, "ОК (Тел. межд.)"
    else:
        return False, "Не тел. межд. (формат)"

def is_likely_coordinates(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    parts = re.split(r'[\s,]+', text)
    if len(parts) == 2:
        try:
            float(parts[0])
            float(parts[1])
            return True, "ОК (Координаты)"
        except ValueError:
            return False, "Координаты не числа"
    return False, "Координаты не из 2х частей"

def is_likely_device_id(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if len(text) > 5 and bool(re.fullmatch(r'[a-zA-Z0-9\-]+', text)):
        return True, "ОК (Device ID)"
    else:
        return False, "Не Device ID (формат/длина)"

def is_likely_serial_number(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if len(text) > 5 and bool(re.fullmatch(r'[a-zA-Z0-9\-]+', text)):
        return True, "ОК (S/N)"
    else:
        return False, "Не S/N (формат/длина)"

def is_likely_imei(tokens_list_or_string):
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if not text.isdigit():
        return False, "IMEI не только цифры"
    if len(text) != 15:
        return False, f"IMEI не 15 цифр (длина {len(text)})"
    return check_imei_luhn(text)

def is_likely_oms_policy_number(tokens_list_or_string):
    # Проверяет номер полиса ОМС (16 цифр, алгоритм Луна) для test.py.
    text = "".join(tokens_list_or_string) if isinstance(tokens_list_or_string, list) else tokens_list_or_string
    if not text.isdigit():
        return False, "Полис ОМС не только цифры"
    if len(text) != 16:
        return False, f"Полис ОМС не 16 цифр (длина {len(text)})"
    return check_luhn(text) # check_luhn уже возвращает (bool, comment)

# Адаптированный словарь функций проверки для test.py
# Ключи - это типы сущностей без B-/I- префиксов
# Некоторые сущности из вашего списка (например, USERNAME, PASSWORD, CITY, STREET и т.д.) 
# не имеют строгих regex проверок в main_generate_conll.py, 
# поэтому для них пока не будет специфичных валидаторов, кроме общих (например, NUMBER для части адреса).
# DATE_TIME, DATE_MONTH, DATE_YEAR будут использовать is_likely_date_full, если нет более специфичных.
format_checkers = {
    "IPV4": is_likely_ipv4,
    "IPV6": is_likely_ipv6,
    "SNILS": is_likely_snils,
    "DATE_TIME": None, # Возвращаем DATE_TIME, но без функции проверки (пока)
    "DATE_MONTH": is_likely_date_month,
    "DATE_YEAR": is_likely_date_year,
    "DATE_DAY": is_likely_date_day,
    "DAY_MONTH": is_likely_day_month,
    "NUMBER": is_likely_number,
    "POSTAL_CODE": is_likely_postal_code,
    "KPP": is_likely_kpp,
    "INDIVIDUAL_TAX_ID": is_likely_inn_individual,
    "LEGAL_TAX_ID": is_likely_inn_legal,
    "OGRN": is_likely_ogrn,
    "OGRNIP": is_likely_ogrnip,
    "MAC": is_likely_mac,
    "EMAIL": is_likely_email,
    "MOBILE_PHONE": is_likely_mobile_phone,
    "LANDLINE_PHONE": is_likely_phone_landline,
    "INTERNATIONAL_PHONE": is_likely_phone_international,
    "URL": is_likely_url,
    "CARD_NUMBER": is_likely_card_number,
    "BIK": is_likely_bik,
    "ACCOUNT": is_likely_account,
    "IBAN": is_likely_iban,
    "LICENSE_PLATE": is_likely_license_plate,
    "COORDINATES": is_likely_coordinates,
    "DEVICE_ID": is_likely_device_id,       # Общий regex
    "SERIAL_NUMBER": is_likely_serial_number, # Общий regex
    "IMEI": is_likely_imei,
    "OMS_POLICY_NUMBER": is_likely_oms_policy_number,
    # Сущности без строгих валидаторов в main_generate_conll.py (пока не добавляем или нужны новые функции):
    # SURNAME, NAME, PATRONYMIC, USERNAME, PASSWORD, PIN, TOKEN, 
    # COUNTRY, REGION, CITY, DISTRICT, LOCALITY, STREET, 
    # ORGANIZATION, DEPARTMENT, SUBDIVISION, AUTHORITY, CARD_TYPE
}

# Константа для максимальной длины последовательности
MAX_LENGTH = 164 

def split_to_tokens(text):
    """Разбивает текст на токены так же, как в генераторе."""
    return re.findall(r'\w+|[.,!?;:\-—()]|\S', text, re.UNICODE)

# --- Функция исправления грамматики (SageFredT5) ---
def correct_text_sagemodel(text_to_correct, tokenizer, model):
    if not tokenizer or not model:
        print("Warning: Модель или токенизатор для исправления грамматики не загружены.")
        return text_to_correct
    
    # inputs = tokenizer(text_to_correct, return_tensors="pt", padding="longest", truncation=True, max_length=512) # Используем truncation для очень длинных строк
    inputs = tokenizer(text_to_correct, return_tensors="pt", padding="longest", truncation=False)
    
    current_device = model.device
    inputs = {k: v.to(current_device) for k, v in inputs.items()}
    
    input_length = inputs["input_ids"].size(1)
    # Устанавливаем output_max_length с некоторым запасом, но не слишком большим, чтобы избежать OOM
    output_max_length = min(int(input_length * 1.5) + 10, 1024) # Ограничиваем максимальную длину
    if input_length == 0: # если пустой инпут
        output_max_length = 50 

    try:
        outputs = model.generate(**inputs, max_length=output_max_length, num_beams=3, early_stopping=True) # num_beams для лучшего качества
        corrected_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return corrected_texts[0] if corrected_texts else text_to_correct
    except Exception as e:
        print(f"Error during grammar correction: {e}")
        return text_to_correct # Возвращаем оригинал в случае ошибки

# --- Функция анализа текста (NER) ---
def analyze_text(current_model_ner, current_tokenizer_ner, current_id2label, text_segment):
    # Используем переданные аргументы вместо глобальных
    current_model_ner.eval()
    original_tokens = split_to_tokens(text_segment)
    if not original_tokens:
        return []

    encoded = current_tokenizer_ner(
        [original_tokens],
        is_split_into_words=True,
        return_tensors='pt',
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    # device_ner - это глобальная переменная, содержащая устройство, на которое загружена модель
    input_ids = encoded['input_ids'].to(device_ner) 
    attention_mask = encoded['attention_mask'].to(device_ner)

    # Эта переменная будет содержать ID метки для каждого субтокена после декодирования
    predicted_subtoken_label_ids_for_loop: list[int] # type: ignore

    with torch.no_grad():
        outputs = current_model_ner(input_ids=input_ids, attention_mask=attention_mask) # Получаем pre-CRF логиты

        if hasattr(current_model_ner, 'decode') and callable(getattr(current_model_ner, 'decode')):
            # Модель имеет CRF, используем ее метод decode
            # outputs['logits'] - это эмиссии для CRF
            decoded_paths_batch = current_model_ner.decode(outputs['logits'], mask=attention_mask.bool() if attention_mask is not None else None)
            
            if decoded_paths_batch and len(decoded_paths_batch) > 0:
                # Так как analyze_text обрабатывает один сегмент за раз, батч равен 1
                subtoken_labels_from_crf = decoded_paths_batch[0] 
                
                # Создаем полный список предсказанных ID до MAX_LENGTH для цикла ниже.
                # crf_output_list будет содержать ID метки для каждого из MAX_LENGTH субтокенов.
                crf_output_list = [-100] * MAX_LENGTH  # -100 для специальных/паддинг токенов по умолчанию
                
                # Заполняем crf_output_list реальными предсказаниями из CRF
                # Длина subtoken_labels_from_crf соответствует количеству активных токенов в attention_mask
                active_subtoken_idx = 0
                for i in range(MAX_LENGTH):
                    if attention_mask is not None and attention_mask[0, i].item() == 1:
                        if active_subtoken_idx < len(subtoken_labels_from_crf):
                            crf_output_list[i] = subtoken_labels_from_crf[active_subtoken_idx]
                            active_subtoken_idx += 1
                        else:
                            # Этого не должно произойти, если CRF отработал корректно для всех активных токенов
                            # Если произошло, используем 'O' как запасной вариант
                            # current_id2label здесь еще не определен, но id2label_ner глобальный
                            o_label_id = 0 # По умолчанию ID для 'O' часто 0
                            for k_id, v_label in id2label_ner.items(): # type: ignore
                                if v_label == 'O':
                                    o_label_id = k_id
                                    break
                            crf_output_list[i] = o_label_id 
                    # Если токен неактивен (паддинг/специальный), он останется -100
                predicted_subtoken_label_ids_for_loop = crf_output_list
            else: 
                # CRF декодирование не удалось или вернуло пустой результат, откат к argmax
                print("ПРЕДУПРЕЖДЕНИЕ: CRF декодирование не вернуло результатов. Используется argmax.")
                argmax_preds = torch.argmax(outputs['logits'], dim=2)
                predicted_subtoken_label_ids_for_loop = argmax_preds[0].tolist()
        else: 
            # Модель не имеет CRF-декодера, используем argmax
            print("ИНФОРМАЦИЯ: Модель не имеет CRF-декодера. Используется argmax.")
            argmax_preds = torch.argmax(outputs['logits'], dim=2)
            predicted_subtoken_label_ids_for_loop = argmax_preds[0].tolist()

    word_ids = encoded.word_ids(batch_index=0) # список длиной MAX_LENGTH
    raw_labels_from_model = []
    last_word_id = None
    for i, word_id in enumerate(word_ids): # i - индекс субтокена от 0 до MAX_LENGTH-1
        if word_id is not None and word_id != last_word_id:
            pred_id = predicted_subtoken_label_ids_for_loop[i] # Используем (потенциально CRF-декодированный) ID метки
            
            if pred_id == -100: # Если для этого субтокена нет предсказания (например, паддинг за пределами CRF)
                label = "O" # По умолчанию O
            else:
                # current_id2label здесь еще не определен, но id2label_ner глобальный
                label = id2label_ner.get(pred_id, "O") # type: ignore # Преобразуем ID в текстовую метку
            raw_labels_from_model.append(label)
            last_word_id = word_id
    
    num_effective_tokens = len(raw_labels_from_model)
    effective_tokens = original_tokens[:num_effective_tokens]

    if len(effective_tokens) != len(raw_labels_from_model):
        min_len = min(len(effective_tokens), len(raw_labels_from_model))
        effective_tokens = effective_tokens[:min_len]
        raw_labels_from_model = raw_labels_from_model[:min_len]
        num_effective_tokens = min_len
        if num_effective_tokens == 0: return []

    corrected_model_bio_labels = []
    for k, label in enumerate(raw_labels_from_model):
        if label.startswith("I-"):
            current_type = label[2:]
            if k == 0 or corrected_model_bio_labels[k-1] == "O" or \
               (not corrected_model_bio_labels[k-1].endswith(current_type)):
                corrected_model_bio_labels.append(f"B-{current_type}")
            else:
                corrected_model_bio_labels.append(label)
        else:
            corrected_model_bio_labels.append(label)

    model_analysis_results = []
    visited_indices_stage1 = set()
    for idx in range(num_effective_tokens):
        if idx in visited_indices_stage1:
            continue

        token_text = effective_tokens[idx]
        model_bio_label = corrected_model_bio_labels[idx]
        model_entity_type = None
        model_span_indices = [idx]
        model_span_is_valid = (True, "ОК (проверки временно отключены)") # Всегда считаем валидным для теста

        if model_bio_label.startswith("B-"):
            model_entity_type = model_bio_label[2:]
            current_entity_span_tokens = [token_text]
            
            temp_idx = idx + 1
            while temp_idx < num_effective_tokens and \
                  corrected_model_bio_labels[temp_idx].startswith("I-") and \
                  corrected_model_bio_labels[temp_idx][2:] == model_entity_type:
                current_entity_span_tokens.append(effective_tokens[temp_idx])
                model_span_indices.append(temp_idx)
                temp_idx += 1
            
            # Проверка формата сущности
            # checker_function = format_checkers.get(model_entity_type) # Временно отключаем
            # if checker_function: # Временно отключаем
            #     is_valid_check, comment_check = checker_function(current_entity_span_tokens) # Временно отключаем
            #     model_span_is_valid = (is_valid_check, comment_check) # Временно отключаем
            # else: # Временно отключаем
            #     model_span_is_valid = (True, "ОК (нет чекера)") # Временно отключаем
            model_span_is_valid = (True, "ОК (проверки временно отключены)") # Всегда считаем валидным для теста

            for i_in_span, abs_idx_in_span in enumerate(model_span_indices):
                model_analysis_results.append({
                    "token_text": effective_tokens[abs_idx_in_span],
                    "token_abs_index": abs_idx_in_span,
                    "model_bio_label": corrected_model_bio_labels[abs_idx_in_span],
                    "model_entity_type": model_entity_type,
                    "model_entity_span_indices": list(model_span_indices),
                    "model_span_is_valid_by_own_type": model_span_is_valid[0] if isinstance(model_span_is_valid, tuple) else model_span_is_valid,
                    "model_span_comment": model_span_is_valid[1] if isinstance(model_span_is_valid, tuple) else "ОК (нет чекера)"
                })
                visited_indices_stage1.add(abs_idx_in_span)
        
        elif model_bio_label == "O":
            model_analysis_results.append({
                "token_text": token_text,
                "token_abs_index": idx,
                "model_bio_label": "O",
                "model_entity_type": None,
                "model_entity_span_indices": list(model_span_indices), 
                "model_span_is_valid_by_own_type": None,
                "model_span_comment": None
            })
            visited_indices_stage1.add(idx)
    
    MAX_REGEX_SEARCH_LEN = 7 
    regex_found_in_o_spans = [] 
    
    idx_o = 0
    while idx_o < num_effective_tokens:
        current_token_model_info_for_o_scan = next((info for info in model_analysis_results if info["token_abs_index"] == idx_o), None)
        if not current_token_model_info_for_o_scan or current_token_model_info_for_o_scan["model_bio_label"] != "O":
            idx_o += 1
            continue

        best_regex_match_at_o_start = None 
        for length in range(1, MAX_REGEX_SEARCH_LEN + 1):
            if idx_o + length > num_effective_tokens: break
            
            candidate_tokens_texts_for_o = [] 
            all_o_in_candidate = True
            for k_offset in range(length):
                token_info_for_candidate = next((info for info in model_analysis_results if info["token_abs_index"] == idx_o + k_offset), None)
                if not token_info_for_candidate or token_info_for_candidate["model_bio_label"] != "O":
                    all_o_in_candidate = False
                    break
                candidate_tokens_texts_for_o.append(token_info_for_candidate["token_text"])
            
            if not all_o_in_candidate: break 
            if not candidate_tokens_texts_for_o: continue

            for entity_type_re, checker_func_re in format_checkers.items():
                if not checker_func_re: continue # Пропускаем, если для типа нет функции проверки
                try:
                    is_valid_re, comment_re = checker_func_re(candidate_tokens_texts_for_o)
                    if is_valid_re:
                        if entity_type_re == "NUMBER" and len(candidate_tokens_texts_for_o) > 0 and candidate_tokens_texts_for_o[0] == ".":
                            if idx_o > 0: 
                                prev_token_info = next((info for info in model_analysis_results if info["token_abs_index"] == idx_o - 1), None)
                                if prev_token_info and prev_token_info["token_text"].lower() in ["д", "корп", "кв"]:
                                    continue 
                        
                        if best_regex_match_at_o_start is None or length > best_regex_match_at_o_start[2]: 
                            best_regex_match_at_o_start = (entity_type_re, list(candidate_tokens_texts_for_o), length, comment_re)
                except Exception:
                    pass 
        
        if best_regex_match_at_o_start:
            found_type, tokens_texts, num_found_tokens, comment_from_re_checker = best_regex_match_at_o_start
            regex_found_in_o_spans.append({
                "entity_type": found_type,
                "tokens_texts_list": tokens_texts,
                "abs_start_index": idx_o,
                "num_tokens": num_found_tokens,
                "comment_from_re_checker": comment_from_re_checker
            })
            idx_o += num_found_tokens
        else:
            idx_o += 1

    final_adjudicated_output = [] 
    processed_indices_final = set()

    for i_tok in range(num_effective_tokens):
        if i_tok in processed_indices_final:
            continue

        is_regex_override_in_o = False
        for regex_hit in regex_found_in_o_spans:
            if regex_hit["abs_start_index"] == i_tok:
                entity_type = regex_hit["entity_type"]
                num_tokens_hit = regex_hit["num_tokens"] # Переименовано, чтобы не конфликтовать
                for k_span, token_text_in_hit in enumerate(regex_hit["tokens_texts_list"]):
                    bio = f"B-{entity_type}" if k_span == 0 else f"I-{entity_type}"
                    final_adjudicated_output.append((token_text_in_hit, bio, f"Regex ({entity_type} в O {regex_hit['comment_from_re_checker']})"))
                    processed_indices_final.add(i_tok + k_span)
                is_regex_override_in_o = True
                break
        if is_regex_override_in_o:
            continue
        
        current_token_model_info = next((info for info in model_analysis_results if info["token_abs_index"] == i_tok), None)
        if not current_token_model_info: 
            final_adjudicated_output.append((effective_tokens[i_tok] if i_tok < len(effective_tokens) else "UNK_TOKEN", "O", "Ошибка анализа"))
            processed_indices_final.add(i_tok)
            continue

        token_text = current_token_model_info["token_text"]
        model_bio = current_token_model_info["model_bio_label"] 
        
        if model_bio != "O":
            model_ent_type = current_token_model_info["model_entity_type"]
            model_span_valid = current_token_model_info["model_span_is_valid_by_own_type"]
            model_span_comment = current_token_model_info["model_span_comment"]
            model_span_indices = current_token_model_info["model_entity_span_indices"]
            model_span_tokens_texts = [info["token_text"] for info in model_analysis_results if info["token_abs_index"] in model_span_indices]

            if model_span_valid:
                for abs_idx_in_span in model_span_indices:
                    tok_info_for_span = next(info for info in model_analysis_results if info["token_abs_index"] == abs_idx_in_span)
                    final_adjudicated_output.append((tok_info_for_span["token_text"], tok_info_for_span["model_bio_label"], f"Модель ({model_ent_type} {model_span_comment})"))
                    processed_indices_final.add(abs_idx_in_span)
            else: 
                corrected_by_other_regex = False
                for re_type_candidate, re_checker_candidate in format_checkers.items():
                    if not re_checker_candidate or re_type_candidate == model_ent_type: continue 
                    try:
                        is_valid_re, comment_re = re_checker_candidate(model_span_tokens_texts) 
                        if is_valid_re:
                            for k_span_idx, abs_idx_in_span_re_corr in enumerate(model_span_indices):
                                bio = f"B-{re_type_candidate}" if k_span_idx == 0 else f"I-{re_type_candidate}"
                                final_adjudicated_output.append((model_span_tokens_texts[k_span_idx], bio, f"Regex (испр. {model_ent_type} на {re_type_candidate} {comment_re})"))
                                processed_indices_final.add(abs_idx_in_span_re_corr)
                            corrected_by_other_regex = True
                            break 
                    except Exception:
                        pass
                
                if not corrected_by_other_regex:
                    for abs_idx_in_span_not_corr in model_span_indices:
                        tok_info_for_span = next(info for info in model_analysis_results if info["token_abs_index"] == abs_idx_in_span_not_corr)
                        final_adjudicated_output.append((tok_info_for_span["token_text"], "O", f"Модель ({model_ent_type} ошибка -> O)"))
                        processed_indices_final.add(abs_idx_in_span_not_corr)
        else: 
            final_adjudicated_output.append((token_text, "O", "Модель (O)"))
            processed_indices_final.add(i_tok)
            
    return final_adjudicated_output

# --- Словарь с примерами замен (заглушка) ---
# Ключи - типы сущностей (без B-/I-), значения - список кортежей токенов для замены
# Важно: для корректной морфологической замены здесь должны быть канонические формы
SAMPLE_REPLACEMENT_ENTITIES = {
    "PER": [("Александр", "Сергеевич", "Пушкин"), ("Мария", "Викторовна", "Склодовская-Кюри"), ("Лев", "Николаевич", "Толстой")],
    "LOC": [("город", "Москва"), ("река", "Нева"), ("гора", "Эльбрус")],
    "ORG": [("Компания", "Яндекс"), ("Университет", "ИТМО"), ("Газета", "Известия")],
    "DATE_DAY": [("первое",), ("десятое",), ("тридцать", "первое")], # Числа как строки
    "DATE_MONTH": [("января",), ("февраля",), ("марта",)],
    "DATE_YEAR": [("две", "тысячи", "двадцать", "третьего"), ("тысяча", "девятьсот", "девяносто", "девятого")],
    # ... другие типы сущностей по необходимости
}

# --- Функция замены сущностей с учетом морфологии (Natasha) ---
def replace_entities_with_morph_agreement(adjudicated_results, current_morph_vocab, current_morph_tagger, replacement_options):
    output_with_replacements = []
    processed_indices_for_replacement = set()
    
    # Сначала группируем токены по сущностям
    entity_spans = []
    current_span = None
    for i, (token_text, bio_label, comment) in enumerate(adjudicated_results):
        if bio_label.startswith("B-"):
            if current_span:
                entity_spans.append(current_span)
            entity_type = bio_label[2:]
            current_span = {"type": entity_type, "tokens": [(token_text, bio_label, comment, i)], "original_indices": [i]}
        elif bio_label.startswith("I-") and current_span and bio_label[2:] == current_span["type"]:
            current_span["tokens"].append((token_text, bio_label, comment, i))
            current_span["original_indices"].append(i)
        else:
            if current_span:
                entity_spans.append(current_span)
            current_span = None
            # Добавляем O-токены или токены, не попавшие в обработку спанов напрямую
            # output_with_replacements.append((token_text, bio_label, comment)) # Будет обработано ниже
    if current_span: # Добавляем последний спан, если он был
        entity_spans.append(current_span)

    # Теперь итерируемся по исходным результатам и заменяем спаны
    # Это более сложный способ обеспечить порядок, можно проще, если аккуратно строить новый список
    
    temp_output = list(adjudicated_results) # Копируем, чтобы изменять

    for span_info in entity_spans:
        entity_type = span_info["type"]
        original_token_objects = span_info["tokens"]
        
        if entity_type in replacement_options and replacement_options[entity_type]:
            replacement_token_texts_canonical = list(random.choice(replacement_options[entity_type]))
            
            original_span_text = " ".join([t[0] for t in original_token_objects])
            doc = Doc(original_span_text)
            # Удаляем сегментацию внутри спана, она здесь не нужна
            # doc.segment(Segmenter()) 
            doc.tag_morph(current_morph_tagger)

            new_span_tokens_inflected = []
            for i, orig_token_info in enumerate(original_token_objects):
                original_token_abs_index = orig_token_info[3]
                
                if i < len(replacement_token_texts_canonical) and i < len(doc.tokens):
                    replacement_canonical_form = replacement_token_texts_canonical[i]
                    original_natasha_token = doc.tokens[i] # Токен Наташи для оригинального слова
                    
                    target_grammemes = set()
                    if original_natasha_token.feats:
                        if 'Case' in original_natasha_token.feats: target_grammemes.add(original_natasha_token.feats['Case'].lower())
                        if 'Number' in original_natasha_token.feats: target_grammemes.add(original_natasha_token.feats['Number'].lower())
                        if 'Gender' in original_natasha_token.feats: target_grammemes.add(original_natasha_token.feats['Gender'].lower())
                    
                    # Пытаемся инфлектировать токен замены
                    # Для этого нужно создать временный Doc для слова замены, чтобы его распарсить Natasha
                    replacement_doc_temp = Doc(replacement_canonical_form)
                    replacement_doc_temp.tag_morph(current_morph_tagger)
                    
                    inflected_word = replacement_canonical_form # По умолчанию
                    if replacement_doc_temp.tokens:
                        parsed_replacement_token = replacement_doc_temp.tokens[0]
                        try:
                            inflected_forms = parsed_replacement_token.inflect(current_morph_vocab, target_grammemes)
                            if inflected_forms:
                                inflected_word = inflected_forms[0].text # Берем первую форму
                        except Exception as e_inflect:
                            # print(f"DEBUG: Could not inflect '{replacement_canonical_form}' with grammemes {target_grammemes}: {e_inflect}")
                            pass # Оставляем каноническую форму
                    
                    new_span_tokens_inflected.append(inflected_word)
                elif i < len(replacement_token_texts_canonical): 
                    # Если оригинальных токенов Наташи меньше, но есть еще слова для замены (редкий случай при текущей логике)
                    new_span_tokens_inflected.append(replacement_token_texts_canonical[i])
                # else: если слов для замены меньше, чем в оригинальном спане, то оригинальные токены не будут заменены
            
            # Заменяем токены в temp_output
            # Важно: эта логика предполагает, что количество токенов в замене равно количеству токенов в оригинале
            # или что мы берем только первые N токенов замены.
            for i, original_token_info in enumerate(original_token_objects):
                original_abs_idx = original_token_info[3]
                if i < len(new_span_tokens_inflected):
                    new_text = new_span_tokens_inflected[i]
                    # BIO метка остается той же, что у оригинального токена в спане
                    temp_output[original_abs_idx] = (new_text, original_token_info[1], f"Замена ({entity_type})")
                # Если замена короче оригинала, остальные оригинальные токены остаются как есть, но это не идеально.
                # По-хорошему, их надо либо удалять, либо менять их BIO-метку на 'O'

    return temp_output

if __name__ == "__main__":
    # --- Загрузка NER модели ---
    model_ner_dir = "./model" 
    if not os.path.isdir(model_ner_dir):
        print(f"Ошибка: Директория с NER моделью не найдена: {model_ner_dir}")
    else:
        print(f"Загрузка NER модели и токенизатора из {model_ner_dir}...")
        try:
            # 1. Загружаем конфигурацию
            model_config = AutoConfig.from_pretrained(model_ner_dir)
            
            original_base_model_path = "DeepPavlov/rubert-base-cased" 

            if model_config.architectures and model_config.architectures[0] == "SpanBertLSTMForTokenClassification":
                print(f"DEBUG: Явное создание SpanBertLSTMForTokenClassification с base_model_path='{original_base_model_path}'")
                model_ner = SpanBertLSTMForTokenClassification(
                    base_model_path=original_base_model_path,
                    num_labels=model_config.num_labels,
                    config=model_config,
                    lstm_hidden_size=getattr(model_config, 'lstm_hidden_size', model_config.hidden_size // 2), 
                    dropout_rate=getattr(model_config, 'dropout_rate', 0.2) 
                )
            elif model_config.architectures and model_config.architectures[0] == "SpanBertForTokenClassificationCRF":
                print(f"DEBUG: Явное создание SpanBertForTokenClassificationCRF с base_model_path='{original_base_model_path}'")
                model_ner = SpanBertForTokenClassificationCRF(
                    base_model_path=original_base_model_path,
                    num_labels=model_config.num_labels,
                    config=model_config
                )
            else:
                print(f"ОШИБКА: Неизвестная или неподдерживаемая архитектура в config.json: {model_config.architectures}. Загрузка через AutoModelForTokenClassification.")
                model_ner = AutoModelForTokenClassification.from_pretrained(model_ner_dir)

            state_dict_path = os.path.join(model_ner_dir, "pytorch_model.bin")
            if os.path.exists(state_dict_path) and isinstance(model_ner, (SpanBertLSTMForTokenClassification, SpanBertForTokenClassificationCRF)):
                print(f"DEBUG: Загрузка state_dict из {state_dict_path} в экземпляр {type(model_ner)}")
                map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                model_ner.load_state_dict(torch.load(state_dict_path, map_location=map_location))
                print(f"DEBUG: state_dict успешно загружен.")
            elif not isinstance(model_ner, (SpanBertLSTMForTokenClassification, SpanBertForTokenClassificationCRF)):
                print(f"ПРЕДУПРЕЖДЕНИЕ: model_ner не является ожидаемым кастомным типом ({type(model_ner)}), state_dict не загружен явно.")
            else:
                print(f"ОШИБКА: Файл весов pytorch_model.bin не найден в {state_dict_path}. Веса не загружены.")

            print(f"DEBUG: Тип загруженной model_ner ПОСЛЕ явного создания: {type(model_ner)}")
            tokenizer_ner = AutoTokenizer.from_pretrained(model_ner_dir)
            id2label_ner = {int(k): v for k, v in model_ner.config.id2label.items()} 
            device_ner = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_ner.to(device_ner)
            print(f"NER модель и токенизатор успешно загружены на устройство: {device_ner}.")
        except Exception as e:
            print(f"Критическая ошибка при загрузке NER модели: {e}")
            model_ner = None 
    
    # --- Загрузка модели для исправления грамматики (SageFredT5) ---
    corrector_model_name = "ai-forever/sage-fredt5-distilled-95m"
    print(f"Загрузка модели исправления грамматики {corrector_model_name}...")
    try:
        tokenizer_corrector = AutoTokenizer.from_pretrained(corrector_model_name)
        model_corrector = AutoModelForSeq2SeqLM.from_pretrained(corrector_model_name)
        if torch.cuda.is_available():
            device_corrector = torch.device("cuda")
            print("Попытка загрузить модель корректора на GPU.")
        else:
            device_corrector = torch.device("cpu")
            print("Модель корректора будет использовать CPU.")
        model_corrector.to(device_corrector)
        print(f"Модель исправления грамматики успешно загружена на устройство: {device_corrector}.")
    except Exception as e:
        print(f"Ошибка при загрузке модели исправления грамматики: {e}")
        print("Предобработка грамматики будет недоступна.")
        model_corrector = None
        tokenizer_corrector = None

    # --- Проверка, загружена ли хотя бы NER модель --- 
    if not model_ner:
        print("Критическая ошибка: NER модель не загружена. Выход.")
        exit()
        
    # --- Основной цикл обработки ввода --- 
    print("\nМодели загружены. Введите текст для анализа (или 'выход' для завершения).")
    while True:
        try:
            input_text = input("> ")
            if input_text.lower() == 'выход':
                break
            if not input_text.strip():
                continue
                
            print("--- Начинаю обработку... ---")
            
            # 1. Предобработка текста
            corrected_text = input_text
            if model_corrector and tokenizer_corrector:
                print("--- Исправление грамматики... ---")
                corrected_text = correct_text_sagemodel(input_text, tokenizer_corrector, model_corrector)
                print(f"--- Исправленный текст: {corrected_text} ---")
            else:
                print("--- Исправление грамматики пропущено (модель не загружена) ---")
                
            # 2. NER анализ
            print("--- NER анализ... ---")
            adjudicated_ner_results = analyze_text(model_ner, tokenizer_ner, id2label_ner, corrected_text)
            
            # 3. Замена сущностей
            print("--- Замена сущностей... ---")
            final_results_after_replacement = replace_entities_with_morph_agreement(
                adjudicated_ner_results, 
                morph_vocab, 
                morph_tagger, 
                SAMPLE_REPLACEMENT_ENTITIES 
            )
            
            # 4. Формирование и вывод результата
            output_data = [
                {"token": res[0], "label": res[1], "comment": res[2]} 
                for res in final_results_after_replacement
            ]
            print("--- Результат обработки: ---")
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
            print("\nВведите следующий текст или 'выход':")
            
        except Exception as e:
            print(f"Произошла ошибка во время обработки: {e}")
            # Можно добавить вывод traceback при необходимости
            # import traceback
            # traceback.print_exc()
            print("Пожалуйста, попробуйте еще раз или введите 'выход'.")
            
    print("Завершение работы.")

    # --- Удаляем запуск Flask --- 
    # print("Запуск Flask-сервера...")
    # app.run(debug=True, host='0.0.0.0', port=5000) 