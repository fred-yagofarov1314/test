# Инструкция по запуску GitGud Anonymizer

## Содержание

1. [Системные требования](#системные-требования)
2. [Быстрый старт](#быстрый-старт)
3. [Подробная инструкция по установке](#подробная-инструкция-по-установке)

## Системные требования

### Минимальные требования:
- Python 3.8+
- 4 ГБ RAM
- 2 ГБ свободного места на диске
- Процессор x86_64 с поддержкой AVX2

### Рекомендуемые требования (для быстрой работы):
- Python 3.10+
- 16+ ГБ RAM
- 8+ ГБ свободного места на диске
- NVIDIA GPU с CUDA поддержкой и 4+ ГБ VRAM
- CUDA 11.7+ и cuDNN

### Поддерживаемые операционные системы:
- Ubuntu 20.04 LTS или новее
- Windows 10/11
- macOS 11 (Big Sur) или новее

## Быстрый старт

### 1. Клонирование репозитория и установка зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/fred-yagofarov1314/test
cd test

# Создание виртуального окружения
python -m venv venv

# Активация окружения
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Распаковка модели

```bash
# Распаковка модели из архива
# Linux/Mac:
7z x model.7z -omodel/

# Windows (с помощью графического интерфейса):
# Распакуйте файл model.7z в папку model/ с помощью 7-Zip или встроенного архиватора
```

### 3. Запуск примера анонимизации

```bash
python text_raw_model_output.py
```

### 4. Использование веб-интерфейса

Для быстрого доступа к анонимизации без локальной установки используйте онлайн веб-интерфейс:

[https://gitgud-site-3ajy.vercel.app/anonymize](https://gitgud-site-3ajy.vercel.app/anonymize)

## Подробная инструкция по установке

### Linux (Ubuntu/Debian)

```bash
# Установка необходимых системных зависимостей
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv p7zip-full git

# Клонирование репозитория
git clone https://github.com/fred-yagofarov1314/test
cd test

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Распаковка модели
7z x model.7z -omodel/

# Проверка установки
python -c "import torch; print('CUDA доступен:', torch.cuda.is_available())"
```

### Windows

1. Установите Python 3.10 с [официального сайта](https://www.python.org/downloads/)
2. Установите 7-Zip с [официального сайта](https://www.7-zip.org/)
3. Установите Git для Windows с [официального сайта](https://git-scm.com/download/win)

```powershell
# Клонирование репозитория
git clone https://github.com/fred-yagofarov1314/test
cd test

# Создание виртуального окружения
python -m venv venv
venv\Scripts\activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Распаковка модели с помощью 7-Zip (через командную строку)
& "C:\Program Files\7-Zip\7z.exe" x model.7z -omodel\

# Проверка установки
python -c "import torch; print('CUDA доступен:', torch.cuda.is_available())"
```

### macOS

```bash
# Установите Homebrew, если его еще нет
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка необходимых зависимостей
brew install python p7zip git

# Клонирование репозитория
git clone https://github.com/fred-yagofarov1314/test
cd test

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Распаковка модели
7z x model.7z -omodel/

# Проверка установки
python -c "import torch; print('MPS доступен:', torch.backends.mps.is_available())"
```