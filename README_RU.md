// # Torch TTS Setup

Этот репозиторий предоставляет инструкции и файлы для установки и работы с `torch` и `TTS` на Python 3.10.16.

## 1️⃣ Установка

### 🔹 1.1 Установка `pyenv`

**Windows:**
```bash
    winget install pyenv-win
```

**Ubuntu:**
```bash
    curl https://pyenv.run | bash
```

**macOS:**
```bash
    brew install pyenv
```

### 🔹 1.2 Установка Python 3.10.16
```bash
    pyenv install 3.10.16
pyenv local 3.10.16
# Привязка версии Python к текущей папке
```

### 🔹 1.3 Создание и активация виртуального окружения
```bash
    pyenv exec python -m venv venv
source venv/bin/activate
```

**Для Windows (PowerShell):**
```powershell
pyenv exec python -m venv venv
venv\Scripts\Activate
```

### 🔹 1.4 Установка зависимостей
```bash
    pip install -r requirements.txt
```

**Обновление зависимостей:**
```bash
    pip install --upgrade -r requirements.txt
```

### 🔹 1.5 Проверка версии `torch`
```bash
    python -c "import torch; print(torch.__version__)"
```

### 🔹 1.6 Проверка доступных устройств (CUDA/MPS/CPU)
```bash
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

📌 **Если CUDA и MPS отсутствуют**, будет использован CPU.

---
## 2️⃣ Установка `torch` с CUDA (только для Windows/Linux с NVIDIA GPU)
```bash
    pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

📌 На macOS использовать стандартную версию `torch`, т.к. CUDA не поддерживается.

---
## 3️⃣ Запуск тестового синтеза речи
```bash
    python gen.py
```

---
## 4️⃣ Автоматическая установка
Для быстрой установки можно использовать скрипты:

**Linux/macOS:**
```bash
    bash setup.sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

---
## 5️⃣ 🚀 Запуск скрипта с разными параметрами

**Для macOS (MPS)**
```bash
    PYTORCH_ENABLE_MPS_FALLBACK=1 FORCE_DEVICE="mps" pyenv exec python gen.py "Привет, мир!"
```

🔹 **PYTORCH_ENABLE_MPS_FALLBACK=1** — включает CPU fallback для неподдерживаемых операций на MPS.  
🔹 **FORCE_DEVICE="mps"** — задает использование Apple Metal (MPS).

---
**Для Windows/Linux с NVIDIA (CUDA)**
```bash
    FORCE_DEVICE="cuda" pyenv exec python gen.py "Hello, world!"
```

🔹 **FORCE_DEVICE="cuda"** — включает GPU (NVIDIA).  
🔹 **Если CUDA не доступна**, автоматически переключится на CPU.

---
**Для работы на CPU**
```bash
    FORCE_DEVICE="cpu" pyenv exec python gen.py "Привет, CPU!"
```

🔹 **FORCE_DEVICE="cpu"** — заставляет работать только на CPU.

---
**Для работы с RTX 4090 / 4080 / 3090**
```bash
    FORCE_DEVICE="cuda" python gen.py "Оптимизировано для 4090!"
```

🔹 Оптимизация через `torch.backends.cudnn.benchmark = True` в коде.

---
**Общий вызов без явного указания (автовыбор устройства)**
```bash
    pyenv exec python gen.py "Привет, автоматический режим!"
```

🔹 **Скрипт сам определит** `cuda`, `mps` или `cpu`, если `FORCE_DEVICE` не указан.

---
## 🔍 6️⃣ Проверка доступности устройств перед запуском:
```bash
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

📌 **Если CUDA и MPS отсутствуют**, будет использован CPU.

🔥 **Теперь скрипт универсален для macOS (MPS), Windows/Linux (CUDA) и CPU!** 🚀

---
# 7️⃣ Text-to-Speech Generation with TTS

## 7.1 Использование

**Генерация аудио из текста**
```bash
    FORCE_DEVICE="cpu" pyenv exec python gen.py 'Привет, автоматический режим. В рот мне ноги!'
```

---
## 7.2 Пример логов

```plaintext
2025-02-27 03:21:07,401 - INFO -

===== Новый запуск скрипта =====
🖥️ Используем устройство для модели: cpu
🎙 Используем устройство для аудио: cpu
2025-02-27 03:21:12,717 - INFO - Используем устройство для модели: cpu
2025-02-27 03:21:12,717 - INFO - Используем устройство для аудио: cpu
> tts_models/multilingual/multi-dataset/xtts_v2 is already downloaded.
> Using model: xtts
⏳ Время загрузки модели: 22.58 сек
2025-02-27 03:21:35,298 - INFO - Время загрузки модели: 22.58 сек
🎙️ Используем голос из семпла daniel.wav
📢 Текст для генерации: Привет, автоматический режим. В рот мне ноги!
> Processing time: 8.536 сек

🔍 **Детальные метрики:**
⏳ Время загрузки модели: 22.58 сек
⚙️ Время генерации аудио: 8.54 сек
📊 FPS (кадры в секунду): 0.12
💾 Использование памяти (до/после): 3571.59 MB → 3658.97 MB
📂 Размер выходного файла: 0.22 MB
⚡ CPU загрузка: 25.3%
🖥️ GPU: N/A
🔥 TFLOPS (теоретические): 0.00 TFLOPS
✅ Файл output.wav сгенерирован!
```

---
📌 **Дополнительно**
1. **Для повышения производительности на CUDA** — `torch.backends.cudnn.benchmark = True` (уже включено в коде).
2. **Для работы на Mac MPS** — `PYTORCH_ENABLE_MPS_FALLBACK=1` (включен по умолчанию).
3. **Если `gen.py` падает из-за несовместимости библиотек** — попробуйте обновить зависимости:
```bash
    pip install --upgrade -r requirements.txt
```
    
🎉 **Теперь ваш проект готов к работе с текст-ту-спич на любом устройстве!** 🚀🔥
