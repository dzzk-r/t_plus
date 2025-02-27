import logging
import sys
import os

# ==============================
# 🔧 ФИКС ЛОГГЕРА (удаляем все старые обработчики)
# ==============================
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

METRICS_FILE = "metrics.log"

# ==============================
# 📝 Создаем новый логгер
# ==============================
logger = logging.getLogger("TTS_LOGGER")
logger.setLevel(logging.INFO)  # Записываем INFO и выше

# Лог в файл (без буферизации!)
file_handler = logging.FileHandler(METRICS_FILE, mode="a", encoding="utf-8", delay=False)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Лог в консоль
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# Принудительная запись логов в реальном времени
def flush_logs():
    for handler in logger.handlers:
        handler.flush()
    sys.stdout.flush()

# Записываем первый лог
logger.info("\n\n===== Новый запуск скрипта =====")
flush_logs()


import torch
import sys
import time
import os
import psutil
import logging
from TTS.api import TTS
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

# ==============================
# НАСТРОЙКИ ОКРУЖЕНИЯ
# ==============================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Для Mac MPS

# ==============================
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# ==============================
FORCE_DEVICE = os.getenv("FORCE_DEVICE", "").lower()
METRICS_FILE = "metrics.log"  # Файл для логов


# ==============================
# ОПРЕДЕЛЕНИЕ УСТРОЙСТВА ДЛЯ МОДЕЛИ
# ==============================
if FORCE_DEVICE == "cuda" and torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True  # Оптимизация для RTX 4090, 4080, 3090
elif FORCE_DEVICE == "mps" and torch.backends.mps.is_available():
    device = "mps"
elif FORCE_DEVICE == "cpu":
    device = "cpu"
else:
    # Автовыбор, если FORCE_DEVICE не указан
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

# ==============================
# ОПРЕДЕЛЕНИЕ УСТРОЙСТВА ДЛЯ АУДИО ОБРАБОТКИ
# ==============================
if device == "cuda" and torch.cuda.is_available():
    audio_device = "cuda"
elif device == "mps" and torch.backends.mps.is_available():
    audio_device = "mps"
else:
    audio_device = "cpu"

print(f"🖥️ Используем устройство для модели: {device}")
print(f"🎙 Используем устройство для аудио: {audio_device}")
logger.info(f"Используем устройство для модели: {device}")
logger.info(f"Используем устройство для аудио: {audio_device}")

# ==============================
# ЗАГРУЗКА МОДЕЛИ
# ==============================
start_load_time = time.time()
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(device)
end_load_time = time.time()
load_time = end_load_time - start_load_time

print(f"⏳ Время загрузки модели: {load_time:.2f} сек")
logger.info(f"Время загрузки модели: {load_time:.2f} сек")

# ==============================
# ЗАМЕР ИСПОЛЬЗОВАНИЯ ПАМЯТИ
# ==============================
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024)  # MB

# ==============================
# ОБРАБОТКА ГОЛОСОВОГО ФАЙЛА
# ==============================
speaker_wav = "daniel.wav"
if os.path.exists(speaker_wav):
    print(f"🎙️ Используем голос из семпла {speaker_wav}")
else:
    print(f"⚠️  Файл {speaker_wav} не найден. Используется случайный голос.")
    speaker_wav = None

# ==============================
# ТЕКСТ ДЛЯ ГЕНЕРАЦИИ
# ==============================
text = sys.argv[1] if len(sys.argv) > 1 else "Привет, Вася! Как твои дела?"
print(f"📢 Текст для генерации: {text}")

# ==============================
# ЗАМЕР ВРЕМЕНИ ГЕНЕРАЦИИ
# ==============================
start_gen_time = time.time()

tts.tts_to_file(
    text=text,
    file_path="output.wav",
    language="ru",
    speaker_wav=speaker_wav if speaker_wav else None,
    speaker="ru" if not speaker_wav else None,
)

end_gen_time = time.time()
gen_time = end_gen_time - start_gen_time

# ==============================
# ЗАМЕР ПАМЯТИ ПОСЛЕ ГЕНЕРАЦИИ
# ==============================
mem_after = process.memory_info().rss / (1024 * 1024)  # MB

# ==============================
# ПРОВЕРКА РАЗМЕРА ФАЙЛА
# ==============================
file_size = os.path.getsize("output.wav") / (1024 * 1024)  # MB

# ==============================
# FPS и TFLOPS (если CUDA доступна)
# ==============================
fps = 1 / gen_time if gen_time > 0 else 0
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_tf_per_sec = (
                             torch.cuda.get_device_properties(0).multi_processor_count * 64 * 2 * 1.8
                     ) / 1000  # TFLOPS
else:
    gpu_name = "N/A"
    gpu_tf_per_sec = 0

# ==============================
# ВЫВОД МЕТРИК
# ==============================
print("\n🔍 **Детальные метрики:**")
print(f"⏳ Время загрузки модели: {load_time:.2f} сек")
print(f"⚙️ Время генерации аудио: {gen_time:.2f} сек")
print(f"📊 FPS (кадры в секунду): {fps:.2f}")
print(f"💾 Использование памяти (до/после): {mem_before:.2f} MB → {mem_after:.2f} MB")
print(f"📂 Размер выходного файла: {file_size:.2f} MB")
print(f"⚡ CPU загрузка: {psutil.cpu_percent()}%")
print(f"🖥️ GPU: {gpu_name}")
print(f"🔥 TFLOPS (теоретические): {gpu_tf_per_sec:.2f} TFLOPS")

if torch.cuda.is_available():
    print(
        f"🔥 Использование GPU памяти: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB"
    )
    print(
        f"⚡ Свободная память GPU: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB"
    )

# ==============================
# ЛОГИРОВАНИЕ В `metrics.log`
# ==============================
logger.info(f"Время загрузки модели: {load_time:.2f} сек")
logger.info(f"Время генерации аудио: {gen_time:.2f} сек")
logger.info(f"FPS: {fps:.2f}")
logger.info(f"Использование памяти (до/после): {mem_before:.2f} MB → {mem_after:.2f} MB")
logger.info(f"Размер выходного файла: {file_size:.2f} MB")
logger.info(f"CPU загрузка: {psutil.cpu_percent()}%")
logger.info(f"GPU: {gpu_name}")
logger.info(f"TFLOPS (теоретические): {gpu_tf_per_sec:.2f} TFLOPS")

if torch.cuda.is_available():
    logger.info(
        f"Использование GPU памяти: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB"
    )
    logger.info(
        f"Свободная память GPU: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB"
    )

print("✅ Файл output.wav сгенерирован!")
logger.info("Файл output.wav сгенерирован!")

# ==============================
# 💾 Принудительная запись логов
# ==============================
flush_logs()
logging.shutdown()  # Завершаем логирование корректно
sys.exit(0)  # Выход из скрипта