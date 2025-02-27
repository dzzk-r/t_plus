#!/bin/bash
set -e

# Проверяем, установлен ли pyenv
if ! command -v pyenv &> /dev/null; then
    echo "pyenv не найден. Устанавливаем..."
    curl https://pyenv.run | bash
    exec $SHELL
fi

# Устанавливаем Python через pyenv
pyenv install -s 3.10.16
pyenv local 3.10.16  # Привязка версии Python к текущей папке

# Создание и активация виртуального окружения
pyenv exec python -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Установка Torch с поддержкой CUDA (если Linux и есть NVIDIA GPU)
if [[ $(uname) == "Linux" && -n "$(command -v nvidia-smi)" ]]; then
    pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
fi

echo "✅ Установка завершена!"
