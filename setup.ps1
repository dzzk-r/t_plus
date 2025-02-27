# Файл: setup.ps1

# Проверяем, установлен ли pyenv
if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
    Write-Output "pyenv не найден. Установите его вручную!"
    exit 1
}

# Устанавливаем Python через pyenv
pyenv install -s 3.10.16
pyenv local 3.10.16  # Привязка версии Python к текущей папке

# Создание и активация виртуального окружения
pyenv exec python -m venv venv
venv\Scripts\Activate

# Установка зависимостей
pip install -r requirements.txt

# Установка Torch с поддержкой CUDA
if ($env:CUDA_PATH) {
    pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
}

Write-Output "✅ Установка завершена!"
