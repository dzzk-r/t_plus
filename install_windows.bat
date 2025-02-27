@echo off
echo 🔧 Installing dependencies...

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
python -m pip install -r requirements.txt

:: Check for CUDA
if "%FORCE_CUDA%"=="1" (
    echo ✅ CUDA enabled for NVIDIA GPU
)

echo ✅ Installation complete!
