# Torch TTS Setup

This repository provides setup instructions and files for `torch` and `TTS` on Python 3.10.16.

## 1️⃣ Installation

### 🔹 1.1 Install `pyenv`

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

### 🔹 1.2 Install Python 3.10.16
```bash
pyenv install 3.10.16
pyenv local 3.10.16
# Bind Python version to the current folder
```

### 🔹 1.3 Create and Activate Virtual Environment
```bash
pyenv exec python -m venv venv
source venv/bin/activate
```

**For Windows (PowerShell):**
```powershell
pyenv exec python -m venv venv
venv\Scripts\Activate
```

### 🔹 1.4 Install Dependencies
```bash
pip install -r requirements.txt
```

**Update dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

### 🔹 1.5 Verify `torch` Version
```bash
python -c "import torch; print(torch.__version__)"
```

### 🔹 1.6 Check Available Devices (CUDA/MPS/CPU)
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

📌 **If CUDA and MPS are unavailable, CPU will be used.**

---
## 2️⃣ Install `torch` with CUDA (Windows/Linux with NVIDIA GPU Only)
```bash
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

📌 On macOS, use the standard `torch` version since CUDA is not supported.

---
## 3️⃣ Run Test Speech Synthesis
```bash
python gen.py
```

---
## 4️⃣ Automatic Setup
To quickly set up the environment, use the scripts:

**Linux/macOS:**
```bash
bash setup.sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

---
## 5️⃣ 🚀 Running the Script with Different Parameters

**For macOS (MPS)**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 FORCE_DEVICE="mps" pyenv exec python gen.py "Hello, world!"
```

🔹 **PYTORCH_ENABLE_MPS_FALLBACK=1** — Enables CPU fallback for unsupported MPS operations.  
🔹 **FORCE_DEVICE="mps"** — Uses Apple Metal (MPS).

---
**For Windows/Linux with NVIDIA (CUDA)**
```bash
FORCE_DEVICE="cuda" pyenv exec python gen.py "Hello, world!"
```

🔹 **FORCE_DEVICE="cuda"** — Enables GPU (NVIDIA).  
🔹 **If CUDA is unavailable, falls back to CPU.**

---
**For CPU-only mode**
```bash
FORCE_DEVICE="cpu" pyenv exec python gen.py "Hello, CPU!"
```

🔹 **FORCE_DEVICE="cpu"** — Forces CPU-only execution.

---
**For RTX 4090 / 4080 / 3090 Optimization**
```bash
FORCE_DEVICE="cuda" python gen.py "Optimized for 4090!"
```

🔹 Optimization via `torch.backends.cudnn.benchmark = True`.

---
**General Execution Without Specifying Device (Auto-Selection)**
```bash
pyenv exec python gen.py "Hello, auto mode!"
```

🔹 **The script will auto-detect** `cuda`, `mps`, or `cpu` if `FORCE_DEVICE` is not specified.

---
## 🔍 6️⃣ Check Device Availability Before Running:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

📌 **If CUDA and MPS are unavailable, CPU will be used.**

🔥 **Now the script is universal for macOS (MPS), Windows/Linux (CUDA), and CPU!** 🚀

---
# 7️⃣ Text-to-Speech Generation with TTS

## 7.1 Usage

**Generate Speech from Text**
```bash
FORCE_DEVICE="cpu" pyenv exec python gen.py 'Hello, automatic mode!'
```

---
## 7.2 Example Logs

```plaintext
2025-02-27 03:21:07,401 - INFO -

===== New Script Execution =====
🖥️ Using Model Device: cpu
🎙 Using Audio Device: cpu
2025-02-27 03:21:12,717 - INFO - Using Model Device: cpu
2025-02-27 03:21:12,717 - INFO - Using Audio Device: cpu
> tts_models/multilingual/multi-dataset/xtts_v2 is already downloaded.
> Using model: xtts
⏳ Model Load Time: 22.58 sec
2025-02-27 03:21:35,298 - INFO - Model Load Time: 22.58 sec
🎙️ Using Speaker Sample: daniel.wav
📢 Text to Generate: Hello, automatic mode!
> Processing time: 8.536 sec

🔍 **Detailed Metrics:**
⏳ Model Load Time: 22.58 sec
⚙️ Audio Generation Time: 8.54 sec
📊 FPS (Frames Per Second): 0.12
💾 Memory Usage (Before/After): 3571.59 MB → 3658.97 MB
📂 Output File Size: 0.22 MB
⚡ CPU Load: 25.3%
🖥️ GPU: N/A
🔥 TFLOPS (Theoretical): 0.00 TFLOPS
✅ File `output.wav` Generated!
2025-02-27 03:21:43,842 - INFO - File `output.wav` Generated!
```

---
📌 **Additional Notes**
1. **For CUDA Performance Optimization** — `torch.backends.cudnn.benchmark = True` (already enabled in code).
2. **For macOS MPS Compatibility** — `PYTORCH_ENABLE_MPS_FALLBACK=1` (enabled by default).
3. **If `gen.py` crashes due to dependency issues** — Try updating dependencies:
```bash
pip install --upgrade -r requirements.txt
```

🎉 **Now your project is ready for text-to-speech on any device!** 🚀🔥

