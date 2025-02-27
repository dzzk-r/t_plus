### **Torch TTS Docker Setup Guide**

This guide provides Docker configurations for running **Torch TTS** across different platforms:
- **CUDA (Linux/Windows with NVIDIA GPU)**
- **MPS (macOS M1/M2/M3)**
- **CPU (Fallback for all systems)**

---

### **📁 Directory Structure**
```
📂 t_plus/
├── 📄 Dockerfile.cuda  # CUDA (Linux/Windows with NVIDIA)
├── 📄 Dockerfile.mps   # MPS (Mac M1/M2/M3)
├── 📄 Dockerfile.cpu   # CPU (Fallback)
├── 📄 gen.py
├── 📄 requirements.txt
├── 📄 setup.sh
├── 📄 setup.ps1
└── ... (other project files)
```

---

## **🛠️ 1️⃣ CUDA Dockerfile (Linux/Windows with NVIDIA GPU)**

**📄 `Dockerfile.cuda`**
@@@bash
# Build and run on Linux/Windows with NVIDIA GPU (CUDA)
docker build -t t_plus_cuda -f Dockerfile.cuda .
docker run --gpus all --rm -it -v $(pwd):/app -w /app t_plus_cuda "Hello from CUDA!"
@@@

🔹 **Ensures access to the NVIDIA GPU**  
🔹 Uses `nvidia/cuda:12.1.1-devel-ubuntu22.04` base image  
🔹 **Requires** `--gpus all` flag when running

---

## **🍏 2️⃣ MPS Dockerfile (macOS M1/M2/M3 - Apple Metal GPU)**

**📄 `Dockerfile.mps`**
@@@bash
# Build and run on macOS (MPS)
docker build -t t_plus_mps -f Dockerfile.mps .
docker run --device=/dev/dri --privileged --rm -it -v $(pwd):/app -w /app t_plus_mps "Hello from MPS!"
@@@

🔹 Uses `pytorch/pytorch:2.1.0`  
🔹 Includes `PYTORCH_ENABLE_MPS_FALLBACK=1` to fallback to CPU for unsupported operations  
🔹 **Requires** `--device=/dev/dri --privileged` flag on macOS

---

## **🖥️ 3️⃣ CPU Dockerfile (Universal Fallback)**

**📄 `Dockerfile.cpu`**
@@@bash
# Build and run on CPU-only
docker build -t t_plus_cpu -f Dockerfile.cpu .
docker run --rm -it -v $(pwd):/app -w /app t_plus_cpu "Hello from CPU!"
@@@

🔹 Uses `python:3.10` base image  
🔹 Does **not** require any special hardware

---

## **🚀 Build All Docker Images**
If you want to **build all three images** at once, create `build_all.sh`:

@@@bash
#!/bin/bash
set -e

echo "🔨 Building all Docker images..."
docker build -t t_plus_cuda -f Dockerfile.cuda .
docker build -t t_plus_mps -f Dockerfile.mps .
docker build -t t_plus_cpu -f Dockerfile.cpu .
echo "✅ All images built!"
@@@

Run it:
@@@bash
chmod +x build_all.sh
./build_all.sh
@@@

🔥 **Now you have three Docker images ready for any platform!** 🚀