# PyTorch Apple arm Metal Performance Shader (M1,M2,M3,M...)
It's a quick adoption of next sources:
- https://www.youtube.com/watch?v=cGEIEnekmRM
- https://github.com/svpino/apple-silicon/blob/main/test.ipynb


## Install within project as different `venv` 
```bash
    python -m venv .menv
    source .menv/bin/activate
    pyenv exec pip install -U pip
    pyenv exec pip install torch torchaudio torchvideo
    pyenv exec pip tensorflow tensorflow-macos tensorflow-metal
    pyenv exec pip install tensorflow tensorflow-macos tensorflow-metal
    pyenv exec pip install jax-metal ml_dtypes==0.2.0 jax==0.4.26  jaxlib==0.4.26
```

## IPython Notebook (Jupyter Notebook / iPyNB)

### PyTorch`pytorch.ipynb`
```python
import torch


def create_torch_tensors(device):
    x = torch.rand((10000, 10000), dtype=torch.float32)
    y = torch.rand((10000, 10000), dtype=torch.float32)
    x = x.to(device)
    y = y.to(device)

    return x, y

device = torch.device("cpu")
x, y = create_torch_tensors(device)

%%timeit
x * y

device = torch.device("mps")
x, y = create_torch_tensors(device)

%%timeit
x * y
```

### PyTorch `pytorch.py`

```python
import timeit
import torch

def create_torch_tensors(device):
    """Create tensors directly on the specified device."""
    x = torch.rand((20000, 20000), dtype=torch.float32, device=device)
    y = torch.rand((20000, 20000), dtype=torch.float32, device=device)
    return x, y

# Time multiplication on CPU
device = torch.device("cpu")
x_cpu, y_cpu = create_torch_tensors(device)

# Warm-up to exclude one-time initialization overhead
_ = x_cpu * y_cpu

cpu_time = timeit.timeit(lambda: x_cpu * y_cpu, number=100)
print(f"CPU Time per multiplication: {cpu_time / 100:.6f} seconds")

# Time multiplication on MPS
device = torch.device("mps")
x_mps, y_mps = create_torch_tensors(device)

# Warm-up and synchronize
_ = x_mps * y_mps
torch.mps.synchronize()

def mps_operation():
    _ = x_mps * y_mps
    torch.mps.synchronize()  # Ensure accurate timing for MPS

mps_time = timeit.timeit(mps_operation, number=100)
print(f"MPS Time per multiplication: {mps_time / 100:.6f} seconds")
```

```bash
    pyenv exec python ./bench/pytorch.py`
```

### TensorFlow `ternserflow.ipynb`

```python
import tensorflow as tf


def create_tf_tensors():
    x = tf.random.uniform((10000, 10000), dtype=tf.float32)
    y = tf.random.uniform((10000, 10000), dtype=tf.float32)

    return x, y


x, y = create_tf_tensors()

%%timeit

with tf.device("/CPU:0"):
    x * y

%%timeit

with tf.device("/GPU:0"):
    x * y
```

### TensorFlow `ternserflow.py`



```bash
    pyenv exec python ./bench/tenserflow.py
```

### Tenderflow `tenserflow.ipynb`

```python
    import tensorflow as tf
    
    
    def create_tf_tensors():
        x = tf.random.uniform((20000, 20000), dtype=tf.float32)
        y = tf.random.uniform((20000, 20000), dtype=tf.float32)
    
        return x, y
    
    
    x, y = create_tf_tensors()
    
    %%timeit
    
    with tf.device("/CPU:0"):
        x * y
    
    %%timeit
    
    with tf.device("/GPU:0"):
        x * y
```

### TensorFlow `tenderflow.py`

```python
import timeit
import tensorflow as tf
import os

# --- Диагностика TensorFlow и MPS ---
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# --- Принудительная настройка памяти MPS ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ MPS memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ Could not set memory growth: {e}")

# --- Создание тензоров ---
def create_tf_tensors(device, size=4096):
    """Создает тензоры на указанном устройстве."""
    with tf.device(device):
        x = tf.random.uniform((size, size), dtype=tf.float32)
        y = tf.random.uniform((size, size), dtype=tf.float32)
    return x, y

def benchmark():
    # --- CPU Benchmark ---
    print("\n🔹 **CPU Benchmark**")
    x_cpu, y_cpu = create_tf_tensors("/CPU:0")
    _ = x_cpu * y_cpu  # Прогрев
    cpu_time = timeit.timeit(lambda: x_cpu * y_cpu, number=10)
    print(f"CPU Time (10 итераций): {cpu_time:.4f}s")

    # --- GPU Benchmark (MPS) ---
    if not gpus:
        print("❌ GPU не обнаружено! Пропускаем тест GPU")
        return

    print("\n🔹 **GPU Benchmark (MPS)**")
    x_gpu, y_gpu = create_tf_tensors("/GPU:0")
    _ = x_gpu * y_gpu  # Прогрев
    tf.identity(x_gpu).numpy()  # Синхронизация

    def gpu_op():
        _ = x_gpu * y_gpu
        tf.identity(x_gpu).numpy()  # Синхронизация

    gpu_time = timeit.timeit(gpu_op, number=10)
    print(f"GPU Time (10 итераций): {gpu_time:.4f}s")

    # --- Metal mGPU Benchmark (без `@tf.function`) ---
    print("\n🔹 **mGPU Benchmark (tf.linalg.matmul, NO @tf.function)**")
    def mgpu_op():
        out = tf.linalg.matmul(x_gpu, y_gpu)
        tf.identity(out).numpy()  # Явная синхронизация
        return out

    _ = mgpu_op()  # Прогрев
    mgpu_time = timeit.timeit(mgpu_op, number=10)
    print(f"mGPU Time (10 итераций): {mgpu_time:.4f}s")

    # --- Очистка кэша в конце (важно для MPS) ---
    tf.keras.backend.clear_session()
    print("\n✅ **Benchmark completed!**\n")

if __name__ == "__main__":
    benchmark()
```

```bash
    pyenv exec python ./bench/tenserflow.py`
```

### SURPRISE!

This code wor mGPU on TenserFlow 2.16.2 brought significant degradation!
```bash
    # pyenv exec python ./tenserflow.py                    
    TensorFlow version: 2.16.2
    GPU available: True
    ✅ MPS memory growth enabled
    
    🔹 CPU Benchmark
    CPU Time (10 итераций): 0.0009s
    
    🔹 GPU Benchmark (MPS)
    GPU Time (10 итераций): 0.0795s
    
    🔹 mGPU Benchmark (tf.linalg.matmul, NO @tf.function)
    mGPU Time (10 итераций): 0.1892s
    
    ✅ Benchmark completed!
```

CPU - 0.0009s / GPU - 0.0795s / mGPU - 0.1892! mGPU 210 times slower than CPU!

...go to the grocery store, wanna sleep ASAP... tomorrow
