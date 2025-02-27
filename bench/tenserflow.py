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
