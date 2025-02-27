# Iteration 1
import timeit
import tensorflow as tf
import numpy as np

def create_tf_tensors(device, size=20000):
    with tf.device(device):
        x = tf.random.uniform((size, size), dtype=tf.float32)
        y = tf.random.uniform((size, size), dtype=tf.float32)
    return x, y

def benchmark_op(device, x, y, num_iters=100):
    # Явная синхронизация через контекст выполнения
    @tf.function(jit_compile=True)
    def func():
        return tf.linalg.matmul(x, y)

    # Прогрев
    func()

    # Замер времени
    times = []
    for _ in range(num_iters):
        start = timeit.default_timer()
        func()
        if 'GPU' in device:
            tf.keras.backend.get_value(x)  # Синхронизация для GPU
        times.append(timeit.default_timer() - start)

    return np.median(times)

def benchmark():
    print(f"TensorFlow: {tf.__version__}")
    print(f"Devices: {tf.config.list_physical_devices()}\n")

    # Размер матрицы (увеличиваем для нагрузки)
    size = 20000

    # CPU Benchmark
    x_cpu, y_cpu = create_tf_tensors("/CPU:0", size)
    cpu_time = benchmark_op("/CPU:0", x_cpu, y_cpu)
    print(f"CPU Time: {cpu_time:.6f}s/iter")

    # GPU Benchmark
    gpu_time = np.inf
    if any(d.device_type == 'GPU' for d in tf.config.list_physical_devices()):
        x_gpu, y_gpu = create_tf_tensors("/GPU:0", size)
        gpu_time = benchmark_op("/GPU:0", x_gpu, y_gpu)
        print(f"GPU Time: {gpu_time:.6f}s/iter")
        print(f"Ratio CPU/GPU: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU недоступен!")

if __name__ == "__main__":
    benchmark()
# Iteration 0
# import tensorflow as tf
# import numpy as np
#
# def create_tf_tensors(device, size=20000):
#     with tf.device(device):
#         return (
#             tf.random.uniform((size, size), dtype=tf.float32),
#             tf.random.uniform((size, size), dtype=tf.float32)
#         )
#
# def benchmark():
#     # Конфигурация
#     print(f"TensorFlow: {tf.__version__}")
#     print(f"Devices: {tf.config.list_physical_devices()}\n")
#
#     # CPU benchmark
#     x_cpu, y_cpu = create_tf_tensors("/CPU:0")
#     cpu_time = tf.test.Benchmark().run_op_benchmark(
#         op=lambda: tf.linalg.matmul(x_cpu, y_cpu),
#         min_iters=100
#     )['wall_time']
#
#     # GPU benchmark (только если GPU доступен)
#     gpu_time = np.inf
#     if any('GPU' in d.device_type for d in tf.config.list_physical_devices()):
#         x_gpu, y_gpu = create_tf_tensors("/GPU:0")
#         bench = tf.test.Benchmark()
#         gpu_time = bench.run_op_benchmark(
#             op=lambda: tf.linalg.matmul(x_gpu, y_gpu),
#             min_iters=100
#         )['wall_time']
#
#     # Результаты
#     print(f"CPU Time: {cpu_time:.6f}s/iter")
#     if not np.isinf(gpu_time):
#         print(f"GPU Time: {gpu_time:.6f}s/iter")
#         print(f"Ratio: {cpu_time/gpu_time:.2f}x")
#     else:
#         print("GPU недоступен!")
#
# if __name__ == "__main__":
#     benchmark()
