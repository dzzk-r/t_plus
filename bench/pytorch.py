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
