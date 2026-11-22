"""Print actionable dependency and GPU diagnostics before starting the demo."""

import importlib.metadata as metadata
import platform
import sys


def main():
    print("Python:", sys.version.split()[0], "Executable:", sys.executable)
    print("Platform:", platform.platform())
    for package in ("gym", "numpy", "ninja", "isaacgym", "torch"):
        try:
            print(f"{package}: {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            print(f"{package}: NOT INSTALLED")
    # Isaac Gym must be imported before PyTorch.
    from isaacgym import gymapi
    import torch
    print("PyTorch CUDA build:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check nvidia-smi and the driver for the running kernel.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Compiled architectures:", torch.cuda.get_arch_list())
    # Availability alone does not prove that the wheel can execute GPU kernels.
    x = torch.arange(16, dtype=torch.float32, device="cuda:0")
    result = (x.square() + 1).sum().item()
    assert result == 1256.0, result
    print("PASS: CUDA allocation and arithmetic kernels. Next run check_env.py.")


if __name__ == "__main__":
    main()
