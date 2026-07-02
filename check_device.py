import torch
import warnings

def get_best_device():
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print("-" * 40)

    # 1. Try CUDA first
    if torch.cuda.is_available():
        try:
            # Force initialization to catch driver mismatch
            _ = torch.cuda.get_device_name(0)
            test = torch.tensor([1.0], device="cuda")
            print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"❌ CUDA detected but failed: {e}")
            print("\n🔧 Likely cause: NVIDIA driver is too old for this PyTorch build.")
            return None

    # 2. MPS (Apple Silicon — not applicable here but kept for completeness)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU)")
        return torch.device("mps")

    # 3. CPU fallback
    print("✅ CPU (fallback)")
    return torch.device("cpu")

device = get_best_device()

if device is None:
    print("\n" + "=" * 50)
    print("QUICK FIXES:")
    print("=" * 50)
    print("\nOption A — Update NVIDIA driver (recommended):")
    print("  sudo apt update")
    print("  sudo ubuntu-drivers autoinstall")
    print("  sudo reboot")
    print("\nOption B — Reinstall PyTorch matching your current driver:")
    print("  1. Check your driver: nvidia-smi")
    print("  2. Go to https://pytorch.org/get-started/locally/")
    print("     and pick the CUDA version that matches your driver.")
    print("\nOption C — Force CPU for now:")
    device = torch.device("cpu")

print(f"\nUsing device: {device}")
x = torch.rand(2, 3, device=device)
print(f"Test tensor device: {x.device}")