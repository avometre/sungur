import sys
import warnings

# PyTorch numpy bulunmadığında internal UserWarning vermesin diye filtreliyoruz
warnings.filterwarnings("ignore", category=UserWarning)

import torch


def main() -> None:
    print(f"Python sürümü: {sys.version.split()[0]}")
    print(f"torch sürümü: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_avail}")

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16_sup = torch.cuda.is_bf16_supported()

        print(f"GPU adı: {gpu_name}")
        print(f"Toplam VRAM (GB): {vram_gb:.2f}")
        print(f"torch.cuda.is_bf16_supported(): {bf16_sup}")

        # GPU üzerinde 2048x2048 bir matris çarpımı yapıp doğrula
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print("GPU OK")
    else:
        print("CUDA mevcut değil!")


if __name__ == "__main__":
    main()
