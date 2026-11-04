"""Toplanan JSONL veri setleri için kapsamlı özet ve kalite raporu betiği.

Kullanım:
    python scripts/data_report.py data/raw/*.jsonl
"""

import argparse
import glob
import random
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any
from src.common.io import read_jsonl


def is_empty_value(val: Any) -> bool:
    """Değerin boş (None, boş metin, boş liste, boş sözlük) olup olmadığını denetler."""
    if val is None:
        return True
    if isinstance(val, (str, list, dict, set)) and len(val) == 0:
        return True
    return False


def generate_report_for_file(file_path: Path, seed: int = 42) -> None:
    """Tek bir JSONL dosyası için ayrıntılı istatistikleri ve 5 rastgele örneği yazdırır."""
    print("=" * 80)
    print(f"DOSYA: {file_path}")
    print("=" * 80)

    if not file_path.exists():
        print(f"HATA: Dosya bulunamadı: {file_path}\n")
        return

    try:
        records = list(read_jsonl(file_path))
    except Exception as e:
        print(f"HATA: Dosya okunurken hata oluştu: {e}\n")
        return

    total = len(records)
    print(f"Toplam Kayıt Sayısı : {total}")
    if total == 0:
        print("Dosyada kayıt bulunamadı.\n")
        return

    # 1. 'arap' alanına göre tekrar oranı
    arap_list = [r.get("arap", "") for r in records if r.get("arap")]
    unique_arap = len(set(arap_list))
    repeat_count = len(arap_list) - unique_arap
    repeat_rate = (repeat_count / len(arap_list) * 100.0) if arap_list else 0.0

    print(f"Benzersiz 'arap'    : {unique_arap}")
    print(f"Tekrar Eden 'arap'  : {repeat_count} (%{repeat_rate:.2f})")

    # 2. Ortalama 'arap' uzunluğu (karakter sayısı)
    avg_len = (sum(len(a) for a in arap_list) / len(arap_list)) if arap_list else 0.0
    print(f"Ortalama 'arap' Uz. : {avg_len:.2f} karakter")

    # 3. Boş / null alan sayıları
    all_keys: set[str] = set()
    for r in records:
        all_keys.update(r.keys())

    print("\nAlan Boşluk/Null İstatistikleri:")
    for k in sorted(all_keys):
        empty_count = sum(1 for r in records if is_empty_value(r.get(k)))
        pct = (empty_count / total) * 100.0
        print(f"  - {k:<15}: {empty_count:>6} / {total} boş (%{pct:5.1f})")

    # 4. Rastgele 5 örnek (seed=42)
    sample_size = min(5, total)
    rng = random.Random(seed)
    samples = rng.sample(records, sample_size)

    print(f"\nRastgele {sample_size} Örnek (seed={seed}):")
    for idx, s in enumerate(samples, 1):
        print(f"  --- Örnek #{idx} ---")
        for k in sorted(s.keys()):
            val = s[k]
            val_repr = str(val)
            if len(val_repr) > 120:
                val_repr = val_repr[:117] + "..."
            print(f"    {k:<12}: {val_repr}")
    print("\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="JSONL veri setleri özet ve kalite raporu")
    parser.add_argument("patterns", nargs="+", help="Raporlanacak JSONL dosyaları veya glob desenleri")
    parser.add_argument("--seed", type=int, default=42, help="Örnekleme için rastgelelik tohumu")
    args = parser.parse_args()

    files: list[Path] = []
    for pat in args.patterns:
        matched = [Path(p) for p in glob.glob(pat)]
        if matched:
            files.extend(matched)
        else:
            p = Path(pat)
            if p.exists():
                files.append(p)

    if not files:
        print("Belirtilen desene uyan dosya bulunamadı.")
        return

    # Tekrar eden dosya yollarını temizle
    unique_files = sorted(list(set(files)), key=lambda p: str(p))
    for f in unique_files:
        generate_report_for_file(f, seed=args.seed)


if __name__ == "__main__":
    main()
