"""Test setinden rastgele 200 çifti inceleme için CSV'ye aktarma betiği."""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl

DEFAULT_TEST_PATH = Path("data/gold/translit_test.jsonl")
DEFAULT_OUTPUT_CSV = Path("data/gold/review_200.csv")
DEFAULT_SEED = 7
DEFAULT_SAMPLE_SIZE = 200


def sample_test_pairs(
    test_path: Path = DEFAULT_TEST_PATH,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> List[Dict[str, Any]]:
    """translit_test.jsonl içerisinden belirtilen seed ile rastgele örneklem alır ve CSV olarak yazar."""
    if not test_path.exists():
        raise FileNotFoundError(f"Test dosyası bulunamadı: {test_path}")

    records = list(read_jsonl(test_path))
    if len(records) < sample_size:
        sample_size = len(records)

    rng = random.Random(seed)
    sampled = rng.sample(records, sample_size)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "arap", "latin", "skor", "tr_baslik"]

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sampled:
            writer.writerow({
                "id": r.get("id", ""),
                "arap": r.get("arap", ""),
                "latin": r.get("latin", ""),
                "skor": r.get("skor", 0.0),
                "tr_baslik": r.get("tr_baslik", ""),
            })

    return sampled


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Altın test setinden rastgele 200 çifti inceleme için CSV'ye aktar")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="Test seti jsonl dosya yolu")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Çıktı CSV dosya yolu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Rastgelelik tohumu (varsayılan: 7)")
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Örneklem büyüklüğü (varsayılan: 200)")
    args = parser.parse_args()

    sampled = sample_test_pairs(
        test_path=args.test_path,
        output_csv=args.output_csv,
        seed=args.seed,
        sample_size=args.size,
    )
    print(f"Başarıyla {len(sampled)} kayıt '{args.output_csv}' dosyasına yazıldı (seed={args.seed}).")


if __name__ == "__main__":
    main()
