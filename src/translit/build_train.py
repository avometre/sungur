"""Eğitim verisi derleyicisi (çeviriyazı, Arap → Latin kelime düzeyi).

Bu modül:
1. Temiz Vikisözlük (supheli=False) ve yer adları (guvenilir=True) verilerini birleştirir.
2. src/gold/leakage.py ile dev/test setlerinde yer alan her türlü metni kesin olarak eler.
3. Tekrarları birleştirir.
4. Kayıt formatı: {"id", "kaynak", "arap", "latin", "latin_sade", "duzey": "kelime", "lisans"}
5. %2'sini rastgele (seed=0) data/processed/translit_val_kelime.jsonl olarak ayırır.
6. Kalan %98'ini data/processed/translit_train_kelime.jsonl olarak yazar.
7. Lisans dağılımını raporlar.
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_latin, normalize_ottoman
from src.gold.leakage import is_heldout, load_heldout_all

logger = logging.getLogger(__name__)

DEFAULT_WIKTIONARY_PATH = Path("data/interim/wiktionary_clean.jsonl")
DEFAULT_GAZETTEER_PATH = Path("data/interim/gazetteer_clean.jsonl")
DEFAULT_TRAIN_OUTPUT = Path("data/processed/translit_train_kelime.jsonl")
DEFAULT_VAL_OUTPUT = Path("data/processed/translit_val_kelime.jsonl")


def build_training_dataset(
    wiktionary_path: Path = DEFAULT_WIKTIONARY_PATH,
    gazetteer_path: Path = DEFAULT_GAZETTEER_PATH,
    train_output_path: Path = DEFAULT_TRAIN_OUTPUT,
    val_output_path: Path = DEFAULT_VAL_OUTPUT,
    val_ratio: float = 0.02,
    seed: int = 0,
) -> Dict[str, Any]:
    """Kelime düzeyi eğitim ve doğrulama veri setlerini oluşturur."""
    heldout_pages, heldout_arap, heldout_titles, heldout_latin = load_heldout_all()

    # 1. Vikisözlük kayıtlarını topla
    wikt_records = []
    if wiktionary_path.exists():
        for r in read_jsonl(wiktionary_path):
            if not r.get("supheli", False):
                wikt_records.append(r)

    # 2. Yer adları kayıtlarını topla
    gaz_records = []
    if gazetteer_path.exists():
        for r in read_jsonl(gazetteer_path):
            if r.get("guvenilir", False):
                gaz_records.append(r)

    # 3. Sızıntı kontrolü ve tekilleştirme
    leaked_count = 0
    unique_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Vikisözlük öncelikli olarak ekle
    for r in wikt_records:
        ar = normalize_ottoman(r.get("arap", "")).strip()
        lat = normalize_latin(r.get("latin", "")).strip()
        if not ar or not lat:
            continue

        if is_heldout(
            arap=ar,
            latin=lat,
            heldout_pages=heldout_pages,
            heldout_arap=heldout_arap,
            heldout_titles=heldout_titles,
            heldout_latin=heldout_latin,
        ):
            leaked_count += 1
            continue

        pair_key = (ar, lat)
        if pair_key not in unique_pairs:
            unique_pairs[pair_key] = {
                "kaynak": "wiktionary",
                "arap": ar,
                "latin": lat,
                "latin_sade": normalize_latin(lat, strip_diacritics=True),
                "duzey": "kelime",
                "lisans": r.get("lisans", "CC BY-SA 4.0"),
            }

    # Yer adlarını ekle
    for r in gaz_records:
        ar = normalize_ottoman(r.get("arap", "")).strip()
        lat = normalize_latin(r.get("latin", "")).strip()
        if not ar or not lat:
            continue

        if is_heldout(
            arap=ar,
            latin=lat,
            heldout_pages=heldout_pages,
            heldout_arap=heldout_arap,
            heldout_titles=heldout_titles,
            heldout_latin=heldout_latin,
        ):
            leaked_count += 1
            continue

        pair_key = (ar, lat)
        if pair_key not in unique_pairs:
            unique_pairs[pair_key] = {
                "kaynak": "gazetteer",
                "arap": ar,
                "latin": lat,
                "latin_sade": normalize_latin(lat, strip_diacritics=True),
                "duzey": "kelime",
                "lisans": r.get("lisans", "cc-by-nc-4.0"),
            }

    all_items = list(unique_pairs.values())

    # Deterministik sıralama ve id atama
    all_items.sort(key=lambda x: (x["arap"], x["latin"]))
    for idx, item in enumerate(all_items, 1):
        item["id"] = f"word:{idx:06d}"

    # 4. Rastgele train / val ayrımı (seed=0)
    rng = random.Random(seed)
    # Elemanların indekslerini karıştırarak ayır
    indices = list(range(len(all_items)))
    rng.shuffle(indices)

    num_val = int(len(all_items) * val_ratio)
    val_indices = set(indices[:num_val])

    val_records = [all_items[i] for i in range(len(all_items)) if i in val_indices]
    train_records = [all_items[i] for i in range(len(all_items)) if i not in val_indices]

    # Dosyalara yaz
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    val_output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_output_path, train_records)
    write_jsonl(val_output_path, val_records)

    total_license = Counter(item["lisans"] for item in all_items)
    train_license = Counter(item["lisans"] for item in train_records)
    val_license = Counter(item["lisans"] for item in val_records)

    summary = {
        "toplam_kayit": len(all_items),
        "egitim_kayit_sayisi": len(train_records),
        "dogrulama_kayit_sayisi": len(val_records),
        "sizinti_nedeniyle_atilan": leaked_count,
        "lisans_dagilimi_toplam": dict(total_license),
        "lisans_dagilimi_train": dict(train_license),
        "lisans_dagilimi_val": dict(val_license),
        "kaynak_dagilimi": dict(Counter(item["kaynak"] for item in all_items)),
    }
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Kelime düzeyi eğitim verisi derleyicisi.")
    parser.add_argument("--wiktionary", type=Path, default=DEFAULT_WIKTIONARY_PATH)
    parser.add_argument("--gazetteer", type=Path, default=DEFAULT_GAZETTEER_PATH)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--val-out", type=Path, default=DEFAULT_VAL_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("KELİME DÜZEYİ ÇEVİRİYAZI EĞİTİM VERİSİ DERLENİYOR")
    print("=" * 80)

    summary = build_training_dataset(
        wiktionary_path=args.wiktionary,
        gazetteer_path=args.gazetteer,
        train_output_path=args.train_out,
        val_output_path=args.val_out,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Toplam Tekil Çift Sayısı   : {summary['toplam_kayit']:,}")
    print(f"Eğitim Kümesi (Train %98) : {summary['egitim_kayit_sayisi']:,} kayıt")
    print(f"Doğrulama Kümesi (Val %2) : {summary['dogrulama_kayit_sayisi']:,} kayıt")
    print(f"Sızıntı Nedeniyle Atılan  : {summary['sizinti_nedeniyle_atilan']} kayıt")
    print("-" * 80)
    print("Lisans Dağılımı (Toplam):")
    for lic, cnt in summary["lisans_dagilimi_toplam"].items():
        print(f"  - {lic:<20}: {cnt:,} kayıt")
    print("-" * 80)
    print("Kaynak Dağılımı (Toplam):")
    for src, cnt in summary["kaynak_dagilimi"].items():
        print(f"  - {src:<20}: {cnt:,} kayıt")
    print("=" * 80)


if __name__ == "__main__":
    main()
