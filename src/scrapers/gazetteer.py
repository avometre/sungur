"""OttomanNLP yer adları sözlüğü (Gazetteer) toplayıcısı.

Hugging Face üzerindeki OttomanNLP/ottoman-place-names-gazetteer veri setini indirir,
Osmanlıca yer adlarını ve Latin transkripsiyonlarını ayrıştırıp standart JSONL şemasına dönüştürür.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import pandas as pd
from huggingface_hub import hf_hub_download

from src.common.io import write_jsonl
from src.common.normalize import normalize_latin, normalize_ottoman

logger = logging.getLogger(__name__)

HF_REPO_ID = "OttomanNLP/ottoman-place-names-gazetteer"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"
DATASET_LICENSE = "cc-by-nc-4.0"


def parse_gazetteer_row(
    row: dict[str, Any], idx: int = 0, repo_url: str = HF_DATASET_URL
) -> dict[str, Any]:
    """Gazetteer veri setinin tek bir satırını standart şemaya eşler."""
    osmanlica_raw = row.get("osmanlica", "")
    arap_ham = str(osmanlica_raw).strip() if pd.notna(osmanlica_raw) else ""
    arap = normalize_ottoman(arap_ham)

    transkripsiyon_raw = row.get("transkripsiyon", "")
    latin: Optional[str] = (
        str(transkripsiyon_raw).strip()
        if pd.notna(transkripsiyon_raw) and str(transkripsiyon_raw).strip()
        else None
    )
    latin_sade: Optional[str] = (
        normalize_latin(latin, strip_diacritics=True) if latin else None
    )

    # osmanlica ve transkripsiyon dışındaki olası ek sütunları 'ek' sözlüğüne topla
    ek_keys = [k for k in row.keys() if k not in ("osmanlica", "transkripsiyon")]
    ek = {k: row[k] for k in ek_keys if pd.notna(row[k])}

    return {
        "kaynak": "gazetteer",
        "id": f"gaz:{idx}",
        "arap_ham": arap_ham,
        "arap": arap,
        "latin": latin,
        "latin_sade": latin_sade,
        "tur": "yer_adi",
        "ek": ek,
        "lisans": DATASET_LICENSE,
        "url": repo_url,
    }


def load_gazetteer_file(
    parquet_path: Union[str, Path], limit: Optional[int] = None
) -> Iterator[dict[str, Any]]:
    """Parquet dosyasını okur ve kayıtları standart şemayla üretir."""
    df = pd.read_parquet(parquet_path)
    count = 0
    for idx, row in df.iterrows():
        record = parse_gazetteer_row(dict(row), idx=count)
        yield record
        count += 1
        if limit is not None and count >= limit:
            break


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="OttomanNLP yer adları sözlüğü toplayıcı")
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/gazetteer.jsonl",
        help="Çıktı JSONL dosya yolu",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw/_cache",
        help="Ham indirmelerin saklanacağı önbellek dizini",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Toplanacak maksimum kayıt sayısı (test amaçlı)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help="Hugging Face repo kimliği",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Hugging Face'ten yer adları veri seti indiriliyor: %s", args.repo_id)

    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_file = hf_hub_download(
            repo_id=args.repo_id,
            filename="train.parquet",
            repo_type="dataset",
            local_dir=str(cache_path),
        )
    except Exception as e:
        logger.error("Hugging Face veri seti indirilemedi: %s", e)
        raise

    parquet_file = Path(downloaded_file)
    logger.info("Yerel parquet dosyası hazır: %s", parquet_file)

    # Sütunları ve örnek satırları kontrol et
    df_sample = pd.read_parquet(parquet_file)
    print("\n--- OttomanNLP Yer Adları (Gazetteer) Bilgisi ---")
    print(f"Toplam satır sayısı: {len(df_sample)}")
    print(f"Sütun adları       : {df_sample.columns.tolist()}")
    print("İlk 5 örnek satır:")
    for i, row in df_sample.head(5).iterrows():
        print(f"  [{i}] osmanlica: {row.get('osmanlica')} | transkripsiyon: {row.get('transkripsiyon')}")
    print("------------------------------------------------\n")

    records: list[dict[str, Any]] = []
    with_latin = 0
    for rec in load_gazetteer_file(parquet_file, limit=args.limit):
        records.append(rec)
        if rec.get("latin"):
            with_latin += 1

    out_path = Path(args.out)
    saved_count = write_jsonl(out_path, records)

    print(f"Kayıt tamamlandı: {saved_count} kayıt yazıldı (Latin okunuşu olan: {with_latin}) -> {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
