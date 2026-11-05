"""Priority 3 değerlendirme ve tutulan (heldout) veri setlerini işleme modülü.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AKIS_DIR = Path("data/gold/akis_uctan_uca")
CHURRO_DIR = Path("data/gold/churro_ocr")
HELDOUT_LATIN_PATH = Path("data/gold/heldout_latin.txt")
BIBLE_SAMPLE_PATH = Path("data/interim/ota_bible_sample.jsonl")


def process_akis() -> int:
    """Akis veri setini uçtan uca test seti olarak kaydeder ve Latin satırları heldout listesine ekler."""
    AKIS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = AKIS_DIR / "akis_meta.jsonl"

    logger.info("Akis-Ottoman-Dataset yükleniyor...")
    ds = load_dataset("OttomanNLP/Akis-Ottoman-Dataset", split="test")

    latin_sentences = []
    with open(meta_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            txt = item["text"].strip()
            im = item["image"]
            record = {
                "id": f"akis_{idx:05d}",
                "text": txt,
                "width": im.width,
                "height": im.height,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if txt:
                latin_sentences.append(txt)

    logger.info(f"Akis: {len(latin_sentences)} satır meta olarak kaydedildi: {meta_path}")

    # heldout_latin.txt dosyasına yaz
    existing_heldout = set()
    if HELDOUT_LATIN_PATH.exists():
        existing_heldout = set(HELDOUT_LATIN_PATH.read_text(encoding="utf-8").splitlines())

    new_heldout = existing_heldout.union(set(latin_sentences))
    with open(HELDOUT_LATIN_PATH, "w", encoding="utf-8") as f:
        for s in sorted(new_heldout):
            f.write(s + "\n")

    logger.info(f"heldout_latin.txt güncellendi: {len(new_heldout)} cümle koruma altında.")
    return len(latin_sentences)


def process_churro() -> int:
    """CHURRO veri setini OCR test seti olarak kaydeder."""
    CHURRO_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CHURRO_DIR / "churro_meta.jsonl"

    logger.info("CHURRO-Ottoman-Turkish-Subset yükleniyor...")
    ds = load_dataset("OttomanNLP/CHURRO-Ottoman-Turkish-Subset", split="train")

    records = []
    with open(meta_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            raw_txt = item.get("original_transcription", "")
            im = item["image"]
            rec = {
                "id": f"churro_{idx:04d}",
                "width": im.width,
                "height": im.height,
                "karakter_sayisi": len(raw_txt),
                "satir_sayisi": len(raw_txt.splitlines()),
                "metin": raw_txt,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"CHURRO: {len(records)} sayfa OCR test seti olarak kaydedildi: {meta_path}")
    return len(records)


def extract_bible_samples() -> List[Dict[str, Any]]:
    """enesyila/ota-bible-parallel setinden 10 örnek çıkarır."""
    logger.info("ota-bible-parallel yükleniyor...")
    ds = load_dataset("enesyila/ota-bible-parallel", split="train")

    samples = []
    indices = [0, 10, 50, 100, 150, 200, 300, 400, 500, 600]
    for idx in indices:
        if idx < len(ds):
            row = ds[idx]
            samples.append({
                "ref": row.get("ref", ""),
                "osmanlica_arap": row.get("text", ""),
                "turkce_modern": row.get("text_tr", ""),
                "ingilizce": row.get("text_en", ""),
            })

    with open(BIBLE_SAMPLE_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"İncil paraleli: 10 örnek kaydedildi: {BIBLE_SAMPLE_PATH}")
    return samples


if __name__ == "__main__":
    n_akis = process_akis()
    n_churro = process_churro()
    bible_samples = extract_bible_samples()
    print("Tamamlandı.")
