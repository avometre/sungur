"""OttomanNLP/ottoman-turkish-printed-pages eşlikçi deposu (OttomanOCR-search) inceleme ve metin çekme modülü.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

USER_AGENT = "SungurResearchBot/0.1 (non-commercial research)"
BASE_URL = "https://raw.githubusercontent.com/Koras0ff/OttomanOCR-search/main/data"

SAMPLE_BOOKS = [
    "Kanunnamei_Ali_Osman.js",
    "as_lm_s__bir_adam.js",
    "divan__nedim.js",
    "ahdimsiar00kr.js",
    "amerika_tarih_kesfi.js",
]

OUTPUT_DIR = Path("data/raw/ottoman_printed_text")
INTERIM_OUTPUT = Path("data/interim/ottoman_printed_samples.jsonl")


def fetch_book_data(book_filename: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/{book_filename}"
    logger.info(f"İndiriliyor: {url}")
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    raw = resp.text
    m = re.search(r'=\s*(\{.*?\});?\s*$', raw, re.DOTALL)
    if not m:
        idx = raw.find('{"id"')
        if idx != -1:
            raw_json = raw[idx:].rstrip("; \n")
            return json.loads(raw_json)
        raise ValueError(f"JSON yapısı çözülemedi: {book_filename}")
    return json.loads(m.group(1))


def fetch_all_book_filenames() -> List[str]:
    """GitHub API üzerinden OttomanOCR-search/data altındaki tüm .js kitap dosyalarını listeler."""
    url = "https://api.github.com/repos/Koras0ff/OttomanOCR-search/contents/data"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    items = resp.json()
    return sorted(x["name"] for x in items if isinstance(x, dict) and x.get("name", "").endswith(".js"))


def process_books(book_filenames: List[str] | None = None, output_path: Path = INTERIM_OUTPUT) -> List[Dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = book_filenames if book_filenames is not None else SAMPLE_BOOKS
    all_lines = []

    for bname in targets:
        try:
            data = fetch_book_data(bname)
            book_id = data.get("id", bname)
            title = data.get("title", "")
            pages = data.get("pages", [])
            logger.info(f"Kitap: {book_id} ('{title}'), sayfa sayısı: {len(pages)}")

            book_lines = []
            for p_idx, page in enumerate(pages):
                page_im = page.get("im", f"page_{p_idx+1}.jpg")
                page_num = page.get("n", str(p_idx + 1))
                # L alanı: [region_type, polygon, text, index]
                for block in page.get("L", []):
                    # Metin 3. elemanda (indeks 2) yer alır
                    text_content = None
                    for elem in block:
                        if isinstance(elem, str) and len(elem) > 1 and any("\u0600" <= c <= "\u06ff" for c in elem):
                            text_content = elem
                            break

                    if text_content:
                        # Satırlara böl
                        raw_lines = [ln.strip() for ln in text_content.split("\n") if ln.strip()]
                        for ln in raw_lines:
                            item = {
                                "kitap": book_id,
                                "baslik": title,
                                "sayfa": page_im,
                                "sayfa_no": page_num,
                                "metin": ln,
                            }
                            book_lines.append(item)
                            all_lines.append(item)

            logger.info(f"  -> Toplam {len(book_lines)} satır çıkarıldı.")
            # Kitap bazında ham özeti kaydet
            raw_out = OUTPUT_DIR / f"{book_id}.json"
            with open(raw_out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "id": book_id,
                        "title": title,
                        "pages_count": len(pages),
                        "lines_count": len(book_lines),
                        "sample_lines": [x["metin"] for x in book_lines[:5]],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            time.sleep(0.2)
        except Exception as e:
            logger.error(f"Hata ({bname}): {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info(f"Toplam {len(all_lines)} satır çıkarıldı ve kaydedildi: {output_path}")
    return all_lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OttomanOCR-search kitap metinlerini çeker.")
    parser.add_argument("--all", action="store_true", help="5 örnek yerine tüm 60 kitabı indir")
    parser.add_argument("--output", type=Path, default=None, help="Çıktı JSONL yolu")
    args = parser.parse_args()

    if args.all:
        books = fetch_all_book_filenames()
        out = args.output or Path("data/interim/ottoman_printed_all.jsonl")
        lines = process_books(book_filenames=books, output_path=out)
    else:
        out = args.output or INTERIM_OUTPUT
        lines = process_books(output_path=out)

    print(f"\n--- 20 ÖRNEK SATIR (Toplam {len(lines)} satır içinden) ---")
    for i, item in enumerate(lines[:20], 1):
        safe_txt = item["metin"].encode("ascii", "backslashreplace").decode("ascii")
        safe_book = item["kitap"].encode("ascii", "backslashreplace").decode("ascii")
        print(f"[{i:02d}] ({safe_book} - {item['sayfa']}) {safe_txt}")
