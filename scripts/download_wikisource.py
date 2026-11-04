"""TrWikisource bz2 dökümünü indirme scripti."""

import sys
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DUMP_URL = "https://dumps.wikimedia.org/trwikisource/latest/trwikisource-latest-pages-articles.xml.bz2"
CACHE_DIR = Path("data/raw/_cache")
OUTPUT_PATH = CACHE_DIR / "trwikisource-latest-pages-articles.xml.bz2"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size == 25376776:
    print(f"Döküm zaten indirilmiş: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bayt)")
    sys.exit(0)

print(f"Döküm indiriliyor: {DUMP_URL} -> {OUTPUT_PATH}")
req = urllib.request.Request(DUMP_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) osmanlica-ai/1.0"})

t0 = time.time()
with urllib.request.urlopen(req, timeout=60) as resp:
    total_size = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(OUTPUT_PATH, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded / total_size * 100
                print(f"İndirilen: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB (%{pct:.1f})", end="\r")

t1 = time.time()
print(f"\nİndirme tamamlandı! Süre: {t1 - t0:.1f} sn, Dosya boyutu: {OUTPUT_PATH.stat().st_size} bayt")
