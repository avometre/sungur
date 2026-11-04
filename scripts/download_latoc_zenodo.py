"""Zenodo'dan LATOC metaveri ve betik dosyalarını indirme aracı."""

import json
import sys
import urllib.request
from pathlib import Path

ZENODO_API_URL = "https://zenodo.org/api/records/19445113"
TARGET_DIR = Path("third_party/latoc")


def download_zenodo_files():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        ZENODO_API_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    files = data.get("files", [])
    print(f"Toplam {len(files)} dosya indirilecek -> {TARGET_DIR}")

    for f in files:
        fname = f["key"]
        download_url = f["links"]["self"]
        fsize = f.get("size", 0)
        target_path = TARGET_DIR / fname

        print(f"İndiriliyor: {fname} ({fsize} bayt)...")
        req_file = urllib.request.Request(
            download_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req_file) as r, open(target_path, "wb") as out_f:
            out_f.write(r.read())

    print("Zenodo dosyaları başarıyla indirildi.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    download_zenodo_files()
