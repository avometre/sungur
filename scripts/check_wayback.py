"""YEK 404 URL'leri için Wayback Machine sorgusu."""

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

stat_path = Path("third_party/latoc/data_statistics.csv")
with open(stat_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

yek_books = [
    (row["file"], row["link"])
    for row in rows
    if "yek.gov.tr" in (row.get("link") or "")
]

print(f"Toplam YEK bağlantısı: {len(yek_books)}")

found = []
for idx, (f, link) in enumerate(yek_books, 1):
    encoded = urllib.parse.quote(link, safe=":/?=&")
    api_url = f"https://archive.org/wayback/available?url={encoded}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            snap = data.get("archived_snapshots", {}).get("closest")
            if snap and snap.get("available") and snap.get("url"):
                snap_url = snap["url"]
                print(f"[{idx}/{len(yek_books)}] SNAPSHOT BULUNDU: {f} -> {snap_url}")
                found.append((f, link, snap_url))
            else:
                print(f"[{idx}/{len(yek_books)}] Yok: {f}")
    except Exception as e:
        print(f"[{idx}/{len(yek_books)}] Hata ({f}): {e}")
    time.sleep(0.3)

print("=" * 60)
print(f"Sonuç: {len(found)}/{len(yek_books)} snapshot bulundu.")
for f, l, s in found:
    print(f"- {f}: {s}")
