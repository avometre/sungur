"""6 bozuk kitaptan 2'şer önce/sonra cümle örneği gösteren script."""

import json
from pathlib import Path
import unicodedata
import importlib.util
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

spec = importlib.util.spec_from_file_location("latoc_prep", "third_party/latoc/2_preprocessing.py")
latoc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latoc_mod)
latoc_mod.CONTROVERSY_CACHE_JSON = str(Path("third_party/latoc/controversy_cache.json").resolve())
latoc_mod.load_priority_rules()
latoc_mod.load_controversy_cache()

from src.scrapers.latoc import load_latoc_metadata

all_books, _ = load_latoc_metadata()

target_books = ["b011", "b017", "b031", "b033", "b035", "b141"]
layout_dir = Path("data/raw/latoc/layout")

for book in all_books:
    if book["kitap_id"] not in target_books:
        continue
    stem = book["stem"]
    lpath = layout_dir / f"{stem}_layout.json"
    file_key = latoc_mod.get_file_key(book["file"])
    
    with open(lpath, "r", encoding="utf-8") as f:
        pages = json.load(f)
        
    found_samples = []
    for page in pages:
        segs = page.get("segments", {})
        for stype in ("plain text", "title"):
            for seg in segs.get(stype, []):
                t = seg.get("text", "").strip()
                if any(c in t for c in ("ù", "À", "ã", "ÿ", "è", "à", "ô", "ú", "å", "òu")):
                    norm = latoc_mod.process_text_with_arabic_marking(t, file_key)
                    if norm != t and len(t) > 20:
                        found_samples.append((t, norm))
                        if len(found_samples) >= 2:
                            break
            if len(found_samples) >= 2:
                break
        if len(found_samples) >= 2:
            break
            
    print(f"\n--- Kitap: {book['kitap_id']} ({book['file']}) ---")
    for i, (orig, fixed) in enumerate(found_samples, 1):
        print(f"  Örnek {i} ÖNCE : {orig[:90]}")
        print(f"  Örnek {i} SONRA: {fixed[:90]}")
