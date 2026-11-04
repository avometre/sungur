"""LATOC 2_preprocessing entegrasyonu, kural uygulama ve önce/sonra karakter analizi scripti."""

import importlib.util
import json
import os
import re
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. 2_preprocessing.py modülünü importlib ile yükle
spec = importlib.util.spec_from_file_location("latoc_prep", "third_party/latoc/2_preprocessing.py")
latoc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latoc_mod)

# Controversy cache yolunu ayarla ve yükle
latoc_mod.CONTROVERSY_CACHE_JSON = str(Path("third_party/latoc/controversy_cache.json").resolve())
latoc_mod.load_priority_rules()
latoc_mod.load_controversy_cache()

from src.scrapers.latoc import load_latoc_metadata

all_books, _ = load_latoc_metadata()

# IJMES izinli karakterler kümesi
IJMES_LETTERS = set(
    "abcçdefgğhıijklmnoöprsştuüvyz"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    "âîûêôÂÎÛÊÔ"
    "āīūōĀĪŪŌ"
    "ḳḲġĠḥḤḫḪṣṢṭṬżŻẓẒẕẔḍḌñÑʿʾ"
    "\u0331"  # s̱ / S̱ için alt çizgi
)
IJMES_ALLOWED_CHARS = IJMES_LETTERS | set("0123456789 .,;:!?()[]{}\"'«»—–-’‘“”/… \t\n\r")

def disallowed_ratio(text: str) -> float:
    if not text:
        return 0.0
    text_norm = unicodedata.normalize("NFC", text)
    disallowed = sum(1 for c in text_norm if c not in IJMES_ALLOWED_CHARS)
    return disallowed / len(text_norm)

# 17 kitabın mevcut layout dosyalarını kontrol et
layout_dir = Path("data/raw/latoc/layout")

print("Kitap bazında kural uygulaması ve izinsiz karakter analizi:")
print("=" * 80)

results = []
for book in all_books:
    stem = book["stem"]
    lpath = layout_dir / f"{stem}_layout.json"
    if not lpath.exists():
        continue
    
    file_key = latoc_mod.get_file_key(book["file"])
    rules_in_controversy = latoc_mod.CONTROVERSY_CACHE.get(file_key, {})
    rules_in_priority = latoc_mod.PRIORITY_RULES_CACHE.get(file_key, {})
    
    with open(lpath, "r", encoding="utf-8") as f:
        pages = json.load(f)
    
    raw_texts = []
    processed_texts = []
    applied_rules_count = 0
    
    for page in pages:
        segs = page.get("segments", {})
        for stype in ("title", "plain text"):
            for seg in segs.get(stype, []):
                t = seg.get("text", "").strip()
                if t:
                    raw_texts.append(t)
                    # Orijinal LATOC işleme adımı
                    norm = latoc_mod.process_text_with_arabic_marking(t, file_key)
                    if norm != t:
                        applied_rules_count += 1
                    processed_texts.append(norm)
    
    all_raw = " ".join(raw_texts)
    all_norm = " ".join(processed_texts)
    
    ratio_before = disallowed_ratio(all_raw) * 100
    ratio_after = disallowed_ratio(all_norm) * 100
    
    results.append({
        "kitap_id": book["kitap_id"],
        "file": book["file"],
        "stem": stem,
        "file_key": file_key,
        "controversy_rules": len(rules_in_controversy),
        "priority_rules": len(rules_in_priority),
        "degisen_blok_sayisi": applied_rules_count,
        "oran_once": ratio_before,
        "oran_sonra": ratio_after,
        "raw_samples": raw_texts[:2],
        "norm_samples": processed_texts[:2],
    })
    
    print(f"{book['kitap_id']} | {book['file'][:35]:<35} | Öncesi: %{ratio_before:5.2f} -> Sonrası: %{ratio_after:5.2f} | Kural: {len(rules_in_controversy)} contr, {len(rules_in_priority)} prio | Değişen: {applied_rules_count}")

print("=" * 80)
