"""Aktif sayfa sayısını ve YOLO gereksinimini hesaplayan script."""

import fitz
import sys
from pathlib import Path
from src.scrapers.latoc import load_latoc_metadata, parse_exclusion_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

all_books, _ = load_latoc_metadata()
excl = parse_exclusion_file(Path("third_party/latoc/exclude_pages.txt"))

total_pages = 0
active_pages = 0

for b in all_books:
    pdf_path = Path("data/raw/latoc/pdf") / (b["stem"] + ".pdf")
    if not pdf_path.exists():
        continue
    doc = fitz.open(pdf_path)
    p_count = len(doc)
    total_pages += p_count
    
    clean_stem = b["stem"].replace("_", "").replace(" ", "").lower()
    b_excl = set()
    for k, p_set in excl.items():
        k_clean = k.replace(".xml", "").replace("_", "").replace(" ", "").lower()
        if k_clean == clean_stem or k_clean in clean_stem:
            b_excl = p_set
            break
            
    act = sum(1 for p in range(1, p_count + 1) if p not in b_excl)
    active_pages += act
    print(f"{b['kitap_id']} | Toplam sayfa: {p_count:4d} | Atılan sayfa: {len(b_excl):3d} | Aktif sayfa: {act:4d} | {b['stem'][:40]}")

print("-" * 60)
print(f"Genel Toplam: {total_pages} sayfa, Atılan: {total_pages - active_pages} sayfa, Aktif: {active_pages} sayfa")
