"""Altın setteki 33 sayfanın Vikikaynak dökümündeki şablon ve kategorilerini analiz eden script."""

import bz2
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PAGE_STYLE_PATH = Path("data/gold/page_style.csv")
DUMP_PATH = Path("data/raw/_cache/trwikisource-latest-pages-articles.xml.bz2")

# 33 başlığı yükle
heldout_titles = set()
with open(PAGE_STYLE_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        t = row.get("tr_baslik", "").strip()
        if t:
            heldout_titles.add(t)

print(f"Hedef altın başlık sayısı: {len(heldout_titles)}")

found_pages = {}
template_counter = Counter()
category_counter = Counter()

def strip_ns(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag

with bz2.open(DUMP_PATH, "rb") as bz2_file:
    # iterparse ile akış halinde oku
    context = ET.iterparse(bz2_file, events=("end",))
    for event, elem in context:
        if strip_ns(elem.tag) == "page":
            ns_elem = elem.find("{*}ns")
            if ns_elem is not None and ns_elem.text == "0":
                title_elem = elem.find("{*}title")
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                
                if title in heldout_titles:
                    # Yönlendirme değilse metni al
                    redirect_elem = elem.find("{*}redirect")
                    if redirect_elem is None:
                        rev_elem = elem.find("{*}revision")
                        text_elem = rev_elem.find("{*}text") if rev_elem is not None else None
                        wikitext = text_elem.text if text_elem is not None and text_elem.text else ""
                        
                        # Kategoriler
                        cats = [c.strip() for c in re.findall(r"\[\[(?:Kategori|Category):([^\]|]+)", wikitext, re.I)]
                        # Şablonlar
                        templates = [t.strip().split("\n")[0] for t in re.findall(r"\{\{([^}|]+)", wikitext)]
                        
                        found_pages[title] = {
                            "categories": cats,
                            "templates": templates,
                        }
                        for c in cats:
                            category_counter[c] += 1
                        for t in templates:
                            clean_t = t.strip()
                            if clean_t:
                                template_counter[clean_t] += 1
            
            # Bellek temizliği
            elem.clear()

print(f"Döküm içinde bulunan altın sayfa sayısı: {len(found_pages)}/{len(heldout_titles)}")
missing = heldout_titles - set(found_pages.keys())
if missing:
    print(f"Bulunamayanlar ({len(missing)}): {missing}")

combined_counter = Counter()
for k, v in category_counter.items():
    combined_counter[f"Kategori:{k}"] += v
for k, v in template_counter.items():
    combined_counter[f"Şablon:{k}"] += v

print("\nEN SIK 40 KATEGORİ VE ŞABLON:")
print("-" * 60)
for idx, (name, count) in enumerate(combined_counter.most_common(40), 1):
    print(f"{idx:2d}. {name:<45}: {count} sayfa")
