"""Wikisource kategorilerini, yazarlarını ve dönem filtreleme kurallarını analiz eden script."""

import bz2
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DUMP_PATH = Path("data/raw/_cache/trwikisource-latest-pages-articles.xml.bz2")

period_keywords = [
    "osmanlı", "divan", "tanzimat", "servet-i fünun", "servetifünun",
    "fecr-i âti", "fecriati", "meşrutiyet", "millî edebiyat", "milli edebiyat",
    "halk edebiyatı", "tekke", "tasavvuf", "halk şiiri", "aşık", "âşık"
]

category_counts = Counter()
author_counts = Counter()
year_counts = Counter()
pages_by_period = 0
pages_by_year = 0
total_ns0_pages = 0

with bz2.open(DUMP_PATH, "rb") as f:
    for _, elem in ET.iterparse(f, events=("end",)):
        if elem.tag.endswith("page"):
            ns = elem.find("{*}ns")
            redirect = elem.find("{*}redirect")
            if ns is not None and ns.text == "0" and redirect is None:
                total_ns0_pages += 1
                title = elem.find("{*}title").text or ""
                rev = elem.find("{*}revision")
                txt = rev.find("{*}text").text if rev is not None else ""
                
                cats = [c.strip() for c in re.findall(r"\[\[(?:Kategori|Category):([^\]|]+)", txt, re.I)]
                for c in cats:
                    category_counts[c] += 1
                
                # Yıl ayıklama
                m_yil = re.search(r"\|\s*yıl\s*=\s*(\d{4})", txt, re.I)
                if m_yil:
                    y = int(m_yil.group(1))
                    year_counts[y] += 1
                    if y <= 1928:
                        pages_by_year += 1
                
                # Yazar ayıklama
                m_author = re.search(r"\|\s*(?:yazar|eser sahibi)\s*=\s*([^|\n\}]+)", txt, re.I)
                if m_author:
                    auth = m_author.group(1).strip()
                    # [[Yazar:Ad]] veya [[Ad]] temizle
                    auth = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", auth).strip()
                    if auth:
                        author_counts[auth] += 1
            
            elem.clear()

print(f"Toplam ns=0 (yönlendirme olmayan) sayfa sayısı: {total_ns0_pages}")
print(f"<= 1928 yılı açıkça belirtilen sayfalar: {pages_by_year}")

print("\nEN ÇOK GEÇEN 30 YAZAR:")
for a, c in author_counts.most_common(30):
    print(f"  {a:<35}: {c} sayfa")

print("\nDÖNEM ANAHTAR KELİMELERİ İÇEREN EN SIK KATEGORİLER:")
period_cats = [(c, cnt) for c, cnt in category_counts.items() if any(k in c.lower() for k in period_keywords)]
for c, cnt in sorted(period_cats, key=lambda x: x[1], reverse=True)[:35]:
    print(f"  {c:<45}: {cnt} sayfa")
