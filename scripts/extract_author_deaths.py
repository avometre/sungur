"""Tüm Kişi sayfalarından yazar ölüm yıllarını çıkaran script."""

import bz2
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DUMP_PATH = Path("data/raw/_cache/trwikisource-latest-pages-articles.xml.bz2")

author_death = {}
total_kisi = 0

with bz2.open(DUMP_PATH, "rb") as f:
    for _, elem in ET.iterparse(f, events=("end",)):
        if elem.tag.endswith("page"):
            ns = elem.find("{*}ns")
            if ns is not None and ns.text == "100":
                t = elem.find("{*}title")
                title = t.text.strip() if t is not None and t.text else ""
                name = title.replace("Kişi:", "").strip()
                total_kisi += 1
                
                rev = elem.find("{*}revision")
                txt = rev.find("{*}text").text if rev is not None else ""
                
                # ölümyılı ayıkla
                m = re.search(r"\|\s*ölümyılı\s*=\s*([^\n\|\}]+)", txt, re.I)
                if m:
                    raw_y = m.group(1).strip()
                    # Rakamları bul (ör. "1920", "1936", "1556", "MÖ 34")
                    digits = re.findall(r"\d{3,4}", raw_y)
                    if digits:
                        dy = int(digits[0])
                        author_death[name] = dy
                        author_death[name.lower()] = dy
            elem.clear()

print(f"Toplam Kişi sayfası: {total_kisi}")
print(f"Ölüm yılı tespit edilen yazar sayısı: {len(author_death) // 2}")

pre_1950 = {k: v for k, v in author_death.items() if isinstance(k, str) and not k.islower() and v <= 1950}
print(f"Ölüm yılı <= 1950 olan yazar sayısı: {len(pre_1950)}")
sample_authors = list(pre_1950.items())[:25]
for a, y in sample_authors:
    print(f"  {a:<30}: {y}")
