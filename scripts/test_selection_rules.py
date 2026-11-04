"""Vikikaynak seçim kurallarını test eden ve eşleşen sayfaları inceleyen simülasyon scripti."""

import bz2
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import mwparserfromhell

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DUMP_PATH = Path("data/raw/_cache/trwikisource-latest-pages-articles.xml.bz2")

# 1. Yazar ölüm yıllarını Kişi sayfalarından topla
author_death = {}
with bz2.open(DUMP_PATH, "rb") as f:
    for _, elem in ET.iterparse(f, events=("end",)):
        if elem.tag.endswith("page"):
            ns = elem.find("{*}ns")
            if ns is not None and ns.text == "100":
                t = elem.find("{*}title")
                title = t.text.strip() if t is not None and t.text else ""
                name = title.replace("Kişi:", "").strip()
                rev = elem.find("{*}revision")
                txt = rev.find("{*}text").text if rev is not None else ""
                m = re.search(r"\|\s*ölümyılı\s*=\s*([^\n\|\}]+)", txt, re.I)
                if m:
                    digits = re.findall(r"\d{3,4}", m.group(1))
                    if digits:
                        author_death[name.lower()] = int(digits[0])
            elem.clear()

# Bilinen Türk/Osmanlı klasik yazarları (ölüm yılı <= 1950)
known_turkish_authors_death = {
    "ömer seyfettin": 1920,
    "mehmet âkif ersoy": 1936,
    "mehmet akif ersoy": 1936,
    "tevfik fikret": 1915,
    "ahmet haşim": 1933,
    "cenab şahabeddin": 1934,
    "cenap şahabettin": 1934,
    "recaizade mahmut ekrem": 1914,
    "ziya gökalp": 1924,
    "namık kemal": 1888,
    "şinasi": 1871,
    "ibrahim şinasi": 1871,
    "abdülhak hamit tarhan": 1937,
    "muallim naci": 1893,
    "ahmet mithat efendi": 1912,
    "ahmet rasim": 1932,
    "süleyman nazif": 1927,
    "ali ekrem bolayır": 1937,
    "mehmet emin yurdakul": 1944,
    "hüseyin rahmi gürpınar": 1944,
    "fuzûlî": 1556,
    "fuzuli": 1556,
    "bâkî": 1600,
    "baki": 1600,
    "nedim": 1730,
    "nef'i": 1635,
    "nefi": 1635,
    "nâbi": 1712,
    "nabi": 1712,
    "şeyh galip": 1799,
    "yunus emre": 1320,
    "ahmed yesevi": 1166,
    "karacaoğlan": 1679,
    "pir sultan abdal": 1550,
    "kul himmet": 1584,
    "kadı burhaneddin": 1398,
    "kanuni sultan süleyman": 1566,
    "süleyman çelebi": 1422,
    "mihrî hatun": 1506,
    "mihri hatun": 1506,
    "mustafa kemal atatürk": 1938,
    "gazi mustafa kemal": 1938,
    "ali şîr nevaî": 1501,
    "evliya çelebi": 1682,
    "kâtip çelebi": 1657,
    "katip çelebi": 1657,
    "şah hatayi": 1524,
    "şah ismail": 1524,
    "eşrefoğlu rumî": 1469,
    "ümmî sinan": 1568,
    "üsküdarlı âşık dertli": 1846,
    "aşık dertli": 1846,
    "âşık veysel": 1973, # > 1950!
    "orhan veli kanık": 1950, # 1950 ama modern cumhuriyet
}

# Yabancı yazarlar (çeviri filtresi)
known_foreign_authors = {
    "ezop", "jean de la fontaine", "la fontaine", "william blake", "virginia woolf",
    "karl marx", "friedrich engels", "vladimir ilyiç lenin", "maksim gorki",
    "william shakespeare", "friedrich schiller", "rudyard kipling", "oscar wilde",
    "daniel defoe", "matsuo basho", "john keats", "sophokles", "sofokles",
    "thomas aquinas", "jean-françois champollion", "j. k. rowling", "che guevara"
}

period_categories = [
    "osmanlı", "divan", "tanzimat", "servet-i fünun", "servetifünun",
    "meşrutiyet", "fecr-i âti", "fecriati", "millî edebiyat", "milli edebiyat"
]

def is_translation(txt: str, cats: list) -> bool:
    # 1. Şablon içi çevirmen parametresi
    m = re.search(r"\|\s*(?:çevirmen|çeviren|mütercim)\s*=\s*([^|\n\}]+)", txt, re.I)
    if m and m.group(1).strip():
        return True
    # 2. Kategori denetimi
    for c in cats:
        c_l = c.lower()
        if "çeviri" in c_l or "tercüme" in c_l:
            return True
    return False

def get_page_year(txt: str) -> int:
    m = re.search(r"\|\s*yıl\s*=\s*(\d{4})", txt, re.I)
    if m:
        return int(m.group(1))
    return 0

def get_page_author(txt: str) -> str:
    m = re.search(r"\|\s*(?:yazar|eser sahibi)\s*=\s*([^|\n\}]+)", txt, re.I)
    if m:
        auth = m.group(1).strip()
        auth = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", auth).strip()
        return auth
    return ""

matched_pages = []
discarded_reasons = Counter()

with bz2.open(DUMP_PATH, "rb") as f:
    for _, elem in ET.iterparse(f, events=("end",)):
        if elem.tag.endswith("page"):
            ns = elem.find("{*}ns")
            redirect = elem.find("{*}redirect")
            if ns is not None and ns.text == "0" and redirect is None:
                title = elem.find("{*}title").text or ""
                rev = elem.find("{*}revision")
                txt = rev.find("{*}text").text if rev is not None else ""
                
                cats = [c.strip() for c in re.findall(r"\[\[(?:Kategori|Category):([^\]|]+)", txt, re.I)]
                cats_lower = [c.lower() for c in cats]
                
                # Çeviri kontrolü
                if is_translation(txt, cats):
                    discarded_reasons["ceviri"] += 1
                    elem.clear()
                    continue
                
                # Yazar ve yıl
                author = get_page_author(txt)
                author_l = author.lower()
                year = get_page_year(txt)
                
                # Yabancı yazar kontrolü
                if any(fa in author_l for fa in known_foreign_authors):
                    discarded_reasons["yabanci_yazar"] += 1
                    elem.clear()
                    continue
                
                selected = False
                reason = ""
                
                # Kural 1: Eser yılı <= 1928
                if 1000 <= year <= 1928:
                    selected = True
                    reason = f"eser_yili_{year}"
                
                # Kural 2: Bilinen Türk yazarlar veya Kişi ölüm yılı <= 1950 (Modern cumhuriyet yazarları hariç)
                if not selected and author_l:
                    # Orhan Veli hariç (1950 ama modern cumhuriyet)
                    if "orhan veli" not in author_l and "âşık veysel" not in author_l and "asik veysel" not in author_l:
                        if author_l in known_turkish_authors_death:
                            selected = True
                            reason = f"turk_yazar_olum_{known_turkish_authors_death[author_l]}"
                        elif author_l in author_death and author_death[author_l] <= 1950:
                            selected = True
                            reason = f"kisi_olum_{author_death[author_l]}"
                
                # Kural 3: Dönem kategorileri veya {{Osmanlıca}} şablonu
                if not selected:
                    has_period_cat = any(any(pk in c for pk in period_categories) for c in cats_lower)
                    has_ottoman_tmpl = "{{osmanlıca" in txt.lower() or "{{osmanlica" in txt.lower()
                    if has_period_cat or has_ottoman_tmpl:
                        selected = True
                        reason = "donem_kategorisi_veya_sablonu"
                
                if selected:
                    matched_pages.append({
                        "title": title,
                        "author": author,
                        "year": year,
                        "reason": reason,
                    })
                else:
                    discarded_reasons["kurala_uymadi"] += 1
            
            elem.clear()

print(f"Toplam seçilen sayfa sayısı: {len(matched_pages)}")
print("Elenme nedenleri:", discarded_reasons)

author_counts = Counter(p["author"] if p["author"] else "Belirtilmemiş" for p in matched_pages)
print("\nSEÇİLEN SAYFALARDA EN ÇOK BULUNAN 20 YAZAR:")
for a, c in author_counts.most_common(20):
    print(f"  {a:<35}: {c} sayfa")

reason_counts = Counter(p["reason"].split("_")[0] for p in matched_pages)
print("\nSEÇİM NEDENLERİ:")
for r, c in reason_counts.items():
    print(f"  {r:<30}: {c} sayfa")
