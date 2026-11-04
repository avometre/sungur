"""Vikikaynak Latin derlemi (geç dönem Osmanlıca) çıkarıcı modülü.

Bu modül:
1. Türkçe Vikikaynak dump dosyasını (trwikisource-latest-pages-articles.xml.bz2) akış halinde okur.
2. Sadece ana ad alanındaki (ns=0) ve yönlendirme olmayan sayfaları inceler.
3. 1928 öncesi Osmanlı dönemi eserlerini (yazar ölüm yılı <= 1950, eser yılı <= 1928 veya dönem kategorileri) seçer.
4. Çeviri eserleri ve yabancı yazarları ayıklar.
5. mwparserfromhell ile wikitext'i temizler, <poem> bloklarını satır satır korur.
6. Altın dev/test seti ile veri sızıntısını (başlık, sayfa kimliği ve Latin cümle düzeyinde) kesin olarak önler.
7. Çıktıyı data/interim/vikikaynak_latin.jsonl dosyasına yazar.
"""

import argparse
import bz2
import json
import logging
import random
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import mwparserfromhell

from src.common.io import write_jsonl
from src.common.normalize import normalize_latin, tr_lower
from src.gold.leakage import LeakageChecker
from src.scrapers.vikikaynak_align import split_sentences

logger = logging.getLogger(__name__)

DEFAULT_DUMP_URL = "https://dumps.wikimedia.org/trwikisource/latest/trwikisource-latest-pages-articles.xml.bz2"
DEFAULT_CACHE_DIR = Path("data/raw/_cache")
DEFAULT_DUMP_PATH = DEFAULT_CACHE_DIR / "trwikisource-latest-pages-articles.xml.bz2"
DEFAULT_OUTPUT_JSONL = Path("data/interim/vikikaynak_latin.jsonl")

# Bilinen Türk/Osmanlı klasik yazarları (ölüm yılı <= 1950)
KNOWN_TURKISH_AUTHORS_DEATH: Dict[str, int] = {
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
    "hoca dehhani": 1350,
    "kadı burhaneddin": 1398,
    "şeyhî": 1431,
    "ahmed paşa": 1497,
    "necati": 1509,
    "zâtî": 1546,
    "bağdadlı rûhî": 1605,
    "taşlıcalı yahya": 1582,
    "nev'î": 1599,
    "şeyhülislam yahya": 1644,
    "nâilî": 1666,
    "nevi-zade atayi": 1635,
}

# Çeviri eserlerdeki yabancı yazarlar
KNOWN_FOREIGN_AUTHORS: Set[str] = {
    "ezop",
    "jean de la fontaine",
    "la fontaine",
    "william blake",
    "virginia woolf",
    "karl marx",
    "friedrich engels",
    "vladimir ilyiç lenin",
    "maksim gorki",
    "william shakespeare",
    "friedrich schiller",
    "rudyard kipling",
    "oscar wilde",
    "daniel defoe",
    "matsuo basho",
    "john keats",
    "sophokles",
    "sofokles",
    "thomas aquinas",
    "jean-françois champollion",
    "j. k. rowling",
    "che guevara",
    "adolf hitler",
    "abraham lincoln",
    "nikita kruşçev",
    "josef stalin",
    "lenin",
}

PERIOD_CATEGORIES: List[str] = [
    "osmanlı",
    "divan",
    "tanzimat",
    "servet-i fünun",
    "servetifünun",
    "meşrutiyet",
    "fecr-i âti",
    "fecriati",
    "millî edebiyat",
    "milli edebiyat",
]


def load_author_deaths_from_dump(dump_path: Path) -> Dict[str, int]:
    """Kişi ad alanındaki (ns=100) sayfalardan yazar ölüm yıllarını yükler."""
    deaths: Dict[str, int] = {}
    if not dump_path.exists():
        return deaths

    with bz2.open(dump_path, "rb") as f:
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
                            y = int(digits[0])
                            deaths[name.lower()] = y
                elem.clear()
    return deaths


def clean_wikitext_content(wikitext: str) -> List[Tuple[str, str]]:
    """Wikitext içeriğini mwparserfromhell kullanarak temizler.

    <poem> bloklarını satır satır mısra olarak korur ('poem', line).
    Düzyazı paragraflarını ('prose', paragraph) olarak döndürür.
    """
    if not wikitext:
        return []

    # 1. <poem> etiketlerini yer tutucu ile sakla
    poem_blocks: List[str] = []

    def poem_repl(match: re.Match) -> str:
        idx = len(poem_blocks)
        content = match.group(1)
        poem_blocks.append(content)
        return f"\n__POEM_BLOCK_{idx}__\n"

    text_with_tokens = re.sub(
        r"<poem[^>]*>(.*?)</poem>", poem_repl, wikitext, flags=re.DOTALL | re.IGNORECASE
    )

    # 2. mwparserfromhell ile parse et
    parsed = mwparserfromhell.parse(text_with_tokens)

    # Şablonları kaldır
    for t in parsed.filter_templates():
        try:
            parsed.remove(t)
        except ValueError:
            pass

    # Dosya, Resim ve Kategori wikilinklerini kaldır
    for link in parsed.filter_wikilinks():
        title_lower = str(link.title).lower()
        if any(
            title_lower.startswith(p)
            for p in ("dosya:", "resim:", "file:", "image:", "kategori:", "category:")
        ):
            try:
                parsed.remove(link)
            except ValueError:
                pass

    # Tabloları, ref'leri ve diğer HTML etiketlerini kaldır
    for tag in parsed.filter_tags():
        tag_name = str(tag.tag).lower()
        if tag_name in ("ref", "table", "td", "tr", "th", "div", "span", "center"):
            try:
                parsed.remove(tag)
            except ValueError:
                pass

    plain = parsed.strip_code()

    # Düzensiz dosya bağlantıları kalmışsa regex ile temizle
    plain = re.sub(
        r"\[\[(?:Dosya|Resim|File|Image):[^\]]+\]\]", "", plain, flags=re.IGNORECASE
    )

    lines: List[Tuple[str, str]] = []
    for line in plain.split("\n"):
        line = line.strip()
        if not line or line.startswith("=="):
            continue

        m = re.match(r"^__POEM_BLOCK_(\d+)__$", line)
        if m:
            p_idx = int(m.group(1))
            poem_raw = poem_blocks[p_idx]
            for p_line in poem_raw.split("\n"):
                p_clean = mwparserfromhell.parse(p_line).strip_code().strip()
                if p_clean and not p_clean.startswith("=="):
                    lines.append(("poem", p_clean))
        else:
            lines.append(("prose", line))

    return lines


def is_translation(wikitext: str, categories: List[str]) -> bool:
    """Sayfanın çeviri eser olup olmadığını denetler."""
    # 1. Şablon çevirmen parametresi
    m = re.search(r"\|\s*(?:çevirmen|çeviren|mütercim)\s*=\s*([^|\n\}]+)", wikitext, re.I)
    if m and m.group(1).strip():
        return True

    # 2. Kategorilerde çeviri denetimi
    for c in categories:
        c_l = c.lower()
        if "çeviri" in c_l or "tercüme" in c_l:
            return True

    return False


def extract_metadata(wikitext: str) -> Tuple[str, Optional[int]]:
    """Wikitext şablonlarından yazar adı ve basım yılını ayıklar."""
    # Yazar
    author = ""
    m_author = re.search(
        r"\|\s*(?:yazar|eser sahibi|bağsızesersahibi)\s*=\s*([^|\n\}]+)", wikitext, re.I
    )
    if m_author:
        raw_a = m_author.group(1).strip()
        author = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", raw_a).strip()

    # Yıl
    year: Optional[int] = None
    m_year = re.search(r"\|\s*yıl\s*=\s*(\d{4})", wikitext, re.I)
    if m_year:
        year = int(m_year.group(1))

    return author, year


def is_target_period(
    wikitext: str,
    categories: List[str],
    author: str,
    year: Optional[int],
    author_deaths: Dict[str, int],
) -> Tuple[bool, str]:
    """Sayfanın 1928 öncesi hedef dönem eseri olup olmadığını belirler."""
    author_l = author.lower()
    cats_lower = [c.lower() for c in categories]

    # Yabancı yazar kontrolü
    if any(fa in author_l for fa in KNOWN_FOREIGN_AUTHORS):
        return False, "yabanci_yazar"

    # Kural 1: Eser yılı <= 1928
    if year is not None and 1000 <= year <= 1928:
        return True, f"eser_yili_{year}"

    # Kural 2: Bilinen Türk yazarlar veya Kişi ölüm yılı <= 1950
    # (Orhan Veli ve Âşık Veysel gibi modern cumhuriyet yazarları hariç)
    if author_l:
        if (
            "orhan veli" not in author_l
            and "âşık veysel" not in author_l
            and "asik veysel" not in author_l
        ):
            if author_l in KNOWN_TURKISH_AUTHORS_DEATH:
                return True, f"turk_yazar_olum_{KNOWN_TURKISH_AUTHORS_DEATH[author_l]}"
            if author_l in author_deaths and author_deaths[author_l] <= 1950:
                return True, f"kisi_olum_{author_deaths[author_l]}"

    # Kural 3: Dönem kategorileri veya {{Osmanlıca}} şablonu
    has_period_cat = any(
        any(pk in c for pk in PERIOD_CATEGORIES) for c in cats_lower
    )
    has_ottoman_tmpl = (
        "{{osmanlıca" in wikitext.lower() or "{{osmanlica" in wikitext.lower()
    )
    if has_period_cat or has_ottoman_tmpl:
        return True, "donem_kategorisi_veya_sablonu"

    return False, "kurala_uymadi"


def run_vikikaynak_latin_pipeline(
    dump_path: Path = DEFAULT_DUMP_PATH,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    min_words: int = 3,
    max_words: int = 80,
) -> Dict[str, Any]:
    """Vikikaynak Latin derlemini oluşturur ve sızıntı korumasını uygular."""
    if not dump_path.exists():
        raise FileNotFoundError(f"Döküm dosyası bulunamadı: {dump_path}")

    dump_size_bytes = dump_path.stat().st_size

    # Sızıntı kontrolcüsü
    leak_checker = LeakageChecker()

    # Kişi ad alanından ölüm yıllarını yükle
    print("Yazar ölüm yılları yükleniyor...")
    author_deaths = load_author_deaths_from_dump(dump_path)
    print(f"Tespit edilen yazar sayısı: {len(author_deaths)}")

    total_scanned_pages = 0
    selected_pages_count = 0
    leaked_pages_count = 0
    leaked_sentences_count = 0
    discarded_translation_count = 0

    all_records: List[Dict[str, Any]] = []
    author_sentence_counter: Counter = Counter()
    total_words = 0

    print("Sayfalar taranıyor ve metinler işleniyor...")
    with bz2.open(dump_path, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag.endswith("page"):
                ns = elem.find("{*}ns")
                redirect = elem.find("{*}redirect")
                if ns is not None and ns.text == "0" and redirect is None:
                    total_scanned_pages += 1
                    title_elem = elem.find("{*}title")
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    id_elem = elem.find("{*}id")
                    pageid = int(id_elem.text.strip()) if id_elem is not None and id_elem.text else 0

                    rev = elem.find("{*}revision")
                    wikitext = rev.find("{*}text").text if rev is not None and rev.find("{*}text") is not None else ""
                    if not wikitext:
                        elem.clear()
                        continue

                    # Sayfa düzeyinde sızıntı kontrolü
                    if leak_checker.is_pageid_heldout(pageid) or leak_checker.is_title_heldout(title):
                        leaked_pages_count += 1
                        elem.clear()
                        continue

                    categories = [
                        c.strip()
                        for c in re.findall(r"\[\[(?:Kategori|Category):([^\]|]+)", wikitext, re.I)
                    ]

                    # Çeviri kontrolü
                    if is_translation(wikitext, categories):
                        discarded_translation_count += 1
                        elem.clear()
                        continue

                    author, year = extract_metadata(wikitext)
                    is_target, _ = is_target_period(
                        wikitext, categories, author, year, author_deaths
                    )

                    if not is_target:
                        elem.clear()
                        continue

                    selected_pages_count += 1

                    # Metin temizleme
                    cleaned_segments = clean_wikitext_content(wikitext)
                    sira = 1

                    for kind, seg_text in cleaned_segments:
                        if kind == "poem":
                            cands = [seg_text]
                        else:
                            cands = split_sentences(seg_text, min_len=0)

                        for cand in cands:
                            cand_clean = cand.strip()
                            if not cand_clean:
                                continue

                            words = [w for w in re.split(r"\s+", cand_clean) if w]
                            if not (min_words <= len(words) <= max_words):
                                continue

                            latin = normalize_latin(cand_clean)
                            latin_sade = normalize_latin(cand_clean, strip_diacritics=True)

                            if not latin:
                                continue

                            # Cümle düzeyinde sızıntı kontrolü (altın dev/test seti ile birebir çakışma)
                            if leak_checker.is_latin_heldout(latin):
                                leaked_sentences_count += 1
                                continue

                            record = {
                                "kaynak": "vikikaynak",
                                "id": f"vk:{pageid}:{sira}",
                                "baslik": title,
                                "yazar": author if author else "Bilinmiyor",
                                "yil": year,
                                "latin": latin,
                                "latin_sade": latin_sade,
                                "uslup": "vikikaynak",
                            }
                            all_records.append(record)
                            author_sentence_counter[record["yazar"]] += 1
                            total_words += len(words)
                            sira += 1

                elem.clear()

    # Çıktı dosyasını yaz
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, all_records)

    # Örnek 10 cümle (seed=3)
    rng = random.Random(3)
    sample_10 = (
        rng.sample(all_records, min(10, len(all_records))) if all_records else []
    )

    summary = {
        "dokum_boyutu_mb": round(dump_size_bytes / (1024 * 1024), 2),
        "taranan_sayfa_sayisi": total_scanned_pages,
        "secilen_sayfa_sayisi": selected_pages_count,
        "toplam_cumle_sayisi": len(all_records),
        "toplam_kelime_sayisi": total_words,
        "sizinti_sayfa_sayisi": leaked_pages_count,
        "sizinti_cumle_sayisi": leaked_sentences_count,
        "atilan_ceviri_sayisi": discarded_translation_count,
        "en_cok_cumle_veren_20_yazar": author_sentence_counter.most_common(20),
        "ornek_10_cumle": sample_10,
    }
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Vikikaynak Latin derlemi (geç dönem Osmanlıca) çıkar"
    )
    parser.add_argument(
        "--dump-file",
        type=Path,
        default=DEFAULT_DUMP_PATH,
        help="TrWikisource bz2 döküm dosyası yolu",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Çıktı JSONL dosya yolu",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("VİKİKAYNAK LATİN DERLEMİ ÇIKARIMI (vikikaynak_latin)")
    print("=" * 80)

    summary = run_vikikaynak_latin_pipeline(
        dump_path=args.dump_file,
        output_jsonl=args.output_file,
    )

    print("-" * 80)
    print(f"Döküm Boyutu (MB)            : {summary['dokum_boyutu_mb']} MB")
    print(f"Taranan Sayfa Sayısı         : {summary['taranan_sayfa_sayisi']}")
    print(f"Seçilen Sayfa Sayısı         : {summary['secilen_sayfa_sayisi']}")
    print(f"Toplam Cümle Sayısı          : {summary['toplam_cumle_sayisi']:,}")
    print(f"Toplam Kelime Sayısı         : {summary['toplam_kelime_sayisi']:,}")
    print(f"Sızıntı Nedeniyle Atılan Sayfa: {summary['sizinti_sayfa_sayisi']}")
    print(f"Sızıntı Nedeniyle Atılan Cümle: {summary['sizinti_cumle_sayisi']}")
    print(f"Atılan Çeviri Sayfa          : {summary['atilan_ceviri_sayisi']}")
    print("-" * 80)
    print("En Çok Cümle Veren 20 Yazar:")
    for auth, cnt in summary["en_cok_cumle_veren_20_yazar"]:
        print(f"  - {auth:<35}: {cnt:,} cümle")
    print("-" * 80)
    print("Rastgele 10 Cümle (seed=3):")
    for idx, s in enumerate(summary["ornek_10_cumle"], 1):
        print(f"{idx}. [{s['id']} | {s['baslik']} | Yazar: {s['yazar']}]")
        print(f"   Latin: {s['latin']}")
        print(f"   Sade : {s['latin_sade']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
