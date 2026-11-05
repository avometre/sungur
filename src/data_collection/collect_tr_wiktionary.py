"""Türkçe Vikisözlük (trwiktionary) dump'ından Osmanlı Türkçesi maddelerini ayrıştırır.

Lisans: CC BY-SA 4.0 / GNU FDL
Kayıt formatı: {"arap": str, "latin": str, "anlam_tr": str, "kaynak": "tr_wiktionary", "lisans": "CC BY-SA 4.0"}
Filtreler:
- Arap betiği oranı > 0.9
- Ünsüz iskelet benzerliği >= 0.5
- Altın set sızıntı kontrolü
Mevcut Kaikki İngilizce Vikisözlük verisi ile çakışma tespiti.
"""

import bz2
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.normalize import arabic_letter_count, normalize_latin, normalize_ottoman
from src.gold.leakage import is_heldout, load_heldout_all
from src.translit.skeleton import skeleton_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DUMP_PATH = Path("data/raw/trwiktionary-latest-pages-articles.xml.bz2")
OUTPUT_PATH = Path("data/interim/tr_wiktionary_ota.jsonl")
EXISTING_KAIKKI_PATH = Path("data/raw/wiktionary_ota.jsonl")

# Şablon kalıpları: {{ota-ad|latin}}, {{ota-eylem|latin}}, {{ota-ön ad|latin}}, vb.
OTA_POS_PATTERN = re.compile(
    r"\{\{ota-(?:ad|eylem|ön ad|önad|zarf|bağlaç|adıl|ünlem|söz|ad-tablo|özel ad)\s*\|\s*([^|}]+)",
    re.IGNORECASE,
)
MEANING_LINE_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)
WIKILINK_PATTERN = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
TEMPLATE_PATTERN = re.compile(r"\{\{[^}]+\}\}")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def clean_wikitext_meaning(raw_text: str) -> str:
    """Wikitext anlam satırından bağlantı ve şablonları temizler."""
    t = TEMPLATE_PATTERN.sub("", raw_text)
    t = WIKILINK_PATTERN.sub(r"\1", t)
    t = HTML_TAG_PATTERN.sub("", t)
    t = re.sub(r"''+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extract_ottoman_section(wikitext: str) -> Optional[str]:
    """Wikitext içerisinden '==Osmanlı Türkçesi==' bölümünü çıkarır."""
    start_match = re.search(r"==\s*Osmanlı Türkçesi\s*==", wikitext, re.IGNORECASE)
    if not start_match:
        return None
    start_pos = start_match.end()
    # Sonraki dil seviyesi (==Dil==) başlığına kadar al
    end_match = re.search(r"\n==\s*[^=]+?\s*==\s*\n", wikitext[start_pos:])
    if end_match:
        return wikitext[start_pos : start_pos + end_match.start()]
    return wikitext[start_pos:]


def parse_tr_wiktionary_dump(
    dump_path: Path = DUMP_PATH,
    output_path: Path = OUTPUT_PATH,
    kaikki_path: Path = EXISTING_KAIKKI_PATH,
) -> Dict[str, Any]:
    """trwiktionary XML dump'ını ayrıştırır ve verileri kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_pages, heldout_arap, heldout_titles, heldout_latin = load_heldout_all()

    # Mevcut Kaikki kelimelerini yükle (çakışma kontrolü için)
    kaikki_words: Set[str] = set()
    if kaikki_path.exists():
        with open(kaikki_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    w = obj.get("word") or obj.get("arap")
                    if w:
                        kaikki_words.add(normalize_ottoman(w.strip()))
                except Exception:
                    pass
        logger.info(f"Mevcut Kaikki sözlükten {len(kaikki_words)} tekil kelime yüklendi.")

    logger.info(f"Dump ayrıştırılıyor: {dump_path}...")
    total_pages = 0
    ota_pages_found = 0
    raw_pairs = 0
    discarded_non_arabic = 0
    discarded_skeleton = 0
    discarded_leakage = 0
    overlap_with_kaikki = 0

    valid_records: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    with bz2.open(dump_path, "rb") as bz_file:
        context = ET.iterparse(bz_file, events=("end",))
        for event, elem in context:
            if elem.tag.endswith("page"):
                total_pages += 1
                ns_elem = elem.find("{*}ns")
                title_elem = elem.find("{*}title")
                revision_elem = elem.find("{*}revision")

                if ns_elem is not None and ns_elem.text == "0" and title_elem is not None and revision_elem is not None:
                    title = title_elem.text or ""
                    text_elem = revision_elem.find("{*}text")
                    wikitext = (text_elem.text or "") if text_elem is not None else ""

                    ota_sec = extract_ottoman_section(wikitext)
                    if ota_sec:
                        ota_pages_found += 1
                        # 1. Başlık Arapça kelime
                        arap_word = title.strip()

                        # 2. Latin okunuşları şablonlardan çek
                        latin_candidates: List[str] = []
                        for m in OTA_POS_PATTERN.finditer(ota_sec):
                            lat = m.group(1).strip()
                            # Bazı şablonlarda | veya tr= olabilir
                            lat = lat.split("|")[0].strip()
                            lat = re.sub(r"^tr\s*=\s*", "", lat)
                            if lat and lat not in latin_candidates:
                                latin_candidates.append(lat)

                        # 3. Anlamları çek
                        meanings: List[str] = []
                        for m in MEANING_LINE_PATTERN.finditer(ota_sec):
                            cleaned_m = clean_wikitext_meaning(m.group(1))
                            if cleaned_m and len(cleaned_m) > 1 and cleaned_m not in meanings:
                                meanings.append(cleaned_m)
                        anlam_str = "; ".join(meanings)

                        # Eğer Latin adayı bulunamadıysa ama sayfa içi transkripsiyon varsa
                        if not latin_candidates:
                            tr_match = re.search(r"\|\s*tr\s*=\s*([^|}]+)", ota_sec)
                            if tr_match:
                                latin_candidates.append(tr_match.group(1).strip())

                        for latin_cand in latin_candidates:
                            raw_pairs += 1
                            pair_key = (arap_word, latin_cand)
                            if pair_key in seen_pairs:
                                continue
                            seen_pairs.add(pair_key)

                            # Filtre 1: Arapça betik kontrolü
                            clean_ar = arap_word.replace(" ", "")
                            if not clean_ar:
                                continue
                            if (arabic_letter_count(arap_word) / len(clean_ar)) < 0.9:
                                discarded_non_arabic += 1
                                continue

                            # Filtre 2: İskelet benzerliği >= 0.5
                            sim = skeleton_similarity(arap_word, latin_cand)
                            if sim < 0.5:
                                discarded_skeleton += 1
                                continue

                            # Filtre 3: Sızıntı kontrolü
                            if is_heldout(
                                arap=arap_word,
                                latin=latin_cand,
                                heldout_pages=heldout_pages,
                                heldout_arap=heldout_arap,
                                heldout_titles=heldout_titles,
                                heldout_latin=heldout_latin,
                            ):
                                discarded_leakage += 1
                                continue

                            norm_ar = normalize_ottoman(arap_word)
                            if norm_ar in kaikki_words:
                                overlap_with_kaikki += 1

                            rec = {
                                "arap": arap_word,
                                "latin": latin_cand,
                                "anlam_tr": anlam_str,
                                "iskelet_benzerlik": round(sim, 3),
                                "kaynak": "tr_wiktionary",
                                "lisans": "CC BY-SA 4.0",
                            }
                            valid_records.append(rec)

                elem.clear()
                if total_pages % 50000 == 0:
                    logger.info(f"Taranan sayfa: {total_pages} | Bulunan Osmanlıca: {ota_pages_found} | Geçerli: {len(valid_records)}")

    # Kaydet
    with open(output_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"tr.wiktionary ayrıştırma bitti. Toplam geçerli: {len(valid_records)} -> {output_path}")

    stats = {
        "kaynak": "tr.wiktionary",
        "lisans": "CC BY-SA 4.0",
        "toplam_sayfa": total_pages,
        "osmanlica_madde": ota_pages_found,
        "ham_cift": raw_pairs,
        "gecerli_cift": len(valid_records),
        "atilan_arap_olmayan": discarded_non_arabic,
        "atilan_iskelet": discarded_skeleton,
        "atilan_sizinti": discarded_leakage,
        "kaikki_ile_cakisan": overlap_with_kaikki,
        "dosya": str(output_path),
    }
    logger.info(f"İstatistikler: {stats}")
    return stats


if __name__ == "__main__":
    parse_tr_wiktionary_dump()
