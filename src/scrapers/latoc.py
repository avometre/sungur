"""LATOC (Latin harfli Osmanlıca derlem) indirici, işleyici ve cümle dönüştürücü modülü.

Bu modül:
1. Zenodo LATOC metaverisini (data_statistics.csv, LATOC_metadata_sample.csv) okur.
2. Kitap PDF'lerini indirir veya mevcut PDF'leri kullanır.
3. DocLayout-YOLO (GPU RTX 3060) ile sadece "plain text" ve "title" sınıflarını alıp,
   dipnot, sayfa başlığı/numarası ("abandon", "footnote" vb.) sınıflarını ayıklar.
4. third_party/latoc/2_preprocessing.py kurallarını (öncelikli kurallar, controversy cache,
   CERTAIN_RULES) importlib ile dinamik yükleyerek orijinal dosya adına göre sırayla uygular.
5. Harfe yapışık dipnot numaralarını (strip_footnote_numbers) ve mısra işaretlerini (strip_verse_markers) temizler.
6. IJMES kalite kapısını uygular (cümle düzeyi > %2 elenir; kitap düzeyi > %0.5 ise kitap kalite_dusuk işaretlenip
   data/interim/excluded/latoc_low_quality.jsonl dosyasına yazılır).
7. Geçerli cümleleri data/interim/latoc_sentences.jsonl dosyasına kaydeder ("uslup": "ijmes", "donem": int|None).
"""

import argparse
import csv
import importlib.util
import json
import logging
import os
import random
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import regex

from src.common.io import write_jsonl
from src.common.normalize import normalize_latin, tr_lower
from src.scrapers.vikikaynak_align import split_sentences

logger = logging.getLogger(__name__)

DEFAULT_THIRD_PARTY_DIR = Path("third_party/latoc")
DEFAULT_RAW_DIR = Path("data/raw/latoc")
DEFAULT_OUTPUT_JSONL = Path("data/interim/latoc_sentences.jsonl")
DEFAULT_EXCLUDED_JSONL = Path("data/interim/excluded/latoc_low_quality.jsonl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# IJMES izinli karakterler kümesi:
# Türkçe harfler + â î û ê ô Â Î Û Ê Ô + ā ī ū ō Ā Ī Ū Ō +
# ḳ Ḳ ġ Ġ ḥ Ḥ ḫ Ḫ ṣ Ṣ ṭ Ṭ ż Ż ẓ Ẓ ẕ Ẕ ḍ Ḍ ñ Ñ ʿ ʾ +
# s̱ / S̱ için \u0331 (combining macron below) + noktalama + rakam + boşluk
IJMES_SPECIAL_LETTERS: Set[str] = set("āīūōḳġḥḫṣṭżẓẕḍñʿʾĀĪŪŌḲĠḤḪṢṬŻẒẔḌÑ")

IJMES_LETTERS = set(
    "abcçdefgğhıijklmnoöprsştuüvyz"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    "âîûêôÂÎÛÊÔ"
    "āīūōĀĪŪŌ"
    "ḳḲġĠḥḤḫḪṣṢṭṬżŻẓẒẕẔḍḌñÑʿʾ"
    "\u0331"
)
IJMES_ALLOWED_CHARS = IJMES_LETTERS | set(
    "0123456789 .,;:!?()[]{}\"'«»—–-’‘“”/… \t\n\r"
)

_PREPROCESSING_MODULE = None


def load_latoc_preprocessing_module(third_party_dir: Path = DEFAULT_THIRD_PARTY_DIR):
    """third_party/latoc/2_preprocessing.py modülünü kopyalamadan dinamik yükler."""
    global _PREPROCESSING_MODULE
    if _PREPROCESSING_MODULE is not None:
        return _PREPROCESSING_MODULE

    script_path = third_party_dir / "2_preprocessing.py"
    if not script_path.exists():
        raise FileNotFoundError(f"LATOC 2_preprocessing.py bulunamadı: {script_path}")

    spec = importlib.util.spec_from_file_location("latoc_preprocessing", str(script_path.resolve()))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Controversy cache yolunu ayarla ve yükle
    cache_path = third_party_dir / "controversy_cache.json"
    mod.CONTROVERSY_CACHE_JSON = str(cache_path.resolve())
    mod.load_priority_rules()
    mod.load_controversy_cache()

    _PREPROCESSING_MODULE = mod
    return _PREPROCESSING_MODULE


def strip_footnote_numbers(text: str) -> str:
    """Harfe yapışık sondaki dipnot rakamlarını siler: 'mümtenièdür1079' → 'mümtenièdür'.

    Bağımsız sayılara ('1079 yılında') dokunmaz.
    Regex: (?<=\\p{L})\\d{1,4}(?=\\W|$)
    """
    if not text:
        return ""
    return regex.sub(r"(?<=\p{L})\d{1,4}(?=\W|$)", "", text)


def strip_verse_markers(text: str) -> str:
    """Kelime olarak geçen S1/S2/s1/s2 ve mısra ayracı ' / ' işaretlerini tek boşluğa dönüştürür."""
    if not text:
        return ""
    # S1, S2, s1, s2 bağımsız kelime olarak
    text = regex.sub(r"\b[Ss][12]\b", " ", text)
    # Mısra ayracı " / "
    text = regex.sub(r"\s+/\s+", " ", text)
    # Fazla boşlukları tek boşluğa indir
    return regex.sub(r"\s+", " ", text).strip()


def disallowed_char_ratio(text: str) -> float:
    """Metindeki IJMES izinli olmayan karakterlerin oranını hesaplar."""
    if not text:
        return 0.0
    text_norm = unicodedata.normalize("NFC", text)
    disallowed = sum(1 for c in text_norm if c not in IJMES_ALLOWED_CHARS)
    return disallowed / len(text_norm)


def is_valid_ijmes_sentence(text: str, max_disallowed_ratio: float = 0.02) -> bool:
    """Cümle düzeyinde izinli olmayan karakter oranının eşiği (varsayılan: %2) aşmadığını denetler."""
    return disallowed_char_ratio(text) <= max_disallowed_ratio


def is_apparatus_line(text: str) -> bool:
    """Nüsha/aparat satırlarını tespit eder (harf olmayan ve tek harfli belirteç oranı > %50).

    Örnek: 'U 3 M 90b', 'aḥkem S 2 Evvel ü:'
    """
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return True
    bad = sum(1 for t in tokens if sum(1 for c in t if c.isalpha()) <= 1)
    return (bad / len(tokens)) > 0.50


def is_editor_prose_translation(text: str) -> bool:
    """Editörün bugünkü Türkçe nesir çevirilerini tespit eder:
    8+ kelimelik ve hiç IJMES özel harfi içermeyen cümleler.
    """
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return len(words) >= 8 and not any(c in IJMES_SPECIAL_LETTERS for c in text)


def load_latoc_metadata(
    third_party_dir: Path = DEFAULT_THIRD_PARTY_DIR,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """data_statistics.csv ve LATOC_metadata_sample.csv dosyalarını yükler."""
    stat_file = third_party_dir / "data_statistics.csv"
    sample_file = third_party_dir / "LATOC_metadata_sample.csv"

    sample_meta: Dict[str, Dict[str, Any]] = {}
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                old_id = row.get("oldID", "").strip().lower()
                clean_old = old_id.replace("__", "_").replace("_", "")
                sample_meta[clean_old] = row
                if row.get("work_name"):
                    clean_name = row["work_name"].strip().lower()
                    sample_meta[clean_name] = row

    all_books: List[Dict[str, Any]] = []
    if stat_file.exists():
        with open(stat_file, "r", encoding="utf-8") as f:
            for idx, row in enumerate(csv.DictReader(f), 1):
                raw_file = row.get("file", "").strip()
                link = (row.get("link") or "").strip()
                stem = raw_file.replace(".xml", "")

                clean_stem = (
                    stem.replace("_", "")
                    .replace(" ", "")
                    .replace("ı", "i")
                    .replace("ü", "u")
                    .replace("ö", "o")
                    .replace("ş", "s")
                    .replace("ç", "c")
                    .replace("ğ", "g")
                    .lower()
                )

                meta = (
                    sample_meta.get(clean_stem)
                    or sample_meta.get(stem.lower())
                    or {}
                )

                century = meta.get("century", "")
                author = meta.get("real_name") or meta.get("pen_name") or ""
                work_name = meta.get("work_name") or stem

                if not century:
                    known_cents = {
                        "esrardede": "18",
                        "yenipazarli": "16",
                        "dedeomer": "15",
                        "ruseni": "15",
                        "vuslati": "17",
                        "cehrin": "17",
                        "aski": "16",
                        "heftpeyker": "16",
                        "latifi": "16",
                        "celili": "16",
                        "sehri": "17",
                        "bahti": "17",
                        "hamiahmed": "18",
                        "sururi": "18",
                        "taslicali": "16",
                        "yahyabey": "16",
                        "firaki": "16",
                        "esiri": "16",
                        "asikcelebi": "16",
                        "mesairu": "16",
                        "sabri": "18",
                        "tacizade": "15",
                        "cakeri": "15",
                        "durri": "18",
                    }
                    for k_pat, c_val in known_cents.items():
                        if k_pat in clean_stem:
                            century = c_val
                            break

                if not author and "_" in stem:
                    parts = stem.split("_", 1)
                    author = parts[0].strip().title()
                    work_name = parts[1].strip().title()
                elif not author:
                    author = stem.title()

                all_books.append({
                    "kitap_id": f"b{idx:03d}",
                    "file": raw_file,
                    "stem": stem,
                    "link": link,
                    "baslik": work_name,
                    "yazar": author,
                    "yuzyil": century if century else "bilinmiyor",
                    "total_words": int(row.get("total_words", 0) or 0),
                    "num_paragraphs": int(row.get("num_paragraphs", 0) or 0),
                })

    return all_books, sample_meta


def parse_exclusion_file(exclusion_path: Path) -> Dict[str, Set[int]]:
    """exclude_pages.txt dosyasını ayrıştırır."""
    exclusions: Dict[str, Set[int]] = {}
    if not exclusion_path.exists():
        return exclusions

    with open(exclusion_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            fname, ranges = line.split(":", 1)
            fname = fname.strip()
            ranges = ranges.strip()
            if not ranges:
                continue

            pages: Set[int] = set()
            for part in ranges.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    try:
                        s_str, e_str = part.split("-", 1)
                        s, e = int(s_str.strip()), int(e_str.strip())
                        pages.update(range(s, e + 1))
                    except ValueError:
                        pass
                else:
                    try:
                        pages.add(int(part))
                    except ValueError:
                        pass

            exclusions[fname] = pages
    return exclusions


def filter_sentence_words(
    sentence: str,
    min_words: int = 3,
    max_words: int = 80,
) -> Tuple[bool, int]:
    """Cümlenin kelime sayısı kuralına (3 <= kelime <= 80) uyup uymadığını denetler."""
    words = [w for w in re.split(r"\s+", sentence.strip()) if w]
    cnt = len(words)
    is_valid = min_words <= cnt <= max_words
    return is_valid, cnt


def clean_control_chars(text: str) -> str:
    """XML kontrol karakterlerini temizler."""
    if not text:
        return ""
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)


def process_layout_to_xml(
    layout_data: List[Dict[str, Any]],
    output_xml_path: Path,
    source_filename: str,
    file_key: str,
    prep_mod: Any,
    exclude_pages: Optional[Set[int]] = None,
) -> int:
    """Layout JSON verisini orijinal 2_preprocessing kuralları ile temizlenmiş XML'e dönüştürür.

    Dönen değer: normalize edilen blok sayısı.
    """
    if exclude_pages is None:
        exclude_pages = set()

    root = ET.Element("document")
    root.set("source", source_filename)

    included_pages = 0
    normalized_blocks = 0

    for page_data in layout_data:
        p_num = int(page_data.get("page_num", 0))
        if p_num in exclude_pages:
            continue

        included_pages += 1
        page_elem = ET.SubElement(root, "page")
        page_elem.set("number", str(p_num))

        segments = page_data.get("segments", {})

        # Başlıklar
        if "title" in segments and segments["title"]:
            titles_elem = ET.SubElement(page_elem, "titles")
            for idx, seg in enumerate(segments["title"], 1):
                orig_t = seg.get("text", "").strip()
                if not orig_t:
                    continue
                norm_t = prep_mod.process_text_with_arabic_marking(orig_t, file_key)
                norm_t = strip_footnote_numbers(norm_t)
                norm_t = strip_verse_markers(norm_t)
                if norm_t != orig_t:
                    normalized_blocks += 1
                t_elem = ET.SubElement(titles_elem, "title")
                t_elem.set("id", str(idx))
                t_elem.text = clean_control_chars(norm_t)

        # Gövde (plain text)
        if "plain text" in segments and segments["plain text"]:
            body_elem = ET.SubElement(page_elem, "body")
            for idx, seg in enumerate(segments["plain text"], 1):
                orig_p = seg.get("text", "").strip()
                if not orig_p:
                    continue
                norm_p = prep_mod.process_text_with_arabic_marking(orig_p, file_key)
                norm_p = strip_footnote_numbers(norm_p)
                norm_p = strip_verse_markers(norm_p)
                if norm_p != orig_p:
                    normalized_blocks += 1
                p_elem = ET.SubElement(body_elem, "paragraph")
                p_elem.set("id", str(idx))
                p_elem.text = clean_control_chars(norm_p)

    root.set("total_pages", str(included_pages))
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    return normalized_blocks


def convert_xml_to_sentences(
    xml_path: Path,
    book_meta: Dict[str, Any],
    min_words: int = 3,
    max_words: int = 80,
    discarded_details: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """XML dosyasından cümleleri ayıklar, kelime filtresi ve cümle IJMES kalite kapısını uygular."""
    if not xml_path.exists():
        return [], 0

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"XML okuma hatası ({xml_path}): {e}")
        return [], 0

    kitap_id = book_meta.get("kitap_id", xml_path.stem)
    baslik = book_meta.get("baslik", xml_path.stem)
    yazar = book_meta.get("yazar", "")
    yuzyil_str = str(book_meta.get("yuzyil", "")).strip()

    # donem: int ya da None
    donem: Optional[int] = None
    if yuzyil_str.isdigit():
        donem = int(yuzyil_str)

    records: List[Dict[str, Any]] = []
    discarded_count = 0
    sira = 1

    for page in root.findall("page"):
        texts_to_process: List[str] = []

        titles = page.find("titles")
        if titles is not None:
            for t in titles.findall("title"):
                if t.text and t.text.strip():
                    texts_to_process.append(t.text.strip())

        body = page.find("body")
        if body is not None:
            for p in body.findall("paragraph"):
                if p.text and p.text.strip():
                    texts_to_process.append(p.text.strip())

        for raw_text in texts_to_process:
            sentences = split_sentences(raw_text, min_len=0)

            for s in sentences:
                is_valid, _ = filter_sentence_words(
                    s, min_words=min_words, max_words=max_words
                )
                if not is_valid:
                    discarded_count += 1
                    continue

                # Nüsha / aparat satırları (harf olmayan ve tek harfli belirteç oranı > %50)
                if is_apparatus_line(s):
                    discarded_count += 1
                    if discarded_details is not None and "apparatus" in discarded_details:
                        discarded_details["apparatus"].append(s)
                    continue

                # Editörün bugünkü Türkçe nesir çevirileri (8+ kelime ve hiç IJMES özel harfi içermeyen)
                if is_editor_prose_translation(s):
                    discarded_count += 1
                    if discarded_details is not None and "editor_prose" in discarded_details:
                        discarded_details["editor_prose"].append(s)
                    continue

                # Cümle düzeyinde IJMES izinli olmayan karakter oranı denetimi (%2)
                if not is_valid_ijmes_sentence(s, max_disallowed_ratio=0.02):
                    discarded_count += 1
                    continue

                latin = normalize_latin(s)
                latin_sade = normalize_latin(s, strip_diacritics=True)

                if not latin:
                    discarded_count += 1
                    continue

                records.append({
                    "kaynak": "latoc",
                    "id": f"latoc:{kitap_id}:{sira}",
                    "kitap_id": kitap_id,
                    "baslik": baslik,
                    "yazar": yazar,
                    "yuzyil": yuzyil_str if yuzyil_str else "bilinmiyor",
                    "donem": donem,
                    "latin": latin,
                    "latin_sade": latin_sade,
                    "uslup": "ijmes",
                })
                sira += 1

    return records, discarded_count


def compute_latoc_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cümle derlemi üzerinde istatistikleri ve frekansları hesaplar."""
    total_sentences = len(records)
    total_words = 0
    century_counts: Dict[str, int] = {}
    word_counter: Counter = Counter()

    for r in records:
        text = r.get("latin", "")
        cy = str(r.get("yuzyil", "bilinmiyor"))
        century_counts[cy] = century_counts.get(cy, 0) + 1

        words = [w for w in re.split(r"\s+", text.strip()) if w]
        total_words += len(words)

        for w in words:
            clean_w = tr_lower(w.strip(".,;:!?\"'()[]{}«»—–-"))
            if len(clean_w) >= 2:
                word_counter[clean_w] += 1

    return {
        "toplam_cumle": total_sentences,
        "toplam_kelime": total_words,
        "yuzyil_dagilimi": century_counts,
        "en_sik_30_kelime": word_counter.most_common(30),
    }


def run_latoc_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    excluded_jsonl: Path = DEFAULT_EXCLUDED_JSONL,
    third_party_dir: Path = DEFAULT_THIRD_PARTY_DIR,
) -> Dict[str, Any]:
    """LATOC boru hattını orijinal karakter kuralları ve kalite kapısıyla uçtan uca çalıştırır."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    layout_dir = raw_dir / "layout"
    xml_cleaned_dir = raw_dir / "xml_cleaned"
    xml_cleaned_dir.mkdir(parents=True, exist_ok=True)

    prep_mod = load_latoc_preprocessing_module(third_party_dir)
    all_books, _ = load_latoc_metadata(third_party_dir)
    exclusions = parse_exclusion_file(third_party_dir / "exclude_pages.txt")

    downloaded_books: List[Dict[str, Any]] = []
    for b in all_books:
        lpath = layout_dir / f"{b['stem']}_layout.json"
        if lpath.exists():
            downloaded_books.append(b)

    book_reports = []
    accepted_records: List[Dict[str, Any]] = []
    excluded_records: List[Dict[str, Any]] = []
    excluded_books: List[str] = []
    total_sentence_discarded = 0
    total_apparatus_discarded = 0
    total_editor_discarded = 0
    apparatus_samples: List[str] = []
    editor_samples: List[str] = []

    print("=" * 80)
    print("LATOC KİTAPLARI İŞLENİYOR (Orijinal Kurallar + IJMES Kalite Kapısı)")
    print("=" * 80)

    for book in downloaded_books:
        stem = book["stem"]
        kitap_id = book["kitap_id"]
        layout_path = layout_dir / f"{stem}_layout.json"
        xml_path = xml_cleaned_dir / f"{stem}.xml"

        file_key = prep_mod.get_file_key(book["file"])
        controversy_rules = prep_mod.CONTROVERSY_CACHE.get(file_key, {})
        priority_rules = prep_mod.PRIORITY_RULES_CACHE.get(file_key, {})

        # Dışlanan sayfaları bul
        clean_stem = stem.replace("_", "").replace(" ", "").lower()
        excl_pages = set()
        for k, p_set in exclusions.items():
            k_clean = k.replace(".xml", "").replace("_", "").replace(" ", "").lower()
            if k_clean == clean_stem or k_clean in clean_stem:
                excl_pages = p_set
                break

        # Ham metinlerdeki izinsiz karakter oranını ölç
        with open(layout_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)

        raw_all_text = []
        for p in layout_data:
            segs = p.get("segments", {})
            for stype in ("plain text", "title"):
                for seg in segs.get(stype, []):
                    raw_all_text.append(seg.get("text", ""))

        ratio_before = disallowed_char_ratio(" ".join(raw_all_text)) * 100

        # XML üretimi (orijinal 2_preprocessing kuralları ile)
        normalized_blocks = process_layout_to_xml(
            layout_data=layout_data,
            output_xml_path=xml_path,
            source_filename=book["file"],
            file_key=file_key,
            prep_mod=prep_mod,
            exclude_pages=excl_pages,
        )

        # XML -> Cümleler (cümle IJMES, aparat ve editör çevirisi filtreleri dahil)
        details: Dict[str, List[str]] = {"apparatus": [], "editor_prose": []}
        sentences, discarded = convert_xml_to_sentences(
            xml_path, book, discarded_details=details
        )
        total_sentence_discarded += discarded
        total_apparatus_discarded += len(details["apparatus"])
        total_editor_discarded += len(details["editor_prose"])
        if len(apparatus_samples) < 5:
            apparatus_samples.extend(details["apparatus"][: 5 - len(apparatus_samples)])
        if len(editor_samples) < 5:
            editor_samples.extend(details["editor_prose"][: 5 - len(editor_samples)])

        # Kalan cümlelerde kitap düzeyinde izinsiz karakter oranı (%0.5)
        combined_sentences_text = " ".join(r["latin"] for r in sentences)
        ratio_after = disallowed_char_ratio(combined_sentences_text) * 100

        is_high_quality = ratio_after <= 0.5

        if is_high_quality:
            accepted_records.extend(sentences)
        else:
            excluded_books.append(kitap_id)
            for r in sentences:
                r["kalite_dusuk"] = True
            excluded_records.extend(sentences)

        book_reports.append({
            "kitap_id": kitap_id,
            "file": book["file"],
            "file_key": file_key,
            "kurallar": len(controversy_rules) + len(priority_rules),
            "oran_once": ratio_before,
            "oran_sonra": ratio_after,
            "cumle_sayisi": len(sentences),
            "kalite": "KABUL" if is_high_quality else "DÜŞÜK",
            "atilan_sayfa_sayisi": len(excl_pages),
        })

    # Dosyaları yaz
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, accepted_records)

    if excluded_records:
        excluded_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(excluded_jsonl, excluded_records)

    stats = compute_latoc_stats(accepted_records)

    return {
        "kitap_sayisi": len(downloaded_books),
        "kabul_kitap_sayisi": len(downloaded_books) - len(excluded_books),
        "atilan_kitap_sayisi": len(excluded_books),
        "atilan_kitaplar": excluded_books,
        "kitap_raporlari": book_reports,
        "toplam_cumle": stats["toplam_cumle"],
        "toplam_kelime": stats["toplam_kelime"],
        "elenen_cumle": total_sentence_discarded,
        "atilan_aparat_sayisi": total_apparatus_discarded,
        "atilan_aparat_ornekler": apparatus_samples,
        "atilan_editor_cevirisi_sayisi": total_editor_discarded,
        "atilan_editor_ornekler": editor_samples,
        "yuzyil_dagilimi": stats["yuzyil_dagilimi"],
        "en_sik_30_kelime": stats["en_sik_30_kelime"],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    print("=" * 80)
    print("LATOC DERLEMİ İŞLEME VE IJMES DÖNÜŞTÜRME")
    print("=" * 80)

    report = run_latoc_pipeline()

    print("\nKİTAP BAZINDA ÖNCE / SONRA İZİNSİZ KARAKTER VE KURAL UYGULAMASI:")
    print("-" * 80)
    print(f"{'ID':<6} {'Kitap Dosyası':<38} {'Kurallar':<9} {'Önceki %':<9} {'Sonraki %':<10} {'Durum':<8}")
    print("-" * 80)
    for r in report["kitap_raporlari"]:
        print(f"{r['kitap_id']:<6} {r['file'][:36]:<38} {r['kurallar']:<9} %{r['oran_once']:<7.2f} %{r['oran_sonra']:<8.2f} {r['kalite']:<8}")
    print("-" * 80)
    print(f"Toplam Kabul Edilen Kitap : {report['kabul_kitap_sayisi']}")
    print(f"Kalite Düşük Atılan Kitap : {report['atilan_kitap_sayisi']} ({report['atilan_kitaplar']})")
    print(f"Toplam Cümle Sayısı       : {report['toplam_cumle']:,}")
    print(f"Toplam Kelime Sayısı      : {report['toplam_kelime']:,}")
    print(f"Elenen Cümle Sayısı       : {report['elenen_cumle']:,}")
    print(f"  - Atılan Aparat/Nüsha   : {report['atilan_aparat_sayisi']:,}")
    for idx, s in enumerate(report["atilan_aparat_ornekler"], 1):
        print(f"      {idx}. [Aparat] {s}")
    print(f"  - Atılan Editör Çevirisi: {report['atilan_editor_cevirisi_sayisi']:,}")
    for idx, s in enumerate(report["atilan_editor_ornekler"], 1):
        print(f"      {idx}. [Editör] {s}")
    print("-" * 80)
    print("En Sık 30 Kelime:")
    words_fmt = ", ".join(f"{w} ({c:,})" for w, c in report["en_sik_30_kelime"])
    print(f"  {words_fmt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
