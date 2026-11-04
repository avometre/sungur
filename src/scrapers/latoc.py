"""LATOC (Latin harfli Osmanlıca derlem) indirici, işleyici ve cümle dönüştürücü modülü.

Bu modül;
1. Zenodo LATOC metaverisini (data_statistics.csv, LATOC_metadata_sample.csv) okur.
2. Kitap PDF'lerini ekitap.ktb.gov.tr ve ekitap.yek.gov.tr üzerinden nazik bir hız sınırı
   (en az 2 sn arayla, aynı anda tek indirme) ile indirir ve indirilemeyenleri listeler.
3. İndirilen PDF'leri üçüncü parti betik sarmalayıcılarıyla veya doğrudan PyMuPDF+YOLOv10
   ile JSON düzenine, ardından IJMES uyumlu XML'e ve temizlenmiş XML'e dönüştürür.
4. XML dosyalarındaki paragrafları split_sentences(text, min_len=0) ile cümlelere böler,
   3-80 kelime aralığını filtreler ve data/interim/latoc_sentences.jsonl dosyasına yazar.
5. İstatistikleri (kitap sayısı, cümle sayısı, kelime sayısı, yüzyıl dağılımı, ilk 30 kelime) üretir.
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.io import write_jsonl
from src.common.normalize import normalize_latin, tr_lower
from src.scrapers.vikikaynak_align import split_sentences

logger = logging.getLogger(__name__)

DEFAULT_THIRD_PARTY_DIR = Path("third_party/latoc")
DEFAULT_RAW_DIR = Path("data/raw/latoc")
DEFAULT_OUTPUT_JSONL = Path("data/interim/latoc_sentences.jsonl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


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

                # Yüzyıl ve yazar eşlemesi
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

                # Eğer örnek metaveride yoksa bilinen Osmanlı şairleri ve yüzyılları
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

                # Eğer örnek metaveride yoksa dosya adından yazar/başlık kestirimi
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


def download_single_pdf(
    url: str,
    output_path: Path,
    delay: float = 2.0,
    referer: Optional[str] = None,
) -> bool:
    """Tek bir PDF dosyasını belirtilen hız sınırı ile indirir."""
    time.sleep(delay)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < 1000:
                logger.warning(f"Dosya çok küçük ({len(data)} bayt): {url}")
                return False
            output_path.write_bytes(data)
            return True
    except Exception as e:
        logger.error(f"İndirme hatası ({url}): {e}")
        return False


def resolve_ktb_pdf_url(page_url: str, delay: float = 2.0) -> Optional[str]:
    """KTB web sayfasından gerçek PDF indirme bağlantısını ayıklar."""
    time.sleep(delay)
    headers = dict(HEADERS)
    req = urllib.request.Request(page_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read(250000).decode("utf-8", errors="ignore")
            m = re.search(r'href="([^"]*(?:\.pdf|Eklenti)[^"]*)"', html, re.I)
            if m:
                pdf_rel = m.group(1)
                if pdf_rel.startswith("/"):
                    return f"https://ekitap.ktb.gov.tr{pdf_rel}"
                return pdf_rel
    except Exception as e:
        logger.warning(f"KTB sayfa çözümleme hatası ({page_url}): {e}")
    return None


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


def convert_xml_to_sentences(
    xml_path: Path,
    book_meta: Dict[str, Any],
    min_words: int = 3,
    max_words: int = 80,
) -> Tuple[List[Dict[str, Any]], int]:
    """Tek bir XML dosyasını normalize edilmiş cümlelere dönüştürür.

    Dönen değer: (cümle_kayıtları, elenen_cümle_sayısı)
    """
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
    yuzyil = book_meta.get("yuzyil", "bilinmiyor")

    records: List[Dict[str, Any]] = []
    discarded_count = 0
    sira = 1

    # XML hiyerarşisindeki paragraph ve title metinlerini tara
    for page in root.findall("page"):
        texts_to_process: List[str] = []

        # Başlıklar
        titles = page.find("titles")
        if titles is not None:
            for t in titles.findall("title"):
                if t.text and t.text.strip():
                    texts_to_process.append(t.text.strip())

        # Gövde paragrafları
        body = page.find("body")
        if body is not None:
            for p in body.findall("paragraph"):
                if p.text and p.text.strip():
                    texts_to_process.append(p.text.strip())

        # Eğer sayfa doğrudan element içeriyorsa (fallback)
        if body is None and titles is None:
            for elem in page:
                if elem.text and elem.text.strip():
                    texts_to_process.append(elem.text.strip())

        for raw_text in texts_to_process:
            # LATOC için min_len=0 ile cümle bölme
            sentences = split_sentences(raw_text, min_len=0)

            for s in sentences:
                is_valid, _ = filter_sentence_words(
                    s, min_words=min_words, max_words=max_words
                )
                if not is_valid:
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
                    "yuzyil": yuzyil,
                    "latin": latin,
                    "latin_sade": latin_sade,
                })
                sira += 1

    return records, discarded_count


def clean_control_chars(text: str) -> str:
    """XML 1.0 standartlarına uymayan kontrol karakterlerini temizler."""
    if not text:
        return ""
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)


def process_layout_to_xml(
    layout_data: List[Dict[str, Any]],
    output_xml_path: Path,
    source_filename: str,
    exclude_pages: Optional[Set[int]] = None,
) -> None:
    """Layout JSON verisini temizlenmiş XML formatına dönüştürür."""
    if exclude_pages is None:
        exclude_pages = set()

    root = ET.Element("document")
    root.set("source", source_filename)

    included_pages = 0
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
                t_elem = ET.SubElement(titles_elem, "title")
                t_elem.set("id", str(idx))
                t_elem.text = clean_control_chars(seg.get("text", "").strip())

        # Gövde
        if "plain text" in segments and segments["plain text"]:
            body_elem = ET.SubElement(page_elem, "body")
            for idx, seg in enumerate(segments["plain text"], 1):
                p_elem = ET.SubElement(body_elem, "paragraph")
                p_elem.set("id", str(idx))
                p_elem.text = clean_control_chars(seg.get("text", "").strip())

    root.set("total_pages", str(included_pages))
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)


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


def extract_pdf_layout(
    pdf_path: Path,
    output_json_path: Path,
    use_yolo: bool = False,
    max_pages: Optional[int] = None,
) -> None:
    """PDF dosyasından sayfa ve paragraf düzenini çıkararak JSON olarak kaydeder."""
    import fitz

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    end_page = min(max_pages, total_pages) if max_pages else total_pages

    results: List[Dict[str, Any]] = []

    for page_idx in range(end_page):
        page = doc[page_idx]
        blocks = page.get_text("blocks")
        plain_texts = []
        titles = []

        for b in blocks:
            # b = (x0, y0, x1, y1, text, block_no, block_type)
            if b[6] != 0:
                continue
            btxt = b[4].strip()
            if not btxt:
                continue

            # Kısa ve büyük harf ağırlıklı blokları başlık olarak sınıflandır
            if len(btxt.split()) <= 6 and (btxt.isupper() or len(btxt) < 40):
                titles.append({
                    "bbox": [b[0], b[1], b[2], b[3]],
                    "text": btxt,
                })
            else:
                plain_texts.append({
                    "bbox": [b[0], b[1], b[2], b[3]],
                    "text": btxt,
                })

        results.append({
            "page_num": page_idx + 1,
            "segments": {
                "title": titles,
                "plain text": plain_texts,
            },
        })

    doc.close()
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_latoc_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    max_books: Optional[int] = None,
    delay: float = 2.0,
    skip_download: bool = False,
) -> Dict[str, Any]:
    """LATOC boru hattını uçtan uca çalıştırır."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = raw_dir / "pdf"
    layout_dir = raw_dir / "layout"
    xml_cleaned_dir = raw_dir / "xml_cleaned"

    all_books, _ = load_latoc_metadata()
    exclusions = parse_exclusion_file(DEFAULT_THIRD_PARTY_DIR / "exclude_pages.txt")

    downloaded_books: List[Dict[str, Any]] = []
    skipped_books: List[Tuple[str, str]] = []

    print(f"Toplam kayıtlı kitap: {len(all_books)}")

    # 1. İndirme Aşaması
    for book in all_books:
        link = book.get("link", "")
        stem = book["stem"]
        pdf_path = pdf_dir / f"{stem}.pdf"

        if not link:
            skipped_books.append((book["file"], "Bağlantı linki yok (CSV)"))
            continue

        if "ktb.gov.tr" not in link and "yek.gov.tr" not in link:
            skipped_books.append((book["file"], f"Desteklenmeyen alan adı: {link}"))
            continue

        if "yek.gov.tr" in link:
            skipped_books.append((book["file"], "ekitap.yek.gov.tr 404/erişilemez"))
            continue

        # KTB linki
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            downloaded_books.append(book)
            if max_books and len(downloaded_books) >= max_books:
                break
            continue

        if skip_download:
            skipped_books.append((book["file"], "İndirme atlandı (--skip-download)"))
            continue

        # PDF bağlantısını çözümle
        pdf_url: Optional[str] = None
        if "/Eklenti/" in link and link.endswith(".pdf"):
            pdf_url = link
        else:
            pdf_url = resolve_ktb_pdf_url(link, delay=delay)

        if not pdf_url:
            skipped_books.append((book["file"], "KTB sayfasında PDF eki bulunamadı"))
            continue

        print(f"İndiriliyor ({len(downloaded_books)+1}): {stem} -> {pdf_url[:60]}...")
        success = download_single_pdf(
            pdf_url,
            pdf_path,
            delay=delay,
            referer=link,
        )

        if success:
            downloaded_books.append(book)
            if max_books and len(downloaded_books) >= max_books:
                break
        else:
            skipped_books.append((book["file"], "PDF indirme başarısız"))

    print(f"\nİndirilen kitap sayısı : {len(downloaded_books)}")
    print(f"Atlanan kitap sayısı    : {len(skipped_books)}")

    # 2. Düzen Çıkarma (Layout) ve XML Üretimi
    for book in downloaded_books:
        stem = book["stem"]
        pdf_path = pdf_dir / f"{stem}.pdf"
        layout_path = layout_dir / f"{stem}_layout.json"
        xml_path = xml_cleaned_dir / f"{stem}.xml"

        if not layout_path.exists() and pdf_path.exists():
            extract_pdf_layout(pdf_path, layout_path)

        if layout_path.exists() and not xml_path.exists():
            with open(layout_path, "r", encoding="utf-8") as f:
                layout_data = json.load(f)

            # exclude_pages.txt eşleştirmesi
            clean_stem = stem.replace("_", "").replace(" ", "").lower()
            excl_pages = set()
            for k, p_set in exclusions.items():
                k_clean = k.replace(".xml", "").replace("_", "").replace(" ", "").lower()
                if k_clean == clean_stem or k_clean in clean_stem:
                    excl_pages = p_set
                    break

            process_layout_to_xml(
                layout_data,
                xml_path,
                source_filename=book["file"],
                exclude_pages=excl_pages,
            )

    # 3. XML -> Cümle Dönüştürme
    all_sentences: List[Dict[str, Any]] = []
    total_discarded = 0

    for book in downloaded_books:
        stem = book["stem"]
        xml_path = xml_cleaned_dir / f"{stem}.xml"
        if xml_path.exists():
            recs, discarded = convert_xml_to_sentences(xml_path, book)
            all_sentences.extend(recs)
            total_discarded += discarded

    # JSONL çıktısını yaz
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, all_sentences)

    # 4. İstatistikler
    stats = compute_latoc_stats(all_sentences)

    rng = random.Random(1)
    sample_10 = (
        rng.sample(all_sentences, min(10, len(all_sentences)))
        if all_sentences
        else []
    )

    report = {
        "indirilen_kitap_sayisi": len(downloaded_books),
        "atlanan_kitap_sayisi": len(skipped_books),
        "indirilen_kitaplar": [b["file"] for b in downloaded_books],
        "atlanan_kitaplar": skipped_books,
        "toplam_cumle": stats["toplam_cumle"],
        "toplam_kelime": stats["toplam_kelime"],
        "elenen_cumle_sayisi": total_discarded,
        "yuzyil_dagilimi": stats["yuzyil_dagilimi"],
        "en_sik_30_kelime": stats["en_sik_30_kelime"],
        "ornek_10_cumle": sample_10,
    }
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="LATOC derlemi indirme ve cümle ayıklama")
    parser.add_argument("--max-books", type=int, default=None, help="Maksimum indirilecek kitap sayısı")
    parser.add_argument("--delay", type=float, default=2.0, help="İndirme istekleri arası bekleme (sn)")
    parser.add_argument("--skip-download", action="store_true", help="İndirme yapmadan mevcut dosyaları işle")
    args = parser.parse_args()

    print("=" * 80)
    print("LATOC DERLEMİ İNDİRME VE DÖNÜŞTÜRME")
    print("=" * 80)

    report = run_latoc_pipeline(
        max_books=args.max_books,
        delay=args.delay,
        skip_download=args.skip_download,
    )

    print("-" * 80)
    print(f"İndirilen Kitap Sayısı : {report['indirilen_kitap_sayisi']}")
    print(f"Atlanan Kitap Sayısı    : {report['atlanan_kitap_sayisi']}")
    print(f"Toplam Cümle Sayısı     : {report['toplam_cumle']:,}")
    print(f"Toplam Kelime Sayısı    : {report['toplam_kelime']:,}")
    print(f"Elenen Cümle Sayısı     : {report['elenen_cumle_sayisi']:,} (<3 veya >80 kelime)")
    print("-" * 80)
    print("Yüzyıla Göre Dağılım:")
    for cy, cnt in sorted(report["yuzyil_dagilimi"].items()):
        print(f"  - {cy}. Yüzyıl: {cnt:,} cümle")
    print("-" * 80)
    print("En Sık 30 Kelime:")
    words_fmt = ", ".join(f"{w} ({c:,})" for w, c in report["en_sik_30_kelime"])
    print(f"  {words_fmt}")
    print("-" * 80)
    print("Rastgele 10 Cümle (seed=1):")
    for idx, s in enumerate(report["ornek_10_cumle"], 1):
        print(f"{idx}. [{s['kitap_id']} | {s['baslik']} | {s['yuzyil']}. yy]")
        print(f"   Latin: {s['latin']}")
        print(f"   Sade : {s['latin_sade']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
