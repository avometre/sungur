"""Internet Archive Osmanlıca metin kataloğu ve deneme indirme modülü.

Bu modül;
1. IA advancedsearch API'sini kullanarak mediatype:texts ve Osmanlıca dil etiketlerine
   (ota, "Ottoman Turkish", "Turkish, Ottoman", "osmanlıca") sahip eserleri sorgular.
2. Sayfalama (rows=500) ile tüm kayıtları çeker, dosya biçimlerini (has_pdf, has_jp2_zip, has_djvu_txt)
   ve toplam boyutları hesaplar.
3. Kamu malı süzgecini (yıl <= 1928 ve açık/boş lisans) uygular; yılı bilinmeyenleri işaretler.
4. Kataloğu data/raw/ia_catalog.jsonl olarak kaydeder.
5. Seçilen 5 farklı kamu malı eserden ilk 10 sayfa görüntüsünü JPG (uzun kenar <= 2000px)
   olarak indirir (data/raw/ia/<identifier>/page_0001.jpg ...).
6. 3 eserin djvu OCR metninden ilk 20 satırı inceleme için ayıklar.
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = Path("data/raw/ia_catalog.jsonl")
DEFAULT_TRIAL_DIR = Path("data/raw/ia")

ADVANCED_SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_API_URL = "https://archive.org/metadata"
DOWNLOAD_BASE_URL = "https://archive.org/download"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,*/*",
}

OTA_QUERY = (
    'mediatype:texts AND (language:ota OR language:"Ottoman Turkish" '
    'OR language:"Turkish, Ottoman" OR language:"osmanlıca")'
)

QUERY_VARIATIONS = {
    "language:ota": "mediatype:texts AND language:ota",
    'language:"Ottoman Turkish"': 'mediatype:texts AND language:"Ottoman Turkish"',
    'language:"Turkish, Ottoman"': 'mediatype:texts AND language:"Turkish, Ottoman"',
    'language:"osmanlıca"': 'mediatype:texts AND language:"osmanlıca"',
    "TÜMÜ (COMBINED)": OTA_QUERY,
}


def query_ia_count(query: str) -> int:
    """Belirtilen IA sorgusunun toplam sonuç sayısını döndürür."""
    params = {
        "q": query,
        "fl[]": "identifier",
        "rows": "0",
        "output": "json",
    }
    url = f"{ADVANCED_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return int(data.get("response", {}).get("numFound", 0))


def query_language_variations() -> Dict[str, int]:
    """Farklı dil etiketi yazımlarının kaçar sonuç verdiğini sorgular."""
    counts = {}
    for label, q in QUERY_VARIATIONS.items():
        try:
            cnt = query_ia_count(q)
            counts[label] = cnt
        except Exception as e:
            logger.warning(f"Sorgu hatası ({label}): {e}")
            counts[label] = -1
    return counts


def parse_year(year_val: Any, date_str: Any) -> Tuple[Optional[int], bool]:
    """Yıl ve tarih alanlarından sayısal yılı ayıklar.

    Dönen değer: (sayısal_yıl, yil_bilinmiyor_mu)
    """
    if year_val is not None:
        try:
            y = int(year_val)
            if 1000 <= y <= 2100:
                return y, False
        except (ValueError, TypeError):
            pass

    if date_str:
        m = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", str(date_str))
        if m:
            return int(m.group(1)), False

    return None, True


def is_public_domain(
    year_val: Optional[int],
    yil_bilinmiyor: bool,
    licenseurl: str,
    rights: str,
) -> bool:
    """Eserin kamu malı kriterini sağlayıp sağlamadığını belirler.

    Kriter:
    - Yıl <= 1928 (Harf İnkılabı ve telif öncesi)
    - VE licenseurl boş veya publicdomain / creativecommons / cc0 içeriyor
      (telif kısıtlaması içermeyenler).
    """
    if yil_bilinmiyor or year_val is None:
        return False

    if year_val > 1928:
        return False

    lic_lower = (licenseurl or "").lower()
    rights_lower = (rights or "").lower()

    # Kısıtlayıcı lisanslar elenir
    if "all rights reserved" in rights_lower or "copyright" in rights_lower:
        return False

    if not lic_lower:
        return True

    if any(k in lic_lower for k in ["publicdomain", "creativecommons.org", "cc0", "pdm", "by", "zero"]):
        return True

    return False


def enrich_item(doc: Dict[str, Any]) -> Dict[str, Any]:
    """IA arama sonucunu katalog şemasına göre zenginleştirir."""
    raw_formats = doc.get("format", [])
    if isinstance(raw_formats, str):
        formats = [raw_formats]
    elif isinstance(raw_formats, list):
        formats = [str(f) for f in raw_formats]
    else:
        formats = []

    has_pdf = any("pdf" in f.lower() for f in formats)
    has_jp2_zip = any("jp2" in f.lower() for f in formats)
    has_djvu_txt = any("djvu" in f.lower() and "txt" in f.lower() for f in formats)

    item_size = int(doc.get("item_size", 0) or 0)
    toplam_boyut_mb = round(item_size / (1024 * 1024), 2)

    year_val, yil_bilinmiyor = parse_year(doc.get("year"), doc.get("date"))
    licenseurl = str(doc.get("licenseurl") or "")
    rights = str(doc.get("rights") or "")

    kamu_mali = is_public_domain(
        year_val=year_val,
        yil_bilinmiyor=yil_bilinmiyor,
        licenseurl=licenseurl,
        rights=rights,
    )

    collections = doc.get("collection", [])
    if isinstance(collections, str):
        collections = [collections]

    return {
        "identifier": doc.get("identifier", ""),
        "title": doc.get("title", ""),
        "creator": doc.get("creator", ""),
        "date": doc.get("date", ""),
        "year": year_val,
        "yil_bilinmiyor": yil_bilinmiyor,
        "kamu_mali": kamu_mali,
        "language": doc.get("language", ""),
        "licenseurl": licenseurl,
        "rights": rights,
        "imagecount": int(doc.get("imagecount", 0) or 0),
        "collection": collections,
        "has_pdf": has_pdf,
        "has_jp2_zip": has_jp2_zip,
        "has_djvu_txt": has_djvu_txt,
        "toplam_boyut_mb": toplam_boyut_mb,
    }


def fetch_all_ia_catalog(
    query: str = OTA_QUERY,
    rows_per_page: int = 500,
    delay: float = 1.0,
) -> List[Dict[str, Any]]:
    """Tüm Osmanlıca metin kataloğunu sayfalama ile çeker."""
    catalog: List[Dict[str, Any]] = []
    page = 1

    fields = [
        "identifier",
        "title",
        "creator",
        "date",
        "year",
        "language",
        "licenseurl",
        "rights",
        "imagecount",
        "collection",
        "format",
        "item_size",
    ]

    while True:
        params = {
            "q": query,
            "rows": str(rows_per_page),
            "page": str(page),
            "output": "json",
        }
        query_string = urllib.parse.urlencode(params)
        for f in fields:
            query_string += f"&fl[]={f}"

        url = f"{ADVANCED_SEARCH_URL}?{query_string}"
        logger.info(f"Sayfa {page} çekiliyor: {url[:80]}...")

        time.sleep(delay)
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"IA arama hatası (sayfa {page}): {e}")
            break

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            break

        for doc in docs:
            enriched = enrich_item(doc)
            catalog.append(enriched)

        num_found = data.get("response", {}).get("numFound", 0)
        print(f"Çekilen kayıt: {len(catalog)} / {num_found} (Sayfa {page})")

        if len(catalog) >= num_found or len(docs) < rows_per_page:
            break

        page += 1

    return catalog


def select_trial_candidates(
    catalog: List[Dict[str, Any]],
    count: int = 5,
) -> List[Dict[str, Any]]:
    """Farklı koleksiyonlardan kamu malı ve has_jp2_zip olan deneme adaylarını seçer."""
    eligible = [
        item for item in catalog
        if item.get("kamu_mali") is True
        and item.get("has_jp2_zip") is True
        and item.get("has_pdf") is True
        and item.get("imagecount", 0) >= 10
    ]

    selected: List[Dict[str, Any]] = []
    seen_collections: Set[str] = set()

    # Önce farklı koleksiyonlardan seç
    for item in eligible:
        colls = item.get("collection", [])
        primary_coll = colls[0] if colls else "genel"
        if primary_coll not in seen_collections:
            seen_collections.add(primary_coll)
            selected.append(item)
            if len(selected) >= count:
                break

    # Eğer yetmezse kalan uygunlardan tamamla
    if len(selected) < count:
        for item in eligible:
            if item not in selected:
                selected.append(item)
                if len(selected) >= count:
                    break

    return selected


def download_trial_item_pages(
    identifier: str,
    output_dir: Path,
    max_pages: int = 10,
    max_edge: int = 2000,
    delay: float = 0.5,
) -> int:
    """Tek bir eserin ilk max_pages sayfasını JPG olarak indirir (uzun kenar <= max_edge px)."""
    item_dir = output_dir / identifier
    item_dir.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0

    # 1. Yöntem (Birincil): IA BookReader sayfa görüntüsü URL'leri
    for p_idx in range(max_pages):
        page_num = p_idx + 1
        out_jpg = item_dir / f"page_{page_num:04d}.jpg"
        if out_jpg.exists() and out_jpg.stat().st_size > 1000:
            downloaded_count += 1
            continue

        page_url = f"{DOWNLOAD_BASE_URL}/{identifier}/page/n{p_idx}_w{max_edge}.jpg"
        time.sleep(delay)
        req_p = urllib.request.Request(page_url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req_p, timeout=20) as resp:
                data = resp.read()
                if len(data) > 1000:
                    out_jpg.write_bytes(data)
                    downloaded_count += 1
        except Exception as e:
            logger.debug(f"Sayfa {p_idx} API ile alınamadı ({identifier}): {e}")

    if downloaded_count >= max_pages:
        return downloaded_count

    # 2. Yöntem (Yedek): PDF üzerinden PyMuPDF ile render et
    try:
        import fitz
        pdf_url = f"{DOWNLOAD_BASE_URL}/{identifier}/{identifier}.pdf"
        temp_pdf = item_dir / f"{identifier}_sample.pdf"
        time.sleep(delay)
        req = urllib.request.Request(pdf_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            temp_pdf.write_bytes(resp.read())

        if temp_pdf.exists():
            doc = fitz.open(temp_pdf)
            pages_to_render = min(max_pages, len(doc))
            for p_idx in range(pages_to_render):
                out_jpg = item_dir / f"page_{p_idx+1:04d}.jpg"
                if not out_jpg.exists():
                    page = doc[p_idx]
                    rect = page.rect
                    scale = max_edge / max(rect.width, rect.height) if max(rect.width, rect.height) > 0 else 1.0
                    mat = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=mat)
                    pix.save(str(out_jpg))
                    downloaded_count += 1
            doc.close()
            temp_pdf.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"PDF fallback render hatası ({identifier}): {e}")

    return downloaded_count


def fetch_ocr_sample_lines(identifier: str, max_lines: int = 20, delay: float = 1.0) -> List[str]:
    """IA djvu.txt OCR dosyasından ilk max_lines satırı çeker."""
    url = f"{DOWNLOAD_BASE_URL}/{identifier}/{identifier}_djvu.txt"
    time.sleep(delay)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read(100000).decode("utf-8", errors="ignore")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return lines[:max_lines]
    except Exception as e:
        logger.warning(f"OCR metni çekme hatası ({identifier}): {e}")
        return []


def compute_catalog_statistics(catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Katalog kayıtları üzerinden özet istatistikleri hesaplar."""
    total_items = len(catalog)
    kamu_mali_count = sum(1 for c in catalog if c.get("kamu_mali") is True)
    yil_bilinmiyor_count = sum(1 for c in catalog if c.get("yil_bilinmiyor") is True)

    total_size_mb = sum(c.get("toplam_boyut_mb", 0.0) for c in catalog)
    total_size_gb = round(total_size_mb / 1024, 2)

    # On yıllık (decade) dağılım
    decade_counts: Counter = Counter()
    for c in catalog:
        y = c.get("year")
        if y is not None:
            dec = (y // 10) * 10
            decade_counts[f"{dec}ler"] += 1

    # Koleksiyonlar
    coll_counts: Counter = Counter()
    for c in catalog:
        for coll in c.get("collection", []):
            coll_counts[coll] += 1

    return {
        "toplam_oge": total_items,
        "kamu_mali_sayisi": kamu_mali_count,
        "yil_bilinmiyor_sayisi": yil_bilinmiyor_count,
        "toplam_boyut_gb": total_size_gb,
        "on_yillik_dagilim": sorted(decade_counts.items(), key=lambda x: x[0]),
        "top_10_koleksiyon": coll_counts.most_common(10),
    }


def run_ia_pipeline(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    trial_dir: Path = DEFAULT_TRIAL_DIR,
    skip_trial_download: bool = False,
) -> Dict[str, Any]:
    """Internet Archive kataloğunu üretir ve deneme indirmesini gerçekleştirir."""
    print("=" * 80)
    print("INTERNET ARCHIVE OSMANLICA METİN KATALOĞU (IA ADVANCED SEARCH)")
    print("=" * 80)

    # 1. Dil Yazımları Karşılaştırması
    print("1. Dil Sorguları Sonuç Sayıları:")
    variation_counts = query_language_variations()
    for lbl, count in variation_counts.items():
        print(f"   - {lbl:<25}: {count:,} eser")
    print("-" * 80)

    # 2. Tam Kataloğu Çekme
    catalog = []
    if catalog_path.exists():
        catalog = list(read_jsonl(catalog_path))
    if not catalog or len(catalog) < 2000:
        print("2. Katalog Çekiliyor (sayfalama rows=500)...")
        catalog = fetch_all_ia_catalog()
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(catalog_path, catalog)
        print(f"Katalog başarıyla '{catalog_path}' dosyasına yazıldı ({len(catalog):,} kayıt).")
    else:
        print(f"2. Mevcut Katalog Yüklendi: '{catalog_path}' ({len(catalog):,} kayıt).")
    print("-" * 80)

    # 3. İstatistikler
    stats = compute_catalog_statistics(catalog)
    print("3. Katalog İstatistikleri:")
    print(f"   Toplam Öğe Sayısı        : {stats['toplam_oge']:,}")
    print(f"   Kamu Malı Öğe Sayısı     : {stats['kamu_mali_sayisi']:,} (yıl <= 1928)")
    print(f"   Yılı Bilinmeyen Sayısı   : {stats['yil_bilinmiyor_sayisi']:,}")
    print(f"   Toplam Boyut Tahmini     : {stats['toplam_boyut_gb']:,} GB ({stats['toplam_boyut_gb']/1024:.2f} TB)")
    print("-" * 80)
    print("On Yıllık Dağılım:")
    for dec, c in stats["on_yillik_dagilim"]:
        print(f"   - {dec:<10}: {c:,} eser")
    print("-" * 80)
    print("İlk 10 Koleksiyon:")
    for coll, c in stats["top_10_koleksiyon"]:
        print(f"   - {coll:<30}: {c:,} eser")
    print("-" * 80)

    # 4. Deneme İndirmesi (5 eser, ilk 10 sayfa)
    candidates = select_trial_candidates(catalog, count=5)
    trial_results = []
    print(f"4. Deneme İndirmesi (5 farklı koleksiyondan eser seçildi):")
    for idx, cand in enumerate(candidates, 1):
        ident = cand["identifier"]
        title = cand["title"]
        coll = cand["collection"][0] if cand.get("collection") else "genel"
        print(f"   [{idx}/5] {ident} | {title[:35]} | Kol: {coll}")

        if not skip_trial_download:
            pages_saved = download_trial_item_pages(
                identifier=ident,
                output_dir=trial_dir,
                max_pages=10,
                max_edge=2000,
            )
            print(f"        -> {pages_saved} sayfa kaydedildi: {trial_dir / ident}")
            trial_results.append({
                "identifier": ident,
                "title": title,
                "collection": coll,
                "saved_pages": pages_saved,
                "dir": str(trial_dir / ident),
            })
        else:
            trial_results.append({
                "identifier": ident,
                "title": title,
                "collection": coll,
                "saved_pages": 0,
                "dir": str(trial_dir / ident),
            })

    # 5. IA OCR Metni İncelemesi (has_djvu_txt olan 3 eser)
    ocr_samples = {}
    djvu_candidates = [c for c in candidates if c.get("has_djvu_txt")][:3]
    if len(djvu_candidates) < 3:
        djvu_candidates = [c for c in catalog if c.get("has_djvu_txt") and c.get("kamu_mali")][:3]

    print("-" * 80)
    print(f"5. IA Arap Harfli OCR Metin Örnekleri (3 Eser, İlk 20 Satır):")
    for c in djvu_candidates:
        ident = c["identifier"]
        print(f"\n--- Eser: {ident} ({c.get('title')}) ---")
        lines = fetch_ocr_sample_lines(ident, max_lines=20)
        ocr_samples[ident] = lines
        for l_idx, line in enumerate(lines[:20], 1):
            print(f"  {l_idx:02d}: {line}")

    return {
        "variation_counts": variation_counts,
        "catalog_size": len(catalog),
        "statistics": stats,
        "trial_results": trial_results,
        "ocr_samples": ocr_samples,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Internet Archive Osmanlıca metin kataloğu ve deneme indirme")
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH, help="Katalog JSONL çıktı yolu")
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR, help="Deneme sayfa görselleri dizini")
    parser.add_argument("--skip-trial-download", action="store_true", help="Görsel indirmeyi atla")
    args = parser.parse_args()

    run_ia_pipeline(
        catalog_path=args.catalog_path,
        trial_dir=args.trial_dir,
        skip_trial_download=args.skip_trial_download,
    )


if __name__ == "__main__":
    main()
