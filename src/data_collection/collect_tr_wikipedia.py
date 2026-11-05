"""Türkçe Vikipedi (tr.wikipedia) sayfalarından Osmanlıca isim ve kavram çiftlerini toplar.

Kalıplar:
- Giriş cümleleri ve bilgi kutuları:
  - "Osmanlıca: <Arap>"
  - "Osmanlı Türkçesi: <Arap>"
  - "{{dil|ota|<Arap>}}"
- Latin ad: Madde başlığı veya parantez öncesindeki kalın yazılmış ad ('''...''')
Filtreler:
- Arapça betik oranı > 0.9
- Ünsüz iskelet benzerliği >= 0.5
- Altın set sızıntı kontrolü
Kayıt formatı: {"sayfa": str, "arap": str, "latin": str, "iskelet_benzerlik": float, "kaynak": "tr_wikipedia", "lisans": "CC BY-SA 4.0"}
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import requests

from src.common.normalize import arabic_letter_count, normalize_latin, normalize_ottoman
from src.gold.leakage import is_heldout, load_heldout_all
from src.translit.skeleton import skeleton_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

USER_AGENT = "SungurResearchBot/0.1 (non-commercial research)"
HEADERS = {"User-Agent": USER_AGENT}
WIKI_API_URL = "https://tr.wikipedia.org/w/api.php"
OUTPUT_PATH = Path("data/interim/tr_wikipedia_ota.jsonl")

# Çıkarım Regex Kalıpları
OTA_TEMPLATE_PATTERN = re.compile(r"\{\{dil\|ota\|([^|}]+)\}\}", re.IGNORECASE)
OTA_TEXT_PATTERN = re.compile(
    r"(?:Osmanlıca|Osmanlı\s*Türkçesi)\s*:\s*(?:'{2,5})?([^\)',;<\n]+)(?:'{2,5})?",
    re.IGNORECASE,
)
BOLD_TITLE_PATTERN = re.compile(r"'{3}([^']{2,60})'{3}")


def safe_request(url: str, params: Dict[str, Any], max_retries: int = 5) -> Optional[requests.Response]:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            time.sleep(1.0)  # Kural: İstekler arası en az 1 sn
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in (429, 502, 503, 504):
                logger.warning(f"Durum kodu {resp.status_code}. {delay:.1f} sn bekleniyor...")
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error(f"Hata kodu: {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.RequestException as e:
            logger.warning(f"İstek hatası: {e}. {delay:.1f} sn bekleniyor...")
            time.sleep(delay)
            delay *= 2.0
    return None


def search_wikipedia_ota_pages(max_pages: int = 1500) -> List[str]:
    """'Osmanlıca:' kalıbını içeren Vikipedi maddelerini arar."""
    titles: List[str] = []
    seen: Set[str] = set()
    offset = 0
    limit = 500

    logger.info(f"tr.wikipedia arama başlatılıyor (hedef: ~{max_pages} sayfa)...")
    while offset < max_pages:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": '"Osmanlıca:" OR "Osmanlı Türkçesi:" OR "insource:dil|ota"',
            "srnamespace": 0,
            "srlimit": min(limit, max_pages - offset),
            "sroffset": offset,
            "format": "json",
        }
        resp = safe_request(WIKI_API_URL, params)
        if not resp:
            break
        data = resp.json()
        search_hits = data.get("query", {}).get("search", [])
        if not search_hits:
            break

        for hit in search_hits:
            t = hit["title"]
            if t not in seen:
                seen.add(t)
                titles.append(t)

        offset += len(search_hits)
        logger.info(f"Bulunan sayfa başlığı: {len(titles)} (offset: {offset})")
        if len(search_hits) < min(limit, max_pages - offset):
            break

    return titles


def extract_pairs_from_wikitext(title: str, wikitext: str) -> List[Tuple[str, str]]:
    """Vikimetinden (Arap, Latin) aday çiftlerini çıkarır."""
    pairs: List[Tuple[str, str]] = []

    # Olası Latin adayları: Sayfa başlığı ve ilk paragraftaki kalın yazılar
    latin_candidates: List[str] = []
    # Parantez içindeki açıklamayı başlıkta temizle (ör. "Mehmed Said Paşa (sadrazam)" -> "Mehmed Said Paşa")
    clean_title = re.sub(r"\s*\([^)]*\)", "", title).strip()
    latin_candidates.append(clean_title)

    # İlk 1500 karakterdeki kalın metinler
    lead_text = wikitext[:2000]
    for b in BOLD_TITLE_PATTERN.findall(lead_text):
        b_clean = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", b).strip()
        if b_clean and len(b_clean) > 1 and b_clean not in latin_candidates:
            # Sadece Latin karakter içerenleri al
            if arabic_letter_count(b_clean) == 0:
                latin_candidates.append(b_clean)

    # 1. {{dil|ota|<Arap>}} şablonları
    for m in OTA_TEMPLATE_PATTERN.finditer(lead_text):
        ar = m.group(1).strip()
        ar = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", ar)
        ar = re.sub(r"'{2,5}", "", ar).strip()
        if ar:
            for lat in latin_candidates:
                pairs.append((ar, lat))

    # 2. "Osmanlıca: <Arap>" metin kalıpları
    for m in OTA_TEXT_PATTERN.finditer(lead_text):
        ar = m.group(1).strip()
        ar = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", ar)
        ar = re.sub(r"'{2,5}", "", ar).strip()
        if ar:
            for lat in latin_candidates:
                pairs.append((ar, lat))

    return pairs


def collect_wikipedia_pairs(
    max_pages: int = 1500,
    output_path: Path = OUTPUT_PATH,
) -> Dict[str, Any]:
    """Vikipedi sayfalarını tarar, çiftleri çıkarır ve doğrular."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_pages, heldout_arap, heldout_titles, heldout_latin = load_heldout_all()

    page_titles = search_wikipedia_ota_pages(max_pages=max_pages)
    logger.info(f"Toplam {len(page_titles)} sayfanın metni çekiliyor...")

    batch_size = 50
    raw_pairs = 0
    discarded_non_arabic = 0
    discarded_skeleton = 0
    discarded_leakage = 0
    valid_records: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for i in range(0, len(page_titles), batch_size):
        chunk = page_titles[i : i + batch_size]
        params = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
        }
        resp = safe_request(WIKI_API_URL, params)
        if not resp:
            continue
        pages_dict = resp.json().get("query", {}).get("pages", {})

        for pid, page in pages_dict.items():
            title = page.get("title", "")
            revs = page.get("revisions", [])
            if not revs:
                continue
            # Slot yapısı veya standart içerik
            wikitext = ""
            if "slots" in revs[0] and "main" in revs[0]["slots"]:
                wikitext = revs[0]["slots"]["main"].get("*", "")
            elif "*" in revs[0]:
                wikitext = revs[0]["*"]

            if not wikitext:
                continue

            extracted = extract_pairs_from_wikitext(title, wikitext)
            for ar, lat in extracted:
                raw_pairs += 1
                pair_key = (ar, lat)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # 1. Filtre: Arapça betik oranı > 0.9
                clean_ar = ar.replace(" ", "").replace("-", "")
                if not clean_ar:
                    continue
                ar_ratio = arabic_letter_count(ar) / len(clean_ar)
                if ar_ratio < 0.9:
                    discarded_non_arabic += 1
                    continue

                # 2. Filtre: İskelet benzerliği >= 0.5
                sim = skeleton_similarity(ar, lat)
                if sim < 0.5:
                    discarded_skeleton += 1
                    continue

                # 3. Filtre: Sızıntı kontrolü
                if is_heldout(
                    arap=ar,
                    latin=lat,
                    tr_title=title,
                    heldout_pages=heldout_pages,
                    heldout_arap=heldout_arap,
                    heldout_titles=heldout_titles,
                    heldout_latin=heldout_latin,
                ):
                    discarded_leakage += 1
                    continue

                rec = {
                    "sayfa": title,
                    "arap": ar,
                    "latin": lat,
                    "iskelet_benzerlik": round(sim, 3),
                    "kaynak": "tr_wikipedia",
                    "lisans": "CC BY-SA 4.0",
                }
                valid_records.append(rec)

        if (i // batch_size) % 5 == 0:
            logger.info(f"İşlenen sayfa: {min(i + batch_size, len(page_titles))}/{len(page_titles)} | Geçerli Çift: {len(valid_records)}")

    # Kaydet
    with open(output_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"tr.wikipedia taraması bitti. Toplam geçerli: {len(valid_records)} -> {output_path}")

    stats = {
        "kaynak": "tr.wikipedia",
        "lisans": "CC BY-SA 4.0",
        "toplam_sayfa": len(page_titles),
        "ham_cift": raw_pairs,
        "gecerli_cift": len(valid_records),
        "atilan_arap_olmayan": discarded_non_arabic,
        "atilan_iskelet": discarded_skeleton,
        "atilan_sizinti": discarded_leakage,
        "dosya": str(output_path),
    }
    logger.info(f"İstatistikler: {stats}")
    return stats


if __name__ == "__main__":
    collect_wikipedia_pairs(max_pages=1500)
