"""Wikidata'dan Osmanlıca (ota) ve Türkçe (tr) etiket/takma ad çiftlerini toplar.

Lisans: CC0 1.0 Universal (Public Domain Dedication)
Kural: User-Agent: 'SungurResearchBot/0.1 (non-commercial research)'
İstekler arası en az 1 sn bekleme, 429 durumunda üstel geri çekilme.
Filtreler:
- Arap betiği oranı > 0.9
- Ünsüz iskelet benzerliği >= 0.5 (çeviri değil çeviriyazı)
- Altın set sızıntı kontrolü (leakage protection)
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
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
OUTPUT_PATH = Path("data/interim/wikidata_ota_tr.jsonl")


def safe_request(url: str, params: Dict[str, Any], max_retries: int = 5) -> Optional[requests.Response]:
    """İstekler arasında 1 sn bekleyen ve 429/503 durumunda üstel geri çekilen GET fonksiyonu."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            time.sleep(1.0)  # Kural: İstekler arası en az 1 sn
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in (429, 502, 503, 504):
                logger.warning(f"Durum kodu {resp.status_code}. {delay:.1f} sn bekleniyor (Deneme {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error(f"Beklenmeyen durum kodu: {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Bağlantı hatası: {e}. {delay:.1f} sn bekleniyor...")
            time.sleep(delay)
            delay *= 2.0
    return None


def fetch_wikidata_ota_entities(max_search_items: int = 2500) -> List[str]:
    """haslabel:ota haslabel:tr aramasını sayfalı olarak yaparak Q-ID listesi döner."""
    qids: List[str] = []
    seen: Set[str] = set()
    offset = 0
    limit = 500  # API üst sınırı

    logger.info(f"Wikidata arama başlatılıyor (hedef: ~{max_search_items} öğe)...")
    while offset < max_search_items:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": "haslabel:ota haslabel:tr",
            "srnamespace": 0,
            "srlimit": min(limit, max_search_items - offset),
            "sroffset": offset,
            "format": "json",
        }
        resp = safe_request(WIKIDATA_API_URL, params)
        if not resp:
            break
        data = resp.json()
        search_hits = data.get("query", {}).get("search", [])
        if not search_hits:
            break

        for hit in search_hits:
            qid = hit["title"]
            if qid not in seen:
                seen.add(qid)
                qids.append(qid)

        offset += len(search_hits)
        logger.info(f"Toplanan Q-ID sayısı: {len(qids)} (offset: {offset})")
        if len(search_hits) < min(limit, max_search_items - offset):
            break

    return qids


def collect_wikidata_pairs(
    max_search_items: int = 2500,
    output_path: Path = OUTPUT_PATH,
) -> Dict[str, Any]:
    """Wikidata öğelerini toplar, filtreler, sızıntı kontrolü yapar ve kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_pages, heldout_arap, heldout_titles, heldout_latin = load_heldout_all()

    qids = fetch_wikidata_ota_entities(max_search_items=max_search_items)
    logger.info(f"Toplam {len(qids)} Q-ID detayları wbgetentities ile çekiliyor...")

    batch_size = 50
    raw_count = 0
    discarded_non_arabic = 0
    discarded_translation = 0
    discarded_leakage = 0
    valid_pairs: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for i in range(0, len(qids), batch_size):
        chunk = qids[i : i + batch_size]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "labels|aliases|claims",
            "languages": "ota|tr",
            "format": "json",
        }
        resp = safe_request(WIKIDATA_API_URL, params)
        if not resp:
            continue
        entities = resp.json().get("entities", {})

        for qid, ent in entities.items():
            labels = ent.get("labels", {})
            aliases = ent.get("aliases", {})
            claims = ent.get("claims", {})

            # P31 sınıfını al
            p31_class = None
            p31_list = claims.get("P31", [])
            if p31_list:
                try:
                    p31_class = p31_list[0]["mainsnak"]["datavalue"]["value"]["id"]
                except Exception:
                    pass

            # ota metinleri (etiket + takma adlar)
            ota_texts: List[str] = []
            if "ota" in labels and labels["ota"].get("value"):
                ota_texts.append(labels["ota"]["value"].strip())
            for al in aliases.get("ota", []):
                val = al.get("value", "").strip()
                if val and val not in ota_texts:
                    ota_texts.append(val)

            # tr metinleri (etiket + takma adlar)
            tr_texts: List[str] = []
            if "tr" in labels and labels["tr"].get("value"):
                tr_texts.append(labels["tr"]["value"].strip())
            for al in aliases.get("tr", []):
                val = al.get("value", "").strip()
                if val and val not in tr_texts:
                    tr_texts.append(val)

            if not ota_texts or not tr_texts:
                continue

            for ota in ota_texts:
                for tr in tr_texts:
                    raw_count += 1
                    pair_key = (ota, tr)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    # 1. Filtre: arap tarafında Arap betiği oranı > 0.9
                    clean_ota = ota.replace(" ", "").replace("-", "").replace("،", "")
                    if not clean_ota:
                        continue
                    ar_ratio = arabic_letter_count(ota) / len(clean_ota)
                    if ar_ratio < 0.9:
                        discarded_non_arabic += 1
                        continue

                    # 2. Filtre: Latin-Arap iskelet benzerliği >= 0.5 (çeviri değil, çeviriyazı)
                    sim = skeleton_similarity(ota, tr)
                    if sim < 0.5:
                        discarded_translation += 1
                        continue

                    # 3. Filtre: Sızıntı kontrolü (gold dev/test)
                    if is_heldout(
                        arap=ota,
                        latin=tr,
                        heldout_pages=heldout_pages,
                        heldout_arap=heldout_arap,
                        heldout_titles=heldout_titles,
                        heldout_latin=heldout_latin,
                    ):
                        discarded_leakage += 1
                        continue

                    rec = {
                        "qid": qid,
                        "arap": ota,
                        "latin": tr,
                        "tur": p31_class,
                        "iskelet_benzerlik": round(sim, 3),
                        "lisans": "CC0-1.0",
                    }
                    valid_pairs.append(rec)

        if (i // batch_size) % 5 == 0:
            logger.info(f"İşlenen Q-ID: {min(i + batch_size, len(qids))}/{len(qids)} | Geçerli Çift: {len(valid_pairs)}")

    # Dosyaya yaz
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in valid_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"Wikidata toplama tamamlandı. Kaydedildi: {output_path}")
    stats = {
        "kaynak": "Wikidata (ota/tr)",
        "lisans": "CC0-1.0",
        "ham_cift": raw_count,
        "gecerli_cift": len(valid_pairs),
        "atilan_arap_olmayan": discarded_non_arabic,
        "atilan_ceviri_veya_dusuk_benzerlik": discarded_translation,
        "atilan_sizinti": discarded_leakage,
        "dosya": str(output_path),
    }
    logger.info(f"İstatistikler: {stats}")
    return stats


if __name__ == "__main__":
    collect_wikidata_pairs(max_search_items=2500)
