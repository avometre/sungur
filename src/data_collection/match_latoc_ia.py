"""LATOC ve Vikikaynak metinlerinin Arap harfli asıllarını (IA ve HF kataloglarında) eşleştirir.

Girdi:
- third_party/latoc/data_statistics.csv (143 LATOC kitabı)
- data/raw/ia_catalog.jsonl (2.856 IA eseri)
- HF OttomanNLP/ottoman-turkish-printed-pages (62 kitap başlığı)
- data/interim/vikikaynak_latin.jsonl (Vikikaynak metinleri)

Çıktılar:
- data/interim/latoc_asil_eslesme.csv (latoc_kitap, kaynak, kimlik, eslesme_guveni, kamu_mali_mi)
- data/interim/vikikaynak_ia_eslesme.csv (tr_baslik, ia_identifier, ia_title, eslesme_guveni, kamu_mali_mi)
"""

import csv
import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.normalize import normalize_latin, tr_lower

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LATOC_STATS_PATH = Path("third_party/latoc/data_statistics.csv")
LATOC_SAMPLE_META_PATH = Path("third_party/latoc/LATOC_metadata_sample.csv")
IA_CATALOG_PATH = Path("data/raw/ia_catalog.jsonl")
VIKIKAYNAK_PATH = Path("data/interim/vikikaynak_latin.jsonl")
WS_PARALLEL_PATH = Path("data/interim/ws_parallel.jsonl")

LATOC_OUT_PATH = Path("data/interim/latoc_asil_eslesme.csv")
VK_OUT_PATH = Path("data/interim/vikikaynak_ia_eslesme.csv")

HF_62_BOOKS = [
    "1035-_Turkoloji_Tarix_Siyahatulkubra_Suleyman_Shukri", "Behcetu_l-Fetava", "Feta_va_-y_Ali_Efendi",
    "Feta_va_-y_Feyziye", "Kanunnamei_Ali_Osman", "Su_leyman-na_me",
    "Turkish-1819-New-Testament-Kitab-al-Ahd-al-Jadid-al-mansub-ila-Rabbina-Isa-al-Masih",
    "Turkish-1857-New-Testament-Turque-Nouveau-New-Testament-Incil-injil-Kitabi-Mukaddes-2-Vol",
    "ahdimsiar00kr", "amerika_tarih_kesfi", "as_lm_s__bir_adam", "bariadram4fasl00bolauoft",
    "divan__nedim", "dmetadaanlibirle00ibne", "dustur", "endaht_bahs__2018672550", "esariziyapasa",
    "esfar_erbaa", "fetavayi_feyziye", "hadisat_hukukiyye", "heft_peyker", "hukuk_idare",
    "kamus_alam", "kamus_turki", "kavaid_osmaniyye", "kitab_mukaddes", "lugat_cudi",
    "lugat_naci", "marifetname", "mecelle", "mecmua_ebuzziya", "menakib_hulefa",
    "mizan_ul_hakk", "muhakemat", "muntehabat_tasvir_efkar", "netayic_vukuat",
    "resimli_kamus", "safahat", "sahh_u_geda", "sekerpare", "serh_divan_hafiz",
    "seyahatname_evliya", "siyer_veysi", "subha_sibyan", "tacut_tevarih",
    "tarih_cevdet", "tarih_g盤ata", "tarih_lutfi", "tarih_naima", "tarih_rasid",
    "tercume_i_hal", "tercume_milal", "tezhire_latifi", "tibbi_adli", "tuhfe_vehbi",
    "usul_fikh", "vakayi_baba_tahir", "vefeyat_ayan", "yeni_turkce_lugat", "zafername"
]


def clean_tokens(text: str) -> Set[str]:
    """Metni küçük harf, noktalama ve eklerden arındırıp anlamlı kelime kümesine böler."""
    if not text:
        return set()
    s = tr_lower(text)
    s = normalize_latin(s, strip_diacritics=True)
    words = re.findall(r"[a-z0-9]+", s)
    stop = {"ve", "ile", "de", "da", "bir", "bu", "o", "ki", "der", "fi", "el", "ul", "un", "al", "i", "u", "yu", "vu", "xml", "divani", "divan"}
    return {w for w in words if len(w) > 2 and w not in stop}


def calculate_match_score(title1: str, title2: str) -> float:
    """İki başlık arasındaki kelime kesişimi ve difflib benzerliğini hesaplar."""
    t1 = clean_tokens(title1)
    t2 = clean_tokens(title2)
    if not t1 or not t2:
        return 0.0

    inter = t1 & t2
    jaccard = len(inter) / max(len(t1), len(t2))

    s1 = " ".join(sorted(t1))
    s2 = " ".join(sorted(t2))
    seq_ratio = difflib.SequenceMatcher(None, s1, s2).ratio()

    return 0.6 * jaccard + 0.4 * seq_ratio


def match_latoc_with_sources(output_path: Path = LATOC_OUT_PATH) -> List[Dict[str, Any]]:
    """LATOC 143 kitabını IA kataloğu ve HF 62 kitap ile eşleştirir."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. LATOC kitaplarını oku
    latoc_books: List[Dict[str, str]] = []
    if LATOC_STATS_PATH.exists():
        with open(LATOC_STATS_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fname = r.get("file", "").strip()
                if fname:
                    link = r.get("link", "") or ""
                    latoc_books.append({"file": fname, "link": link})
    logger.info(f"LATOC'tan {len(latoc_books)} kitap yüklendi.")

    # 2. IA kataloğunu oku
    ia_items: List[Dict[str, Any]] = []
    if IA_CATALOG_PATH.exists():
        with open(IA_CATALOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ia_items.append(json.loads(line))
                except Exception:
                    pass
    logger.info(f"IA kataloğundan {len(ia_items)} eser yüklendi.")

    matches: List[Dict[str, Any]] = []

    for lbook in latoc_books:
        fname = lbook["file"]
        # Dosya adını temizle (ör. "celili_leyla vü mecnun.xml" -> yazar: celili, eser: leyla vu mecnun)
        clean_fname = fname.replace(".xml", "").replace("_", " ")
        clean_fname = normalize_latin(clean_fname, strip_diacritics=True)

        best_source = None
        best_id = None
        best_score = 0.0
        is_pd = True  # Klasik Osmanlı eserleri kamu malı

        # a) HF 62 kitap kontrolü
        for hf_book in HF_62_BOOKS:
            score = calculate_match_score(clean_fname, hf_book.replace("_", " "))
            if score > best_score:
                best_score = score
                best_source = "HF (ottoman-turkish-printed-pages)"
                best_id = hf_book

        # b) IA kataloğu kontrolü
        for item in ia_items:
            ia_title = item.get("title", "")
            ia_creator = item.get("creator", "")
            ia_id = item.get("identifier", "")
            combined_ia = f"{ia_creator} {ia_title}"

            score = calculate_match_score(clean_fname, combined_ia)
            if score > best_score:
                best_score = score
                best_source = "Internet Archive"
                best_id = ia_id
                is_pd = item.get("kamu_mali", True)

        # Güven sınıflaması
        if best_score >= 0.70:
            guven = "yuksek"
        elif best_score >= 0.40:
            guven = "orta"
        else:
            guven = "dusuk"
            if not best_source:
                best_source = "Eşleşme bulunamadı"
                best_id = "-"

        matches.append({
            "latoc_kitap": fname,
            "kaynak": best_source,
            "kimlik": best_id,
            "skor": round(best_score, 3),
            "eslesme_guveni": guven,
            "kamu_mali_mi": is_pd,
        })

    # CSV olarak yaz
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["latoc_kitap", "kaynak", "kimlik", "skor", "eslesme_guveni", "kamu_mali_mi"]
        )
        writer.writeheader()
        writer.writerows(matches)

    logger.info(f"LATOC asıl eşleşmesi kaydedildi: {output_path}")
    return matches


def match_vikikaynak_with_ia(output_path: Path = VK_OUT_PATH) -> List[Dict[str, Any]]:
    """Eşleşmemiş Vikikaynak sayfalarını IA kataloğuyla eşleştirir."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Önceden eşleşmiş başlıklar
    already_parallel: Set[str] = set()
    if WS_PARALLEL_PATH.exists():
        with open(WS_PARALLEL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    already_parallel.add(json.loads(line).get("tr_baslik", ""))
                except Exception:
                    pass

    # Tüm Vikikaynak başlıkları
    vk_titles: Set[str] = set()
    if VIKIKAYNAK_PATH.exists():
        with open(VIKIKAYNAK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    b = json.loads(line).get("baslik", "")
                    if b and b not in already_parallel:
                        vk_titles.add(b)
                except Exception:
                    pass

    # IA Kataloğu
    ia_items: List[Dict[str, Any]] = []
    if IA_CATALOG_PATH.exists():
        with open(IA_CATALOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ia_items.append(json.loads(line))
                except Exception:
                    pass

    vk_matches: List[Dict[str, Any]] = []

    for vtitle in vk_titles:
        # Bölüm alt başlıklarını at
        main_title = vtitle.split("/")[0].strip()
        best_ia_id = None
        best_ia_title = None
        best_score = 0.0

        for item in ia_items:
            ia_t = item.get("title", "")
            ia_c = item.get("creator", "")
            score = calculate_match_score(main_title, f"{ia_c} {ia_t}")
            if score > best_score:
                best_score = score
                best_ia_id = item.get("identifier")
                best_ia_title = ia_t

        if best_score >= 0.50:
            guven = "yuksek" if best_score >= 0.70 else "orta"
            vk_matches.append({
                "tr_baslik": vtitle,
                "ia_identifier": best_ia_id,
                "ia_title": best_ia_title,
                "skor": round(best_score, 3),
                "eslesme_guveni": guven,
                "kamu_mali_mi": True,
            })

    # CSV yaz
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tr_baslik", "ia_identifier", "ia_title", "skor", "eslesme_guveni", "kamu_mali_mi"]
        )
        writer.writeheader()
        writer.writerows(vk_matches)

    logger.info(f"Vikikaynak IA eşleşmesi kaydedildi: {len(vk_matches)} eşleşme -> {output_path}")
    return vk_matches


if __name__ == "__main__":
    match_latoc_with_sources()
    match_vikikaynak_with_ia()
