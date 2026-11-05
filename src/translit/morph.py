"""Çekim üretici (morphological inflection generator).

Bu modül:
1. Vikisözlük'teki isim, sıfat, ad ve fiil köklerini kurallarla çekimleyerek
   ters model (Latin -> Arap) için zenginleştirilmiş eğitim verisi üretir.
2. İsimler (noun, adj, name) için 7 güvenli çekim eki üretir:
   - çoğul: -lar / -ler -> لر
   - bulunma: -da / -de (-ta / -te) -> ده
   - ayrılma: -dan / -den (-tan / -ten) -> دن
   - lı: -lı / -li / -lu / -lü -> لی
   - sız: -sız / -siz / -suz / -süz -> سز
   - lık: -lık / -lik / -luk / -lük -> kalın: لق, ince: لك
   - cı: -cı / -ci / -cu / -cü (-çı...) -> جی
   Arap kök ه (he) ile bitiyorsa ekten önce ZWNJ (U+200C) eklenir.
3. Fiiller (verb) için mastar (-mak/-mek, مق/مك/مک) köklerinden 3 çekim üretir:
   - geçmiş: -dı / -di / -du / -dü (-tı...) -> دی
   - miş'li geçmiş: -mış / -miş / -muş / -müş -> مش
   - şimdiki zaman: -ıyor / -iyor / -uyor / -üyor (-yor) -> یور
4. Üretilen her kayıt:
   {"kaynak": "morph", "kok_id": ..., "ek": ..., "arap": ..., "latin": ..., "latin_sade": ..., "duzey": "kelime"}
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_latin, normalize_ottoman

logger = logging.getLogger(__name__)

DEFAULT_WIKTIONARY_PATH = Path("data/interim/wiktionary_clean.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/translit_train_morph.jsonl")

# Ünlü ses kümeleri
BACK_VOWELS = set("aıouâûāū")      # Kalın ünlüler
FRONT_VOWELS = set("eiöüîêīō")     # İnce ünlüler
ROUNDED_VOWELS = set("oöuüûôū")    # Yuvarlak ünlüler
UNROUNDED_VOWELS = set("aeıiâîā")  # Düz ünlüler
ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS

# Sert ünsüzler (benzeşme için: ç, f, h, k, p, s, ş, t ve transkripsiyon diyakritikleri)
SERT_UNSUZLER = set("çfhkpsştḳḥḫṣṭ")

ZWNJ = "\u200C"


def find_last_vowel(text: str) -> Optional[str]:
    """Latin metindeki son ünlüyü döndürür."""
    for c in reversed(text.lower()):
        if c in ALL_VOWELS:
            return c
    return None


def get_vowel_harmony(last_vowel: Optional[str]) -> Tuple[str, str]:
    """Son ünlüye göre 2'li (e/a) ve 4'lü (i/ı/u/ü) ünlü karşılıklarını döndürür."""
    is_back = (last_vowel in BACK_VOWELS) if last_vowel else False
    is_rounded = (last_vowel in ROUNDED_VOWELS) if last_vowel else False

    e_a = "a" if is_back else "e"
    if is_back:
        i_u = "u" if is_rounded else "ı"
    else:
        i_u = "ü" if is_rounded else "i"

    return e_a, i_u


def inflect_noun(
    root_latin: str,
    root_arap: str,
    kok_id: str,
) -> List[Dict[str, Any]]:
    """İsim, sıfat ve özel adlar için 7 çekimli biçim üretir."""
    if not root_latin or not root_arap:
        return []

    # Latin kökün son karakterleri
    clean_lat = root_latin.strip()
    last_v = find_last_vowel(clean_lat)
    e_a, i_u = get_vowel_harmony(last_v)
    is_back = (last_v in BACK_VOWELS) if last_v else False

    # Sert ünsüz benzeşmesi (d -> t, c -> ç)
    last_char = clean_lat[-1].lower() if clean_lat else ""
    is_sert = last_char in SERT_UNSUZLER
    d_t = "t" if is_sert else "d"
    c_ç = "ç" if is_sert else "c"

    # Arapça kök he ile mi bitiyor?
    clean_ar = normalize_ottoman(root_arap).strip()
    needs_zwnj = clean_ar.endswith("ه") or clean_ar.endswith("ۀ")
    ar_prefix = (clean_ar + ZWNJ) if needs_zwnj else clean_ar

    inflections: List[Dict[str, Any]] = []

    # 1. Çoğul (-lar / -ler -> لر)
    lat_cogul = f"{clean_lat}l{e_a}r"
    ar_cogul = f"{ar_prefix}لر"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "cogul",
        "arap": ar_cogul,
        "latin": lat_cogul,
        "latin_sade": normalize_latin(lat_cogul, strip_diacritics=True),
        "duzey": "kelime",
    })

    # 2. Bulunma (-da / -de / -ta / -te -> ده)
    lat_bulunma = f"{clean_lat}{d_t}{e_a}"
    ar_bulunma = f"{ar_prefix}ده"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "bulunma",
        "arap": ar_bulunma,
        "latin": lat_bulunma,
        "latin_sade": normalize_latin(lat_bulunma, strip_diacritics=True),
        "duzey": "kelime",
    })

    # 3. Ayrılma (-dan / -den / -tan / -ten -> دن)
    lat_ayrilma = f"{clean_lat}{d_t}{e_a}n"
    ar_ayrilma = f"{ar_prefix}دن"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "ayrilma",
        "arap": ar_ayrilma,
        "latin": lat_ayrilma,
        "latin_sade": normalize_latin(lat_ayrilma, strip_diacritics=True),
        "duzey": "kelime",
    })

    # 4. -lı / -li / -lu / -lü -> لی
    lat_li = f"{clean_lat}l{i_u}"
    ar_li = f"{ar_prefix}لی"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "li",
        "arap": ar_li,
        "latin": lat_li,
        "latin_sade": normalize_latin(lat_li, strip_diacritics=True),
        "duzey": "kelime",
    })

    # 5. -sız / -siz / -suz / -süz -> سز
    lat_siz = f"{clean_lat}s{i_u}z"
    ar_siz = f"{ar_prefix}سز"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "siz",
        "arap": ar_siz,
        "latin": lat_siz,
        "latin_sade": normalize_latin(lat_siz, strip_diacritics=True),
        "duzey": "kelime",
    })

    # 6. -lık / -lik / -luk / -lük -> kalın: لق, ince: لك
    lat_lik = f"{clean_lat}l{i_u}k"
    ar_lik_suffix = "لق" if is_back else "لك"
    ar_lik = f"{ar_prefix}{ar_lik_suffix}"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "lik",
        "arap": ar_lik,
        "latin": lat_lik,
        "latin_sade": normalize_latin(lat_lik, strip_diacritics=True),
        "duzey": "kelime",
    })

    # 7. -cı / -ci / -cu / -cü (-çı...) -> جی
    lat_ci = f"{clean_lat}{c_ç}{i_u}"
    ar_ci = f"{ar_prefix}جی"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "ci",
        "arap": ar_ci,
        "latin": lat_ci,
        "latin_sade": normalize_latin(lat_ci, strip_diacritics=True),
        "duzey": "kelime",
    })

    return inflections


def inflect_verb(
    infinitive_latin: str,
    infinitive_arap: str,
    kok_id: str,
) -> List[Dict[str, Any]]:
    """Fiiller için geçmiş zaman, miş'li geçmiş ve şimdiki zaman çekimlerini üretir."""
    lat = infinitive_latin.strip()
    ar = normalize_ottoman(infinitive_arap).strip()

    # Mastar eki kontrolü: -mak / -mek
    lat_lower = lat.lower()
    if not (lat_lower.endswith("mak") or lat_lower.endswith("mek")):
        return []
    if not (ar.endswith("مق") or ar.endswith("مك") or ar.endswith("مک")):
        return []

    # Kök ayıklama
    stem_lat = lat[:-3]
    stem_ar = ar[:-2].rstrip(ZWNJ).strip()

    if not stem_lat or not stem_ar:
        return []

    last_v = find_last_vowel(stem_lat)
    e_a, i_u = get_vowel_harmony(last_v)

    # Sert ünsüz benzeşmesi (d -> t)
    last_char = stem_lat[-1].lower() if stem_lat else ""
    is_sert = last_char in SERT_UNSUZLER
    d_t = "t" if is_sert else "d"

    # Kök he ile mi bitiyor?
    needs_zwnj = stem_ar.endswith("ه") or stem_ar.endswith("ۀ")
    ar_prefix = (stem_ar + ZWNJ) if needs_zwnj else stem_ar

    inflections: List[Dict[str, Any]] = []

    # 1. Geçmiş zaman (-dı / -di / -du / -dü / -tı...) -> دی
    lat_gecmis = f"{stem_lat}{d_t}{i_u}"
    ar_gecmis = f"{ar_prefix}دی"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "fiil_gecmis",
        "arap": ar_gecmis,
        "latin": lat_gecmis,
        "latin_sade": normalize_latin(lat_gecmis, strip_diacritics=True),
        "duzey": "kelime",
    })

    # 2. Miş'li geçmiş (-mış / -miş / -muş / -müş) -> مش
    lat_mis = f"{stem_lat}m{i_u}ş"
    ar_mis = f"{ar_prefix}مش"
    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "fiil_mis",
        "arap": ar_mis,
        "latin": lat_mis,
        "latin_sade": normalize_latin(lat_mis, strip_diacritics=True),
        "duzey": "kelime",
    })

    # 3. Şimdiki zaman:
    # Ünlüyle biten kökte -yor -> یور
    # Ünsüzle biten kökte -[i/ı/u/ü]yor -> araya ی eklenir: یاز + ی + ور = یازیور
    stem_ends_vowel = last_char in ALL_VOWELS
    if stem_ends_vowel:
        lat_simdiki = f"{stem_lat}yor"
        ar_simdiki = f"{ar_prefix}یور"
    else:
        lat_simdiki = f"{stem_lat}{i_u}yor"
        ar_simdiki = f"{stem_ar}یور"

    inflections.append({
        "kaynak": "morph",
        "kok_id": kok_id,
        "ek": "fiil_simdiki",
        "arap": ar_simdiki,
        "latin": lat_simdiki,
        "latin_sade": normalize_latin(lat_simdiki, strip_diacritics=True),
        "duzey": "kelime",
    })

    return inflections


def generate_all_morph_inflections(
    wiktionary_path: Path = DEFAULT_WIKTIONARY_PATH,
    output_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
) -> List[Dict[str, Any]]:
    """Vikisözlük'teki uygun kayıtlardan tüm çekimleri üretir."""
    if not wiktionary_path.exists():
        logger.error(f"Vikisözlük dosyası bulunamadı: {wiktionary_path}")
        return []

    all_inflections: List[Dict[str, Any]] = []
    allowed_turs = {"noun", "adj", "name", "verb"}

    for r in read_jsonl(wiktionary_path):
        if r.get("supheli", False):
            continue

        tur = r.get("tur", "")
        if tur not in allowed_turs:
            continue

        kok_id = r.get("id", "")
        lat = r.get("latin", "").strip()
        ar = r.get("arap", "").strip()

        if not lat or not ar:
            continue

        # Bileşik / boşluklu kalıpları hariç tut (örn. "harâb etmek")
        if " " in lat or " " in ar:
            continue

        if tur in {"noun", "adj", "name"}:
            items = inflect_noun(root_latin=lat, root_arap=ar, kok_id=kok_id)
            all_inflections.extend(items)
        elif tur == "verb":
            items = inflect_verb(infinitive_latin=lat, infinitive_arap=ar, kok_id=kok_id)
            all_inflections.extend(items)

    # id atama
    for idx, item in enumerate(all_inflections, 1):
        item["id"] = f"morph:{idx:06d}"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_path, all_inflections)

    return all_inflections


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Çekim üretici (morphological generator).")
    parser.add_argument("--wiktionary", type=Path, default=DEFAULT_WIKTIONARY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    print("=" * 80)
    print("ÇEKİM ÜRETİCİ (VERİ ZENGİNLEŞTİRME)")
    print("=" * 80)

    inflections = generate_all_morph_inflections(args.wiktionary, args.output)

    ek_counts = Counter(item["ek"] for item in inflections)

    print(f"Toplam Üretilen Çekim Sayısı : {len(inflections):,}")
    print("-" * 80)
    print("Ek Dağılımı:")
    for ek_name, cnt in ek_counts.most_common():
        print(f"  - {ek_name:<15}: {cnt:,} kayıt")
    print("-" * 80)
    print("İlk 10 Örnek:")
    for idx, item in enumerate(inflections[:10], 1):
        print(f"{idx:2d}. [Kök: {item['kok_id']} | Ek: {item['ek']}] Arap: {item['arap']} | Latin: {item['latin']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
