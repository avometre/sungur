"""Vikisözlük Osmanlıca (kaikki.org) veri toplayıcısı.

Kaikki.org üzerindeki Osmanlı Türkçesi (ota) sözlük verisini akış halinde indirir,
Arapça başsözcükleri normalize eder, Latin transkripsiyonu ve Türkçe torun/karşılık
sözcükleri çıkararak JSONL formatında kaydeder.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from src.common.io import write_jsonl
from src.common.normalize import arabic_ratio, normalize_latin, normalize_ottoman
from src.scrapers.http import download_cached

logger = logging.getLogger(__name__)

KAIKKI_OTA_URL = (
    "https://kaikki.org/dictionary/Ottoman%20Turkish/kaikki.org-dictionary-OttomanTurkish.jsonl"
)


def extract_turkish_words(data: Any) -> list[str]:
    """Verilen veri yapısı (descendants/senses) içindeki 'tr' dil kodlu Türkçe kelimeleri toplar."""
    results: list[str] = []

    def _extract(node: Any) -> None:
        if isinstance(node, dict):
            lang_code = node.get("lang_code")
            lang_name = node.get("lang")
            word = node.get("word")
            if (lang_code == "tr" or lang_name == "Turkish") and word and isinstance(word, str):
                cleaned = word.strip()
                if cleaned and cleaned not in results:
                    results.append(cleaned)
            for v in node.values():
                _extract(v)
        elif isinstance(node, list):
            for item in node:
                _extract(item)

    _extract(data)
    return results


def parse_wiktionary_entry(entry: dict[str, Any], idx: int = 0) -> Optional[dict[str, Any]]:
    """Tek bir Kaikki JSON kaydını ayrıştırıp standart şemaya dönüştürür.

    Arap harfi içermeyen (arabic_ratio < 0.5) başsözcükleri None döndürerek eler.
    """
    word = entry.get("word", "")
    if not isinstance(word, str) or arabic_ratio(word) < 0.5:
        return None

    arap_ham = word.strip()
    arap = normalize_ottoman(arap_ham)
    pos = str(entry.get("pos", "") or "")

    # 1. Latin okunuşu bulma
    latin: Optional[str] = None
    for form_obj in entry.get("forms", []):
        if isinstance(form_obj, dict) and "romanization" in form_obj.get("tags", []):
            form_val = form_obj.get("form")
            if form_val and isinstance(form_val, str) and form_val.strip():
                latin = form_val.strip()
                break

    if not latin:
        for ht in entry.get("head_templates", []):
            if isinstance(ht, dict):
                args = ht.get("args", {})
                if isinstance(args, dict) and args.get("tr"):
                    val = str(args["tr"]).strip()
                    if val:
                        latin = val
                        break

    latin_sade: Optional[str] = (
        normalize_latin(latin, strip_diacritics=True) if latin else None
    )

    # 2. Anlamlar (glosses)
    anlam: list[str] = []
    for sense in entry.get("senses", []):
        if isinstance(sense, dict):
            for g in sense.get("glosses", []):
                if isinstance(g, str) and g.strip() and g.strip() not in anlam:
                    anlam.append(g.strip())

    # 3. Günümüz Türkçe karşılıkları (descendants)
    turkce: list[str] = extract_turkish_words(entry.get("descendants", []))
    for sense in entry.get("senses", []):
        if isinstance(sense, dict):
            for tr_word in extract_turkish_words(sense.get("translations", [])):
                if tr_word not in turkce:
                    turkce.append(tr_word)

    # 4. Köken (etymology)
    koken = entry.get("etymology_text")
    if koken is not None and not isinstance(koken, str):
        koken = str(koken)

    # 5. Çekimli biçim (form_of) bilgisi
    form_of: list[dict[str, Any]] = []
    for sense in entry.get("senses", []):
        if isinstance(sense, dict):
            tags = sense.get("tags", [])
            if not isinstance(tags, list):
                tags = [tags] if tags else []
            for fo in sense.get("form_of", []):
                if isinstance(fo, dict) and fo.get("word"):
                    lemma = str(fo["word"]).strip()
                    if lemma:
                        item = {"lemma": lemma, "etiketler": list(tags)}
                        if item not in form_of:
                            form_of.append(item)

    cekim_mi = bool(form_of)

    return {
        "kaynak": "wiktionary",
        "id": f"wikt:{arap_ham}:{pos}:{idx}",
        "arap_ham": arap_ham,
        "arap": arap,
        "latin": latin,
        "latin_sade": latin_sade,
        "tur": pos,
        "anlam": anlam,
        "turkce": turkce,
        "koken": koken,
        "form_of": form_of,
        "cekim_mi": cekim_mi,
        "lisans": "CC BY-SA 4.0",
        "url": f"https://en.wiktionary.org/wiki/{arap_ham}",
    }


def parse_wiktionary_file(
    file_path: Union[str, Path], limit: Optional[int] = None
) -> Iterator[dict[str, Any]]:
    """Kaikki JSONL dosyasını satır satır okur ve geçerli kayıtları üretir."""
    path = Path(file_path)
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw_entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            parsed = parse_wiktionary_entry(raw_entry, idx=count)
            if parsed is not None:
                yield parsed
                count += 1
                if limit is not None and count >= limit:
                    break


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Vikisözlük Osmanlıca (kaikki.org) toplayıcı")
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/wiktionary_ota.jsonl",
        help="Çıktı JSONL dosya yolu",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw/_cache",
        help="Ham indirmelerin saklanacağı önbellek dizini",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Toplanacak maksimum kayıt sayısı (test amaçlı)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=KAIKKI_OTA_URL,
        help="Kaikki.org JSONL indirme URL'si",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Kaikki.org Osmanlıca verisi kontrol ediliyor: %s", args.url)

    cached_path = download_cached(
        args.url,
        cache_dir=args.cache_dir,
        filename="kaikki.org-dictionary-OttomanTurkish.jsonl",
    )
    logger.info("Ham veri hazır: %s", cached_path)

    records: list[dict[str, Any]] = []
    with_latin = 0
    with_turkce = 0

    for rec in parse_wiktionary_file(cached_path, limit=args.limit):
        records.append(rec)
        if rec.get("latin"):
            with_latin += 1
        if rec.get("turkce"):
            with_turkce += 1

    out_path = Path(args.out)
    saved_count = write_jsonl(out_path, records)

    print("\n--- Vikisözlük (Kaikki.org) Özeti ---")
    print(f"Toplam geçerli kayıt : {saved_count}")
    print(f"Latin okunuşu olan   : {with_latin}")
    print(f"Türkçe karşılığı olan: {with_turkce}")
    print(f"Çıktı dosyası        : {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
