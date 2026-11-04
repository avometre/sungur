"""B1_sozluk — Sözlük tabanlı ve ek soyma (stemming) destekli çeviriyazı çözümü (baseline).

Bu modül:
1. Temiz Vikisözlük (supheli=False) ve yer adları (guvenilir=True) verilerini birleştirerek
   Arapça kelime -> en sık Latin karşılığı sözlüğü oluşturur.
   Eşitlik durumunda Vikisözlük önceliklidir.
2. Kelime sözlükte doğrudan yoksa:
   (a) Türkçe çekim eklerini sondan soyar (en uzun eşleşme önce):
       لر/لری/لرک/لرڭ/لرینی/ده/دن/دە/یه/نی/نک/نڭ/ڭ/سی/لق/لك/جه/جق/مش/دی/یور vb.
   (b) Kök sözlükte bulunursa kök + ekin ses uyumlu Latin karşılığını üretir.
   (c) Bulunamazsa B0_kural harf eşlemelerine başvurur.
3. Dev ve test kümelerini çevirir, data/eval/B1_sozluk/ altına yazar ve değerlendirir.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_ottoman
from src.eval.run_eval import print_score_table, run_evaluation
from src.translit.baseline_rules import transliterate_rules, transliterate_word_rules

DEFAULT_WIKTIONARY_PATH = Path("data/interim/wiktionary_clean.jsonl")
DEFAULT_GAZETTEER_PATH = Path("data/interim/gazetteer_clean.jsonl")

# Ünlü ses sınıfları
BACK_VOWELS = set("aıouâûāū")      # Kalın ünlüler
FRONT_VOWELS = set("eiöüîêīō")     # İnce ünlüler
ROUNDED_VOWELS = set("oöuüûôū")    # Yuvarlak ünlüler
UNROUNDED_VOWELS = set("aeıiâîā")  # Düz ünlüler
ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS

# Açık ek tablosu (Suffix Table)
# Format: (Arapça ek, Tip/Şablon adı)
# Sıralama: En uzun eşleşme önce denenecek şekilde kod içinde len(ek) tersiyle sıralanır.
SUFFIX_TABLE: List[Tuple[str, str]] = [
    ("لرینی", "ler_in_i"),
    ("لرينی", "ler_in_i"),
    ("لرک", "ler_in"),
    ("لرڭ", "ler_in"),
    ("لری", "ler_i"),
    ("لرين", "ler_in"),
    ("لر", "ler"),
    ("یور", "iyor"),
    ("دە", "de"),
    ("ده", "de"),
    ("دن", "den"),
    ("یه", "ye"),
    ("نی", "ni"),
    ("نک", "nin"),
    ("نڭ", "nin"),
    ("سی", "si"),
    ("لق", "lik_kalin"),
    ("لك", "lik_ince"),
    ("جه", "ce"),
    ("جق", "cik"),
    ("مش", "mis"),
    ("دی", "di"),
    ("ڭ", "in_tekil"),
]


def find_last_vowel(text: str) -> Optional[str]:
    """Latin metindeki son ünlüyü döndürür."""
    for c in reversed(text.lower()):
        if c in ALL_VOWELS:
            return c
    return None


def resolve_suffix(suffix_type: str, stem_latin: str) -> str:
    """Kökün son ünlüsüne göre ses uyumlu Latin ekini döndürür."""
    last_v = find_last_vowel(stem_latin)
    is_back = (last_v in BACK_VOWELS) if last_v else False
    is_rounded = (last_v in ROUNDED_VOWELS) if last_v else False

    # 2'li uyum (e / a)
    e_a = "a" if is_back else "e"

    # 4'lü uyum (i / ı / u / ü)
    if is_back:
        i_ı = "u" if is_rounded else "ı"
    else:
        i_ı = "ü" if is_rounded else "i"

    # Stem sonu ünlü ile mi bitiyor?
    stem_ends_vowel = (stem_latin[-1].lower() in ALL_VOWELS) if stem_latin else False

    if suffix_type == "ler":
        return f"l{e_a}r"
    elif suffix_type == "ler_i":
        return f"l{e_a}r{i_ı}"
    elif suffix_type == "ler_in":
        return f"l{e_a}r{i_ı}n"
    elif suffix_type == "ler_in_i":
        return f"l{e_a}r{i_ı}n{i_ı}"
    elif suffix_type == "de":
        return f"d{e_a}"
    elif suffix_type == "den":
        return f"d{e_a}n"
    elif suffix_type == "ye":
        return f"y{e_a}"
    elif suffix_type == "ni":
        return f"n{i_ı}"
    elif suffix_type == "nin":
        if stem_ends_vowel:
            return f"n{i_ı}n"
        return f"{i_ı}n"
    elif suffix_type == "si":
        return f"s{i_ı}"
    elif suffix_type == "lik_kalin":
        return "luk" if is_rounded else "lık"
    elif suffix_type == "lik_ince":
        return "lük" if is_rounded else "lik"
    elif suffix_type == "ce":
        return f"c{e_a}"
    elif suffix_type == "cik":
        return f"c{i_ı}k"
    elif suffix_type == "mis":
        return f"m{i_ı}ş"
    elif suffix_type == "di":
        return f"d{i_ı}"
    elif suffix_type == "iyor":
        return f"{i_ı}yor"
    elif suffix_type == "in_tekil":
        if stem_ends_vowel:
            return "n"
        return f"{i_ı}n"

    return ""


class BaselineDictionary:
    """Temiz Vikisözlük ve yer adları sözlüğünü yükler ve kelime çevirisi yapar."""

    def __init__(
        self,
        wiktionary_path: Path = DEFAULT_WIKTIONARY_PATH,
        gazetteer_path: Path = DEFAULT_GAZETTEER_PATH,
    ) -> None:
        self.wiktionary_path = wiktionary_path
        self.gazetteer_path = gazetteer_path
        self.dict_map: Dict[str, str] = {}
        # Ekleri uzunluk sırasına göre sırala
        self.sorted_suffixes = sorted(SUFFIX_TABLE, key=lambda x: len(x[0]), reverse=True)
        self._load_dictionaries()

    def _load_dictionaries(self) -> None:
        """Sözlükleri okuyup frekans ve eşitlik kuralına göre dict_map'i oluşturur."""
        wikt_counts: Dict[str, Counter] = defaultdict(Counter)
        gaz_counts: Dict[str, Counter] = defaultdict(Counter)

        if self.wiktionary_path.exists():
            for r in read_jsonl(self.wiktionary_path):
                if not r.get("supheli", False):
                    arap = normalize_ottoman(r.get("arap", ""))
                    latin = r.get("latin", "").strip()
                    if arap and latin:
                        wikt_counts[arap][latin] += 1

        if self.gazetteer_path.exists():
            for r in read_jsonl(self.gazetteer_path):
                if r.get("guvenilir", False):
                    arap = normalize_ottoman(r.get("arap", ""))
                    latin = r.get("latin", "").strip()
                    if arap and latin:
                        gaz_counts[arap][latin] += 1

        all_keys = set(wikt_counts.keys()) | set(gaz_counts.keys())
        for arap in all_keys:
            all_cands = set(wikt_counts[arap].keys()) | set(gaz_counts[arap].keys())

            def sort_key(cand: str):
                w_cnt = wikt_counts[arap][cand]
                g_cnt = gaz_counts[arap][cand]
                total = w_cnt + g_cnt
                # total önce, sonra wiktionary önceliği, sonra deterministik alfabe
                return (total, w_cnt, cand)

            best_latin = max(all_cands, key=sort_key)
            self.dict_map[arap] = best_latin

    def transliterate_word(self, word: str) -> str:
        """Tek bir kelimeyi sözlük, ek soyma ve kural sırasıyla çevirir."""
        if not word:
            return ""

        w_norm = normalize_ottoman(word)

        # 1. Doğrudan sözlük eşleşmesi
        if w_norm in self.dict_map:
            return self.dict_map[w_norm]

        # 2. Ek soyma (en uzun eşleşme önce)
        for ar_suf, suf_type in self.sorted_suffixes:
            if w_norm.endswith(ar_suf) and len(w_norm) > len(ar_suf):
                stem = w_norm[:-len(ar_suf)]
                if stem in self.dict_map:
                    stem_lat = self.dict_map[stem]
                    resolved = resolve_suffix(suf_type, stem_lat)
                    return stem_lat + resolved

        # 3. Kural tabanlı yedek (B0_kural)
        return transliterate_word_rules(word)

    def transliterate_text(self, text: str) -> str:
        """Metin düzeyinde çeviriyazı uygular."""
        if not text:
            return ""

        tokens = re.split(r"(\s+|[.,!?:;\"'«»()\[\]—–\-])", text)
        result = []
        for tok in tokens:
            if not tok:
                continue
            if re.match(r"^(\s+|[.,!?:;\"'«»()\[\]—–\-])$", tok):
                result.append(tok)
            else:
                result.append(self.transliterate_word(tok))

        return "".join(result)


def run_baseline_dict(
    split: str = "dev",
    wiktionary_path: Path = DEFAULT_WIKTIONARY_PATH,
    gazetteer_path: Path = DEFAULT_GAZETTEER_PATH,
) -> Dict[str, Any]:
    """B1_sozluk baseline modelini çalıştırır ve değerlendirir."""
    gold_path = Path(f"data/gold/translit_{split}.jsonl")
    pred_dir = Path("data/eval/B1_sozluk")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}.jsonl"
    skor_path = pred_dir / f"{split}_skor.json"

    b_dict = BaselineDictionary(wiktionary_path, gazetteer_path)
    gold_records = list(read_jsonl(gold_path))
    pred_records = []

    for item in gold_records:
        rec_id = item["id"]
        arap = item.get("arap", "")
        hyp = b_dict.transliterate_text(arap)
        pred_records.append({"id": rec_id, "tahmin": hyp})

    write_jsonl(pred_path, pred_records)

    eval_result = run_evaluation(
        system_name="B1_sozluk",
        split=split,
        gold_path=gold_path,
        pred_path=pred_path,
        output_path=skor_path,
    )
    return eval_result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="B1_sozluk çeviriyazı çözümü.")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both", help="Değerlendirme kümesi")
    args = parser.parse_args()

    splits = ["dev", "test"] if args.split == "both" else [args.split]
    for sp in splits:
        print(f"\nB1_sozluk çalıştırılıyor ({sp})...")
        res = run_baseline_dict(sp)
        print_score_table(res)


if __name__ == "__main__":
    main()
