"""B0_kural — Harf harf kural tabanlı temel çeviriyazı çözümü (baseline).

Bu modül:
- Arap harfli kelimeleri kural tabanlı olarak Latin alfabesine çevirir.
- skeleton.py eşlemelerini genişletir:
  - Ünsüzler: ب->b, پ->p, ت/ط->t, د->d, ث/س/ص->s, ج->c, چ->ç, ح/خ->h,
             ذ/ز/ض/ظ->z, ر->r, ژ->j, ش->ş, غ->g, ف->f, ق/ك/ک->k, گ->g,
             ڭ/ݣ->n, ل->l, م->m, ن->n, ع->'
  - Ünlüler ve yarı ünlüler:
    - آ -> â, ا -> a (kelime başında او -> u, ای -> i)
    - و -> kelime başında v, ünsüzden sonra u, ünlüden sonra v
    - ی -> kelime başında y, ünsüzden sonra i, ünlüden sonra y
    - kelime sonu ه -> e, kelime içi ه -> h
    - ۀ -> -i (izafet için)
- Ses uyumu YOKTUR.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_ottoman
from src.eval.run_eval import print_score_table, run_evaluation

# Sayı eşlemeleri
DIGIT_MAP = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

# Temel ünsüz harf eşlemeleri
BASE_CONSONANTS: Dict[str, str] = {
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ط": "t",
    "ة": "t",
    "د": "d",
    "ث": "s",
    "س": "s",
    "ص": "s",
    "ج": "c",
    "چ": "ç",
    "ح": "h",
    "خ": "h",
    "ذ": "z",
    "ز": "z",
    "ض": "z",
    "ظ": "z",
    "ر": "r",
    "ژ": "j",
    "ش": "ş",
    "غ": "g",
    "ف": "f",
    "ق": "k",
    "ك": "k",
    "ک": "k",
    "گ": "g",
    "ڭ": "n",
    "ݣ": "n",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ع": "'",
    "ء": "'",
    "ئ": "y",
}


def is_consonant(char: str) -> bool:
    """Karakterin ünsüz Arap harfi olup olmadığını belirler."""
    return char in BASE_CONSONANTS or char in ("h", "v", "y", "b", "p", "t", "d", "s", "c", "ç", "z", "r", "j", "ş", "g", "f", "k", "n", "l", "m")


def transliterate_word_rules(word: str) -> str:
    """Tek bir Arap harfli kelimeyi kural tabanlı Latin çevirisine dönüştürür."""
    if not word:
        return ""

    # Sayılardan oluşuyorsa çevir
    if all(c in DIGIT_MAP or c.isdigit() for c in word):
        return "".join(DIGIT_MAP.get(c, c) for c in word)

    w = normalize_ottoman(word)
    chars = list(w)
    n = len(chars)
    out: List[str] = []

    i = 0
    while i < n:
        c = chars[i]
        prev_c = chars[i - 1] if i > 0 else ""
        next_c = chars[i + 1] if i + 1 < n else ""
        is_first = (i == 0)
        is_last = (i == n - 1)

        # 1. ۀ (izafet hemzesi veya he-i resmiyye izafeti)
        if c == "ۀ":
            out.append("-i")
            i += 1
            continue

        # 2. آ (meddeli elif)
        if c == "آ":
            out.append("â" if not is_first else "a")
            i += 1
            continue

        # 3. ا (elif)
        if c == "ا":
            if is_first:
                # او -> u
                if next_c == "و":
                    out.append("u")
                    i += 2
                    continue
                # ای -> i
                if next_c == "ی":
                    out.append("i")
                    i += 2
                    continue
                out.append("a")
            else:
                out.append("a")
            i += 1
            continue

        # 4. و (vav)
        if c == "و":
            if is_first:
                out.append("v")
            elif prev_c in BASE_CONSONANTS or prev_c == "ه":
                out.append("u")
            else:
                out.append("v")
            i += 1
            continue

        # 5. ی (ye)
        if c == "ی":
            if is_first:
                out.append("y")
            elif prev_c in BASE_CONSONANTS or prev_c == "ه":
                out.append("i")
            else:
                out.append("y")
            i += 1
            continue

        # 6. ه (he)
        if c == "ه":
            if is_last:
                out.append("e")
            else:
                out.append("h")
            i += 1
            continue

        # 7. Temel ünsüzler
        if c in BASE_CONSONANTS:
            out.append(BASE_CONSONANTS[c])
            i += 1
            continue

        # 8. Rakamlar
        if c in DIGIT_MAP:
            out.append(DIGIT_MAP[c])
            i += 1
            continue

        # Diğer karakterler (noktalama, Latin harfi vb.)
        out.append(c)
        i += 1

    return "".join(out)


def transliterate_rules(text: str) -> str:
    """Metin düzeyinde kural tabanlı çeviriyazı uygular."""
    if not text:
        return ""

    tokens = re.split(r"(\s+|[.,!?:;\"'«»()\[\]—–\-])", text)
    result = []
    for tok in tokens:
        if not tok:
            continue
        # Boşluk veya noktalama ise aynen bırak
        if re.match(r"^(\s+|[.,!?:;\"'«»()\[\]—–\-])$", tok):
            result.append(tok)
        else:
            result.append(transliterate_word_rules(tok))

    return "".join(result)


def run_baseline_rules(split: str = "dev") -> Dict[str, Any]:
    """B0_kural baseline modelini çalıştırır, data/eval/B0_kural/ altına yazar ve değerlendirir."""
    gold_path = Path(f"data/gold/translit_{split}.jsonl")
    pred_dir = Path("data/eval/B0_kural")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}.jsonl"
    skor_path = pred_dir / f"{split}_skor.json"

    gold_records = list(read_jsonl(gold_path))
    pred_records = []

    for item in gold_records:
        rec_id = item["id"]
        arap = item.get("arap", "")
        hyp = transliterate_rules(arap)
        pred_records.append({"id": rec_id, "tahmin": hyp})

    write_jsonl(pred_path, pred_records)

    eval_result = run_evaluation(
        system_name="B0_kural",
        split=split,
        gold_path=gold_path,
        pred_path=pred_path,
        output_path=skor_path,
    )
    return eval_result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="B0_kural çeviriyazı çözümü.")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both", help="Değerlendirme kümesi")
    args = parser.parse_args()

    splits = ["dev", "test"] if args.split == "both" else [args.split]
    for sp in splits:
        print(f"\nB0_kural çalıştırılıyor ({sp})...")
        res = run_baseline_rules(sp)
        print_score_table(res)


if __name__ == "__main__":
    main()
