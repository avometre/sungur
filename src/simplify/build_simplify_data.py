"""Sadeleştirme (Osmanlıca Latin → Bugünkü Türkçe) eğitim verisi derleyicisi (Görev 14).

Kaynaklar:
1. Ottoman2Turkish sözlüğü (data/interim/sade_sozluk.jsonl) — yalnız yerel
2. Türkçe Vikisözlük anlamları (data/interim/tr_wiktionary_ota.jsonl)
3. İngilizce Vikisözlük Türkçe karşılıkları (data/interim/wiktionary_clean.jsonl)
4. Opsiyonel cümle çiftleri (data/interim/vikikaynak_sade_pairs.jsonl)

Çıktılar:
- data/processed/simplify_train.jsonl
- data/processed/simplify_val.jsonl
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_latin, tr_lower
from src.gold.leakage import is_heldout, load_heldout_all

logger = logging.getLogger(__name__)

DEFAULT_SADE_SOZLUK_PATH = Path("data/interim/sade_sozluk.jsonl")
DEFAULT_TR_WIKTIONARY_PATH = Path("data/interim/tr_wiktionary_ota.jsonl")
DEFAULT_WIKTIONARY_PATH = Path("data/interim/wiktionary_clean.jsonl")
DEFAULT_VK_SADE_PAIRS_PATH = Path("data/interim/vikikaynak_sade_pairs.jsonl")

DEFAULT_TRAIN_OUT = Path("data/processed/simplify_train.jsonl")
DEFAULT_VAL_OUT = Path("data/processed/simplify_val.jsonl")


def clean_gloss(text: str) -> str:
    """Sözlük anlamındaki fazla boşlukları ve gürültüyü temizler."""
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    return s


def build_simplify_dataset(
    sade_sozluk_path: Path = DEFAULT_SADE_SOZLUK_PATH,
    tr_wiktionary_path: Path = DEFAULT_TR_WIKTIONARY_PATH,
    wiktionary_path: Path = DEFAULT_WIKTIONARY_PATH,
    vk_sade_pairs_path: Path = DEFAULT_VK_SADE_PAIRS_PATH,
    train_output_path: Path = DEFAULT_TRAIN_OUT,
    val_output_path: Path = DEFAULT_VAL_OUT,
    val_ratio: float = 0.02,
    seed: int = 0,
) -> Dict[str, Any]:
    """Sadeleştirme eğitim ve doğrulama veri setlerini derler."""
    heldout_pages, heldout_arap, heldout_titles, heldout_latin = load_heldout_all()

    unique_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    leaked_count = 0

    def _add_pair(
        kaynak: str,
        osmanlica: str,
        turkce: str,
        duzey: str,
        lisans: str,
    ) -> None:
        nonlocal leaked_count
        osm = normalize_latin(osmanlica).strip()
        tr = clean_gloss(turkce)
        if not osm or not tr:
            return
        if tr_lower(osm) == tr_lower(tr):
            return
        if is_heldout(
            latin=osm,
            heldout_pages=heldout_pages,
            heldout_arap=heldout_arap,
            heldout_titles=heldout_titles,
            heldout_latin=heldout_latin,
        ):
            leaked_count += 1
            return

        key = (tr_lower(osm), tr_lower(tr))
        if key not in unique_pairs:
            unique_pairs[key] = {
                "kaynak": kaynak,
                "osmanlica_latin": osm,
                "turkce_sade": tr,
                "duzey": duzey,
                "lisans": lisans,
            }

    # 1. Ottoman2Turkish sözlüğü (yalnız yerel)
    if sade_sozluk_path.exists():
        for r in read_jsonl(sade_sozluk_path):
            _add_pair(
                kaynak="sade_sozluk",
                osmanlica=r.get("kelime_latin", ""),
                turkce=r.get("anlam_tr", ""),
                duzey="kelime",
                lisans=r.get("lisans", "yalnız yerel"),
            )

    # 2. Türkçe Vikisözlük anlamları
    if tr_wiktionary_path.exists():
        for r in read_jsonl(tr_wiktionary_path):
            _add_pair(
                kaynak="tr_wiktionary",
                osmanlica=r.get("latin", ""),
                turkce=r.get("anlam_tr", ""),
                duzey="kelime",
                lisans=r.get("lisans", "CC BY-SA 4.0"),
            )

    # 3. İngilizce Vikisözlük Türkçe karşılıkları
    if wiktionary_path.exists():
        for r in read_jsonl(wiktionary_path):
            if r.get("supheli", False):
                continue
            tr_list = r.get("turkce") or []
            if isinstance(tr_list, list) and tr_list:
                _add_pair(
                    kaynak="wiktionary_tr",
                    osmanlica=r.get("latin", ""),
                    turkce=", ".join(str(x).strip() for x in tr_list if str(x).strip()),
                    duzey="kelime",
                    lisans=r.get("lisans", "CC BY-SA 4.0"),
                )

    # 4. Opsiyonel Vikikaynak sadeleştirme cümle çiftleri
    if vk_sade_pairs_path.exists():
        for r in read_jsonl(vk_sade_pairs_path):
            _add_pair(
                kaynak="vikikaynak_sade",
                osmanlica=r.get("osmanlica_latin") or r.get("latin") or "",
                turkce=r.get("turkce_sade") or r.get("modern_turkce") or "",
                duzey="cumle",
                lisans=r.get("lisans", "CC BY-SA 4.0"),
            )

    all_items = list(unique_pairs.values())
    all_items.sort(key=lambda x: (x["osmanlica_latin"], x["turkce_sade"]))
    for idx, item in enumerate(all_items, 1):
        item["id"] = f"simp:{idx:06d}"

    rng = random.Random(seed)
    indices = list(range(len(all_items)))
    rng.shuffle(indices)

    num_val = int(len(all_items) * val_ratio)
    val_indices = set(indices[:num_val])

    val_records = [all_items[i] for i in range(len(all_items)) if i in val_indices]
    train_records = [all_items[i] for i in range(len(all_items)) if i not in val_indices]

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    val_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_output_path, train_records)
    write_jsonl(val_output_path, val_records)

    return {
        "toplam_kayit": len(all_items),
        "egitim_kayit_sayisi": len(train_records),
        "dogrulama_kayit_sayisi": len(val_records),
        "sizinti_nedeniyle_atilan": leaked_count,
        "kaynak_dagilimi": dict(Counter(item["kaynak"] for item in all_items)),
        "duzey_dagilimi": dict(Counter(item["duzey"] for item in all_items)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sadeleştirme veri derleyicisi (Görev 14)")
    parser.add_argument("--sade-sozluk", type=Path, default=DEFAULT_SADE_SOZLUK_PATH)
    parser.add_argument("--tr-wiktionary", type=Path, default=DEFAULT_TR_WIKTIONARY_PATH)
    parser.add_argument("--wiktionary", type=Path, default=DEFAULT_WIKTIONARY_PATH)
    parser.add_argument("--vk-sade", type=Path, default=DEFAULT_VK_SADE_PAIRS_PATH)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--val-out", type=Path, default=DEFAULT_VAL_OUT)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = build_simplify_dataset(
        sade_sozluk_path=args.sade_sozluk,
        tr_wiktionary_path=args.tr_wiktionary,
        wiktionary_path=args.wiktionary,
        vk_sade_pairs_path=args.vk_sade,
        train_output_path=args.train_out,
        val_output_path=args.val_out,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Sadeleştirme verisi hazır: {summary}")


if __name__ == "__main__":
    main()
