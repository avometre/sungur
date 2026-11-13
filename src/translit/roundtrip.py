"""Sentetik veri için gidiş-dönüş (round-trip) CER skoru ve filtreleme modülü (Görev 11).

Sentetik olarak üretilen (arap_sentetik, latin_referans) çiftlerinin kalitesini
ölçmek için ileri çeviriyazı modelini (ör. S1b) çalıştırır, tahmin edilen
Latin metni sadeleştirilmiş referansla CER üzerinden kıyaslar (`rt_cer`) ve
belirlenen eşiğin üzerindeki gürültülü örnekleri eler.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import jiwer

from src.common.io import read_jsonl, write_jsonl
from src.eval.metrics import normalize_eval_text


def compute_roundtrip_cer(latin_ref: str, latin_pred: str) -> float:
    """Referans Latin ile ileri model tahmininin sade moddaki CER değerini döndürür."""
    ref_norm = normalize_eval_text(latin_ref, mode="sade")
    pred_norm = normalize_eval_text(latin_pred, mode="sade")

    if not ref_norm and not pred_norm:
        return 0.0
    if not ref_norm or not pred_norm:
        return 1.0

    try:
        return float(jiwer.cer([ref_norm], [pred_norm]))
    except Exception:
        return 1.0


def annotate_roundtrip_cer(
    records: Sequence[Dict[str, Any]],
    transliterate_fn: Callable[[List[str]], List[str]],
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """Kayıtlardaki Arapça sentetik metinleri çevirerek her kayda `rt_cer` ekler."""
    annotated: List[Dict[str, Any]] = []
    recs = list(records)

    for start in range(0, len(recs), batch_size):
        chunk = recs[start : start + batch_size]
        arap_texts = [r.get("arap", "") for r in chunk]
        preds = transliterate_fn(arap_texts)

        for rec, pred in zip(chunk, preds):
            ref = rec.get("latin", "")
            rt_cer = compute_roundtrip_cer(ref, pred)
            new_rec = dict(rec)
            new_rec["rt_latin_tahmin"] = pred
            new_rec["rt_cer"] = round(rt_cer, 4)
            annotated.append(new_rec)

    return annotated


def filter_by_roundtrip_cer(
    records: Iterable[Dict[str, Any]],
    max_rt_cer: float = 0.25,
) -> List[Dict[str, Any]]:
    """`rt_cer <= max_rt_cer` koşulunu sağlayan sentetik kayıtları döndürür."""
    kept: List[Dict[str, Any]] = []
    for rec in records:
        val = rec.get("rt_cer")
        if val is not None and float(val) <= max_rt_cer:
            kept.append(rec)
    return kept


def score_synthetic_file(
    input_path: Path,
    output_path: Path,
    model_dir: Path = Path("models/sungur-ceviriyazi-small-v1"),
    prefix: str = "<vk> ",
    max_rt_cer: Optional[float] = 0.25,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Sentetik JSONL dosyasını S1b ile skorlar, isterse `max_rt_cer` ile filtreler."""
    from src.translit.infer import SungurTransliterator

    records = list(read_jsonl(input_path))
    engine = SungurTransliterator(model_dir=model_dir)
    scored = annotate_roundtrip_cer(
        records,
        transliterate_fn=lambda texts: engine.transliterate_batch(
            texts, prefix=prefix, batch_size=batch_size
        ),
        batch_size=batch_size,
    )

    final_records = (
        filter_by_roundtrip_cer(scored, max_rt_cer=max_rt_cer)
        if max_rt_cer is not None
        else scored
    )
    write_jsonl(output_path, final_records)

    avg_cer = (
        sum(r["rt_cer"] for r in scored) / len(scored)
        if scored
        else 0.0
    )
    return {
        "girdi": len(records),
        "cikti": len(final_records),
        "ortalama_rt_cer": round(avg_cer, 4),
        "max_rt_cer": max_rt_cer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentetik veri gidiş-dönüş (rt_cer) skorlayıcı")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/sungur-ceviriyazi-small-v1"))
    parser.add_argument("--prefix", type=str, default="<vk> ")
    parser.add_argument("--max-rt-cer", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    res = score_synthetic_file(
        input_path=args.input,
        output_path=args.output,
        model_dir=args.model,
        prefix=args.prefix,
        max_rt_cer=args.max_rt_cer,
        batch_size=args.batch_size,
    )
    print(f"rt_cer tamamlandı: {res}")


if __name__ == "__main__":
    main()
