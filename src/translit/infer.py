"""Sungur çeviriyazı modeli çıkarım (inference) ve değerlendirme modülü.

Kullanım:
  # Tek metin çevirisi
  python -m src.translit.infer --model models/sungur-ceviriyazi-small-v1 --text "عثمانلی توركجه‌سی"

  # Dosya çevirisi
  python -m src.translit.infer --model models/sungur-ceviriyazi-small-v1 --file girdi.txt -o cikti.txt

  # Altın test/dev kümesinde tam değerlendirme (S1_sungur_small_v1)
  python -m src.translit.infer --model models/sungur-ceviriyazi-small-v1 --split dev
  python -m src.translit.infer --model models/sungur-ceviriyazi-small-v1 --split test
"""

import argparse
import functools
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_latin, normalize_ottoman
from src.eval.run_eval import print_score_table, run_evaluation
from src.translit.decode import (
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    compute_output_limit,
    generate_with_guard,
    is_degenerate,
    truncate_to_limit,
)

print = functools.partial(print, flush=True)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models/sungur-ceviriyazi-small-v1")


class SungurTransliterator:
    """Sungur ByT5 çeviriyazı çıkarım motoru."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model klasörü bulunamadı: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

    def transliterate_sentence(
        self,
        text: str,
        prefix: str = "<vk> ",
        beam_size: int = 4,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ) -> str:
        """Tek bir Arapça cümleyi korumalı olarak Latin harflerine çevirir."""
        clean_text = normalize_ottoman(text).strip()
        if not clean_text:
            return ""

        res, _ = generate_with_guard(
            model=self.model,
            tokenizer=self.tokenizer,
            sentences=[clean_text],
            prefix=prefix,
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=self.device,
            batch_size=1,
            apply_retry=True,
            apply_limit=True,
        )
        return normalize_latin(res[0]).strip()

    def transliterate_batch(
        self,
        sentences: List[str],
        prefix: str = "<vk> ",
        beam_size: int = 4,
        batch_size: int = 16,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        apply_retry: bool = True,
        apply_limit: bool = True,
    ) -> List[str]:
        """Cümle listesini uzunluk sınırı ve tekrar korumasıyla toplu halde çevirir."""
        res, _ = generate_with_guard(
            model=self.model,
            tokenizer=self.tokenizer,
            sentences=sentences,
            prefix=prefix,
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=self.device,
            batch_size=batch_size,
            apply_retry=apply_retry,
            apply_limit=apply_limit,
        )
        return [normalize_latin(s).strip() for s in res]

    def transliterate_text(
        self,
        text: str,
        prefix: str = "<vk> ",
        beam_size: int = 4,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ) -> str:
        """Çok satırlı serbest metni satır satır çevirir."""
        lines = text.splitlines()
        translated_lines = self.transliterate_batch(
            lines,
            prefix=prefix,
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return "\n".join(translated_lines)


def run_evaluation_sungur(
    split: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
    system_name: str = "S1b_sungur_small_v1",
    beam_size: int = 4,
    batch_size: int = 16,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
) -> Dict[str, Any]:
    """Altın kümede modeli çalıştırır, data/eval/<system_name> altına yazar ve değerlendirir."""
    gold_path = Path(f"data/gold/translit_{split}.jsonl")
    pred_dir = Path(f"data/eval/{system_name}")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}.jsonl"
    skor_path = pred_dir / f"{split}_skor.json"

    print(f"{system_name} modeli yükleniyor ve {split.upper()} kümesi çevriliyor...")
    engine = SungurTransliterator(model_dir=model_dir)

    gold_records = list(read_jsonl(gold_path))
    sentences = [r.get("arap", "") for r in gold_records]

    hypotheses = engine.transliterate_batch(
        sentences,
        prefix="<vk> ",
        beam_size=beam_size,
        batch_size=batch_size,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    pred_records = [
        {"id": r["id"], "tahmin": hyp}
        for r, hyp in zip(gold_records, hypotheses)
    ]
    write_jsonl(pred_path, pred_records)

    eval_result = run_evaluation(
        system_name=system_name,
        split=split,
        gold_path=gold_path,
        pred_path=pred_path,
        output_path=skor_path,
    )
    return eval_result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Sungur çeviriyazı çıkarım aracı.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR, help="Model klasör yolu")
    parser.add_argument("--text", type=str, default=None, help="Çevrilecek metin")
    parser.add_argument("--file", type=Path, default=None, help="Çevrilecek metin dosyası")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Çıktı dosya yolu")
    parser.add_argument("--split", choices=["dev", "test", "both"], default=None, help="Değerlendirme kümesi")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam arama genişliği")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch boyutu")
    parser.add_argument("--system-name", default="S1b_sungur_small_v1", help="Değerlendirme sistem adı")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=DEFAULT_NO_REPEAT_NGRAM_SIZE, help="Tekrar koruması ngram boyutu")
    args = parser.parse_args()

    if args.split:
        splits = ["dev", "test"] if args.split == "both" else [args.split]
        for sp in splits:
            res = run_evaluation_sungur(
                split=sp,
                model_dir=args.model,
                system_name=args.system_name,
                beam_size=args.beam_size,
                batch_size=args.batch_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            print_score_table(res)
        return

    if args.text:
        engine = SungurTransliterator(model_dir=args.model)
        result = engine.transliterate_text(args.text, beam_size=args.beam_size)
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        else:
            print(result)
        return

    if args.file:
        content = args.file.read_text(encoding="utf-8")
        engine = SungurTransliterator(model_dir=args.model)
        result = engine.transliterate_text(content, beam_size=args.beam_size)
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        else:
            print(result)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
