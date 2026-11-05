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
        max_length: int = 512,
    ) -> str:
        """Tek bir Arapça cümleyi Latin harflerine çevirir."""
        clean_text = normalize_ottoman(text).strip()
        if not clean_text:
            return ""

        prompt = f"{prefix}{clean_text}"
        inputs = self.tokenizer(
            [prompt],
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_beams=beam_size,
                    do_sample=False,
                )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return normalize_latin(decoded).strip()

    def transliterate_batch(
        self,
        sentences: List[str],
        prefix: str = "<vk> ",
        beam_size: int = 4,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> List[str]:
        """Cümle listesini toplu halde verimli biçimde çevirir."""
        results = []
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i : i + batch_size]
            prompts = [f"{prefix}{normalize_ottoman(s).strip()}" for s in chunk]

            inputs = self.tokenizer(
                prompts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        num_beams=beam_size,
                        do_sample=False,
                    )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for d in decoded:
                results.append(normalize_latin(d).strip())

        return results

    def transliterate_text(
        self,
        text: str,
        prefix: str = "<vk> ",
        beam_size: int = 4,
    ) -> str:
        """Çok satırlı serbest metni satır satır çevirir."""
        lines = text.splitlines()
        translated_lines = self.transliterate_batch(lines, prefix=prefix, beam_size=beam_size)
        return "\n".join(translated_lines)


def run_evaluation_sungur(
    split: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
    beam_size: int = 4,
) -> Dict[str, Any]:
    """Altın kümede modeli çalıştırır, data/eval/S1_sungur_small_v1 altına yazar ve değerlendirir."""
    gold_path = Path(f"data/gold/translit_{split}.jsonl")
    pred_dir = Path("data/eval/S1_sungur_small_v1")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}.jsonl"
    skor_path = pred_dir / f"{split}_skor.json"

    print(f"S1_sungur_small_v1 modeli yükleniyor ve {split.upper()} kümesi çevriliyor...")
    engine = SungurTransliterator(model_dir=model_dir)

    gold_records = list(read_jsonl(gold_path))
    sentences = [r.get("arap", "") for r in gold_records]

    hypotheses = engine.transliterate_batch(
        sentences,
        prefix="<vk> ",
        beam_size=beam_size,
        batch_size=32,
    )

    pred_records = [
        {"id": r["id"], "tahmin": hyp}
        for r, hyp in zip(gold_records, hypotheses)
    ]
    write_jsonl(pred_path, pred_records)

    eval_result = run_evaluation(
        system_name="S1_sungur_small_v1",
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
    args = parser.parse_args()

    if args.split:
        splits = ["dev", "test"] if args.split == "both" else [args.split]
        for sp in splits:
            res = run_evaluation_sungur(split=sp, model_dir=args.model, beam_size=args.beam_size)
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
