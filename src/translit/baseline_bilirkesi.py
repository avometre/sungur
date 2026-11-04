"""B2_bilirkesi — Hugging Face 'bilirkesi/osmanlica-v1' (ottoman-transliterator) temel çözümü.

Paket İnceleme Özeti:
- Paket Adı: ottoman-transliterator (v0.0.0, CLI/kod sürümü 1.0.0)
- Bağımlılıklar: click, openai (opsiyonel neural mod), turkicnlp (opsiyonel nlp mod)
- Mimari: Varsayılan modda (hybrid/neural) DeepSeek V4 Flash API (https://api.deepseek.com/v1)
  üzerinden dış ağ çağrısı yapan bir LLM sarmalayıcısıdır.
- Çevrimdışı Çalışma: Dış API anahtarı olmadan ve ağa çıkmadan mode="nlp" fallback'i ile
  çevrimdışı karakter kural eşlemesi (_rule_based_transliterate) çalıştırılır.
- Güvenlik: Kötü niyetli arka kapı, telemetry veya dosya tahribatı kodu bulunmamaktadır.
  Ana ortamın kütüphanelerini korumak amacıyla izole .venv-bilirkesi sanal ortamında çalışır.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.common.io import read_jsonl, write_jsonl
from src.eval.run_eval import print_score_table, run_evaluation

logger = logging.getLogger(__name__)

VENV_BILIRKESI_PYTHON = Path(".venv-bilirkesi/Scripts/python.exe")


def run_bilirkesi_prediction(split: str) -> Path:
    """İzole .venv-bilirkesi ortamında dev/test kümesini çevirir ve jsonl olarak kaydeder."""
    gold_path = Path(f"data/gold/translit_{split}.jsonl")
    pred_dir = Path("data/eval/B2_bilirkesi")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}.jsonl"

    if not VENV_BILIRKESI_PYTHON.exists():
        raise FileNotFoundError(
            f".venv-bilirkesi ortamı bulunamadı ({VENV_BILIRKESI_PYTHON}). "
            "Lütfen 'uv venv .venv-bilirkesi' ve paketi kurun."
        )

    # İzole sanal ortamda çeviri yapan betik kodu
    worker_code = """
import sys, json
from pathlib import Path
from ottoman_transliterator import OttomanTransliterationPipeline, OttomanPipelineConfig

gold_path = sys.argv[1]
pred_path = sys.argv[2]

config = OttomanPipelineConfig(use_hybrid=False, use_nlp_fallback=True)
pipeline = OttomanTransliterationPipeline(config)

records = []
with open(gold_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        arap = item.get("arap", "")
        # Çevrimdışı mod: mode='nlp' (dış ağ çağrısı yok)
        res = pipeline.transliterate(arap, mode="nlp")
        records.append({"id": item["id"], "tahmin": res.modern_turkish})

with open(pred_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
"""

    cmd = [
        str(VENV_BILIRKESI_PYTHON.resolve()),
        "-c",
        worker_code,
        str(gold_path.resolve()),
        str(pred_path.resolve()),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Bilirkesi tahmin çalıştırma hatası:\n{res.stderr}")

    return pred_path


def run_baseline_bilirkesi(split: str = "dev") -> Dict[str, Any]:
    """B2_bilirkesi modelini çalıştırır, tahminleri kaydeder ve değerlendirir."""
    gold_path = Path(f"data/gold/translit_{split}.jsonl")
    pred_path = run_bilirkesi_prediction(split)
    skor_path = Path(f"data/eval/B2_bilirkesi/{split}_skor.json")

    eval_result = run_evaluation(
        system_name="B2_bilirkesi",
        split=split,
        gold_path=gold_path,
        pred_path=pred_path,
        output_path=skor_path,
    )
    return eval_result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="B2_bilirkesi çeviriyazı çözümü.")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both", help="Değerlendirme kümesi")
    args = parser.parse_args()

    splits = ["dev", "test"] if args.split == "both" else [args.split]
    for sp in splits:
        print(f"\nB2_bilirkesi çalıştırılıyor ({sp})...")
        res = run_baseline_bilirkesi(sp)
        print_score_table(res)


if __name__ == "__main__":
    main()
