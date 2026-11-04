"""CLI değerlendirme çalıştırıcısı (run_eval).

Kullanım:
  python -m src.eval.run_eval --system <ad> --split dev|test

Girdi:
  data/eval/<sistem>/<split>.jsonl ({"id": str, "tahmin": str})
  data/gold/translit_<split>.jsonl

Çıktı:
  data/eval/<sistem>/<split>_skor.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.io import read_jsonl
from src.eval.metrics import evaluate_both_modes

logger = logging.getLogger(__name__)


def run_evaluation(
    system_name: str,
    split: str,
    gold_path: Optional[Path] = None,
    pred_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Sistem tahminlerini altın veri ile karşılaştırıp skorları hesaplar ve kaydeder."""
    if gold_path is None:
        gold_path = Path(f"data/gold/translit_{split}.jsonl")
    if pred_path is None:
        pred_path = Path(f"data/eval/{system_name}/{split}.jsonl")
    if output_path is None:
        output_path = Path(f"data/eval/{system_name}/{split}_skor.json")

    if not gold_path.exists():
        raise FileNotFoundError(f"Altın veri seti bulunamadı: {gold_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Sistem tahmin dosyası bulunamadı: {pred_path}")

    gold_records = list(read_jsonl(gold_path))
    pred_records = list(read_jsonl(pred_path))

    # id -> tahmin haritası
    predictions: Dict[str, str] = {}
    for r in pred_records:
        rec_id = r.get("id")
        if rec_id:
            # tahmin veya latin_tahmin desteği
            pred_text = r.get("tahmin") if "tahmin" in r else r.get("latin_tahmin", "")
            predictions[rec_id] = str(pred_text or "")

    eval_result = evaluate_both_modes(gold_records, predictions)

    final_payload = {
        "system": system_name,
        "split": split,
        "adet": len(gold_records),
        "eslesen_tahmin_sayisi": len(predictions),
        "skorlar": eval_result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    return final_payload


def print_score_table(result: Dict[str, Any]) -> None:
    """Sonuçları konsola biçimlendirilmiş bir tablo halinde yazdırır."""
    sys_name = result["system"]
    split = result["split"]
    scores = result["skorlar"]

    print("=" * 84)
    print(f"DEĞERLENDİRME RAPORU: {sys_name} ({split.upper()} KÜMESİ, {result['adet']} çift)")
    print("=" * 84)

    header = f"{'Mod':<8} {'Bölüm':<16} {'CER (%)':<10} {'WER (%)':<10} {'chrF2':<10} {'Tam Eşl (%)':<12} {'Adet':<6}"
    print(header)
    print("-" * 84)

    for mode_key, mode_title in [("sade", "SADE (Ana)"), ("kati", "KATI")]:
        mode_data = scores[mode_key]
        top = mode_data["toplam"]
        print(
            f"{mode_title:<8} {'Toplam':<16} "
            f"%{top['cer_pct']:<9.2f} %{top['wer_pct']:<9.2f} {top['chrf']:<10.2f} "
            f"%{top['tam_eslesme_pct']:<11.2f} {top['adet']:<6}"
        )

        for u_name, u_data in mode_data.get("uslup", {}).items():
            print(
                f"{'':<8} {f'  Üslup: {u_name}':<16} "
                f"%{u_data['cer_pct']:<9.2f} %{u_data['wer_pct']:<9.2f} {u_data['chrf']:<10.2f} "
                f"%{u_data['tam_eslesme_pct']:<11.2f} {u_data['adet']:<6}"
            )

        macro = mode_data.get("sayfa_makro", {})
        print(
            f"{'':<8} {'  Sayfa Makro':<16} "
            f"%{macro['cer_pct']:<9.2f} %{macro['wer_pct']:<9.2f} {macro['chrf']:<10.2f} "
            f"%{macro['tam_eslesme_pct']:<11.2f} {macro.get('sayfa_sayisi', '-'):<6}"
        )
        print("-" * 84)
    print("=" * 84)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Çeviriyazı değerlendirme betiği.")
    parser.add_argument("--system", required=True, help="Sistem adı (örn: B0_kural, B1_sozluk)")
    parser.add_argument("--split", required=True, choices=["dev", "test"], help="Değerlendirme kümesi")
    parser.add_argument("--gold-file", type=Path, default=None, help="Altın veri jsonl yolu")
    parser.add_argument("--pred-file", type=Path, default=None, help="Tahmin jsonl yolu")
    parser.add_argument("--output-file", type=Path, default=None, help="Çıktı skor json yolu")
    args = parser.parse_args()

    result = run_evaluation(
        system_name=args.system,
        split=args.split,
        gold_path=args.gold_file,
        pred_path=args.pred_file,
        output_path=args.output_file,
    )
    print_score_table(result)


if __name__ == "__main__":
    main()
