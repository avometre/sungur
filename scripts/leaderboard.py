"""Tüm temel çözümlerin ve modellerin değerlendirme skorlarını özetleyen liderlik tablosu betiği.

Kullanım:
  python scripts/leaderboard.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_all_scores(eval_dir: Path = Path("data/eval")) -> List[Dict[str, Any]]:
    """data/eval altındaki tüm *_skor.json dosyalarını okur."""
    if not eval_dir.exists():
        return []

    all_scores = []
    for score_path in sorted(eval_dir.glob("*/*_skor.json")):
        try:
            with open(score_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_file"] = str(score_path)
                all_scores.append(data)
        except Exception as e:
            print(f"Hata ({score_path}): {e}", file=sys.stderr)

    return all_scores


def print_leaderboard(scores: List[Dict[str, Any]]) -> None:
    """Liderlik tablosunu biçimlendirilmiş olarak konsola basar."""
    if not scores:
        print("Henüz kaydedilmiş değerlendirme skoru bulunamadı (data/eval boş).")
        return

    print("=" * 115)
    print("OSMANLICA -> TÜRKÇE ÇEVİRİYAZI LİDERLİK TABLOSU (LEADERBOARD)")
    print("=" * 115)

    header = (
        f"{'Sistem':<15} {'Split':<6} {'Mod':<6} "
        f"{'CER (%)':<9} {'WER (%)':<9} {'chrF2':<8} {'TamEşl(%)':<10} "
        f"{'Birebir CER':<12} {'Serbest CER':<12} {'Makro CER':<10}"
    )
    print(header)
    print("-" * 115)

    # Sistem ve split'e göre sırala
    sorted_scores = sorted(
        scores,
        key=lambda s: (s.get("system", ""), s.get("split", ""))
    )

    for entry in sorted_scores:
        sys_name = entry.get("system", "-")
        split = entry.get("split", "-")
        skorlar = entry.get("skorlar", {})

        for mode_key in ["sade", "kati"]:
            if mode_key not in skorlar:
                continue

            m_data = skorlar[mode_key]
            top = m_data.get("toplam", {})
            uslup = m_data.get("uslup", {})
            birebir_cer = uslup.get("birebir", {}).get("cer_pct", 0.0)
            serbest_cer = uslup.get("serbest", {}).get("cer_pct", 0.0)
            makro_cer = m_data.get("sayfa_makro", {}).get("cer_pct", 0.0)

            print(
                f"{sys_name:<15} {split:<6} {mode_key:<6} "
                f"%{top.get('cer_pct', 0.0):<8.2f} %{top.get('wer_pct', 0.0):<8.2f} "
                f"{top.get('chrf', 0.0):<8.2f} %{top.get('tam_eslesme_pct', 0.0):<9.2f} "
                f"%{birebir_cer:<11.2f} %{serbest_cer:<11.2f} %{makro_cer:<9.2f}"
            )
        print("-" * 115)
    print("=" * 115)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    scores = load_all_scores()
    print_leaderboard(scores)


if __name__ == "__main__":
    main()
