"""B3_llm — Gemini API ile çeviriyazı çözümü (isteğe bağlı).

Not:
- Ortam değişkeni GEMINI_API_KEY yoksa bu adım temiz bir şekilde atlanır ve "anahtar yok, atlandı" bildirilir.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY ortam değişkeni bulunamadı. Adım temiz şekilde atlandı: anahtar yok, atlandı.")
        sys.exit(0)

    # API anahtarı varsa model parametresi denetlenir
    parser = argparse.ArgumentParser(description="B3_llm Gemini çeviriyazı çözümü.")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL"), help="Gemini model adı")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Değerlendirme kümesi")
    args = parser.parse_args()

    if not args.model:
        print("HATA: Model adı --model parametresiyle ya da GEMINI_MODEL ortam değişkeniyle verilmelidir.", file=sys.stderr)
        sys.exit(1)

    print(f"B3_llm çalıştırılıyor (Model: {args.model})...")


if __name__ == "__main__":
    main()
