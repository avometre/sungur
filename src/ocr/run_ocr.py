"""Uçtan uca OCR ve Çeviriyazı CLI modülü (src/ocr/run_ocr.py).

Kullanım:
  python -m src.ocr.run_ocr --image sayfa.jpg [--segmenter klasik|kraken] [--translit models/sungur-ceviriyazi-small-v1]

Çıktı: Satır satır Arap metni; --translit verilirse yanında Latin transkripsiyonu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from src.common.normalize import normalize_ottoman
from src.ocr.eval_ocr import AzraOCRRunner
from src.ocr.segment import segment_page


class EndToEndPipeline:
    """Görüntü -> Satır Bölütleme -> Azra-1-Mini -> S1b Çeviriyazı boru hattı."""

    def __init__(
        self,
        ocr_runner: Optional[AzraOCRRunner] = None,
        translit_model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ocr_runner = ocr_runner or AzraOCRRunner(device=self.device)
        self.translit_model = None
        self.translit_tokenizer = None

        if translit_model_path:
            self._load_translit_model(translit_model_path)

    def _load_translit_model(self, model_path: str | Path) -> None:
        """S1b ByT5 çeviriyazı modelini yükler."""
        from transformers import AutoTokenizer, ByT5Tokenizer, T5ForConditionalGeneration

        path_str = str(model_path)
        print(f"[TRANSLIT] Çeviriyazı modeli yükleniyor: {path_str}...")
        try:
            self.translit_tokenizer = AutoTokenizer.from_pretrained(path_str)
        except Exception:
            self.translit_tokenizer = ByT5Tokenizer.from_pretrained(path_str)

        self.translit_model = T5ForConditionalGeneration.from_pretrained(
            path_str,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.translit_model.eval()
        print("[TRANSLIT] Çeviriyazı modeli yüklendi!")

    def process_image(
        self,
        image: Image.Image | str | Path,
        segmenter: str = "klasik",
        prefix: str = "<vk> ",
    ) -> List[Tuple[Tuple[int, int, int, int], str, Optional[str]]]:
        """Sayfayı işler ve her satır için (kutu, arapca_metin, latin_metin) döndürür."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # 1. Satır bölütleme
        lines = segment_page(image, method=segmenter)
        if not lines:
            return []

        line_boxes = [box for box, _ in lines]
        line_images = [img for _, img in lines]

        # 2. OCR Çıkarımı (Azra-1-Mini)
        raw_ocr_texts = self.ocr_runner.predict_batch(
            line_images, use_dynamic_tokens=True
        )

        # 3. Normalizasyon
        normalized_texts = [
            normalize_ottoman(t, keep_harakat=False) for t in raw_ocr_texts
        ]

        # 4. İsteğe bağlı S1b Çeviriyazı
        latin_texts: List[Optional[str]] = [None] * len(normalized_texts)
        if self.translit_model is not None and self.translit_tokenizer is not None:
            from src.translit.decode import generate_with_guard

            latin_preds, _ = generate_with_guard(
                model=self.translit_model,
                tokenizer=self.translit_tokenizer,
                sentences=normalized_texts,
                prefix=prefix,
                beam_size=4,
                no_repeat_ngram_size=12,
                device=self.device,
            )
            latin_texts = latin_preds

        return list(zip(line_boxes, normalized_texts, latin_texts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Osmanlıca Görüntü -> OCR -> Çeviriyazı Boru Hattı"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="İşlenecek sayfa görüntüsünün yolu"
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="klasik",
        choices=["klasik", "kraken"],
        help="Satır bölütleme yöntemi (varsayılan: klasik)",
    )
    parser.add_argument(
        "--translit",
        type=str,
        default=None,
        help="S1b ByT5 çeviriyazı modelinin yolu (ör. models/sungur-ceviriyazi-small-v1)",
    )

    args = parser.parse_args()
    pipeline = EndToEndPipeline(
        translit_model_path=args.translit,
    )

    results = pipeline.process_image(
        image=args.image,
        segmenter=args.segmenter,
    )

    print("\n" + "=" * 80)
    print(f"SAYFA İŞLEME SONUCU ({len(results)} SATIR): {args.image}")
    print("=" * 80)

    for i, (box, arap, latin) in enumerate(results, 1):
        x0, y0, x1, y1 = box
        if latin is not None:
            print(f"[{i:02d}] ({x0},{y0}-{x1},{y1})")
            print(f"     Arap:  {arap}")
            print(f"     Latin: {latin}")
        else:
            print(f"[{i:02d}] ({x0},{y0}-{x1},{y1}) {arap}")


if __name__ == "__main__":
    main()
