"""Sungur uçtan uca boru hattı (Görev 17): Görüntü -> OCR -> Çeviriyazı -> Sadeleştirme.

Modüler tasarım:
- OCR: Opsiyonel AzraOCRRunner + satır bölütleme (`segment_page`)
- Çeviriyazı: SungurTransliterator (ByT5) veya model yoksa kural/sözlük tabanlı yedek
- Sadeleştirme: Opsiyonel model/fonksiyon veya kelime bazlı sözlük sadeleştirmesi
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.common.normalize import normalize_latin, normalize_ottoman


@dataclass
class LineResult:
    """Tek bir satır için uçtan uca işlem sonucu."""

    box: Optional[Tuple[int, int, int, int]]
    arap: str
    latin: str
    sade: str


class SungurPipeline:
    """Osmanlıca görüntü veya metni günümüz Türkçesine aktaran uçtan uca boru hattı."""

    def __init__(
        self,
        ocr_runner: Optional[Any] = None,
        translit_model_dir: Optional[Path | str] = None,
        translit_fn: Optional[Callable[[List[str]], List[str]]] = None,
        simplify_fn: Optional[Callable[[List[str]], List[str]]] = None,
    ) -> None:
        self.ocr_runner = ocr_runner
        self._translit_engine: Optional[Any] = None
        self.translit_fn = translit_fn
        self.simplify_fn = simplify_fn

        if self.translit_fn is None and translit_model_dir is not None:
            model_path = Path(translit_model_dir)
            if model_path.exists():
                from src.translit.infer import SungurTransliterator

                self._translit_engine = SungurTransliterator(model_dir=model_path)
                self.translit_fn = lambda texts: self._translit_engine.transliterate_batch(texts)

        if self.translit_fn is None:
            from src.translit.baseline_rules import transliterate_rules

            self.translit_fn = lambda texts: [transliterate_rules(t) for t in texts]

        if self.simplify_fn is None:
            self.simplify_fn = lambda texts: [normalize_latin(t).strip() for t in texts]

    def process_arabic_lines(self, lines: List[str]) -> List[LineResult]:
        """Arap harfli satır listesini Çeviriyazı + Sadeleştirme adımlarından geçirir."""
        norm_lines = [normalize_ottoman(ln, keep_harakat=False).strip() for ln in lines]
        latin_lines = self.translit_fn(norm_lines) if norm_lines else []
        sade_lines = self.simplify_fn(latin_lines) if latin_lines else []

        return [
            LineResult(box=None, arap=ar, latin=lat, sade=sad)
            for ar, lat, sad in zip(norm_lines, latin_lines, sade_lines)
        ]

    def process_text(self, arabic_text: str) -> Dict[str, Any]:
        """Çok satırlı Arap harfli metni işler."""
        raw_lines = [ln for ln in arabic_text.splitlines() if ln.strip()]
        results = self.process_arabic_lines(raw_lines)
        return {
            "satir_sayisi": len(results),
            "satirlar": results,
            "arap": "\n".join(r.arap for r in results),
            "latin": "\n".join(r.latin for r in results),
            "sade": "\n".join(r.sade for r in results),
        }

    def process_image(
        self,
        image: Any,
        segmenter: str = "klasik",
        backend: str = "sirali",
    ) -> Dict[str, Any]:
        """Sayfa görüntüsünü OCR + Çeviriyazı + Sadeleştirme ile işler."""
        from PIL import Image
        from src.ocr.eval_ocr import AzraOCRRunner
        from src.ocr.segment import segment_page

        if self.ocr_runner is None:
            self.ocr_runner = AzraOCRRunner()

        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                img = im.convert("RGB")
        else:
            img = image.convert("RGB")

        segmented = segment_page(img, method=segmenter)
        if not segmented:
            return {"satir_sayisi": 0, "satirlar": [], "arap": "", "latin": "", "sade": ""}

        boxes = [b for b, _ in segmented]
        crops = [crop for _, crop in segmented]
        raw_ocr = self.ocr_runner.predict_batch(
            crops, use_dynamic_tokens=True, backend=backend
        )
        line_results = self.process_arabic_lines(raw_ocr)
        for res, box in zip(line_results, boxes):
            res.box = box

        return {
            "satir_sayisi": len(line_results),
            "satirlar": line_results,
            "arap": "\n".join(r.arap for r in line_results),
            "latin": "\n".join(r.latin for r in line_results),
            "sade": "\n".join(r.sade for r in line_results),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sungur uçtan uca boru hattı")
    parser.add_argument("--text", type=str, default=None, help="Arap harfli Osmanlıca metin")
    parser.add_argument("--image", type=Path, default=None, help="Sayfa görüntüsü")
    parser.add_argument("--translit-model", type=Path, default=Path("models/sungur-ceviriyazi-small-v1"))
    parser.add_argument("--segmenter", choices=["klasik", "kraken"], default="klasik")
    parser.add_argument("--backend", choices=["sirali", "batch"], default="sirali")
    args = parser.parse_args()

    pipe = SungurPipeline(translit_model_dir=args.translit_model)
    if args.image is not None:
        out = pipe.process_image(args.image, segmenter=args.segmenter, backend=args.backend)
    elif args.text is not None:
        out = pipe.process_text(args.text)
    else:
        parser.print_help()
        return

    print("--- ARAP ---")
    print(out["arap"])
    print("--- LATİN ---")
    print(out["latin"])
    print("--- SADE ---")
    print(out["sade"])


if __name__ == "__main__":
    main()
