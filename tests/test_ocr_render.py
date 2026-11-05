from __future__ import annotations

from pathlib import Path
import pytest

from src.ocr.render import validate_font_glyphs


def test_validate_font_glyphs_amiri_and_noto():
    fonts_dir = Path("data/fonts")
    amiri = fonts_dir / "Amiri-Regular.ttf"
    noto = fonts_dir / "NotoNaskhArabic-Regular.ttf"

    if not amiri.exists() or not noto.exists():
        pytest.skip("Font dosyaları data/fonts altında bulunamadı.")

    # Amiri ve Noto sunum formlarını (U+FB50-FEFF) içerir, True dönmeli
    assert validate_font_glyphs(amiri, sample_text="کتاب", require_reshaped_forms=True) is True
    assert validate_font_glyphs(noto, sample_text="کتاب", require_reshaped_forms=True) is True


def test_validate_font_glyphs_scheherazade_missing_reshaped():
    fonts_dir = Path("data/fonts")
    scheherazade = fonts_dir / "ScheherazadeNew-Regular.ttf"

    if not scheherazade.exists():
        pytest.skip("ScheherazadeNew font dosyası bulunamadı.")

    # ScheherazadeNew modern OpenType fontudur ve sunum formlarını (U+FB50-FEFF) içermez.
    # Bu nedenle require_reshaped_forms=True ile çağrıldığında ValueError vermeli.
    with pytest.raises(ValueError, match="glif doğrulaması başarısız"):
        validate_font_glyphs(scheherazade, sample_text="کتاب", require_reshaped_forms=True)

    # Ancak doğrudan temel Arapça kod noktaları kontrol edildiğinde (Raqm modu) başarılı olmalı
    assert validate_font_glyphs(scheherazade, sample_text="کتاب", require_reshaped_forms=False) is True
