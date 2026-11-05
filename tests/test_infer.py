"""Sungur çıkarım aracı (src/translit/infer.py) birim testleri."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.translit.infer import SungurTransliterator


def test_sungur_transliterator_mock():
    """SungurTransliterator metin bölme ve ön ek biçimlendirmesini sınar."""
    # Model yüklemesini mock'la
    engine = SungurTransliterator.__new__(SungurTransliterator)
    engine.model_dir = Path("models/sungur-ceviriyazi-small-v1")
    engine.device = "cpu"
    engine.tokenizer = MagicMock()
    engine.model = MagicMock()

    # Mock transliterate_batch
    engine.transliterate_batch = MagicMock(return_value=["satır bir", "satır iki"])

    multiline = "\u0628\u0631\u0646\u062c\u06cc \u0633\u0637\u0631\n\u0627\u06cc\u06a9\u0646\u062c\u06cc \u0633\u0637\u0631"
    res = engine.transliterate_text(multiline, prefix="<vk> ")

    assert "satır bir" in res
    assert "satır iki" in res
    assert "\n" in res
