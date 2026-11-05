"""OCR değerlendirme ve metrik fonksiyonları için birim testleri (tests/test_ocr_eval.py).

Ağsız ve GPU'suz çalışır. Arapça metinler \\uXXXX kaçış dizileri ile yazılmıştır.
"""

import pytest

from src.ocr.eval_ocr import calculate_ocr_metrics, compute_dynamic_max_tokens


def test_dynamic_max_tokens_calculation():
    """Dinamik max_new_tokens hesabını farklı en/boy oranlarında test eder."""
    # En: 400, Boy: 40 -> En/Boy = 10 -> int(2 * 10 * 8 + 32) = 192
    tokens = compute_dynamic_max_tokens(400, 40)
    assert tokens == 192

    # Çok dar / kare görsel (En: 50, Boy: 50 -> En/Boy = 1.0 -> int(16 + 32) = 48)
    tokens_square = compute_dynamic_max_tokens(50, 50)
    assert tokens_square == 48

    # Min sınır testi (min_tokens=32)
    tokens_min = compute_dynamic_max_tokens(10, 100, min_tokens=32)
    assert tokens_min == 40  # aspect=0.5 -> 2*0.5*8 + 32 = 40 >= 32

    # Max sınır testi (max_tokens=512)
    tokens_max = compute_dynamic_max_tokens(5000, 20, max_tokens=512)
    assert tokens_max == 512


def test_ocr_metrics_exact_match():
    """Birebir aynı Osmanlıca metinlerde CER 0.0 olmalıdır."""
    # "\u0639\u062b\u0645\u0627\u0646\u0644\u06cc\u062c\u0647 \u0645\u062a\u064a\u0646" = "عثمانلیجه متین"
    ref = ["\u0639\u062b\u0645\u0627\u0646\u0644\u06cc\u062c\u0647 \u0645\u062a\u064a\u0646"]
    hyp = ["\u0639\u062b\u0645\u0627\u0646\u0644\u06cc\u062c\u0647 \u0645\u062a\u064a\u0646"]

    res = calculate_ocr_metrics(ref, hyp)
    assert res["cer"] == 0.0
    assert res["cer_bosluksuz"] == 0.0
    assert res["adet"] == 1


def test_ocr_metrics_harakat_insensitivity():
    """Harekeli referans ile harekesiz tahmin normalize_ottoman sonrası eşit sayılmalıdır."""
    # ref: "\u0639\u064e\u062b\u0652\u0645\u064e\u0627\u0646\u0652\u0644\u0650\u06cc\u062c\u064e\u0647" (harekeli: üstün, cezm, esre)
    # hyp: "\u0639\u062b\u0645\u0627\u0646\u0644\u06cc\u062c\u0647" (harekesiz)
    ref = ["\u0639\u064e\u062b\u0652\u0645\u064e\u0627\u0646\u0652\u0644\u0650\u06cc\u062c\u064e\u0647"]
    hyp = ["\u0639\u062b\u0645\u0627\u0646\u0644\u06cc\u062c\u0647"]

    res = calculate_ocr_metrics(ref, hyp)
    assert res["cer"] == 0.0
    assert res["cer_bosluksuz"] == 0.0


def test_ocr_metrics_whitespace_difference():
    """Boşluk farkı olduğunda standart CER > 0 ancak cer_bosluksuz == 0 olmalıdır."""
    # ref: "پادشاه لر" ("\u067e\u0627\u062f\u0634\u0627\u0647 \u0644\u0631")
    # hyp: "پادشاه‌لر" (zwnj ile bitişik veya boşluksuz: "\u067e\u0627\u062f\u0634\u0627\u0647\u0644\u0631")
    ref = ["\u067e\u0627\u062f\u0634\u0627\u0647 \u0644\u0631"]
    hyp = ["\u067e\u0627\u062f\u0634\u0627\u0647\u0644\u0631"]

    res = calculate_ocr_metrics(ref, hyp)
    assert res["cer"] > 0.0
    assert res["cer_bosluksuz"] == 0.0


def test_ocr_metrics_empty_inputs():
    """Boş girdi durumunda 0 metrik dönmelidir."""
    res = calculate_ocr_metrics([], [])
    assert res["cer"] == 0.0
    assert res["adet"] == 0
