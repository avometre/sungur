import pytest
from src.translit.decode import (
    compute_output_limit,
    truncate_to_limit,
    is_degenerate,
    clean_fallback,
)


def test_compute_output_limit():
    # Ön ek sayılmamalı
    ar = "عثمانلی"
    lim1 = compute_output_limit(ar)
    lim2 = compute_output_limit("<vk> " + ar)
    lim3 = compute_output_limit("<ijmes> " + ar)
    assert lim1 == lim2 == lim3
    # Arapça 7 harf = 14 UTF-8 baytı -> ceil(1.3 * 14) + 16 = 19 + 16 = 35
    assert lim1 == 35


def test_is_degenerate_positive():
    limit = 100
    # 3+ karakterlik tekrar
    assert is_degenerate("Belter:Belter:Belter:", limit) is True
    # 5+ aynı karakter tekrarı
    assert is_degenerate("Didi.i0000000", limit) is True
    # Geçersiz karakter
    assert is_degenerate("Tak! Tak! ±±±±", limit) is True
    # Limiti aşma
    assert is_degenerate("Çok uzun bir metin" * 10, 20) is True


def test_is_degenerate_negative():
    limit = 100
    # 2 kez tekrar meşrudur, bozuk SAYILMAZ
    assert is_degenerate("hemen hemen on sene", limit) is False
    assert is_degenerate("Aman, aman!", limit) is False
    assert is_degenerate("Normal bir Türkçe cümle.", limit) is False


def test_clean_fallback():
    limit = 100
    # Tekrar eden birimi teke indirir
    res = clean_fallback("Belter:Belter:Belter:", limit)
    assert res == "Belter:"

    # 5+ karakter tekrarını teke indirir
    res2 = clean_fallback("Didi.i0000000", limit)
    assert "00000" not in res2

    # Yabancı simgeyi temizler
    res3 = clean_fallback("“Tak!”±±±±“Tak!“Tak…", limit)
    assert "±" not in res3
