"""Temel çözümler (B0_kural ve B1_sozluk) birim testleri."""

import pytest
from src.translit.baseline_dict import (
    SUFFIX_TABLE,
    BaselineDictionary,
    find_last_vowel,
    resolve_suffix,
)
from src.translit.baseline_rules import transliterate_rules, transliterate_word_rules


def test_b0_kural_transliteration():
    """B0_kural harf eşleme ve kural çevirisini sınar."""
    # كتاب -> ktab / kitab
    # \u0643\u062a\u0627\u0628 (k-t-a-b)
    assert transliterate_word_rules("\u0643\u062a\u0627\u0628") == "ktab"

    # باب -> bab
    # \u0628\u0627\u0628
    assert transliterate_word_rules("\u0628\u0627\u0628") == "bab"

    # Kelime sonu he -> e
    # نامە -> nam-i veya نامه -> name
    # \u0646\u0627\u0645\u0647 (n-a-m-h)
    assert transliterate_word_rules("\u0646\u0627\u0645\u0647") == "name"

    # İzafet hemzesi ۀ -> -i
    # \u062e\u0627\u0646\u06c0 (h-a-n-izafet)
    assert transliterate_word_rules("\u062e\u0627\u0646\u06c0") == "han-i"

    # Cümle düzeyinde noktalama korunumu
    res = transliterate_rules("\u0628\u0627\u0628, \u0643\u062a\u0627\u0628!")
    assert res == "bab, ktab!"


def test_suffix_table_and_vowel_harmony():
    """B1_sozluk ek tablosu ve ses uyumu işlevlerini sınar."""
    # Ek tablosu boş olmamalı ve beklenen ekleri içermeli
    assert len(SUFFIX_TABLE) > 10
    suffixes_set = {s[0] for s in SUFFIX_TABLE}
    assert "\u0644\u0631" in suffixes_set  # لر
    assert "\u062f\u0647" in suffixes_set  # ده
    assert "\u062f\u0646" in suffixes_set  # دن

    # Son ünlü tespiti
    assert find_last_vowel("kitap") == "a"
    assert find_last_vowel("ev") == "e"
    assert find_last_vowel("göz") == "ö"
    assert find_last_vowel("yolcu") == "u"

    # 2'li ses uyumu (e / a)
    assert resolve_suffix("ler", "kitap") == "lar"
    assert resolve_suffix("ler", "ev") == "ler"
    assert resolve_suffix("de", "kitap") == "da"
    assert resolve_suffix("de", "ev") == "de"
    assert resolve_suffix("den", "kitap") == "dan"
    assert resolve_suffix("den", "ev") == "den"

    # 4'lü ses uyumu (i / ı / u / ü)
    assert resolve_suffix("lik_kalin", "baş") == "lık"
    assert resolve_suffix("lik_kalin", "kol") == "luk"
    assert resolve_suffix("lik_ince", "göz") == "lük"
    assert resolve_suffix("lik_ince", "ev") == "lik"


def test_baseline_dict_transliteration():
    """BaselineDictionary nesnesinin sözlük eşleşmesi ve ek soymasını sınar."""
    b_dict = BaselineDictionary()

    # Eğer sözlükte 'devlet' veya bilinen bir kelime varsa
    # \u062f\u0648\u0644\u062a (devlet)
    t = b_dict.transliterate_word("\u062f\u0648\u0644\u062a")
    assert t != ""
