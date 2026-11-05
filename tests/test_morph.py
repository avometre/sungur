"""Çekim üretici (src/translit/morph.py) birim testleri."""

import pytest
from src.translit.morph import (
    find_last_vowel,
    get_vowel_harmony,
    inflect_noun,
    inflect_verb,
)


def test_morph_noun_vowel_harmony_and_consonants():
    """İsim çekimlerinde 2'li ve 4'lü ses uyumu ile sert ünsüz benzeşmesini sınar."""
    # kitab + lar -> \u06a9\u062a\u0627\u0628\u0644\u0631 / kitablar
    # Arapça: ک = \u06a9, ت = \u062a, ا = \u0627, ب = \u0628
    items = inflect_noun("kitab", "\u06a9\u062a\u0627\u0628", "test:1")
    by_ek = {it["ek"]: it for it in items}
    assert by_ek["cogul"]["latin"] == "kitablar"
    assert by_ek["cogul"]["arap"] == "\u06a9\u062a\u0627\u0628\u0644\u0631"

    # ev + de -> \u0627\u0648\u062f\u0647 / evde
    # Arapça: ا = \u0627, و = \u0648
    items_ev = inflect_noun("ev", "\u0627\u0648", "test:2")
    by_ek_ev = {it["ek"]: it for it in items_ev}
    assert by_ek_ev["bulunma"]["latin"] == "evde"
    assert by_ek_ev["bulunma"]["arap"] == "\u0627\u0648\u062f\u0647"

    # Sert ünsüz benzeşmesi: fas + da -> fasta, fas + dan -> fastan
    # Arapça: فاسده, فاسدن (Arapçada değişmez)
    items_fas = inflect_noun("fas", "\u0641\u0627\u0633", "test:3")
    by_ek_fas = {it["ek"]: it for it in items_fas}
    assert by_ek_fas["bulunma"]["latin"] == "fasta"
    assert by_ek_fas["bulunma"]["arap"] == "\u0641\u0627\u0633\u062f\u0647"
    assert by_ek_fas["ayrilma"]["latin"] == "fastan"
    assert by_ek_fas["ayrilma"]["arap"] == "\u0641\u0627\u0633\u062f\u0646"


def test_morph_noun_zwnj_on_he():
    """Arapça kök he (\u0647) ile bitiyorsa ZWNJ (\u200c) eklendiğini sınar."""
    # hâne -> \u062e\u0627\u0646\u0647
    items = inflect_noun("hâne", "\u062e\u0627\u0646\u0647", "test:4")
    by_ek = {it["ek"]: it for it in items}
    # خانه‌لر -> \u062e\u0627\u0646\u0647\u200c\u0644\u0631
    assert by_ek["cogul"]["arap"] == "\u062e\u0627\u0646\u0647\u200c\u0644\u0631"
    assert by_ek["cogul"]["latin"] == "hâneler"
    assert by_ek["bulunma"]["arap"] == "\u062e\u0627\u0646\u0647\u200c\u062f\u0647"


def test_morph_verb_inflections():
    """Fiil çekimlerini sınar: yazmak -> yazdı, yazmış, yazıyor."""
    # yazmak -> \u06cc\u0627\u0632\u0645\u0642
    items = inflect_verb("yazmak", "\u06cc\u0627\u0632\u0645\u0642", "test:5")
    by_ek = {it["ek"]: it for it in items}

    # geçmiş zaman: yazdı -> \u06cc\u0627\u0632\u062f\u06cc
    assert by_ek["fiil_gecmis"]["latin"] == "yazdı"
    assert by_ek["fiil_gecmis"]["arap"] == "\u06cc\u0627\u0632\u062f\u06cc"

    # miş'li geçmiş: yazmış -> \u06cc\u0627\u0632\u0645\u0634
    assert by_ek["fiil_mis"]["latin"] == "yazmış"
    assert by_ek["fiil_mis"]["arap"] == "\u06cc\u0627\u0632\u0645\u0634"

    # şimdiki zaman: yazıyor -> \u06cc\u0627\u0632\u06cc\u0648\u0631
    assert by_ek["fiil_simdiki"]["latin"] == "yazıyor"
    assert by_ek["fiil_simdiki"]["arap"] == "\u06cc\u0627\u0632\u06cc\u0648\u0631"
