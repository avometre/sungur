"""Geri çeviri, sık kelimeler ve filtre kuralları (src/translit/backtranslate.py) birim testleri."""

import pytest
from src.translit.backtranslate import (
    BackTransliterator,
    ReverseDictionary,
    is_valid_arabic_prediction,
    load_sik_kelimeler,
)


def test_reverse_dictionary_lookup():
    """Ters sözlüğün latin_sade anahtarıyla en sık Arapça karşılığı verdiğini sınar."""
    r_dict = ReverseDictionary()

    # "kitab" veya "istanbul" gibi bilinen bir kelime
    ar_kitab = r_dict.get("kitab")
    assert ar_kitab is not None
    assert "\u06a9\u062a\u0627\u0628" in ar_kitab or "\u0643\u062a\u0627\u0628" in ar_kitab

    # Şapkalı kelimenin sade biçimle de bulunabilmesi
    ar_alem = r_dict.get("alem")
    assert ar_alem is not None


def test_special_rules_conjunction_and_numbers():
    """Bağlaç ve rakam kurallarını sınar."""
    bt = BackTransliterator(model_path=None)

    # Bağlaç: ü, vü, u -> \u0648 (vav)
    assert bt.translate_token("ü") == "\u0648"
    assert bt.translate_token("vü") == "\u0648"
    assert bt.translate_token("u") == "\u0648"

    # Rakamlar korunmalı
    assert bt.translate_token("1920") == "1920"
    assert bt.translate_token("10") == "10"


def test_special_rules_izafet():
    """İzafet kurallarını sınar (-i/-ı ek yazılmaz; -yi/-yı he ile bitende ۀ, diğerinde ی)."""
    bt = BackTransliterator(model_path=None)

    # 1. Standart izafet: "devlet-i" -> Arapçada izafet eki yazılmaz, sadece "devlet"
    res_devlet = bt.translate_token("devlet-i")
    assert res_devlet == bt.translate_single_word("devlet")

    # 2. Ünlüyle biten izafet:
    # hâne-yi -> hâne sonu he (\u0647) olduğu için son he yerine ۀ (\u06c0) gelir
    bt.word_cache["hâne"] = "\u062e\u0627\u0646\u0647"
    res_hane = bt.translate_token("hâne-yi")
    # خانۀ -> \u062e\u0627\u0646\u06c0
    assert res_hane == "\u062e\u0627\u0646\u06c0"

    # he ile bitmeyen kök: safâ-yı -> safâ + ی
    bt.word_cache["safâ"] = "\u0635\u0641\u0627"
    res_safa = bt.translate_token("safâ-yı")
    # صفای -> \u0635\u0641\u0627\u06cc
    assert res_safa == "\u0635\u0641\u0627\u06cc"


def test_special_rules_hyphenated_compound():
    """Tireli bileşiklerin ayrı çevrilip birleştirildiğini sınar."""
    bt = BackTransliterator(model_path=None)
    bt.word_cache["testkelime"] = "\u062a\u0633\u062a"
    bt.word_cache["ornekkelime"] = "\u0627\u0648\u0631\u0646\u06a9"

    # testkelime-ornekkelime -> testkelime + " " + ornekkelime
    res = bt.translate_token("testkelime-ornekkelime")
    assert res == "\u062a\u0633\u062a \u0627\u0648\u0631\u0646\u06a9"


def test_sentence_tokenization_and_punctuation():
    """Cümle düzeyinde noktalama işaretlerinin ve tırnakların korunduğunu sınar."""
    bt = BackTransliterator(model_path=None)
    bt.word_cache["kitap"] = "\u06a9\u062a\u0627\u0628"

    # Noktalama ve boşluklar korunmalı
    sent = "«kitap», 1920!"
    res, is_valid = bt.translate_sentence(sent)
    assert res.startswith("«")
    assert "»" in res
    assert "1920!" in res
    assert is_valid is True


def test_output_filter_validity_rules():
    """GÖREV 07b model çıktı filtresi kurallarını sınar."""
    # 1. "بروس" -> sözdizimsel olarak geçerli sayılmalı (filtre fonksiyonu bunu yakalayamaz, sözlük çözer)
    # بروس -> \u0628\u0631\u0648\u0633
    assert is_valid_arabic_prediction("\u0628\u0631\u0648\u0633", "o") is True

    # 2. "پی「......................." -> geçersiz (Arapça dışı karakter ve aşırı nokta tekrarı)
    assert is_valid_arabic_prediction("\u067e\u06cc\u300c.......................", "o") is False

    # 3. "ییییی" -> geçersiz (4+ aynı harf tekrarı)
    # \u06cc * 5
    assert is_valid_arabic_prediction("\u06cc\u06cc\u06cc\u06cc\u06cc", "y") is False

    # 4. Geçerli kelimeler
    # او -> \u0627\u0648
    assert is_valid_arabic_prediction("\u0627\u0648", "o") is True
    # فقط -> \u0641\u0642\u0637
    assert is_valid_arabic_prediction("\u0641\u0642\u0637", "fakat") is True


def test_sik_kelimeler_loader_and_priority():
    """sik_kelimeler.tsv dosyasının 'o' ve 'fakat' gibi kelimeleri doğru yüklediğini ve öncelikli olduğunu sınar."""
    mapping = load_sik_kelimeler()
    # o -> او (\u0627\u0648)
    assert mapping.get("o") == "\u0627\u0648"
    # fakat -> فقط (\u0641\u0642\u0637)
    assert mapping.get("fakat") == "\u0641\u0642\u0637"

    # BackTransliterator hiyerarşi sınaması:
    # sik_kelimeler ters sözlükten ve modelden önce gelmeli
    bt = BackTransliterator(model_path=None)
    assert bt.translate_single_word("o") == "\u0627\u0648"
    assert bt.translate_single_word("O") == "\u0627\u0648"
    assert bt.translate_single_word("fakat") == "\u0641\u0642\u0637"
