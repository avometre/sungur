import pytest
from src.common.normalize import (
    arabic_ratio,
    normalize_latin,
    normalize_ottoman,
    tr_lower,
)


class TestNormalizeOttoman:
    def test_letter_unification_kaf(self) -> None:
        # ك (U+0643) -> ک (U+06A9)
        assert normalize_ottoman("باتك") == "باتک"

    def test_letter_unification_yeh(self) -> None:
        # ي (U+064A) -> ی (U+06CC)
        assert normalize_ottoman("يلع") == "یلع"

    def test_letter_unification_alef_maksura(self) -> None:
        # ى (U+0649) -> ی (U+06CC)
        assert normalize_ottoman("ىفطصم") == "یفطصم"

    def test_letter_unification_urdu_heh(self) -> None:
        # ہ (U+06C1) -> ه (U+0647)
        assert normalize_ottoman("ہماره") == "هماره"

    def test_harakat_removal_default(self) -> None:
        # keep_harakat=False ise harekeler silinir, ك -> ک olur
        assert normalize_ottoman("بٌاتَكِ", keep_harakat=False) == "باتک"

    def test_keep_harakat_preserves_vowels_but_unifies_letters(self) -> None:
        # keep_harakat=True ise harekeler kalır ama ك yine ک olur
        assert normalize_ottoman("بٌاتَكِ", keep_harakat=True) == "بٌاتَکِ"

    def test_zwnj_preserved(self) -> None:
        # U+200C (ZWNJ) korunur
        text_with_zwnj = "پادشاه\u200cلر"
        res = normalize_ottoman(text_with_zwnj)
        assert "\u200c" in res
        assert res == "پادشاه\u200cلر"

    def test_presentation_forms(self) -> None:
        # Sunum formları (Arabic Presentation Forms B: ﺏﺎﺘﻛ) -> باتک
        presentation_str = "\ufe8f\ufe8e\ufe98\ufedb"
        assert normalize_ottoman(presentation_str) == "باتک"

    def test_ascii_digits_conversion(self) -> None:
        # Farsça ve Arap-Hint rakamları 0-9'a çevrilir
        assert normalize_ottoman("۱۹۲۸", ascii_digits=True) == "1928"
        assert normalize_ottoman("١٢٩٣ هنس", ascii_digits=True) == "1293 هنس"
        # ascii_digits=False iken rakamlar korunur
        assert normalize_ottoman("۱۹۲۸", ascii_digits=False) == "۱۹۲۸"

    def test_protected_ottoman_letters(self) -> None:
        # گ, ڭ, چ, پ, ژ, ؤ, ئ, آ, ة, ۋ korunur
        assert normalize_ottoman("گ") == "گ"
        assert normalize_ottoman("ڭ") == "ڭ"
        assert normalize_ottoman("چ") == "چ"
        assert normalize_ottoman("پ") == "پ"
        assert normalize_ottoman("ژ") == "ژ"
        assert normalize_ottoman("آغاز") == "آغاز"
        assert normalize_ottoman("مسئول") == "مسئول"
        assert normalize_ottoman("مؤلف") == "مؤلف"

    def test_tatweel_removal(self) -> None:
        # Tatvil / keşide silinir
        assert normalize_ottoman("رلـــــباتك") == "رلباتک"

    def test_fold_hamza(self) -> None:
        # fold_hamza=False iken hemzeli elifler korunur
        assert normalize_ottoman("أحمد", fold_hamza=False) == "أحمد"
        assert normalize_ottoman("إسلام", fold_hamza=False) == "إسلام"
        # fold_hamza=True iken أ, إ, ٱ -> ا olur
        assert normalize_ottoman("أحمد", fold_hamza=True) == "احمد"
        assert normalize_ottoman("إسلام", fold_hamza=True) == "اسلام"
        assert normalize_ottoman("ٱلله", fold_hamza=True) == "الله"

    def test_punctuation_mapping(self) -> None:
        # ، -> ,   ؟ -> ?   ؛ -> ;
        assert normalize_ottoman("كتاب، قلم؟ دفتر؛") == "کتاب, قلم? دفتر;"

    def test_directional_and_zero_width_removal(self) -> None:
        # U+200E, U+200B, U+200D, U+FEFF silinir
        dirty = "\u200e\ufeff\u200bباتك\u200d"
        assert normalize_ottoman(dirty) == "باتک"

    def test_whitespace_collapse(self) -> None:
        # Fazla ve farklı boşluklar tek boşluğa iner, kırpılır
        text = "   باتک    \u00a0\t  قلم   "
        assert normalize_ottoman(text) == "باتک قلم"


class TestNormalizeLatin:
    def test_strip_diacritics_true(self) -> None:
        assert (
            normalize_latin("devlet-i ʿaliyye", strip_diacritics=True)
            == "devlet-i aliyye"
        )
        assert (
            normalize_latin("Ḥażret-i Pâdişâh", strip_diacritics=True)
            == "Hazret-i Padişah"
        )

    def test_turkish_letters_preserved_when_stripping_diacritics(self) -> None:
        # ç ğ ı İ ö ş ü Ç Ğ Ö Ş Ü korunur
        assert normalize_latin("şükür ve ğ", strip_diacritics=True) == "şükür ve ğ"
        assert (
            normalize_latin("ÇİĞDEM ÖRDEK ŞÜPHE", strip_diacritics=True)
            == "ÇİĞDEM ÖRDEK ŞÜPHE"
        )

    def test_ayn_and_hamza_preserved_default(self) -> None:
        # strip_diacritics=False iken ʿ ve ʾ korunur
        assert (
            normalize_latin("devlet-i ʿaliyye", strip_diacritics=False)
            == "devlet-i ʿaliyye"
        )
        assert (
            normalize_latin("mesʾele", strip_diacritics=False)
            == "mesʾele"
        )

    def test_ayn_and_hamza_variants_standardized(self) -> None:
        # ʻ (U+02BB) -> ʿ (U+02BF), ʼ (U+02BC) -> ʾ (U+02BE)
        assert (
            normalize_latin("devlet-i \u02bbaliyye", strip_diacritics=False)
            == "devlet-i ʿaliyye"
        )
        assert (
            normalize_latin("mes\u02bcele", strip_diacritics=False)
            == "mesʾele"
        )

    def test_quote_unification(self) -> None:
        # ’ ‘ ` ´ ʹ -> '
        quotes = "’‘`´ʹ"
        assert normalize_latin(quotes) == "' ' ' ' '" or normalize_latin("’kelime‘") == "'kelime'"


class TestTrLower:
    def test_turkish_lowercase_rules(self) -> None:
        assert tr_lower("IŞIK İSTANBUL") == "ışık istanbul"
        assert tr_lower("KİTAP VE SIRÇA") == "kitap ve sırça"
        assert tr_lower("") == ""


class TestArabicRatio:
    def test_pure_arabic(self) -> None:
        assert arabic_ratio("باتک") == 1.0

    def test_pure_latin(self) -> None:
        assert arabic_ratio("devlet") == 0.0

    def test_empty_and_whitespace(self) -> None:
        assert arabic_ratio("") == 0.0
        assert arabic_ratio("   \t\n") == 0.0

    def test_mixed_text(self) -> None:
        # "باتک okumak" -> 4 arapça, 6 latin = 4/10 = 0.4
        ratio = arabic_ratio("باتک okumak")
        assert pytest.approx(ratio, 0.001) == 0.4
