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
        # "\u0643\u062a\u0627\u0628" (kitâb) -> "\u06a9\u062a\u0627\u0628" (kitâb)
        assert normalize_ottoman("\u0643\u062a\u0627\u0628") == "\u06a9\u062a\u0627\u0628"

    def test_letter_unification_yeh(self) -> None:
        # ي (U+064A) -> ی (U+06CC)
        # "\u0639\u0644\u064a" (Ali) -> "\u0639\u0644\u06cc" (Ali)
        assert normalize_ottoman("\u0639\u0644\u064a") == "\u0639\u0644\u06cc"

    def test_letter_unification_alef_maksura(self) -> None:
        # ى (U+0649) -> ی (U+06CC)
        # "\u0645\u0635\u0637\u0641\u0649" (Mustafa) -> "\u0645\u0635\u0637\u0641\u06cc" (Mustafa)
        assert normalize_ottoman("\u0645\u0635\u0637\u0641\u0649") == "\u0645\u0635\u0637\u0641\u06cc"

    def test_letter_unification_urdu_heh(self) -> None:
        # ہ (U+06C1) -> ه (U+0647)
        # "\u06c1\u0645\u0627\u0631\u0647" (hemâre) -> "\u0647\u0645\u0627\u0631\u0647" (hemâre)
        assert normalize_ottoman("\u06c1\u0645\u0627\u0631\u0647") == "\u0647\u0645\u0627\u0631\u0647"

    def test_letter_unification_urdu_heh_with_hamza(self) -> None:
        # ۂ (U+06C2) -> ۀ (U+06C0)
        assert normalize_ottoman("\u06c2") == "\u06c0"

    def test_harakat_removal_default(self) -> None:
        # "\u0643\u0650\u062a\u064e\u0627\u0628\u064c" (kitâbun)
        # keep_harakat=False iken harekeler silinir, ك -> ک olur
        assert (
            normalize_ottoman("\u0643\u0650\u062a\u064e\u0627\u0628\u064c", keep_harakat=False)
            == "\u06a9\u062a\u0627\u0628"
        )

    def test_keep_harakat_preserves_vowels_but_unifies_letters(self) -> None:
        # keep_harakat=True iken harekeler kalır ama ك yine ک olur
        assert (
            normalize_ottoman("\u0643\u0650\u062a\u064e\u0627\u0628\u064c", keep_harakat=True)
            == "\u06a9\u0650\u062a\u064e\u0627\u0628\u064c"
        )

    def test_izafet_hamza_combination(self) -> None:
        # "\u062e\u0627\u0646\u0647\u0654" (hâne-i, ه + U+0654) -> "\u062e\u0627\u0646\u06c0" (خانۀ)
        assert normalize_ottoman("\u062e\u0627\u0646\u0647\u0654") == "\u062e\u0627\u0646\u06c0"

    def test_yeh_with_hamza_combination(self) -> None:
        # "\u06cc\u0654" (ی + U+0654) -> "\u0626" (ئ)
        assert normalize_ottoman("\u06cc\u0654") == "\u0626"
        assert normalize_ottoman("\u064a\u0654") == "\u0626"

    def test_waw_and_alef_with_hamza_combination(self) -> None:
        # و + U+0654 -> ؤ, ا + U+0654 -> أ, ا + U+0655 -> إ
        assert normalize_ottoman("\u0648\u0654") == "\u0624"
        assert normalize_ottoman("\u0627\u0654") == "\u0623"
        assert normalize_ottoman("\u0627\u0655") == "\u0625"

    def test_hamza_with_intermediate_harakat(self) -> None:
        # "\u0647\u064e\u0654" (ه + üstün + hemze)
        # keep_harakat=False iken üstün silinir, ۀ kalır
        assert normalize_ottoman("\u0647\u064e\u0654", keep_harakat=False) == "\u06c0"
        # keep_harakat=True iken üstün korunur
        assert normalize_ottoman("\u0647\u064e\u0654", keep_harakat=True) == "\u06c0\u064e"

    def test_precomposed_hamza_words_preserved(self) -> None:
        # "\u0645\u0633\u0626\u0644\u0647" (mes'ele) keep_harakat=False iken değişmez
        assert (
            normalize_ottoman("\u0645\u0633\u0626\u0644\u0647", keep_harakat=False)
            == "\u0645\u0633\u0626\u0644\u0647"
        )

    def test_harakat_presentation_forms(self) -> None:
        # U+FE70 (fathatan sunum formu) kelimeyi bölmez (boşluk eklenmez)
        # "\u06a9\u062a\u0627\u0628\ufe70 \u0642\u0644\u0645" -> "\u06a9\u062a\u0627\u0628 \u0642\u0644\u0645"
        assert (
            normalize_ottoman("\u06a9\u062a\u0627\u0628\ufe70 \u0642\u0644\u0645")
            == "\u06a9\u062a\u0627\u0628 \u0642\u0644\u0645"
        )
        # keep_harakat=True ile U+FE70 -> U+064B (fathatan) birleşik harekeye çevrilir
        assert (
            normalize_ottoman("\u06a9\u062a\u0627\u0628\ufe70", keep_harakat=True)
            == "\u06a9\u062a\u0627\u0628\u064b"
        )

    def test_zwnj_preserved(self) -> None:
        # U+200C (ZWNJ) korunur: "\u067e\u0627\u062f\u0634\u0627\u0647\u200c\u0644\u0631" (pâdişâh-lar)
        text_with_zwnj = "\u067e\u0627\u062f\u0634\u0627\u0647\u200c\u0644\u0631"
        res = normalize_ottoman(text_with_zwnj)
        assert "\u200c" in res
        assert res == "\u067e\u0627\u062f\u0634\u0627\u0647\u200c\u0644\u0631"

    def test_presentation_forms_letters(self) -> None:
        # Sunum formları (Arabic Presentation Forms B: "\ufedb\ufe98\ufe8e\ufe8f", kitâb) -> "\u06a9\u062a\u0627\u0628"
        presentation_str = "\ufedb\ufe98\ufe8e\ufe8f"
        assert normalize_ottoman(presentation_str) == "\u06a9\u062a\u0627\u0628"

    def test_ascii_digits_conversion(self) -> None:
        # Farsça ve Arap-Hint rakamları 0-9'a çevrilir
        # "\u06f1\u06f9\u06f2\u06f8" (1928) -> "1928"
        assert normalize_ottoman("\u06f1\u06f9\u06f2\u06f8", ascii_digits=True) == "1928"
        # "\u0633\u0646\u0647 \u0661\u0662\u0669\u0663" (sene 1293) -> "\u0633\u0646\u0647 1293"
        assert (
            normalize_ottoman("\u0633\u0646\u0647 \u0661\u0662\u0669\u0663", ascii_digits=True)
            == "\u0633\u0646\u0647 1293"
        )
        # ascii_digits=False iken rakamlar korunur
        assert normalize_ottoman("\u06f1\u06f9\u06f2\u06f8", ascii_digits=False) == "\u06f1\u06f9\u06f2\u06f8"

    def test_protected_ottoman_letters(self) -> None:
        # گ, ڭ, چ, پ, ژ, ؤ, ئ, آ, ة, ۋ, ۀ korunur
        assert normalize_ottoman("\u06af") == "\u06af"  # گ (Gef)
        assert normalize_ottoman("\u06ad") == "\u06ad"  # ڭ (Nef)
        assert normalize_ottoman("\u0686") == "\u0686"  # چ (Tche)
        assert normalize_ottoman("\u067e") == "\u067e"  # پ (Peh)
        assert normalize_ottoman("\u0698") == "\u0698"  # ژ (Zhe)
        assert normalize_ottoman("\u0622\u063a\u0627\u0632") == "\u0622\u063a\u0627\u0632"  # آغاز (âğâz)
        assert normalize_ottoman("\u0645\u0633\u0626\u0648\u0644") == "\u0645\u0633\u0626\u0648\u0644"  # مسئول (mes'ul)
        assert normalize_ottoman("\u0645\u0624\u0644\u0641") == "\u0645\u0624\u0644\u0641"  # مؤلف (müellif)
        assert normalize_ottoman("\u06c0") == "\u06c0"  # ۀ (Heh with Yeh above)

    def test_tatweel_removal(self) -> None:
        # Tatvil / keşide silinir: "\u0643\u062a\u0627\u0628\u0640\u0640\u0640\u0644\u0631" (kitâb---ler)
        assert (
            normalize_ottoman("\u0643\u062a\u0627\u0628\u0640\u0640\u0640\u0644\u0631")
            == "\u06a9\u062a\u0627\u0628\u0644\u0631"
        )

    def test_fold_hamza(self) -> None:
        # fold_hamza=False iken hemzeli elifler korunur
        assert normalize_ottoman("\u0623\u062d\u0645\u062f", fold_hamza=False) == "\u0623\u062d\u0645\u062f"  # أحمد
        assert normalize_ottoman("\u0625\u0633\u0644\u0627\u0645", fold_hamza=False) == "\u0625\u0633\u0644\u0627\u0645"  # إسلام
        # fold_hamza=True iken أ, إ, ٱ -> ا olur
        assert normalize_ottoman("\u0623\u062d\u0645\u062f", fold_hamza=True) == "\u0627\u062d\u0645\u062f"  # احمد
        assert normalize_ottoman("\u0625\u0633\u0644\u0627\u0645", fold_hamza=True) == "\u0627\u0633\u0644\u0627\u0645"  # اسلام
        assert normalize_ottoman("\u0671\u0644\u0644\u0647", fold_hamza=True) == "\u0627\u0644\u0644\u0647"  # الله

    def test_punctuation_mapping(self) -> None:
        # ، -> ,   ؟ -> ?   ؛ -> ;
        # "\u0643\u062a\u0627\u0628\u060c \u0642\u0644\u0645\u061f \u062f\u0641\u062a\u0631\u061b"
        assert (
            normalize_ottoman("\u0643\u062a\u0627\u0628\u060c \u0642\u0644\u0645\u061f \u062f\u0641\u062a\u0631\u061b")
            == "\u06a9\u062a\u0627\u0628, \u0642\u0644\u0645? \u062f\u0641\u062a\u0631;"
        )

    def test_directional_and_zero_width_removal(self) -> None:
        # U+200E, U+200B, U+200D, U+FEFF silinir
        dirty = "\u200e\ufeff\u200b\u0643\u062a\u0627\u0628\u200d"
        assert normalize_ottoman(dirty) == "\u06a9\u062a\u0627\u0628"

    def test_whitespace_collapse(self) -> None:
        # Fazla ve farklı boşluklar tek boşluğa iner, kırpılır
        text = "   \u06a9\u062a\u0627\u0628    \u00a0\t  \u0642\u0644\u0645   "
        assert normalize_ottoman(text) == "\u06a9\u062a\u0627\u0628 \u0642\u0644\u0645"


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
        # Kesin eşleşme testi: 'kelime' 'x' 'y
        assert (
            normalize_latin("\u2019kelime\u2018 `x\u00b4 \u02b9y")
            == "'kelime' 'x' 'y"
        )


class TestTrLower:
    def test_turkish_lowercase_rules(self) -> None:
        assert tr_lower("IŞIK İSTANBUL") == "ışık istanbul"
        assert tr_lower("KİTAP VE SIRÇA") == "kitap ve sırça"
        assert tr_lower("") == ""


class TestArabicRatio:
    def test_pure_arabic(self) -> None:
        assert arabic_ratio("\u06a9\u062a\u0627\u0628") == 1.0

    def test_pure_latin(self) -> None:
        assert arabic_ratio("devlet") == 0.0

    def test_empty_and_whitespace(self) -> None:
        assert arabic_ratio("") == 0.0
        assert arabic_ratio("   \t\n") == 0.0

    def test_mixed_text(self) -> None:
        # "\u06a9\u062a\u0627\u0628 okumak" -> 4 arapça, 6 latin = 4/10 = 0.4
        ratio = arabic_ratio("\u06a9\u062a\u0627\u0628 okumak")
        assert pytest.approx(ratio, 0.001) == 0.4
