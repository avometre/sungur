"""Adversarial stress-test suite for Milestone 1 (Clean & Gazetteer).

Covers:
1. Non-standard Ottoman orthography with mixed Arabic/Persian/Turkish letters (پ, چ, ژ, گ, ڭ, ݣ).
2. Boundary similarity cases where score diff is exactly 0.019, 0.020, 0.021.
3. Robustness against punctuation, empty strings, digits, non-Arabic diacritics, and ZWNJ.
4. Empirical verification of Habsamana vs Hapsamana resolution.
5. Empirical verification of Wiktionary non-Arabic filter on digits and vocabulary words.
"""

import json
from pathlib import Path
import pytest

from src.clean.clean_raw import clean_gazetteer, clean_wiktionary
from src.common.normalize import (
    PROTECTED_OTTOMAN_LETTERS,
    arabic_letter_count,
    normalize_ottoman,
)
from src.translit.skeleton import (
    arabic_skeleton,
    fine_arabic_skeleton,
    fine_latin_skeleton,
    fine_skeleton,
    fine_skeleton_similarity,
    latin_skeleton,
    skeleton_similarity,
)


class TestAdversarialFineSkeletonOrthography:
    """Stress tests for Ottoman orthography with Persian/Turkish consonant distinctions."""

    def test_peh_distinction_across_positions(self) -> None:
        # Initial position: پادشاه (pâdişâh) vs بادشاه (bâdişâh)
        assert fine_skeleton("پادشاه") == "PTX"
        assert fine_skeleton("بادشاه") == "BTX"
        assert arabic_skeleton("پادشاه") == "BTX"  # coarse conflates

        # Medial position: توپراق (toprak) vs توبره (tobre)
        assert fine_skeleton("توپراق") == "TPRK"
        assert fine_skeleton("توبره") == "TBR"  # final heh stripped

        # Final position: الپ (alp) vs الب (alb)
        assert fine_skeleton("الپ") == "LP"
        assert fine_skeleton("الب") == "LB"

    def test_tche_distinction_across_positions(self) -> None:
        # Initial position: چشمه (çeşme) vs جشمه (ceşme)
        assert fine_skeleton("چشمه") == "ÇXM"
        assert fine_skeleton("جشمه") == "CXM"
        assert arabic_skeleton("چشمه") == "CXM"  # coarse conflates

        # Medial position: باغچه (bağçe) vs باغجه (bağce)
        assert fine_skeleton("باغچه") == "BKÇ"
        assert fine_skeleton("باغجه") == "BKC"

        # Final position: آغاج (ağac) vs آغاچ (ağaç)
        assert fine_skeleton("آغاچ") == "KÇ"
        assert fine_skeleton("آغاج") == "KC"

    def test_zhe_and_gef_mapping(self) -> None:
        # ژ (zhe) -> J in both coarse and fine
        assert fine_skeleton("ژاله") == "JL"
        # د (dal) maps to T class
        assert fine_skeleton("مژده") == "MJT"
        assert arabic_skeleton("ژاله") == "JL"

        # گ (gef) -> K in both coarse and fine
        assert fine_skeleton("گوزل") == "KZL"
        assert fine_skeleton("گورمک") == "KRMK"
        assert arabic_skeleton("گوزل") == "KZL"

    def test_nef_and_nef_variant_mapping(self) -> None:
        # ڭ (Ottoman nef U+06AD) -> N
        assert fine_skeleton("صوڭرا") == "SNR"
        assert fine_skeleton("دیڭله") == "TNL"

        # ݣ (Nef variant U+0763) -> N
        assert fine_skeleton("صوݣرا") == "SNR"
        assert fine_skeleton("دیڭله") == "TNL"

    def test_mixed_quad_persian_turkish_letters(self) -> None:
        # Word containing all 4 letters: پ, چ, ژ, گ
        word = "پچژگ"
        assert fine_skeleton(word) == "PÇJK"
        assert arabic_skeleton(word) == "BCJK"

        # Latin matching: 'p'->P, 'ç'->Ç, 'j'->J, 'g'->K
        assert fine_skeleton("peçejeg") == "PÇJK"
        assert fine_skeleton_similarity(word, "peçejeg") == 1.0
        assert fine_skeleton_similarity(word, "beçejeg") == 6 / 8  # B instead of P -> 0.75
        assert fine_skeleton_similarity(word, "pecejeg") == 6 / 8  # C instead of Ç -> 0.75

    def test_final_heh_stripping_rules(self) -> None:
        # Final heh stripped
        assert fine_skeleton("خیره بولی") == "HRBL"
        assert fine_skeleton("خانه") == "HN"
        assert fine_skeleton("خانۀ") == "HN"

        # Internal heh NOT stripped
        assert fine_skeleton("شهر") == "XHR"
        assert fine_skeleton("جهنم") == "CHNM"

        # Teh marbuta (ة) NOT stripped (mapped to T)
        assert fine_skeleton("بلدة") == "BLTT"
        assert fine_skeleton("قوة") == "KT"
        assert fine_skeleton("قوه") == "K"  # heh is stripped

        # Isolated heh word becomes empty
        assert fine_skeleton("ه") == ""
        assert fine_skeleton("ۀ") == ""


class TestAdversarialBoundaryCases:
    """Mathematical stress tests on the <= 0.02 threshold in clean_gazetteer."""

    def test_boundary_diff_0019_triggers_tie_breaker(self) -> None:
        # len = 105. Two mismatches in coarse -> diff = 4 / 210 = 0.0190476... -> round(diff, 4) == 0.0190
        # diff <= 0.02 -> Tie breaker MUST trigger and switch to candidate 2 with higher fine score
        arap = "پ" * 105
        cand1 = "b" * 105
        cand2 = "b" * 102 + "t" * 2 + "p"

        s1 = skeleton_similarity(arap, cand1)
        s2 = skeleton_similarity(arap, cand2)
        diff = round(s1 - s2, 4)
        assert diff == 0.0190

        f1 = fine_skeleton_similarity(arap, cand1)
        f2 = fine_skeleton_similarity(arap, cand2)
        assert f2 > f1

        record = [{"arap_ham": arap, "latin": f"{cand1} / {cand2}"}]
        kept, _, _ = clean_gazetteer(record)
        assert len(kept) == 1
        assert kept[0]["latin"] == cand2, "Tie-breaker should activate for diff=0.0190"

    def test_boundary_diff_0020_triggers_tie_breaker(self) -> None:
        # len = 50. One mismatch in coarse -> diff = 2 / 100 = 0.0200
        # diff == 0.02 -> round(diff, 4) <= 0.02 is True -> Tie-breaker MUST trigger
        arap = "پ" * 50
        cand1 = "b" * 50
        cand2 = "b" * 48 + "t" + "p"

        s1 = skeleton_similarity(arap, cand1)
        s2 = skeleton_similarity(arap, cand2)
        diff = round(s1 - s2, 4)
        assert diff == 0.0200

        f1 = fine_skeleton_similarity(arap, cand1)
        f2 = fine_skeleton_similarity(arap, cand2)
        assert f2 > f1

        record = [{"arap_ham": arap, "latin": f"{cand1} / {cand2}"}]
        kept, _, _ = clean_gazetteer(record)
        assert len(kept) == 1
        assert kept[0]["latin"] == cand2, "Tie-breaker should activate for exact diff=0.0200"

    def test_boundary_diff_0021_does_not_trigger_tie_breaker(self) -> None:
        # len = 95. Two mismatches in coarse -> diff = 4 / 190 = 0.0210526... -> round(diff, 4) == 0.0211
        # diff > 0.02 -> Tie breaker MUST NOT trigger; coarse winner (cand1) must be preserved
        arap = "پ" * 95
        cand1 = "b" * 95
        cand2 = "b" * 92 + "t" * 2 + "p"

        s1 = skeleton_similarity(arap, cand1)
        s2 = skeleton_similarity(arap, cand2)
        diff = round(s1 - s2, 4)
        assert diff == 0.0211
        assert diff > 0.02

        f1 = fine_skeleton_similarity(arap, cand1)
        f2 = fine_skeleton_similarity(arap, cand2)
        assert f2 > f1  # fine would prefer cand2, but coarse gap prevents tie-breaking

        record = [{"arap_ham": arap, "latin": f"{cand1} / {cand2}"}]
        kept, _, _ = clean_gazetteer(record)
        assert len(kept) == 1
        assert kept[0]["latin"] == cand1, "Tie-breaker must NOT activate when diff > 0.02"

    def test_tie_boundary_equal_fine_scores_preserves_first(self) -> None:
        # Both coarse and fine scores are tied -> first candidate strictly preserved
        record = [{"arap_ham": "قلم", "latin": "Kalem / Kalam"}]
        kept, _, _ = clean_gazetteer(record)
        assert len(kept) == 1
        assert kept[0]["latin"] == "Kalem"


class TestAdversarialInputsAndRobustness:
    """Stress tests on empty, numeric, punctuation, diacritics, and malformed inputs."""

    def test_empty_and_whitespace_inputs(self) -> None:
        assert fine_skeleton("") == ""
        assert fine_skeleton("   ") == ""
        assert fine_skeleton("\t\n\r") == ""
        assert fine_skeleton_similarity("", "") == 0.0
        assert fine_skeleton_similarity("   ", "   ") == 0.0
        assert fine_skeleton_similarity("", "kitap") == 0.0
        assert fine_skeleton_similarity("کتاب", "") == 0.0

    def test_numeric_inputs(self) -> None:
        # Western digits
        assert fine_skeleton("1234567890") == ""
        assert fine_skeleton_similarity("12345", "12345") == 0.0

        # Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩)
        assert fine_skeleton("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669") == ""

        # Eastern Arabic-Indic digits (۰۱۲۳۴۵۶۷۸۹)
        assert fine_skeleton("\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9") == ""

        # Mixed letters and digits
        assert fine_skeleton("پادشاه 123") == "PTX"
        assert fine_skeleton("پادشاه ۱۲۳") == "PTX"

    def test_punctuation_and_symbols(self) -> None:
        assert fine_skeleton(".,:;!?\"'()[]{}-–—/") == ""
        assert fine_skeleton("،؛؟٪«»") == ""
        assert fine_skeleton("😀🎉🚀❤️") == ""
        assert fine_skeleton("===+++***###") == ""

    def test_zwnj_handling(self) -> None:
        # ZWNJ (U+200C) in Ottoman word suffixes
        # When written as compound: پادشاه‌لر (with ZWNJ before ler)
        # heh is followed by ZWNJ and ler, so heh is not final and preserved as H
        assert fine_skeleton("پادشاه\u200Cلر") == "PTXHLR"

        # When separated by space: پادشاه لر
        # heh is at end of first word, so stripped
        assert fine_skeleton("پادشاه لر") == "PTXLR"

    def test_vowel_only_strings(self) -> None:
        # All-vowel strings produce empty skeletons and 0.0 similarity
        assert fine_skeleton("اوای") == ""
        assert fine_skeleton("aeiou") == ""
        assert fine_skeleton_similarity("اوای", "aeiou") == 0.0

    def test_casing_and_transcription_diacritics(self) -> None:
        assert fine_skeleton("İPSİLİ") == "PSL"
        assert fine_skeleton("ipsili") == "PSL"
        assert fine_skeleton("ÇEŞME") == "ÇXM"
        assert fine_skeleton("çeşme") == "ÇXM"
        assert fine_skeleton("PÂDİŞÂH") == "PTXH"
        assert fine_skeleton("pâdişâh") == "PTXH"
        assert fine_skeleton("ḥatt-ı hümâyûn") == "HTTHMN"


class TestEmpiricalHabsamanaResolution:
    """Strict verification that Habsamana vs Hapsamana resolves to Hapsamana."""

    def test_synthetic_habsamana_both_orders(self) -> None:
        arap = "حاپسمانه"
        # Order 1: Habsamana / Hapsamana
        rec1 = [{"id": "t1", "arap_ham": arap, "latin": "Habsamana / Hapsamana"}]
        kept1, _, _ = clean_gazetteer(rec1)
        assert len(kept1) == 1
        assert kept1[0]["latin"] == "Hapsamana"

        # Order 2: Hapsamana / Habsamana
        rec2 = [{"id": "t2", "arap_ham": arap, "latin": "Hapsamana / Habsamana"}]
        kept2, _, _ = clean_gazetteer(rec2)
        assert len(kept2) == 1
        assert kept2[0]["latin"] == "Hapsamana"

    def test_real_dataset_gaz_7432(self) -> None:
        raw_path = Path("data/raw/gazetteer.jsonl")
        if not raw_path.exists():
            pytest.skip("data/raw/gazetteer.jsonl not present")

        records = [json.loads(line) for line in open(raw_path, encoding="utf-8")]
        target = [r for r in records if r.get("id") == "gaz:7432"]
        assert len(target) == 1, "gaz:7432 must exist in raw gazetteer"

        kept, _, _ = clean_gazetteer(target)
        assert len(kept) == 1
        assert kept[0]["latin"] == "Hapsamana"
        assert kept[0]["eslesme_skoru"] == 1.0
        assert kept[0]["guvenilir"] is True

    def test_real_dataset_gaz_15280(self) -> None:
        raw_path = Path("data/raw/gazetteer.jsonl")
        if not raw_path.exists():
            pytest.skip("data/raw/gazetteer.jsonl not present")

        records = [json.loads(line) for line in open(raw_path, encoding="utf-8")]
        target = [r for r in records if r.get("id") == "gaz:15280"]
        assert len(target) == 1

        kept, _, _ = clean_gazetteer(target)
        assert len(kept) == 1
        # اولوبک مع حپسمانه has peh in حپسمانه, candidate Ulubey maa Hapsamana has 'p'
        assert kept[0]["latin"] == "Ulubey maa Hapsamana"


class TestEmpiricalWiktionaryNonArabicFilter:
    """Strict verification of Wiktionary non-Arabic filter."""

    def test_ten_digits_0_to_9_all_rejected(self) -> None:
        raw_digits = [
            ("۰", "sıfr"), ("۱", "bir"), ("۲", "iki"), ("۳", "üç"),
            ("٤", "dört"), ("٥", "beş"), ("٦", "altı"),
            ("۷", "yedi"), ("۸", "sekiz"), ("۹", "dokuz"),
        ]
        records = [
            {"id": f"w:d:{i}", "arap_ham": char, "latin": lat, "tur": "num"}
            for i, (char, lat) in enumerate(raw_digits)
        ]
        kept, excluded, stats = clean_wiktionary(records)
        assert len(kept) == 0
        assert len(excluded) == 10
        assert stats["nedenler"]["arap_harfi_yok"] == 10
        assert all(ex["neden"] == "arap_harfi_yok" for ex in excluded)

    def test_all_protected_ottoman_letters_preserved(self) -> None:
        # Every protected Ottoman letter must have arabic_letter_count > 0 and not be rejected
        records = [
            {"id": f"w:prot:{i}", "arap_ham": ch, "latin": f"let_{i}", "tur": "noun"}
            for i, ch in enumerate(sorted(PROTECTED_OTTOMAN_LETTERS))
        ]
        kept, excluded, stats = clean_wiktionary(records)
        assert len(kept) == len(PROTECTED_OTTOMAN_LETTERS)
        assert len(excluded) == 0
        assert stats["nedenler"].get("arap_harfi_yok", 0) == 0

    def test_vowel_only_words_preserved(self) -> None:
        records = [
            {"id": "w:aya", "arap_ham": "آیا", "latin": "aya", "tur": "noun"},
            {"id": "w:iki", "arap_ham": "ایکی", "latin": "iki", "tur": "num"},
            {"id": "w:on", "arap_ham": "اون", "latin": "on", "tur": "num"},
        ]
        kept, excluded, stats = clean_wiktionary(records)
        assert len(kept) == 3
        assert len(excluded) == 0

    def test_real_wiktionary_dataset_exact_exclusions(self) -> None:
        raw_path = Path("data/raw/wiktionary_ota.jsonl")
        if not raw_path.exists():
            pytest.skip("data/raw/wiktionary_ota.jsonl not present")

        records = [json.loads(line) for line in open(raw_path, encoding="utf-8")]
        kept, excluded, stats = clean_wiktionary(records)

        # 1. Exactly 10 records rejected for arap_harfi_yok
        excluded_non_arabic = [r for r in excluded if r.get("neden") == "arap_harfi_yok"]
        assert len(excluded_non_arabic) == 10

        # 2. All 10 are digits 0-9
        for r in excluded_non_arabic:
            ham = r.get("arap_ham", "")
            assert arabic_letter_count(ham) == 0
            assert r.get("tur") == "num"

        # 3. Exactly 9 records rejected for latin_yok
        excluded_no_latin = [r for r in excluded if r.get("neden") == "latin_yok"]
        assert len(excluded_no_latin) == 9
        for r in excluded_no_latin:
            assert not r.get("latin")

        # 4. Total excluded is exactly 19
        assert len(excluded) == 19

        # 5. Every kept record has at least 1 Arabic letter
        assert all(arabic_letter_count(r["arap"]) > 0 for r in kept)
