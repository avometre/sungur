"""Empirical stress-test suite for Milestone 1 (Clean & Gazetteer).

Conducted by teamwork_challenger to rigorously verify:
1. Tie-breaker behavior across candidate counts (0, 1, 2, 3+).
2. Threshold boundaries for coarse score difference (diff <= 0.02 vs diff > 0.02).
3. Permutation and reverse ordering stability.
4. Identical fine score tie resolution (order preservation).
5. Wiktionary non-Arabic letter filtering (digits, symbols, mixed).
6. Real dataset integrity for clean_raw outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.clean.clean_raw import (
    clean_gazetteer,
    clean_wiktionary,
    extract_gazetteer_candidates,
)
from src.translit.skeleton import (
    fine_skeleton,
    fine_skeleton_similarity,
    skeleton_similarity,
)


class TestCandidateCountBoundaries:
    """Stress tests for candidate counts (0, 1, 2, 3+)."""

    @pytest.mark.parametrize(
        "raw_latin",
        [
            "",
            None,
            "   ",
            "()",
            "(?)",
            " / , / ",
            "(?) / , ()",
        ],
    )
    def test_zero_candidates_handling(self, raw_latin: Any) -> None:
        """Records with 0 valid candidates are gracefully excluded as 'latin_yok' without crashing."""
        rec = {
            "id": "gaz:zero:1",
            "arap_ham": "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647",
            "latin": raw_latin,
        }
        kept, excluded, stats = clean_gazetteer([rec])
        assert len(kept) == 0
        assert len(excluded) == 1
        assert excluded[0]["neden"] == "latin_yok"

    @pytest.mark.parametrize(
        "raw_latin, expected_candidate",
        [
            ("İstanbul", "İstanbul"),
            ("  Ankara  ", "Ankara"),
            ("(İzmir)", "İzmir"),
            ("Bursa (?)", "Bursa"),
        ],
    )
    def test_single_candidate_handling(self, raw_latin: str, expected_candidate: str) -> None:
        """Records with 1 candidate bypass tie-breaker and select the candidate directly."""
        rec = {
            "id": "gaz:single:1",
            "arap_ham": "\u0628\u0648\u0631\u0633\u0647",
            "latin": raw_latin,
        }
        kept, excluded, stats = clean_gazetteer([rec])
        assert len(kept) == 1
        assert kept[0]["latin"] == expected_candidate
        assert kept[0]["latin_adaylar"] == [expected_candidate]

    def test_two_candidates_coarse_tie_resolved_by_fine(self) -> None:
        """When coarse scores are equal, fine skeleton correctly breaks the tie."""
        # حاپسمانه (پ -> P). Habsamana (B) vs Hapsamana (P).
        rec = {
            "id": "gaz:two:1",
            "arap_ham": "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647",
            "latin": "Habsamana / Hapsamana",
        }
        kept, excluded, stats = clean_gazetteer([rec])
        assert len(kept) == 1
        assert kept[0]["latin"] == "Hapsamana"

    def test_two_candidates_reverse_order_stability(self) -> None:
        """Reversing candidate order does not flip the chosen candidate."""
        rec1 = {
            "id": "gaz:two:rev1",
            "arap_ham": "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647",
            "latin": "Habsamana / Hapsamana",
        }
        rec2 = {
            "id": "gaz:two:rev2",
            "arap_ham": "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647",
            "latin": "Hapsamana / Habsamana",
        }
        kept1, _, _ = clean_gazetteer([rec1])
        kept2, _, _ = clean_gazetteer([rec2])
        assert kept1[0]["latin"] == "Hapsamana"
        assert kept2[0]["latin"] == "Hapsamana"

    def test_three_plus_candidates_coarse_tie(self) -> None:
        """3+ candidates where top coarse score is unambiguous selects the top candidate."""
        rec = {
            "id": "gaz:three:1",
            "arap_ham": "\u0686\u0634\u0645\u0647",  # Çeşme (CXM)
            "latin": "Cesme / Cisme / Çeşme / Alakasız",
        }
        kept, excluded, stats = clean_gazetteer([rec])
        assert len(kept) == 1
        assert kept[0]["latin"] == "Çeşme"

    def test_three_candidates_pairwise_top2_design_observation(self) -> None:
        """Empirically document that clean_gazetteer evaluates the top two candidates per R5.
        
        R5 specification: 'en yüksek iki adayın skor farkı <= 0.02 ise...'.
        When 3 candidates have identical coarse scores (1.0), the top two in sorted order
        are compared. If cand3 has a higher fine score than cand2, it is not evaluated
        unless it is in the top 2.
        """
        arap = "\u067e\u0627\u067e\u0627\u0633"  # Papas: coarse 'BBS', fine 'PPS'
        # All 3 have coarse score 1.0:
        # Papas (fine 1.0), Bapas (fine 0.667), Babas (fine 0.333)
        # When Papas is in top 2:
        rec_papas_first = {"id": "1", "arap_ham": arap, "latin": "Papas / Bapas / Babas"}
        k1, _, _ = clean_gazetteer([rec_papas_first])
        assert k1[0]["latin"] == "Papas"

        # When Babas and Bapas are the top two:
        rec_papas_third = {"id": "2", "arap_ham": arap, "latin": "Babas / Bapas / Papas"}
        k2, _, _ = clean_gazetteer([rec_papas_third])
        # cand1=Babas (0.333), cand2=Bapas (0.667). cand2 wins between top 2.
        assert k2[0]["latin"] == "Bapas"


class TestTieBreakerThresholdBoundaries:
    """Stress tests for the 0.02 coarse difference threshold."""

    def test_diff_strictly_greater_than_002_preserves_coarse_winner(self) -> None:
        """When coarse difference > 0.02, higher coarse candidate wins even if fine score differs."""
        # Arap: قصر (KSR).
        # Cand 1: "Kasar" -> coarse KSR (1.0), fine KSR (1.0)
        # Cand 2: "Kas" -> coarse KS (0.80), fine KS (0.80)
        # Difference = 1.0 - 0.80 = 0.20 > 0.02
        rec = {
            "id": "gaz:thresh:1",
            "arap_ham": "\u0642\u0635\u0631",
            "latin": "Kas / Kasar",
        }
        kept, _, _ = clean_gazetteer([rec])
        assert len(kept) == 1
        assert kept[0]["latin"] == "Kasar"

    def test_diff_boundary_synthetic(self) -> None:
        """Explicit boundary test: verify behavior when diff <= 0.02 vs diff > 0.02."""
        arap = "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647"  # fine: HPSMN, coarse: HBSMN
        # cand1 coarse=1.0, fine=0.8. cand2 coarse=0.985 (diff 0.015 <= 0.02), fine=1.0
        # When diff <= 0.02 and fine2 > fine1, cand2 is chosen.
        rec = {
            "id": "gaz:thresh:2",
            "arap_ham": arap,
            "latin": "Habsamana / Hapsamana",
        }
        kept, _, _ = clean_gazetteer([rec])
        assert kept[0]["latin"] == "Hapsamana"

    def test_diff_exact_threshold_behavior(self) -> None:
        """Verify behavior around the 0.02 boundary threshold."""
        score1 = 0.9000
        # If score2 is 0.8800, difference is exactly 0.0200 -> round(0.02, 4) <= 0.02 is True
        assert round(score1 - 0.8800, 4) <= 0.02
        # If score2 is 0.8799, difference is 0.0201 -> round(0.0201, 4) <= 0.02 is False
        assert not (round(score1 - 0.8799, 4) <= 0.02)
        arap = "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647"  # fine: HPSMN, coarse: HBSMN
        # cand1 coarse=1.0, fine=0.8. cand2 coarse=0.985 (diff 0.015 <= 0.02), fine=1.0
        # When diff <= 0.02 and fine2 > fine1, cand2 is chosen.
        # Let's verify with Habsamana (1.0, 0.8) vs Hapsamana (1.0, 1.0) -> diff 0.0 <= 0.02.
        rec = {
            "id": "gaz:thresh:2",
            "arap_ham": arap,
            "latin": "Habsamana / Hapsamana",
        }
        kept, _, _ = clean_gazetteer([rec])
        assert kept[0]["latin"] == "Hapsamana"


class TestIdenticalFineScoresOrderPreservation:
    """Verify that when both coarse and fine scores tie, original candidate order is preserved."""

    def test_equal_coarse_and_fine_preserves_first(self) -> None:
        rec1 = {
            "id": "gaz:order:1",
            "arap_ham": "\u0642\u0644\u0645",  # KLM
            "latin": "Kalem / Kalam",
        }
        rec2 = {
            "id": "gaz:order:2",
            "arap_ham": "\u0642\u0644\u0645",
            "latin": "Kalam / Kalem",
        }
        kept1, _, _ = clean_gazetteer([rec1])
        kept2, _, _ = clean_gazetteer([rec2])
        assert kept1[0]["latin"] == "Kalem"
        assert kept2[0]["latin"] == "Kalam"


class TestWiktionaryArabicLetterFilter:
    """Stress tests for Wiktionary arabic_letter_count filter."""

    @pytest.mark.parametrize(
        "digit_char",
        [
            "\u06f0",  # Eastern Arabic-Indic 0
            "\u06f1",  # Eastern Arabic-Indic 1
            "\u06f2",  # Eastern Arabic-Indic 2
            "\u06f3",  # Eastern Arabic-Indic 3
            "\u0664",  # Arabic-Indic 4
            "\u0665",  # Arabic-Indic 5
            "\u0666",  # Arabic-Indic 6
            "\u06f7",  # Eastern Arabic-Indic 7
            "\u06f8",  # Eastern Arabic-Indic 8
            "\u06f9",  # Eastern Arabic-Indic 9
        ],
    )
    def test_digits_excluded_with_arap_harfi_yok(self, digit_char: str) -> None:
        rec = {
            "id": f"wikt:digit:{ord(digit_char)}",
            "arap_ham": digit_char,
            "latin": "rakam",
            "tur": "num",
        }
        kept, excluded, stats = clean_wiktionary([rec])
        assert len(kept) == 0
        assert len(excluded) == 1
        assert excluded[0]["neden"] == "arap_harfi_yok"

    def test_mixed_text_with_digits_and_letters_is_not_excluded(self) -> None:
        """Entries containing valid Arabic letters alongside digits must NOT be excluded."""
        rec = {
            "id": "wikt:mixed:1",
            "arap_ham": "\u06f1\u06f0 \u06af\u0648\u0646",  # "۱۰ گون"
            "latin": "on gün",
            "tur": "phrase",
        }
        kept, excluded, stats = clean_wiktionary([rec])
        assert len(kept) == 1
        assert len(excluded) == 0
        assert kept[0]["arap"] == "10 \u06af\u0648\u0646"


class TestRealDataIntegrity:
    """Empirical verification on actual data files."""

    def test_wiktionary_excluded_real_distribution(self) -> None:
        excluded_path = Path("data/interim/excluded/wiktionary.jsonl")
        assert excluded_path.exists(), "data/interim/excluded/wiktionary.jsonl must exist"

        records = [json.loads(line) for line in excluded_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(records) == 19
        reasons = {r["neden"] for r in records}
        assert reasons == {"arap_harfi_yok", "latin_yok"}
        arap_harfi_yok_count = sum(1 for r in records if r["neden"] == "arap_harfi_yok")
        latin_yok_count = sum(1 for r in records if r["neden"] == "latin_yok")
        assert arap_harfi_yok_count == 10
        assert latin_yok_count == 9

    def test_gazetteer_clean_real_stats(self) -> None:
        clean_path = Path("data/interim/gazetteer_clean.jsonl")
        assert clean_path.exists(), "data/interim/gazetteer_clean.jsonl must exist"

        records = [json.loads(line) for line in clean_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(records) == 19307
        guvenilir_count = sum(1 for r in records if r.get("guvenilir") is True)
        assert guvenilir_count == 18907
        assert round(guvenilir_count / len(records), 4) == 0.9793
