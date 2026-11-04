"""Empirical Challenger Test Suite for Milestone 2 (Scraper & Alignment v2).

Conducted by challenger_m2_2 to rigorously challenge and verify:
1. Candidate evaluation and scoring logic (`evaluate_candidate_alignment` and `process_gold_candidates`).
2. Tie-breaking when candidates have identical coverage (kapsama): selects highest average score (ort_skor).
3. Boundary threshold edge cases:
   - kapsama == 0.399 (rejected) vs kapsama == 0.400 (accepted if ort_skor >= 0.70)
   - ort_skor == 0.699 (rejected) vs ort_skor == 0.700 (accepted)
4. Full schema, row count, and data integrity verification for:
   - data/interim/ws_search_log.jsonl (259 rows, 7 fields, strict acceptance criteria)
   - data/interim/ws_page_links.jsonl (33 rows, 6 fields, valid methods, recovery of 85999 & 155879)
   - data/interim/ws_parallel.jsonl (1424 rows, 12 fields, reliable flag rule skor >= 0.55)
5. Adversarial input cases: empty lines, malformed titles, and graceful fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from src.scrapers.vikikaynak_align import (
    align_lines,
    clean_latin_html,
    evaluate_candidate_alignment,
    fetch_latin_page_html,
    process_gold_candidates,
    search_candidate_titles,
    split_sentences,
)


# ============================================================================
# 1. CANDIDATE EVALUATION & SCORING LOGIC TESTS
# ============================================================================


class TestCandidateEvaluationScoring:
    """Rigorous verification of `evaluate_candidate_alignment`."""

    def test_empty_or_whitespace_input_returns_zero(self) -> None:
        """Empty, whitespace-only or unparseable input returns (0.0, 0.0, [])."""
        assert evaluate_candidate_alignment([], []) == (0.0, 0.0, [])
        assert evaluate_candidate_alignment(["   "], ["\n\t"]) == (0.0, 0.0, [])
        assert evaluate_candidate_alignment(["كتاب"], []) == (0.0, 0.0, [])
        assert evaluate_candidate_alignment([], ["kitap"]) == (0.0, 0.0, [])

    def test_zero_reliable_pairs_returns_zero_metrics_but_preserves_pairs(self) -> None:
        """When all aligned pairs score below 0.55, kapsama and ort_skor must be 0.0."""
        # Unrelated texts will align with very low similarity
        arap = ["كتاب پادشاه دولت عليّه"]
        latin = ["xyz wvu tsr qpo"]
        kapsama, ort_skor, pairs = evaluate_candidate_alignment(arap, latin)
        assert len(pairs) >= 1
        assert all(p["skor"] < 0.55 for p in pairs)
        assert kapsama == 0.0
        assert ort_skor == 0.0

    def test_perfect_alignment_returns_unit_metrics(self) -> None:
        """Identical or perfectly transliterated lines give kapsama=1.0 and ort_skor >= 0.95."""
        arap = ["كتاب", "دولت عليه", "پادشاه"]
        latin = ["kitap", "devlet-i aliyye", "padişah"]
        kapsama, ort_skor, pairs = evaluate_candidate_alignment(arap, latin)
        assert len(pairs) == 3
        assert kapsama == 1.0
        assert ort_skor >= 0.90

    def test_alignment_2_to_1_counts_both_arabic_lines_for_coverage(self) -> None:
        """In a 2-1 alignment, the single reliable pair covers 2 Arabic lines."""
        # Two Arabic lines merged into one Latin line
        arap = ["استانبول", "پادشاه"]
        latin = ["İstanbul padişah"]
        kapsama, ort_skor, pairs = evaluate_candidate_alignment(arap, latin)
        assert len(pairs) == 1
        assert pairs[0]["hizalama"] == "2-1"
        assert pairs[0]["skor"] >= 0.55
        # kapsanan_arap should be 2 out of 2 lines -> 1.0
        assert kapsama == 1.0

    def test_alignment_1_to_2_counts_one_arabic_line_for_coverage(self) -> None:
        """In a 1-2 alignment, the single reliable pair covers 1 Arabic line."""
        arap = ["كتاب پادشاه"]
        latin = ["kitap", "padişah"]
        kapsama, ort_skor, pairs = evaluate_candidate_alignment(arap, latin)
        assert len(pairs) == 1
        assert pairs[0]["hizalama"] == "1-2"
        assert pairs[0]["skor"] >= 0.55
        # kapsanan_arap is 1 out of 1 -> 1.0
        assert kapsama == 1.0

    def test_partial_coverage_mathematical_precision(self) -> None:
        """Mathematical precision of kapsama and ort_skor with partial matching."""
        # 4 Arabic lines: 2 matching, 2 non-matching gibberish
        arap = ["كتاب", "پادشاه", "غغغغغغغغغ", "ظظظظظظظظظ"]
        latin = ["kitap", "padişah", "abcdef", "ghijkl"]
        kapsama, ort_skor, pairs = evaluate_candidate_alignment(arap, latin)
        guvenilir = [p for p in pairs if p["skor"] >= 0.55]
        assert len(guvenilir) == 2
        # Kapsama = 2 / 4 = 0.5
        assert kapsama == 0.5
        expected_avg = round(sum(p["skor"] for p in guvenilir) / 2, 4)
        assert ort_skor == expected_avg


# ============================================================================
# 2. TIE-BREAKING & THRESHOLD BOUNDARY TESTS
# ============================================================================


class TestCandidateTieBreakingAndThresholds:
    """Rigorous tests for tie-breaking and threshold edge cases in candidate selection."""

    def test_tie_breaking_equal_kapsama_picks_highest_ort_skor(self, tmp_path: Path) -> None:
        """When two candidates have the same kapsama, it MUST pick the candidate with highest ort_skor."""
        csv_file = tmp_path / "title_map.csv"
        csv_file.write_text(
            "pageid,baslik_arap,tr_baslik,not\n"
            "501,شعر,Aday Dusuk Skor|Aday Yuksek Skor,claude_aday\n",
            encoding="utf-8",
        )
        search_log_file = tmp_path / "ws_search_log.jsonl"
        pages_lines = {501: [{"arap": "شعر بيرنجي"}]}

        with mock.patch("src.scrapers.vikikaynak_align.search_candidate_titles") as mock_search, \
             mock.patch("src.scrapers.vikikaynak_align.fetch_latin_page_html") as mock_fetch, \
             mock.patch("src.scrapers.vikikaynak_align.clean_latin_html") as mock_clean, \
             mock.patch("src.scrapers.vikikaynak_align.evaluate_candidate_alignment") as mock_eval:

            mock_search.side_effect = lambda c, **kw: [c]
            mock_fetch.return_value = "<p>text</p>"
            mock_clean.return_value = ["text"]

            # Both candidates have identical kapsama (0.8500), but candidate 2 has higher ort_skor (0.9200 vs 0.7400)
            mock_eval.side_effect = [
                (0.8500, 0.7400, []),  # Aday Dusuk Skor
                (0.8500, 0.9200, []),  # Aday Yuksek Skor
            ]

            accepted, logs = process_gold_candidates(
                gold_csv_path=csv_file,
                pages_lines=pages_lines,
                cache_dir=tmp_path,
                search_log_path=search_log_file,
            )

            assert 501 in accepted
            assert accepted[501]["tr_baslik"] == "Aday Yuksek Skor"
            assert accepted[501]["kapsama"] == 0.8500
            assert accepted[501]["ort_skor"] == 0.9200

    def test_tie_breaking_reverse_order_stability(self, tmp_path: Path) -> None:
        """Regardless of evaluation order, highest ort_skor is chosen when kapsama is equal."""
        csv_file = tmp_path / "title_map.csv"
        csv_file.write_text(
            "pageid,baslik_arap,tr_baslik,not\n"
            "502,شعر,Aday Yuksek Skor|Aday Dusuk Skor,claude_aday\n",
            encoding="utf-8",
        )
        search_log_file = tmp_path / "ws_search_log.jsonl"
        pages_lines = {502: [{"arap": "شعر بيرنجي"}]}

        with mock.patch("src.scrapers.vikikaynak_align.search_candidate_titles") as mock_search, \
             mock.patch("src.scrapers.vikikaynak_align.fetch_latin_page_html") as mock_fetch, \
             mock.patch("src.scrapers.vikikaynak_align.clean_latin_html") as mock_clean, \
             mock.patch("src.scrapers.vikikaynak_align.evaluate_candidate_alignment") as mock_eval:

            mock_search.side_effect = lambda c, **kw: [c]
            mock_fetch.return_value = "<p>text</p>"
            mock_clean.return_value = ["text"]

            # Reverse evaluation order
            mock_eval.side_effect = [
                (0.8500, 0.9200, []),  # Aday Yuksek Skor evaluated first
                (0.8500, 0.7400, []),  # Aday Dusuk Skor evaluated second
            ]

            accepted, logs = process_gold_candidates(
                gold_csv_path=csv_file,
                pages_lines=pages_lines,
                cache_dir=tmp_path,
                search_log_path=search_log_file,
            )

            assert 502 in accepted
            assert accepted[502]["tr_baslik"] == "Aday Yuksek Skor"
            assert accepted[502]["ort_skor"] == 0.9200

    def test_three_way_tie_and_kapsama_precedence(self, tmp_path: Path) -> None:
        """Higher kapsama always takes precedence over higher ort_skor, tie-break only applies on equal kapsama."""
        csv_file = tmp_path / "title_map.csv"
        csv_file.write_text(
            "pageid,baslik_arap,tr_baslik,not\n"
            "503,شعر,Cand A|Cand B|Cand C,claude_aday\n",
            encoding="utf-8",
        )
        search_log_file = tmp_path / "ws_search_log.jsonl"
        pages_lines = {503: [{"arap": "شعر"}]}

        with mock.patch("src.scrapers.vikikaynak_align.search_candidate_titles") as mock_search, \
             mock.patch("src.scrapers.vikikaynak_align.fetch_latin_page_html") as mock_fetch, \
             mock.patch("src.scrapers.vikikaynak_align.clean_latin_html") as mock_clean, \
             mock.patch("src.scrapers.vikikaynak_align.evaluate_candidate_alignment") as mock_eval:

            mock_search.side_effect = lambda c, **kw: [c]
            mock_fetch.return_value = "<p>text</p>"
            mock_clean.return_value = ["text"]

            # Cand A: (0.75, 0.99) -> very high score, lower coverage
            # Cand B: (0.90, 0.72) -> higher coverage, lower score
            # Cand C: (0.90, 0.88) -> same coverage as B, higher score than B
            mock_eval.side_effect = [
                (0.7500, 0.9900, []),
                (0.9000, 0.7200, []),
                (0.9000, 0.8800, []),
            ]

            accepted, logs = process_gold_candidates(
                gold_csv_path=csv_file,
                pages_lines=pages_lines,
                cache_dir=tmp_path,
                search_log_path=search_log_file,
            )

            assert 503 in accepted
            assert accepted[503]["tr_baslik"] == "Cand C"
            assert accepted[503]["kapsama"] == 0.9000
            assert accepted[503]["ort_skor"] == 0.8800

    @pytest.mark.parametrize(
        "kapsama, ort_skor, expected_accepted",
        [
            # kapsama == 0.399 (rejected even with high ort_skor)
            (0.399, 0.950, False),
            (0.3999, 1.000, False),
            # kapsama == 0.400, ort_skor >= 0.70 (accepted)
            (0.400, 0.700, True),
            (0.400, 0.850, True),
            # ort_skor == 0.699 (rejected even with high kapsama)
            (0.850, 0.699, False),
            (1.000, 0.6999, False),
            (0.400, 0.699, False),
            # Both below threshold
            (0.350, 0.650, False),
        ],
    )
    def test_threshold_edge_cases(
        self, tmp_path: Path, kapsama: float, ort_skor: float, expected_accepted: bool
    ) -> None:
        """Verify boundary thresholds: kapsama >= 0.40 and ort_skor >= 0.70 strictly enforced."""
        csv_file = tmp_path / "title_map.csv"
        csv_file.write_text(
            f"pageid,baslik_arap,tr_baslik,not\n"
            f"601,حد,Boundary Test Title,claude_aday\n",
            encoding="utf-8",
        )
        search_log_file = tmp_path / "ws_search_log.jsonl"
        pages_lines = {601: [{"arap": "حد"}]}

        with mock.patch("src.scrapers.vikikaynak_align.search_candidate_titles", return_value=["Boundary Test Title"]), \
             mock.patch("src.scrapers.vikikaynak_align.fetch_latin_page_html", return_value="<p>Metin</p>"), \
             mock.patch("src.scrapers.vikikaynak_align.clean_latin_html", return_value=["Metin"]), \
             mock.patch("src.scrapers.vikikaynak_align.evaluate_candidate_alignment", return_value=(kapsama, ort_skor, [])):

            accepted, logs = process_gold_candidates(
                gold_csv_path=csv_file,
                pages_lines=pages_lines,
                cache_dir=tmp_path,
                search_log_path=search_log_file,
            )

            assert len(logs) == 1
            assert logs[0]["kabul"] is expected_accepted
            if expected_accepted:
                assert 601 in accepted
                assert accepted[601]["kapsama"] == kapsama
                assert accepted[601]["ort_skor"] == ort_skor
            else:
                assert 601 not in accepted


# ============================================================================
# 3. INTERIM DATASETS INTEGRITY & SCHEMA CHALLENGES
# ============================================================================


class TestInterimWsSearchLogIntegrity:
    """Direct empirical validation of data/interim/ws_search_log.jsonl."""

    SEARCH_LOG_PATH = Path("data/interim/ws_search_log.jsonl")

    def test_search_log_file_exists(self) -> None:
        assert self.SEARCH_LOG_PATH.exists(), f"{self.SEARCH_LOG_PATH} does not exist"

    def test_search_log_exact_row_count(self) -> None:
        lines = self.SEARCH_LOG_PATH.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 259, f"Expected 259 rows, got {len(lines)}"

    def test_search_log_schema_and_types(self) -> None:
        required_keys = {
            "pageid",
            "baslik_arap",
            "aday",
            "bulunan_baslik",
            "kapsama",
            "ort_skor",
            "kabul",
        }
        for idx, line in enumerate(self.SEARCH_LOG_PATH.read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            assert required_keys.issubset(record.keys()), f"Row {idx} missing fields"
            assert isinstance(record["pageid"], int), f"Row {idx} pageid must be int"
            assert isinstance(record["baslik_arap"], str), f"Row {idx} baslik_arap must be str"
            assert isinstance(record["aday"], str), f"Row {idx} aday must be str"
            assert isinstance(record["bulunan_baslik"], str), f"Row {idx} bulunan_baslik must be str"
            assert isinstance(record["kapsama"], (int, float)), f"Row {idx} kapsama must be numeric"
            assert isinstance(record["ort_skor"], (int, float)), f"Row {idx} ort_skor must be numeric"
            assert isinstance(record["kabul"], bool), f"Row {idx} kabul must be bool"

    def test_search_log_no_acceptance_violations(self) -> None:
        """Verify whether ANY candidate marked kabul=True violated kapsama >= 0.40 and ort_skor >= 0.70,
        or whether any candidate marked kabul=False qualified for acceptance."""
        violations = []
        for idx, line in enumerate(self.SEARCH_LOG_PATH.read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            kapsama = record["kapsama"]
            ort_skor = record["ort_skor"]
            kabul = record["kabul"]

            expected_kabul = bool(kapsama >= 0.40 and ort_skor >= 0.70)
            if kabul != expected_kabul:
                violations.append((idx, record, expected_kabul))

        assert len(violations) == 0, f"Found {len(violations)} threshold violations: {violations}"


class TestInterimWsPageLinksIntegrity:
    """Direct empirical validation of data/interim/ws_page_links.jsonl."""

    PAGE_LINKS_PATH = Path("data/interim/ws_page_links.jsonl")

    def test_page_links_file_exists(self) -> None:
        assert self.PAGE_LINKS_PATH.exists(), f"{self.PAGE_LINKS_PATH} does not exist"

    def test_page_links_exact_count_and_methods(self) -> None:
        records = [
            json.loads(line)
            for line in self.PAGE_LINKS_PATH.read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 33, f"Expected 33 accepted pages, got {len(records)}"

        method_counts = {}
        for r in records:
            m = r["yontem"]
            method_counts[m] = method_counts.get(m, 0) + 1

        assert method_counts.get("langlinks") == 11, f"Expected 11 langlinks, got {method_counts.get('langlinks')}"
        assert method_counts.get("arama") == 22, f"Expected 22 arama, got {method_counts.get('arama')}"

    def test_page_links_schema_and_scores(self) -> None:
        required_keys = {"pageid", "baslik_arap", "tr_baslik", "yontem", "kapsama", "ort_skor"}
        for idx, line in enumerate(self.PAGE_LINKS_PATH.read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            assert required_keys.issubset(r.keys()), f"Row {idx} missing keys"
            assert isinstance(r["pageid"], int) and r["pageid"] > 0
            assert len(r["baslik_arap"].strip()) > 0
            assert len(r["tr_baslik"].strip()) > 0
            assert r["yontem"] in {"langlinks", "arama"}
            assert r["kapsama"] >= 0.40, f"Row {idx} kapsama < 0.40: {r}"
            assert r["ort_skor"] >= 0.70, f"Row {idx} ort_skor < 0.70: {r}"

    def test_recovered_pages_presence_and_attributes(self) -> None:
        """Confirm recovered pages:
        - pageid=85999 ("Sen Olmasan" -> Rübâb-ı Şikeste/Sen Olmasan)
        - pageid=155879 ("Gazel" -> Bâğ-ı Dehrin Hem Hazânın Hem Bahârın Görmüşüz)
        """
        records_by_id = {
            json.loads(line)["pageid"]: json.loads(line)
            for line in self.PAGE_LINKS_PATH.read_text(encoding="utf-8").splitlines()
        }

        # Check pageid 85999
        assert 85999 in records_by_id, "pageid=85999 missing from ws_page_links.jsonl"
        rec_85999 = records_by_id[85999]
        assert rec_85999["tr_baslik"] == "Rübâb-ı Şikeste/Sen Olmasan"
        assert rec_85999["yontem"] == "arama"
        assert rec_85999["kapsama"] >= 0.40
        assert rec_85999["ort_skor"] >= 0.70

        # Check pageid 155879
        assert 155879 in records_by_id, "pageid=155879 missing from ws_page_links.jsonl"
        rec_155879 = records_by_id[155879]
        assert rec_155879["tr_baslik"] == "Bâğ-ı Dehrin Hem Hazânın Hem Bahârın Görmüşüz"
        assert rec_155879["yontem"] == "arama"
        assert rec_155879["kapsama"] >= 0.40
        assert rec_155879["ort_skor"] >= 0.70


class TestInterimWsParallelIntegrity:
    """Direct empirical validation of data/interim/ws_parallel.jsonl."""

    PARALLEL_PATH = Path("data/interim/ws_parallel.jsonl")
    PAGE_LINKS_PATH = Path("data/interim/ws_page_links.jsonl")

    def test_parallel_file_exists(self) -> None:
        assert self.PARALLEL_PATH.exists(), f"{self.PARALLEL_PATH} does not exist"

    def test_parallel_exact_row_count(self) -> None:
        lines = self.PARALLEL_PATH.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1424, f"Expected 1424 pairs, got {len(lines)}"

    def test_parallel_all_12_required_fields_and_types(self) -> None:
        required_keys = {
            "kaynak",
            "pageid",
            "yontem",
            "kapsama",
            "baslik_arap",
            "tr_baslik",
            "arap",
            "latin",
            "hizalama",
            "skor",
            "guvenilir",
            "lisans",
        }
        for idx, line in enumerate(self.PARALLEL_PATH.read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            assert required_keys.issubset(r.keys()), f"Pair {idx} missing fields"
            assert r["kaynak"] == "wikisource+vikikaynak"
            assert isinstance(r["pageid"], int)
            assert r["yontem"] in {"langlinks", "arama"}
            assert isinstance(r["kapsama"], (int, float))
            assert isinstance(r["arap"], str) and len(r["arap"].strip()) > 0
            assert isinstance(r["latin"], str) and len(r["latin"].strip()) > 0
            assert r["hizalama"] in {"1-1", "1-2", "2-1"}
            assert isinstance(r["skor"], (int, float))
            assert isinstance(r["guvenilir"], bool)
            assert r["lisans"] == "CC BY-SA 4.0"

    def test_parallel_reliable_pairs_flag_strict_rule(self) -> None:
        """Every single pair must satisfy guvenilir == (skor >= 0.55) without exception."""
        flag_violations = []
        guvenilir_count = 0

        for idx, line in enumerate(self.PARALLEL_PATH.read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            expected = bool(r["skor"] >= 0.55)
            if r["guvenilir"] != expected:
                flag_violations.append((idx, r["skor"], r["guvenilir"], expected))
            if r["guvenilir"]:
                guvenilir_count += 1

        assert len(flag_violations) == 0, f"Found {len(flag_violations)} flag violations: {flag_violations}"
        assert guvenilir_count == 1348, f"Expected 1348 reliable pairs, got {guvenilir_count}"

    def test_parallel_all_33_pages_represented(self) -> None:
        """ws_parallel.jsonl contains records from all 33 accepted pages."""
        expected_pageids = {
            json.loads(line)["pageid"]
            for line in self.PAGE_LINKS_PATH.read_text(encoding="utf-8").splitlines()
        }
        actual_pageids = {
            json.loads(line)["pageid"]
            for line in self.PARALLEL_PATH.read_text(encoding="utf-8").splitlines()
        }
        assert actual_pageids == expected_pageids, "Pageid sets between ws_page_links and ws_parallel must match"

    def test_parallel_score_distribution(self) -> None:
        """Check score distribution: high confidence pairs (skor >= 0.80) form the dominant majority."""
        pairs = [
            json.loads(line)
            for line in self.PARALLEL_PATH.read_text(encoding="utf-8").splitlines()
        ]
        high_score_pairs = [p for p in pairs if p["skor"] >= 0.80]
        # Over 90% of all pairs have skor >= 0.80
        ratio = len(high_score_pairs) / len(pairs)
        assert ratio >= 0.90, f"Expected >= 90% pairs with score >= 0.80, got {ratio * 100:.2f}%"
