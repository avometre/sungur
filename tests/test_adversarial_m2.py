"""Adversarial stress-test suite for Milestone 2 (Scraper & Alignment v2).

Tests:
1. Sentence splitting stress testing:
   - Ellipsis with varied spacing and trailing quotes/parentheses.
   - Single-letter and multi-letter abbreviation chains in Latin and Arabic.
   - Strict boundary tests (119, 120, 121 chars, exactly 120 with dense punctuation).
   - Arabic punctuation (؟, ؛, ۔) with quotation marks and parentheses.
   - Monolithic text with zero punctuation (>1000 chars, >5000 chars).
2. HTML Navigation cleanup adversarial testing:
   - Whitespace padding, casing variations for navigation terms.
   - Arrow variants (isolated arrows, trailing arrows, leading arrows).
   - Single-word authors (Fuzûlî, Bâkî, Nedîm, Kemal, Ömer, etc.).
   - Nested tables, mixed table types, multiple cquote tables.
3. DP Alignment edge cases:
   - Extreme line count disparities, zero similarity fallback, all-whitespace lines.
"""

from pathlib import Path
import pytest
from bs4 import BeautifulSoup

from src.scrapers.vikikaynak_align import (
    align_lines,
    clean_latin_html,
    evaluate_candidate_alignment,
    split_sentences,
    SPLIT_SENTENCE_PATTERN,
    NAVIGATION_EXACT_TITLES,
)


class TestAdversarialSentenceSplitting:
    """Stress tests for split_sentences."""

    def test_boundary_lengths_119_120_121(self) -> None:
        # 119 chars: line with punctuation MUST NOT split
        s119 = "A" * 60 + ". " + "B" * 57
        assert len(s119) == 119
        res119 = split_sentences(s119)
        assert len(res119) == 1
        assert res119[0] == s119

        # Exactly 120 chars with punctuation: MUST NOT split (<= 120)
        s120 = "A" * 60 + ". " + "B" * 58
        assert len(s120) == 120
        res120 = split_sentences(s120)
        assert len(res120) == 1
        assert res120[0] == s120

        # Exactly 120 chars with dense punctuation (. ? !): STILL MUST NOT split
        s120_multi = "A" * 30 + ". " + "B" * 30 + "? " + "C" * 30 + "! " + "D" * 24
        assert len(s120_multi) == 120
        res120_m = split_sentences(s120_multi)
        assert len(res120_m) == 1
        assert res120_m[0] == s120_multi

        # Exactly 121 chars with sentence end: MUST split into 2
        s121 = "A" * 60 + ". " + "B" * 59
        assert len(s121) == 121
        res121 = split_sentences(s121)
        assert len(res121) == 2
        assert res121[0] == "A" * 60 + "."
        assert res121[1] == "B" * 59

        # Exactly 121 chars WITHOUT punctuation: MUST NOT split
        s121_nopunct = "a" * 121
        res121_no = split_sentences(s121_nopunct)
        assert len(res121_no) == 1
        assert res121_no[0] == s121_nopunct

    def test_ellipsis_variations_with_quotes_and_brackets(self) -> None:
        prefix = "Gönlümdeki derin ızdırap ve dinmeyen fırtına hiç bitmeyecek gibiydi ve ruhumu derinden sarıyordu"
        suffix = "Fakat nihayet ufukta beklenen aydınlık şafak belirdi ve yeni bir umut doğdu."

        # Case 1: Standard triple dot with attached double quote
        line1 = f'{prefix}..." {suffix}'
        assert len(line1) > 120
        res1 = split_sentences(line1)
        assert len(res1) == 2
        assert res1[0] == f'{prefix}..."'
        assert res1[1] == suffix

        # Case 2: Standard triple dot with attached single quote
        line2 = f"{prefix}...' {suffix}"
        assert len(line2) > 120
        res2 = split_sentences(line2)
        assert len(res2) == 2
        assert res2[0] == f"{prefix}...'"
        assert res2[1] == suffix

        # Case 3: Standard triple dot with attached closing paren
        line3 = f"{prefix}...) {suffix}"
        assert len(line3) > 120
        res3 = split_sentences(line3)
        assert len(res3) == 2
        assert res3[0] == f"{prefix}...)"
        assert res3[1] == suffix

        # Case 4: Standard triple dot with right double quotation mark (”)
        line4 = f"{prefix}...” {suffix}"
        assert len(line4) > 120
        res4 = split_sentences(line4)
        assert len(res4) == 2
        assert res4[0] == f"{prefix}...”"
        assert res4[1] == suffix

        # Case 5: Standard triple dot with guillemet (»)
        line5 = f"{prefix}...» {suffix}"
        assert len(line5) > 120
        res5 = split_sentences(line5)
        assert len(res5) == 2
        assert res5[0] == f"{prefix}...»"
        assert res5[1] == suffix

        # Case 6: Four dots (....)
        line6 = f"{prefix}.... {suffix}"
        assert len(line6) > 120
        res6 = split_sentences(line6)
        assert len(res6) == 2
        assert res6[0] == f"{prefix}...."
        assert res6[1] == suffix

    def test_ellipsis_with_spaced_quotes(self) -> None:
        """When an ellipsis is followed by a space before the quote: '... " '"""
        prefix = "Gönlümdeki derin ızdırap ve dinmeyen fırtına hiç bitmeyecek gibiydi ve ruhumu derinden sarıyordu"
        suffix = "Fakat nihayet ufukta beklenen aydınlık şafak belirdi ve yeni bir umut doğdu."
        line = f'{prefix}... " {suffix}'
        assert len(line) > 120
        res = split_sentences(line)
        # Note: If split happens at '... ', quote remains attached to second sentence
        # Let's verify it does not crash or lose content
        assert "".join(res).replace(" ", "") == line.replace(" ", "")

    def test_chains_of_abbreviations_latin(self) -> None:
        # Single-letter chains: M. Kemal, M. K. Atatürk, A. H. Tanpınar
        s1 = ("Büyük kumandan M. K. Atatürk önderliğinde kazanılan bu muazzam zafer "
              "tarih boyunca milletimizin en büyük iftiharı olarak kalacaktır.")
        s2 = "Yeni nesiller bu kutlu emaneti daima muhafaza edeceklerdir."
        line = f"{s1} {s2}"
        assert len(line) > 120
        res = split_sentences(line)
        assert len(res) == 2
        assert res[0] == s1
        assert res[1] == s2
        assert "M. K. Atatürk" in res[0]

    def test_chains_of_abbreviations_multi_letter(self) -> None:
        """Multi-letter abbreviations (Prof., Dr.) vs specification:
        The prompt specifies: 'tek harf + nokta kısaltmalarını koru'.
        Multi-letter abbreviations (e.g. Prof., Dr.) are NOT single letter.
        We test how split_sentences handles them."""
        s = ("Prof. Dr. M. Kemal Paşa tarafından kaleme alınan bu tarihi vesika "
             "milletimizin sinesinde ebediyen yaşayacak büyük bir şaheserdir ve pek kıymetlidir.")
        assert len(s) > 120
        res = split_sentences(s)
        # Empirical observation: 'Prof.' and 'Dr.' split because they have >1 letter before dot.
        # But 'M. Kemal' is preserved because 'M' is single letter.
        assert any("M. Kemal" in r for r in res)

    def test_chains_of_abbreviations_arabic(self) -> None:
        # Arabic single letter abbreviations: م. ع. افندی
        s1 = ("حضرت م. ع. افندی اولوب بو کتابده اوزون اوزادییه بیان قلنمشدر "
              "و هر نقطه سی دقته شایan اولان بر اثردر.")
        s2 = "نهایتنده ده بو مبارک جمله علاوه اولنمشدر که یوز یگرمی حرفدن طاشقندر."
        line = f"{s1} {s2}"
        assert len(line) > 120
        res = split_sentences(line)
        assert len(res) == 2
        assert res[0] == s1
        assert res[1] == s2
        assert "م. ع. افندی" in res[0]

    def test_arabic_punctuation_with_quotes(self) -> None:
        prefix = "بو برنجی جمله در که عثمانلیجه یازیلمشدر و نهایتنده صورو اشارتی بولنمقده در"
        suffix = "بو ده ایکنجی جمله در که یوز یگرمی حرفدن اوزون اولمسی ایچون بورایه علاوه قلنمشدر."

        # Arabic question mark with quote: ؟"
        line_q = f'{prefix}؟" {suffix}'
        assert len(line_q) > 120
        res_q = split_sentences(line_q)
        assert len(res_q) == 2
        assert res_q[0] == f'{prefix}؟"'
        assert res_q[1] == suffix

        # Arabic semicolon with quote: ؛"
        line_semi = f'{prefix}؛" {suffix}'
        assert len(line_semi) > 120
        res_semi = split_sentences(line_semi)
        assert len(res_semi) == 2
        assert res_semi[0] == f'{prefix}؛"'
        assert res_semi[1] == suffix

        # Arabic full stop with quote: ۔"
        line_dot = f'{prefix}۔" {suffix}'
        assert len(line_dot) > 120
        res_dot = split_sentences(line_dot)
        assert len(res_dot) == 2
        assert res_dot[0] == f'{prefix}۔"'
        assert res_dot[1] == suffix

        # Arabic punctuation with guillemet: ؟»
        line_g = f'{prefix}؟» {suffix}'
        assert len(line_g) > 120
        res_g = split_sentences(line_g)
        assert len(res_g) == 2
        assert res_g[0] == f'{prefix}؟»'
        assert res_g[1] == suffix

    def test_monolithic_text_no_punctuation_extreme_lengths(self) -> None:
        # 1000 characters with zero punctuation
        text_1000 = "kelime " * 143  # 7 * 143 = 1001 chars
        assert len(text_1000.strip()) >= 1000
        res_1000 = split_sentences(text_1000)
        assert len(res_1000) == 1
        assert res_1000[0] == text_1000.strip()

        # 5000 characters monolithic text
        text_5000 = "vatan " * 834  # 6 * 834 = 5004 chars
        assert len(text_5000.strip()) >= 5000
        res_5000 = split_sentences(text_5000)
        assert len(res_5000) == 1
        assert res_5000[0] == text_5000.strip()

    def test_consecutive_punctuation_marks(self) -> None:
        # Multiple exclamation/question marks: !?, ??, !!!
        s1 = ("Bu nasil bir cesarettir ve bu ne buyuk bir kahramanliktir boyle!?! "
              "Butun dunya bu essiz fedakarliga hayran kalmistir.")
        s2 = "Tarih bu serefli mucadeleyi altin harflerle kaydetmistir."
        line = f"{s1} {s2}"
        assert len(line) > 120
        res = split_sentences(line)
        # s1 contains 2 sentences ('...boyle!?!' and '...kalmistir.'), s2 is 1 sentence. Total: 3.
        assert len(res) == 3
        assert "!?!" in res[0]
        assert res[2] == s2


class TestAdversarialHtmlNavigationCleanup:
    """Stress tests for clean_latin_html."""

    def test_whitespace_and_casing_variations_navigation(self) -> None:
        html = """
        <p>   Sayfa Başı   </p>
        <p>  KAYNAKÇA  </p>
        <p>   kaynak   </p>
        <p> NOTLAR </p>
        <p>  DİPNOTLAR </p>
        <p>   Ayrıca Bakınız   </p>
        <p> DIŞ BAĞLANTILAR </p>
        <p>   İÇİNDEKİLER   </p>
        <p>Bu metin gercektir ve kaybolmamalidir, gayet sahihtir.</p>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 1
        assert lines[0] == "Bu metin gercektir ve kaybolmamalidir, gayet sahihtir."

    def test_arrow_variations(self) -> None:
        html = """
        <p>← Önceki Metin</p>
        <p>Sonraki Metin →</p>
        <p>←</p>
        <p>→</p>
        <p>←  </p>
        <p>   →</p>
        <p>Gerçek metin mısrası burada yer almaktadır ve muteberdir.</p>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 1
        assert lines[0] == "Gerçek metin mısrası burada yer almaktadır ve muteberdir."

    def test_single_word_authors_preserved(self) -> None:
        authors = ["Fuzûlî", "Bâkî", "Nedîm", "Kemal", "Ömer", "Nâbî", "Nef'î", "Şinâsi"]
        html_parts = [f"<p>{author}</p>\n<p>Dost bağının goncası açıldı gül yüzüne.</p>" for author in authors]
        full_html = "\n".join(html_parts)
        lines = clean_latin_html(full_html)
        for author in authors:
            assert author in lines, f"Author {author} should be preserved!"

    def test_author_collision_with_nav_words(self) -> None:
        """If a text has a line containing a navigation word as part of a sentence, it is kept."""
        html = """
        <p>Kaynak ve menba araştırması ilmin temelidir.</p>
        <p>Notlar ve haşiyeler eseri tefsir eder.</p>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 2
        assert "Kaynak ve menba araştırması ilmin temelidir." in lines
        assert "Notlar ve haşiyeler eseri tefsir eder." in lines

    def test_nested_cquote_tables(self) -> None:
        """Nested tables: cquote inside cquote, or normal table inside cquote."""
        html = """
        <table class="cquote">
            <tr>
                <td>
                    <p>Dış mısra burada yer alır.</p>
                    <table class="cquote">
                        <tr><td>İç mısra burada yer alır.</td></tr>
                    </table>
                </td>
            </tr>
        </table>
        """
        lines = clean_latin_html(html)
        assert "Dış mısra burada yer alır." in lines
        assert "İç mısra burada yer alır." in lines

    def test_multiple_cquote_tables_mixed_with_wikitables(self) -> None:
        html = """
        <table class="wikitable"><tr><td>Silinecek Tablo 1</td></tr></table>
        <table class="cquote"><tr><td>Birinci Şiir Kıymetli Mısra</td></tr></table>
        <table class="navbox"><tr><td>Silinecek Navigasyon</td></tr></table>
        <table class="cquote"><tr><td>İkinci Şiir Kıymetli Mısra</td></tr></table>
        <table class="infobox"><tr><td>Silinecek Bilgi Kutusu</td></tr></table>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 2
        assert lines[0] == "Birinci Şiir Kıymetli Mısra"
        assert lines[1] == "İkinci Şiir Kıymetli Mısra"

    def test_malformed_html_robustness(self) -> None:
        # Unclosed tags, broken tables
        html = """
        <table class="cquote"><tr><td>Yarım kalmış dize<br>ikinci dize
        <p>Paragraf kapanmamış
        <table class="wikitable"><tr><td>tablo da kapanmamış
        """
        lines = clean_latin_html(html)
        assert len(lines) >= 2
        assert any("Yarım kalmış dize" in l for l in lines)


class TestAdversarialAlignmentDP:
    """Stress tests on dynamic programming align_lines."""

    def test_extreme_disparity_in_line_counts(self) -> None:
        # 1 Arabic line vs 100 Latin lines
        arap = ["کتاب"]
        latin = [f"farkli satir {i}" for i in range(100)]
        pairs, stats = align_lines(arap, latin)
        assert stats["atlanan_latin"] >= 99
        assert len(pairs) <= 1

    def test_zero_similarity_all_lines(self) -> None:
        # Completely disjoint alphabet / letters with 0 similarity
        arap = ["ففف", "ققق"]
        latin = ["zzzz", "yyyy"]
        pairs, stats = align_lines(arap, latin)
        # All lines should be skipped rather than producing false pairs if cost of skip is lower
        for p in pairs:
            # If any pair is matched, its score must be accurately calculated
            assert p["skor"] == 0.0

    def test_evaluate_candidate_alignment_robustness(self) -> None:
        # Candidate evaluation with mismatched empty inputs
        k, s, p = evaluate_candidate_alignment([], ["latin satir"])
        assert k == 0.0
        assert s == 0.0
        assert p == []

        k2, s2, p2 = evaluate_candidate_alignment(["عثمانلیجه"], [])
        assert k2 == 0.0
        assert s2 == 0.0
        assert p2 == []
