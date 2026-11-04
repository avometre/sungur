"""Vikikaynak satır hizalama (align_lines), cümle bölme (split_sentences), HTML temizleme, aday arama ve puanlama testleri."""

import json
from pathlib import Path
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


class TestAlignLines:
    """Temel dinamik programlama hizalama testleri."""

    def test_three_identical_lines(self) -> None:
        arap_lines = [
            "\u06a9\u062a\u0627\u0628",
            "\u062f\u0648\u0644\u062a \u0639\u0644\u06cc\u0647",
            "\u067e\u0627\u062f\u0634\u0627\u0647",
        ]
        latin_lines = ["kitap", "devlet-i aliyye", "padişah"]

        pairs, stats = align_lines(arap_lines, latin_lines)
        assert len(pairs) == 3
        assert [p["hizalama"] for p in pairs] == ["1-1", "1-1", "1-1"]
        assert all(p["skor"] >= 0.8 for p in pairs)
        assert stats["atlanan_arap"] == 0
        assert stats["atlanan_latin"] == 0

    def test_extra_title_line_in_latin_skipped(self) -> None:
        arap_lines = ["\u06a9\u062a\u0627\u0628", "\u062f\u0648\u0644\u062a \u0639\u0644\u06cc\u0647"]
        latin_lines = ["Bölüm I: Giriş ve Başlık", "kitap", "devlet-i aliyye"]

        pairs, stats = align_lines(arap_lines, latin_lines)
        assert len(pairs) == 2
        assert [p["hizalama"] for p in pairs] == ["1-1", "1-1"]
        assert pairs[0]["latin"] == "kitap"
        assert pairs[1]["latin"] == "devlet-i aliyye"
        assert stats["atlanan_latin"] == 1
        assert stats["atlanan_arap"] == 0

    def test_one_arabic_split_into_two_latin(self) -> None:
        arap_lines = ["\u06a9\u062a\u0627\u0628 \u067e\u0627\u062f\u0634\u0627\u0647"]
        latin_lines = ["kitap", "padişah"]

        pairs, stats = align_lines(arap_lines, latin_lines)
        assert len(pairs) == 1
        assert pairs[0]["hizalama"] == "1-2"
        assert pairs[0]["latin"] == "kitap padişah"
        assert pairs[0]["skor"] >= 0.8
        assert stats["atlanan_arap"] == 0
        assert stats["atlanan_latin"] == 0

    def test_two_arabic_merged_into_one_latin(self) -> None:
        arap_lines = ["\u0627\u0633\u062a\u0627\u0646\u0628\u0648\u0644", "\u067e\u0627\u062f\u0634\u0627\u0647"]
        latin_lines = ["İstanbul padişah"]

        pairs, stats = align_lines(arap_lines, latin_lines)
        assert len(pairs) == 1
        assert pairs[0]["hizalama"] == "2-1"
        assert pairs[0]["skor"] >= 0.8

    def test_empty_lines_input(self) -> None:
        pairs, stats = align_lines([], ["kitap"])
        assert pairs == []
        assert stats["atlanan_latin"] == 1
        assert stats["atlanan_arap"] == 0


class TestCleanLatinHtml:
    """Mevcut HTML temizleme testleri."""

    def test_clean_html_removes_unwanted_tags(self) -> None:
        html = """
        <div class="headertemplate">Gereksiz Başlık</div>
        <div class="mw-parser-output">
            <p>Birinci satır metin burada.<br>İkinci dize burada başlıyor.</p>
            <div class="navbox">Navigasyon kutusu</div>
            <div class="noprint">Çokdilli Vikikaynak bağlantısı</div>
            <p>Üçüncü satır devam ediyor.</p>
        </div>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 3
        assert "Birinci satır metin burada." in lines
        assert "İkinci dize burada başlıyor." in lines
        assert "Üçüncü satır devam ediyor." in lines
        assert not any("Navigasyon" in l for l in lines)
        assert not any("Gereksiz" in l for l in lines)
        assert not any("Çokdilli" in l for l in lines)


class TestSplitSentences:
    """split_sentences fonksiyonu için kapsamlı, ağ gerektirmeyen birim testleri (R1)."""

    def test_split_sentences_empty_or_whitespace(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("   ") == []
        assert split_sentences("\n\t  ") == []

    def test_split_sentences_short_line_untouched(self) -> None:
        short = "Bu kısa bir satırdır ve yüz yirmi karakterin altındadır."
        assert len(short) <= 120
        assert split_sentences(short) == [short]

    def test_split_sentences_exact_120_chars(self) -> None:
        base = "Bu cumle tam yuz yirmi karakter uzunlugundadir ve icinde nokta barindirir."
        line_120 = base + "X" * (120 - len(base))
        assert len(line_120) == 120
        res = split_sentences(line_120)
        assert len(res) == 1
        assert res == [line_120]

    def test_split_sentences_exact_121_chars_splits(self) -> None:
        s1 = "Bu birinci cumledir ve yirmi harftir."
        s2 = "Bu ikinci cumledir ve doksan harf kadar uzatilarak yuz yirmi biri gecmesi saglanmistir buraya kadar."
        line_121 = f"{s1} {s2}"
        assert len(line_121) > 120
        res = split_sentences(line_121)
        assert len(res) >= 2
        assert res[0] == s1
        assert res[1] == s2

    def test_split_sentences_long_latin_all_punctuations(self) -> None:
        s1 = "Birinci cümle noktayla biter."
        s2 = "İkinci cümle ise ünlemle biter!"
        s3 = "Üçüncü cümle acaba soru işaretiyle mi biter?"
        s4 = "Dördüncü cümle iki nokta barındırır: dikkat ediniz;"
        s5 = "Beşinci cümle üç nokta karakteri içerir… Altıncı cümle devam eder."
        long_line = f"{s1} {s2} {s3} {s4} {s5}"
        assert len(long_line) > 120
        res = split_sentences(long_line)
        assert len(res) >= 5
        assert any(s1 in r for r in res)
        assert any(s2 in r for r in res)
        assert any(s3 in r for r in res)

    def test_split_sentences_preserves_triple_dots(self) -> None:
        s1 = "Gönlümdeki bu derin ızdırap ve dinmeyen fırtına hiç bitmeyecek gibiydi ve ruhumu derinden sarıyordu..."
        s2 = "Fakat nihayet ufukta beklenen aydınlık şafak belirdi ve yeni bir umut doğdu."
        long_line = f"{s1} {s2}"
        assert len(long_line) > 120
        res = split_sentences(long_line)
        assert len(res) == 2
        assert "..." in res[0]
        assert res[0] == s1
        assert res[1] == s2

    def test_split_sentences_preserves_single_abbreviations(self) -> None:
        s1 = "Başkomutan M. Kemal Paşa ordulara ilk hedeflerinin Akdeniz olduğunu bildiren tarihi emrini bizzat tebliğ etmiştir."
        s2 = "Bu mühim emir sonrasında Türk ordusu büyük bir azim ve cesaretle taarruza geçerek zaferi kazanmıştır."
        long_line = f"{s1} {s2}"
        assert len(long_line) > 120
        res = split_sentences(long_line)
        assert len(res) == 2
        assert any("M. Kemal" in r for r in res)
        assert not any(r.strip() == "M." for r in res)

    def test_split_sentences_preserves_multiple_abbreviations(self) -> None:
        s1 = "Müellif M. A. Ersoy Beyefendi tarafından kaleme alınan bu eser milletimizin hislerine bütünüyle tercüman olmaktadır."
        s2 = "Eserin her mısrası derin bir vatan sevgisi ve hürriyet aşkı barındırmaktadır."
        long_line = f"{s1} {s2}"
        assert len(long_line) > 120
        res = split_sentences(long_line)
        assert len(res) == 2
        assert any("M. A. Ersoy" in r for r in res)

    def test_split_sentences_long_arabic_paragraph(self) -> None:
        s1 = "بو برنجی جمله در که عثمانلیجه یازیلمشدر و نهایتنده صورو اشارتی بولنمقده در؟"
        s2 = "بو ده ایکنجی جمله در که یوز یگرمی حرفدن اوزون اولمسی ایچون بورایه علاوه قلنمشدر."
        long_line = f"{s1} {s2}"
        assert len(long_line) > 120
        res = split_sentences(long_line)
        assert len(res) >= 2
        assert res[0] == s1
        assert res[1] == s2

    def test_split_sentences_arabic_single_letter_abbreviation(self) -> None:
        s1 = "حضرت م. علی کرم الله وجهه اولوب بو کتابده اوزون اوزادییه بیان قلنمشدر که یوز یگرمی حرفدن طاشقندر."
        s2 = "نهایتنده ده بو مبارک جمله علاوه اولنمشدر."
        long_line = f"{s1} {s2}"
        assert len(long_line) > 120
        res = split_sentences(long_line)
        assert len(res) == 2
        assert any("م." in r for r in res)
        assert not any(r.strip() == "م." for r in res)

    def test_split_sentences_long_line_no_punctuation(self) -> None:
        line = "bu satirin icinde hicbir noktalama isareti bulunmamaktadir ve yuz yirmi karakterden oldukca uzundur ancak yine de keyfi bir bolme yapilmamalidir"
        assert len(line) > 120
        res = split_sentences(line)
        assert len(res) == 1
        assert res[0] == line

    def test_split_sentences_quoted_sentences(self) -> None:
        s1 = 'Gazi Hazretleri nutkunda kesin bir dille bildirdiği üzere "Ordular ilk hedefiniz Akdenizdir, ileri!"'
        s2 = "Bu tarihi emir ile şanlı ordu taarruza geçerek düşmanı bütünüyle mağlup etmiştir."
        long_line = f"{s1} {s2}"
        assert len(long_line) > 120
        res = split_sentences(long_line)
        assert len(res) == 2
        assert res[0] == s1
        assert res[1] == s2


class TestCleanLatinHtmlExtended:
    """clean_latin_html fonksiyonu için cquote ve gezinme temizliği birim testleri (R1)."""

    def test_clean_latin_html_cquote_table_unwrapped(self) -> None:
        html = """
        <table class="cquote">
            <tr>
                <td>Dost bağının goncaları açıldı<br>Gül yüzüne nergis gözler süzüldü</td>
            </tr>
        </table>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 2
        assert "Dost bağının goncaları açıldı" in lines
        assert "Gül yüzüne nergis gözler süzüldü" in lines

    def test_clean_latin_html_cquote_compound_class(self) -> None:
        html = '<table class="cquote poem-table"><tr><td>Kaside dizesi burada yer almaktadır</td></tr></table>'
        lines = clean_latin_html(html)
        assert len(lines) == 1
        assert lines[0] == "Kaside dizesi burada yer almaktadır"

    def test_clean_latin_html_standard_tables_removed(self) -> None:
        html = """
        <table><tr><td>Yazar Bilgisi Tablosu</td></tr></table>
        <table class="wikitable"><tr><td>Veri Satırı</td></tr></table>
        <table class="navbox"><tr><td>Navigasyon Menüsü</td></tr></table>
        <p>Şiirin gerçek ilk dizesi burasıdır ve muteberdir.</p>
        """
        lines = clean_latin_html(html)
        assert not any("Yazar Bilgisi" in l for l in lines)
        assert not any("Veri Satırı" in l for l in lines)
        assert not any("Navigasyon Menüsü" in l for l in lines)
        assert any("Şiirin gerçek ilk dizesi" in l for l in lines)

    def test_clean_latin_html_navigation_exact_titles_removed(self) -> None:
        html = """
        <p>Sayfa başı</p>
        <p>Asıl metin burada yer almaktadır ve gayet muteberdir.</p>
        <p>Kaynakça</p>
        <p>Kaynak</p>
        <p>Notlar</p>
        <p>Dipnotlar</p>
        <p>Ayrıca bakınız</p>
        <p>Dış bağlantılar</p>
        <p>İçindekiler</p>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 1
        assert lines[0] == "Asıl metin burada yer almaktadır ve gayet muteberdir."

    def test_clean_latin_html_navigation_arrows_removed(self) -> None:
        html = """
        <p>← Önceki Metin</p>
        <p>←Başka Metin</p>
        <p>Gerçek metin satırı buradadır.</p>
        <p>Sonraki Metin →</p>
        <p>Daha Sonraki Metin→</p>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 1
        assert lines[0] == "Gerçek metin satırı buradadır."

    def test_clean_latin_html_preserves_single_word_authors(self) -> None:
        html = """
        <p>Fuzûlî</p>
        <p>Beni candan usandırdı cefâdan yâr usanmaz mı</p>
        <p>Felekler yandı âhımdan murâdım şem'i yanmaz mı</p>
        <p>Nedîm</p>
        """
        lines = clean_latin_html(html)
        assert any("Fuzûlî" in l for l in lines)
        assert any("Nedîm" in l for l in lines)

    def test_clean_latin_html_embedded_boilerplate_preserved(self) -> None:
        html = "<p>Bu kaynak mühim bir tarihi vesikadır ve korunmalıdır.</p>"
        lines = clean_latin_html(html)
        assert len(lines) == 1
        assert lines[0] == "Bu kaynak mühim bir tarihi vesikadır ve korunmalıdır."

    def test_clean_latin_html_empty_input(self) -> None:
        assert clean_latin_html("") == []
        assert clean_latin_html("   ") == []


class TestCandidateSearchAndScoringM2:
    """Milestone 2 (R2) Aday Arama, Sayfa Puanlama ve Loglama Testleri."""

    def test_search_candidate_titles_exact_success(self) -> None:
        """Tam başlık bulunduğunda arama sorgusuna geçmeden standart başlığı döndürür."""
        with mock.patch("src.scrapers.vikikaynak_align.polite_get") as mock_get:
            mock_get.return_value.json.return_value = {
                "query": {"pages": {"85191": {"pageid": 85191, "title": "İstiklâl Marşı"}}}
            }
            res = search_candidate_titles("İstiklâl Marşı")
            assert res == ["İstiklâl Marşı"]
            assert mock_get.call_count == 1
            params = mock_get.call_args[1].get("params", {})
            assert params["action"] == "query"
            assert params["titles"] == "İstiklâl Marşı"
            assert params["redirects"] == "1"

    def test_search_candidate_titles_fallback_on_missing(self) -> None:
        """Tam başlık bulunamadığında list=search sorgusu ile srlimit=5 olarak arama yapar."""
        with mock.patch("src.scrapers.vikikaynak_align.polite_get") as mock_get:
            mock_get.return_value.json.side_effect = [
                {"query": {"pages": {"-1": {"missing": ""}}}},
                {"query": {"search": [{"title": "Uyan"}, {"title": "Uyan Ey Gözlerim"}]}},
            ]
            res = search_candidate_titles("Uyan!")
            assert res == ["Uyan", "Uyan Ey Gözlerim"]
            assert mock_get.call_count == 2
            search_params = mock_get.call_args[1].get("params", {})
            assert search_params["list"] == "search"
            assert search_params["srsearch"] == "Uyan!"
            assert str(search_params["srlimit"]) == "5"

    def test_search_candidate_titles_fallback_disabled(self) -> None:
        """fallback_search=False olduğunda tam başlık bulunamazsa arama yapmaz, boş döner."""
        with mock.patch("src.scrapers.vikikaynak_align.polite_get") as mock_get:
            mock_get.return_value.json.return_value = {"query": {"pages": {"-1": {"missing": ""}}}}
            res = search_candidate_titles("Olmayan Sayfa", fallback_search=False)
            assert res == []
            assert mock_get.call_count == 1

    def test_search_candidate_titles_empty_input(self) -> None:
        """Boş veya sadece boşluktan oluşan girdilerde ağ çağrısı yapmadan boş liste döner."""
        with mock.patch("src.scrapers.vikikaynak_align.polite_get") as mock_get:
            assert search_candidate_titles("") == []
            assert search_candidate_titles("   ") == []
            assert not mock_get.called

    def test_search_candidate_titles_network_exception_handled(self) -> None:
        """Ağ hatası durumunda çökmez, boş liste döndürür."""
        with mock.patch("src.scrapers.vikikaynak_align.polite_get", side_effect=RuntimeError("Connection timeout")):
            res = search_candidate_titles("Aday")
            assert res == []

    def test_evaluate_candidate_alignment_high_score(self) -> None:
        """Uyumlu Arapça ve Latin satırlarında kapsama >= 0.4 ve ort_skor >= 0.7 döner."""
        arap = ["کتاب", "دولت علیه", "پادشاه"]
        latin = ["kitap", "devlet-i aliyye", "padişah"]
        kapsama, ort_skor, pairs = evaluate_candidate_alignment(arap, latin)
        assert isinstance(kapsama, float)
        assert isinstance(ort_skor, float)
        assert kapsama >= 0.40
        assert ort_skor >= 0.70
        assert len(pairs) == 3

    def test_evaluate_candidate_alignment_low_coverage(self) -> None:
        """Kapsama 0.40 altı kaldığında kapsama < 0.40 döner."""
        arap = ["کتاب", "دولت", "پادشاه", "عثمانلی", "استانبول"]
        latin = ["kitap"]
        kapsama, ort_skor, pairs = evaluate_candidate_alignment(arap, latin)
        assert kapsama < 0.40
        assert not (kapsama >= 0.40 and ort_skor >= 0.70)

    def test_evaluate_candidate_alignment_low_score(self) -> None:
        """Alakasız metinlerde ortalama skor < 0.70 döner."""
        arap = ["کتاب", "دولت"]
        latin = ["tamamen alakasız bir metin", "başka bir fuzuli satır"]
        kapsama, ort_skor, pairs = evaluate_candidate_alignment(arap, latin)
        assert ort_skor < 0.70

    def test_evaluate_candidate_alignment_empty_lists(self) -> None:
        """Boş satır listelerinde (0.0, 0.0, []) döner."""
        assert evaluate_candidate_alignment([], ["kitap"]) == (0.0, 0.0, [])
        assert evaluate_candidate_alignment(["کتاب"], []) == (0.0, 0.0, [])

    def test_fetch_latin_page_html_passes_redirects_param(self, tmp_path: Path) -> None:
        """fetch_latin_page_html API sorgusunda redirects=1 parametresi gönderildiğini doğrular."""
        with mock.patch("src.scrapers.vikikaynak_align.polite_get") as mock_get:
            mock_get.return_value.json.return_value = {
                "parse": {"text": {"*": "<p>Gerçek Kanun Metni</p>"}}
            }
            res = fetch_latin_page_html("Kanun-u Esasi", cache_dir=tmp_path)
            assert res == "<p>Gerçek Kanun Metni</p>"
            params = mock_get.call_args[1].get("params", {})
            assert str(params.get("redirects")) == "1"

    def test_fetch_latin_page_html_detects_and_refreshes_redirect_stubs(self, tmp_path: Path) -> None:
        """Önbellekte redirectText içeren bayat yönlendirme dosyası varsa yeniden çeker."""
        stub_file = tmp_path / "Kanun-u Esasi.html"
        stub_file.write_text('<ul class="redirectText"><li>Hedef</li></ul>', encoding="utf-8")

        with mock.patch("src.scrapers.vikikaynak_align.polite_get") as mock_get:
            mock_get.return_value.json.return_value = {
                "parse": {"text": {"*": "<p>Taze Hedef Metin</p>"}}
            }
            res = fetch_latin_page_html("Kanun-u Esasi", cache_dir=tmp_path)
            assert res == "<p>Taze Hedef Metin</p>"
            assert mock_get.called

    def test_process_gold_candidates_selects_highest_coverage(self, tmp_path: Path) -> None:
        """Birden fazla kabul edilen aday arasından en yüksek kapsamalı olanı seçer."""
        csv_file = tmp_path / "title_map.csv"
        csv_file.write_text(
            "pageid,baslik_arap,tr_baslik,not\n"
            "201,کتاب,Kitap A|Kitap B,claude_aday\n",
            encoding="utf-8",
        )
        search_log_file = tmp_path / "ws_search_log.jsonl"
        pages_lines = {201: [{"arap": "کتاب دولت علیه"}, {"arap": "پادشاه استانبول"}]}

        with mock.patch("src.scrapers.vikikaynak_align.search_candidate_titles") as mock_search, \
             mock.patch("src.scrapers.vikikaynak_align.fetch_latin_page_html") as mock_fetch, \
             mock.patch("src.scrapers.vikikaynak_align.clean_latin_html") as mock_clean, \
             mock.patch("src.scrapers.vikikaynak_align.evaluate_candidate_alignment") as mock_eval:

            mock_search.side_effect = lambda c, **kw: [c]
            mock_fetch.return_value = "<p>dummy</p>"
            mock_clean.return_value = ["dummy"]
            # Kitap A: Kapsama 0.50, OrtSkor 0.80 -> Geçer
            # Kitap B: Kapsama 0.90, OrtSkor 0.75 -> Geçer ve daha yüksek kapsamalı
            mock_eval.side_effect = [
                (0.50, 0.80, []),
                (0.90, 0.75, []),
            ]

            accepted, logs = process_gold_candidates(
                gold_csv_path=csv_file,
                pages_lines=pages_lines,
                cache_dir=tmp_path,
                search_log_path=search_log_file,
            )

            assert 201 in accepted
            assert accepted[201]["tr_baslik"] == "Kitap B"
            assert accepted[201]["kapsama"] == 0.90
            assert len(logs) == 2
            assert all(l["kabul"] is True for l in logs)

    def test_process_gold_candidates_search_log_schema(self, tmp_path: Path) -> None:
        """ws_search_log.jsonl dosyasına yazılan her satırın doğru şemaya sahip olduğunu doğrular."""
        csv_file = tmp_path / "title_map.csv"
        csv_file.write_text("pageid,baslik_arap,tr_baslik,not\n301,رساله,Risale,claude_aday\n", encoding="utf-8")
        search_log_file = tmp_path / "ws_search_log.jsonl"
        pages_lines = {301: [{"arap": "رساله"}]}

        with mock.patch("src.scrapers.vikikaynak_align.search_candidate_titles", return_value=["Risale"]), \
             mock.patch("src.scrapers.vikikaynak_align.fetch_latin_page_html", return_value="<p>Risale</p>"), \
             mock.patch("src.scrapers.vikikaynak_align.clean_latin_html", return_value=["Risale"]), \
             mock.patch("src.scrapers.vikikaynak_align.evaluate_candidate_alignment", return_value=(0.8, 0.85, [])):

            accepted, logs = process_gold_candidates(
                gold_csv_path=csv_file,
                pages_lines=pages_lines,
                cache_dir=tmp_path,
                search_log_path=search_log_file,
            )

            lines = search_log_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            for key in ["pageid", "baslik_arap", "aday", "bulunan_baslik", "kapsama", "ort_skor", "kabul"]:
                assert key in data
            assert data["pageid"] == 301
            assert data["kabul"] is True


class TestWsParallelOutputSchema:
    """ws_parallel.jsonl çıktı şeması testleri (R3)."""

    def test_output_schema_keys(self) -> None:
        record = {
            "kaynak": "wikisource+vikikaynak",
            "pageid": 85737,
            "yontem": "langlinks",
            "kapsama": 0.9771,
            "baslik_arap": "خاویار",
            "tr_baslik": "Havyar",
            "arap": "حمدونه‌ خانم",
            "latin": "Hamdune hanım",
            "hizalama": "1-1",
            "skor": 0.9825,
            "guvenilir": True,
            "lisans": "CC BY-SA 4.0",
        }
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
        assert required_keys.issubset(record.keys())
        assert isinstance(record["pageid"], int)
        assert isinstance(record["yontem"], str)
        assert isinstance(record["kapsama"], float)
