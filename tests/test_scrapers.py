import json
from pathlib import Path
from src.common.normalize import arabic_ratio
from src.scrapers.gazetteer import parse_gazetteer_row
from src.scrapers.wikisource import clean_and_extract_lines, parse_page_records
from src.scrapers.wiktionary import extract_turkish_words, parse_wiktionary_entry

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestWiktionaryScraper:
    def test_parse_valid_entry(self) -> None:
        fixture_file = FIXTURES_DIR / "kaikki_sample.jsonl"
        with fixture_file.open("r", encoding="utf-8") as f:
            first_line = f.readline()
            entry = json.loads(first_line)

        parsed = parse_wiktionary_entry(entry, idx=0)
        assert parsed is not None
        assert parsed["kaynak"] == "wiktionary"
        assert parsed["id"] == "wikt:آیا:noun:0"
        assert parsed["arap_ham"] == "آیا"
        assert parsed["arap"] == "آیا"
        assert parsed["latin"] == "aya"
        assert parsed["latin_sade"] == "aya"
        assert parsed["tur"] == "noun"
        assert "aya" in parsed["turkce"]
        assert len(parsed["anlam"]) > 0
        assert parsed["koken"] is not None and "Proto-Turkic" in parsed["koken"]
        assert parsed["lisans"] == "CC BY-SA 4.0"
        assert parsed["url"] == "https://en.wiktionary.org/wiki/آیا"
        assert parsed["form_of"] == []
        assert parsed["cekim_mi"] is False

    def test_parse_form_of_inflection(self) -> None:
        fixture_file = FIXTURES_DIR / "kaikki_sample.jsonl"
        with fixture_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        # Son eklenen 6. satır çekimli biçim kaydıdır
        entry = json.loads(lines[-1])
        parsed = parse_wiktionary_entry(entry, idx=5)
        assert parsed is not None
        assert parsed["cekim_mi"] is True
        assert len(parsed["form_of"]) == 1
        assert parsed["form_of"][0]["lemma"] == "كاتب"
        assert "form-of" in parsed["form_of"][0]["etiketler"]
        assert "plural" in parsed["form_of"][0]["etiketler"]

    def test_parse_skips_non_arabic(self) -> None:
        fixture_file = FIXTURES_DIR / "kaikki_sample.jsonl"
        skipped_count = 0
        with fixture_file.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                parsed = parse_wiktionary_entry(entry, idx=0)
                if parsed is None:
                    skipped_count += 1
                    assert arabic_ratio(entry.get("word", "")) < 0.5

        assert skipped_count >= 1

    def test_extract_turkish_words(self) -> None:
        sample_data = [
            {"lang": "Turkish", "lang_code": "tr", "word": "kelime1"},
            {"lang": "French", "lang_code": "fr", "word": "mot"},
            {"nested": {"lang_code": "tr", "word": "kelime2"}},
            {"lang_code": "tr", "word": "kelime1"},  # Tekrar
        ]
        tr_words = extract_turkish_words(sample_data)
        assert tr_words == ["kelime1", "kelime2"]


class TestWikisourceScraper:
    def test_clean_and_extract_lines(self) -> None:
        html_file = FIXTURES_DIR / "wikisource_sample.html"
        html = html_file.read_text(encoding="utf-8")

        lines = clean_and_extract_lines(html, min_arabic_ratio=0.6)
        assert len(lines) > 20

        # Her satırın Arapça oranı >= 0.6 olmalı
        for line in lines:
            assert arabic_ratio(line) >= 0.6

        # Başlık ve şair satırları bulunmalı
        assert any("استقلال مارشی" in l for l in lines)
        assert any("محمد عاكف" in l for l in lines)

        # Latin alfabesindeki başlık veya açıklamalar elenmiş olmalı
        for line in lines:
            assert "Modern Turkish" not in line
            assert "TemplateStyles" not in line

    def test_parse_page_records(self) -> None:
        html_file = FIXTURES_DIR / "wikisource_sample.html"
        html = html_file.read_text(encoding="utf-8")

        pageid = 85191
        title = "استقلال مارشی"
        records, summary = parse_page_records(pageid, title, html)

        assert len(records) > 20
        assert summary["pageid"] == pageid
        assert summary["baslik"] == title
        assert summary["satir_sayisi"] == len(records)

        first_rec = records[0]
        assert first_rec["kaynak"] == "wikisource"
        assert first_rec["id"] == f"ws:{pageid}:0"
        assert first_rec["baslik"] == title
        assert "arap_ham" in first_rec
        assert "arap" in first_rec
        assert first_rec["lisans"] == "CC BY-SA 4.0 / kamu malı (sayfaya göre)"
        assert first_rec["url"].startswith("https://wikisource.org/wiki/")


class TestGazetteerScraper:
    def test_parse_gazetteer_row_schema(self) -> None:
        sample_row = {
            "osmanlica": "آب کارون",
            "transkripsiyon": "Ab-ı Karun",
        }
        record = parse_gazetteer_row(sample_row, idx=42)

        assert record["kaynak"] == "gazetteer"
        assert record["id"] == "gaz:42"
        assert record["arap_ham"] == "آب کارون"
        assert record["arap"] == "آب کارون"
        assert record["latin"] == "Ab-ı Karun"
        assert record["latin_sade"] == "Ab-ı Karun"
        assert record["tur"] == "yer_adi"
        assert record["lisans"] == "cc-by-nc-4.0"
        assert record["url"] == "https://huggingface.co/datasets/OttomanNLP/ottoman-place-names-gazetteer"
        assert record["ek"] == {}

    def test_parse_gazetteer_row_with_extra_columns(self) -> None:
        sample_row = {
            "osmanlica": "كتاب",
            "transkripsiyon": "kitâb",
            "il": "Konya",
            "tur_detay": "karye",
        }
        record = parse_gazetteer_row(sample_row, idx=1)
        assert record["arap"] == "کتاب"
        assert record["latin_sade"] == "kitab"
        assert record["ek"]["il"] == "Konya"
        assert record["ek"]["tur_detay"] == "karye"
