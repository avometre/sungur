"""Altın test ve geliştirme veri seti üretim mantığı birim testleri."""

import pytest
from src.gold.build_gold import is_reliable_pair, partition_dev_test


def test_is_reliable_pair_threshold():
    """Skor eşik filtresini sınar."""
    # Arapça: "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627" (selam dunya), Latin: "selam dunya" -> skeleton >= 4
    pair_pass = {
        "arap": "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627",
        "latin": "selam dunya",
        "skor": 0.75,
    }
    assert is_reliable_pair(pair_pass, min_score=0.75) is True

    pair_fail_low_score = {
        "arap": "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627",
        "latin": "selam dunya",
        "skor": 0.749,
    }
    assert is_reliable_pair(pair_fail_low_score, min_score=0.75) is False

    # Özel min_score testi
    assert is_reliable_pair(pair_fail_low_score, min_score=0.70) is True


def test_is_reliable_pair_short_line_rule():
    """Kısa satırlarda (iskelet < 4) sadece tam eşleşmelerin (1.0) kabul edilmesini sınar."""
    # Kısa satır örneği: Arapça "\u0635\u0648\u0631\u06cc" (sr: 2 harf), Latin: "Sordu" (srd: 3 harf)
    short_pair_imperfect = {
        "arap": "\u0635\u0648\u0631\u06cc",
        "latin": "Sordu",
        "skor": 0.80,
    }
    # Skor 0.80 (0.75'ten büyük) olsa dahi iskelet < 4 olduğundan elenmeli
    assert is_reliable_pair(short_pair_imperfect, min_score=0.75) is False

    # Tam eşleşen kısa satır (skor == 1.0) kabul edilmeli
    short_pair_perfect = {
        "arap": "\u0635\u0648\u0631\u062f\u06cc",  # srdy
        "latin": "sordu",
        "skor": 1.0,
    }
    assert is_reliable_pair(short_pair_perfect, min_score=0.75) is True


def test_partition_dev_test_sayfa_ort_skor():
    """Bölümleme sırasında sayfa ortalama skorunun kayıtlara doğru yansıtıldığını sınar."""
    sample_records = [
        {
            "arap": "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627 1",
            "latin": "selam dunya 1",
            "pageid": 101,
            "skor": 0.80,
            "tr_baslik": "Test Baslik 1",
        },
        {
            "arap": "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627 2",
            "latin": "selam dunya 2",
            "pageid": 101,
            "skor": 0.90,
            "tr_baslik": "Test Baslik 1",
        },
    ]

    page_avg_scores = {101: 0.85}
    dev, test = partition_dev_test(
        sample_records,
        dev_ratio=0.3,
        page_avg_scores=page_avg_scores,
    )

    combined = dev + test
    assert len(combined) == 2
    for record in combined:
        assert "sayfa_ort_skor" in record
        assert record["sayfa_ort_skor"] == 0.85
        assert "latin_sade" in record
        assert record["uslup"] == "birebir"


def test_partition_dev_test_uslup_mapping_and_validation():
    """page_style_map üzerinden üslup atamasını ve eksik sayfa durumunda ValueError fırlatılmasını sınar."""
    sample_records = [
        {
            "arap": "\u0633\u0644\u0627\u0645 1",
            "latin": "selam 1",
            "pageid": 101,
            "skor": 0.85,
            "tr_baslik": "Birebir Sayfa",
        },
        {
            "arap": "\u0633\u0644\u0627\u0645 2",
            "latin": "selam 2",
            "pageid": 202,
            "skor": 0.90,
            "tr_baslik": "Serbest Sayfa",
        },
    ]

    style_map = {101: "birebir", 202: "serbest"}
    dev, test = partition_dev_test(sample_records, page_style_map=style_map)
    combined = dev + test
    assert len(combined) == 2
    r_map = {r["pageid"]: r["uslup"] for r in combined}
    assert r_map[101] == "birebir"
    assert r_map[202] == "serbest"

    # Eksik pageid -> ValueError
    incomplete_map = {101: "birebir"}
    with pytest.raises(ValueError, match="Eşleşmeyen sayfa kimliği: 202"):
        partition_dev_test(sample_records, page_style_map=incomplete_map)
