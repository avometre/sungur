"""LATOC veri işleme ve cümle dönüştürme birim testleri."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from src.scrapers.latoc import (
    compute_latoc_stats,
    convert_xml_to_sentences,
    filter_sentence_words,
    parse_exclusion_file,
    process_layout_to_xml,
)


def test_filter_sentence_words():
    """Kelime sayısı süzgecini (3 <= kelime <= 80) sınar."""
    # 2 kelime -> elenmeli
    valid, cnt = filter_sentence_words("iki kelime", min_words=3, max_words=80)
    assert valid is False
    assert cnt == 2

    # 3 kelime -> kabul edilmeli
    valid, cnt = filter_sentence_words("tam üç kelime", min_words=3, max_words=80)
    assert valid is True
    assert cnt == 3

    # 80 kelime -> kabul edilmeli
    words_80 = " ".join(["kelime"] * 80)
    valid, cnt = filter_sentence_words(words_80, min_words=3, max_words=80)
    assert valid is True
    assert cnt == 80

    # 81 kelime -> elenmeli
    words_81 = " ".join(["kelime"] * 81)
    valid, cnt = filter_sentence_words(words_81, min_words=3, max_words=80)
    assert valid is False
    assert cnt == 81


def test_parse_exclusion_file():
    """exclude_pages.txt formatındaki sayfa sınırlarının ayrıştırılmasını sınar."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("# Yorum satırı\n")
        f.write("test_doc.xml: 1-3, 5, 8-10\n")
        f.write("empty_doc.xml:\n")
        temp_path = Path(f.name)

    try:
        exclusions = parse_exclusion_file(temp_path)
        assert "test_doc.xml" in exclusions
        assert exclusions["test_doc.xml"] == {1, 2, 3, 5, 8, 9, 10}
        assert "empty_doc.xml" not in exclusions
    finally:
        temp_path.unlink(missing_ok=True)


def test_convert_xml_to_sentences_and_normalization():
    """XML içeriğinden cümle ayıklama, transkripsiyon sadeleştirme ve alan formatlarını sınar."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<document source="ornek.xml" total_pages="1">
  <page number="1">
    <titles>
      <title id="1">D\u00ceV\u00c2N-I H\u00c2FIZ BA\u015eLANGICI</title>
    </titles>
    <body id="1">
      <paragraph id="1">Bu bir deneme c\u00fcmlesidir. \u015e\u00e2ir burada g\u00fczel bir s\u00f6z s\u00f6ylemi\u015ftir!</paragraph>
      <paragraph id="2">K\u0131sa.</paragraph>
    </body>
  </page>
</document>
"""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(xml_content)
        temp_xml = Path(f.name)

    try:
        book_meta = {
            "kitap_id": "b001",
            "baslik": "H\u00e2f\u0131z D\u00eev\u00e2n\u0131",
            "yazar": "H\u00e2f\u0131z",
            "yuzyil": "15",
        }
        records, discarded = convert_xml_to_sentences(temp_xml, book_meta)

        # "Kısa." (1 kelime) elenmeli
        assert discarded >= 1
        assert len(records) >= 2  # Başlık ve 2 cümle

        first = records[0]
        assert first["kaynak"] == "latoc"
        assert first["kitap_id"] == "b001"
        assert first["id"].startswith("latoc:b001:")
        assert first["yuzyil"] == "15"
        assert "latin" in first
        assert "latin_sade" in first
        # latin_sade transkripsiyon işaretlerini sadeleştirmiş olmalı (şapkalı â -> a)
        sentence_record = [r for r in records if "Şâir" in r["latin"] or "şâir" in r["latin"].lower()][0]
        assert "şair" in sentence_record["latin_sade"].lower()
    finally:
        temp_xml.unlink(missing_ok=True)


def test_compute_latoc_stats():
    """Derlem istatistiklerinin doğru hesaplandığını sınar."""
    sample_records = [
        {"latin": "Devlet-i aliyye payidar olsun.", "yuzyil": "16"},
        {"latin": "Adalet ile hükmetmek esastır.", "yuzyil": "16"},
        {"latin": "Kitab-ı aşk nihayetsiz bir deryadır.", "yuzyil": "17"},
    ]
    stats = compute_latoc_stats(sample_records)
    assert stats["toplam_cumle"] == 3
    assert stats["toplam_kelime"] >= 12
    assert stats["yuzyil_dagilimi"]["16"] == 2
    assert stats["yuzyil_dagilimi"]["17"] == 1
    assert len(stats["en_sik_30_kelime"]) > 0
