"""Vikikaynak Latin derlemi (vikikaynak_latin.py) modülü testleri."""

import pytest
from src.scrapers.vikikaynak_latin import (
    clean_wikitext_content,
    is_translation,
    is_target_period,
    extract_metadata,
)


def test_clean_wikitext_content_poem_and_prose():
    """<poem> içeriğinin satır satır korunmasını ve şablon/bağlantı temizliğini sınar."""
    wikitext = """
{{eser başlığı|başlık=Deneme}}
[[Dosya:Ornek.jpg|küçük|Resim yazısı]]
Bu bir düzyazı giriş paragrafıdır. Burası ikinci cümledir.<ref>Dipnot bilgisi</ref>
<poem>
Birinci mısra buradadır,
İkinci mısra devam eder!
</poem>
{| class="wikitable"
|-
| Tablo içi
|}
Son paragraf metni.
"""
    cleaned = clean_wikitext_content(wikitext)
    assert len(cleaned) >= 4

    kinds = [k for k, _ in cleaned]
    texts = [t for _, t in cleaned]

    assert "poem" in kinds
    assert "prose" in kinds

    # Poem satırları bölünmeden korunmalı
    assert "Birinci mısra buradadır," in texts
    assert "İkinci mısra devam eder!" in texts

    # Resim ve tablo temizlenmeli
    for t in texts:
        assert "[[Dosya:" not in t
        assert "wikitable" not in t
        assert "Dipnot bilgisi" not in t


def test_is_translation_detection():
    """Çevirmen parametresi veya çeviri kategorisi içeren sayfaların tespitini sınar."""
    # Çevirmen parametresi olan
    txt_with_trans = "{{eser|başlık=Falanca|çevirmen=Ahmet Mithat Efendi}}"
    assert is_translation(txt_with_trans, ["Kategori:Eserler"]) is True

    # Çeviri kategorisi olan
    txt_no_trans = "{{eser|başlık=Falanca}}"
    assert is_translation(txt_no_trans, ["Kategori:Fransızcadan çeviriler"]) is True
    assert is_translation(txt_no_trans, ["Kategori:Tercüme eserler"]) is True

    # Özgün Türkçe eser
    assert is_translation(txt_no_trans, ["Kategori:Ömer Seyfettin hikayeleri"]) is False


def test_extract_metadata_author_and_year():
    """Wikitext şablonundan yazar ve yıl ayıklamayı sınar."""
    wikitext = "{{eser başlığı|başlık=Diyet|eser sahibi=[[Ömer Seyfettin]]|yıl=1919}}"
    author, year = extract_metadata(wikitext)
    assert author == "Ömer Seyfettin"
    assert year == 1919


def test_is_target_period_rules():
    """1928 öncesi Osmanlı dönemi seçim kurallarını sınar."""
    author_deaths = {"ömer seyfettin": 1920, "namık kemal": 1888}

    # 1. Eser yılı <= 1928
    is_target, reason = is_target_period(
        wikitext="",
        categories=["Kategori:Genel"],
        author="Bilinmeyen Yazar",
        year=1924,
        author_deaths=author_deaths,
    )
    assert is_target is True
    assert "eser_yili" in reason

    # 2. Yazar ölüm yılı <= 1950 (bilinen yazar)
    is_target, reason = is_target_period(
        wikitext="",
        categories=["Kategori:Hikâyeler"],
        author="Ömer Seyfettin",
        year=None,
        author_deaths=author_deaths,
    )
    assert is_target is True

    # 3. Dönem kategorisi (Divan, Osmanlı, Tanzimat vb.)
    is_target, reason = is_target_period(
        wikitext="",
        categories=["Kategori:Divan şiiri"],
        author="Bilinmeyen Şair",
        year=None,
        author_deaths=author_deaths,
    )
    assert is_target is True
    assert "donem_kategorisi" in reason

    # 4. Yabancı yazar elenmeli
    is_target, reason = is_target_period(
        wikitext="",
        categories=["Kategori:Şiirler"],
        author="William Shakespeare",
        year=1600,
        author_deaths=author_deaths,
    )
    assert is_target is False
    assert reason == "yabanci_yazar"

    # 5. Modern cumhuriyet yazarı elenmeli (Orhan Veli, Âşık Veysel)
    is_target, reason = is_target_period(
        wikitext="",
        categories=["Kategori:Cumhuriyet dönemi şiirleri"],
        author="Orhan Veli Kanık",
        year=1940,
        author_deaths=author_deaths,
    )
    assert is_target is False
