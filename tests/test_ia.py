"""Internet Archive kataloğu, kamu malı süzgeci ve metaveri zenginleştirme birim testleri."""

import pytest
from src.scrapers.ia import (
    compute_catalog_statistics,
    enrich_item,
    is_public_domain,
    parse_year,
    select_trial_candidates,
)


def test_parse_year():
    """Yıl ve tarih alanlarından sayısal yıl çıkarma mantığını sınar."""
    # Doğrudan yıl alanı
    y, unknown = parse_year(1890, None)
    assert y == 1890
    assert unknown is False

    # Metinsel ISO tarihi
    y, unknown = parse_year(None, "1890-01-01T00:00:00Z")
    assert y == 1890
    assert unknown is False

    # Düzensiz metin içinde yıl
    y, unknown = parse_year(None, "Basım yılı 1894 civarı")
    assert y == 1894
    assert unknown is False

    # Bilinmeyen / geçersiz yıl
    y, unknown = parse_year(None, "tarihsiz")
    assert y is None
    assert unknown is True


def test_is_public_domain_filter():
    """Kamu malı süzgecini (1928 öncesi ve lisans kuralı) sınar."""
    # 1890 basımı, lisans boş -> Kamu malı
    assert is_public_domain(year_val=1890, yil_bilinmiyor=False, licenseurl="", rights="") is True

    # 1928 basımı, CC lisansı -> Kamu malı
    assert (
        is_public_domain(
            year_val=1928,
            yil_bilinmiyor=False,
            licenseurl="https://creativecommons.org/publicdomain/mark/1.0/",
            rights="",
        )
        is True
    )

    # 1929 basımı (Harf İnkılabı sonrası) -> Kamu malı değil
    assert is_public_domain(year_val=1929, yil_bilinmiyor=False, licenseurl="", rights="") is False

    # 1900 basımı fakat "All rights reserved" telif uyarısı -> Kamu malı değil
    assert (
        is_public_domain(
            year_val=1900,
            yil_bilinmiyor=False,
            licenseurl="",
            rights="All Rights Reserved",
        )
        is False
    )

    # Yılı bilinmeyen eserler kamu malı olarak varsayılmaz (ayrı işaretlenir)
    assert is_public_domain(year_val=None, yil_bilinmiyor=True, licenseurl="", rights="") is False


def test_enrich_item_schema():
    """Ham arama kaydının dosya formatları ve boyut bilgisiyle zenginleştirilmesini sınar."""
    doc = {
        "identifier": "ornek_kitap_1890",
        "title": "D\u00eev\u00e2n-i Osm\u00e2n\u00ee",
        "creator": "M\u00fcellif",
        "date": "1890-01-01T00:00:00Z",
        "year": 1890,
        "language": "ota",
        "licenseurl": "",
        "rights": "",
        "imagecount": 150,
        "collection": ["toronto", "robarts"],
        "format": [
            "Text PDF",
            "Single Page Processed JP2 ZIP",
            "DjVuTXT",
            "Metadata",
        ],
        "item_size": 104857600,  # 100 MB
    }

    enriched = enrich_item(doc)
    assert enriched["identifier"] == "ornek_kitap_1890"
    assert enriched["year"] == 1890
    assert enriched["yil_bilinmiyor"] is False
    assert enriched["kamu_mali"] is True
    assert enriched["has_pdf"] is True
    assert enriched["has_jp2_zip"] is True
    assert enriched["has_djvu_txt"] is True
    assert enriched["toplam_boyut_mb"] == 100.0


def test_select_trial_candidates():
    """Farklı koleksiyonlardan deneme adaylarının seçilmesini sınar."""
    catalog = [
        {
            "identifier": "item1",
            "title": "Kitap 1",
            "kamu_mali": True,
            "has_jp2_zip": True,
            "has_pdf": True,
            "imagecount": 20,
            "collection": ["colA"],
        },
        {
            "identifier": "item2",
            "title": "Kitap 2",
            "kamu_mali": True,
            "has_jp2_zip": True,
            "has_pdf": True,
            "imagecount": 30,
            "collection": ["colA"],  # Aynı koleksiyon
        },
        {
            "identifier": "item3",
            "title": "Kitap 3",
            "kamu_mali": True,
            "has_jp2_zip": True,
            "has_pdf": True,
            "imagecount": 40,
            "collection": ["colB"],  # Farklı koleksiyon
        },
    ]

    selected = select_trial_candidates(catalog, count=2)
    assert len(selected) == 2
    # Farklı koleksiyonlardan tercih edilmeli: colA ve colB
    colls = {s["collection"][0] for s in selected}
    assert colls == {"colA", "colB"}


def test_compute_catalog_statistics():
    """Katalog özet istatistiklerinin hesaplanmasını sınar."""
    catalog = [
        {
            "year": 1892,
            "kamu_mali": True,
            "yil_bilinmiyor": False,
            "toplam_boyut_mb": 50.0,
            "collection": ["col1"],
        },
        {
            "year": 1915,
            "kamu_mali": True,
            "yil_bilinmiyor": False,
            "toplam_boyut_mb": 150.0,
            "collection": ["col1", "col2"],
        },
        {
            "year": None,
            "kamu_mali": False,
            "yil_bilinmiyor": True,
            "toplam_boyut_mb": 20.0,
            "collection": ["col2"],
        },
    ]

    stats = compute_catalog_statistics(catalog)
    assert stats["toplam_oge"] == 3
    assert stats["kamu_mali_sayisi"] == 2
    assert stats["yil_bilinmiyor_sayisi"] == 1
    assert stats["toplam_boyut_gb"] == round(220.0 / 1024, 2)
    # On yıllık dağılım
    decade_dict = dict(stats["on_yillik_dagilim"])
    assert decade_dict["1890ler"] == 1
    assert decade_dict["1910ler"] == 1
    # Koleksiyonlar
    top_colls = dict(stats["top_10_koleksiyon"])
    assert top_colls["col1"] == 2
    assert top_colls["col2"] == 2
