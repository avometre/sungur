"""Veri sızıntısı koruma modülü (leakage.py) testleri."""

import pytest
from pathlib import Path
from src.gold.leakage import (
    is_title_heldout,
    is_heldout,
    LeakageChecker,
)


def test_is_title_heldout_exact_and_subpage():
    """Tam başlık ve alt sayfa (<başlık>/...) eşleşmelerini sınar."""
    heldout = {"Diyet", "Nutuk/19. bölüm", "Rübâb-ı Şikeste/Sen Olmasan"}

    # Tam eşleşmeler
    assert is_title_heldout("Diyet", heldout) is True
    assert is_title_heldout("Nutuk/19. bölüm", heldout) is True

    # Alt sayfalar
    assert is_title_heldout("Diyet/1. Bölüm", heldout) is True
    assert is_title_heldout("Diyet/Giriş", heldout) is True
    assert is_title_heldout("Nutuk/19. bölüm/Kısım 1", heldout) is True

    # Eşleşmeyenler
    assert is_title_heldout("Diyetisyen", heldout) is False
    assert is_title_heldout("Nutuk", heldout) is False  # Üst sayfa heldout değilse engellenmez
    assert is_title_heldout("Bambaşka Bir Eser", heldout) is False
    assert is_title_heldout("", heldout) is False


def test_is_heldout_with_all_dimensions():
    """Sayfa kimliği, Türkçe başlık, Arapça metin ve Latin cümle denetimlerini sınar."""
    pages = {85358, 98687}
    araps = {"\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627"}
    titles = {"Diyet", "Gizli Mâbed"}
    latins = {"Minimini bir hayvandır", "Karıncayı tanırsınız"}

    # Sayfa kimliği
    assert is_heldout(pageid=85358, heldout_pages=pages) is True
    assert is_heldout(pageid=11111, heldout_pages=pages) is False

    # Başlık ve alt sayfa
    assert is_heldout(tr_title="Diyet", heldout_titles=titles) is True
    assert is_heldout(tr_title="Diyet/Bölüm 2", heldout_titles=titles) is True
    assert is_heldout(tr_title="Farklı Başlık", heldout_titles=titles) is False

    # Arapça metin
    assert is_heldout(arap="\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627", heldout_arap=araps) is True
    assert is_heldout(arap="\u0628\u0627\u0634\u0642\u0627 \u0645\u062a\u0646", heldout_arap=araps) is False

    # Latin cümle
    assert is_heldout(latin="Minimini bir hayvandır", heldout_latin=latins) is True
    assert is_heldout(latin="Farklı bir latin cümle", heldout_latin=latins) is False

    # Hiçbir şey verilmezse
    assert is_heldout() is False


def test_leakage_checker_with_real_gold_data():
    """Gerçek altın veri seti dosyaları üzerinden LeakageChecker entegrasyon testi."""
    checker = LeakageChecker()
    # Altın sette 85358 Diyet sayfası var
    assert checker.is_pageid_heldout(85358) is True
    assert checker.is_title_heldout("Diyet") is True
    assert checker.is_title_heldout("Diyet/Ek") is True
    # Altın setteki bir cümle:
    assert checker.is_latin_heldout("Karıncayı tanırsınız") is True
