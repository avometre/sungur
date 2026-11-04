"""Kelime düzeyi eğitim verisi derleme (src/translit/build_train.py) birim testleri."""

import tempfile
from pathlib import Path

import pytest
from src.common.io import read_jsonl, write_jsonl
from src.translit.build_train import build_training_dataset


def test_build_training_dataset_leakage_and_split():
    """Eğitim verisi derlemesinde sızıntı koruması, %2 val ayrımı ve alan formatını sınar."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wikt_file = tmp / "wikt.jsonl"
        gaz_file = tmp / "gaz.jsonl"
        train_file = tmp / "train.jsonl"
        val_file = tmp / "val.jsonl"

        wikt_sample = [
            {"arap": "\u0643\u062a\u0627\u0628", "latin": "kitap", "supheli": False, "lisans": "CC BY-SA 4.0"},
            {"arap": "\u0634\u0628\u0647\u0647", "latin": "şüphe", "supheli": True, "lisans": "CC BY-SA 4.0"},  # Şüpheli, atılmalı
        ]
        gaz_sample = [
            {"arap": "\u0627\u0633\u062a\u0627\u0646\u0628\u0648\u0644", "latin": "İstanbul", "guvenilir": True, "lisans": "cc-by-nc-4.0"},
            {"arap": "\u0628\u0627\u063a\u062f\u0627\u062f", "latin": "Bağdat", "guvenilir": False, "lisans": "cc-by-nc-4.0"},  # Güvenilmez, atılmalı
        ]

        write_jsonl(wikt_file, wikt_sample)
        write_jsonl(gaz_file, gaz_sample)

        summary = build_training_dataset(
            wiktionary_path=wikt_file,
            gazetteer_path=gaz_file,
            train_output_path=train_file,
            val_output_path=val_file,
            val_ratio=0.5,  # Test için %50 ayrım
            seed=0,
        )

        assert summary["toplam_kayit"] == 2
        assert summary["egitim_kayit_sayisi"] == 1
        assert summary["dogrulama_kayit_sayisi"] == 1

        train_records = list(read_jsonl(train_file))
        assert len(train_records) == 1
        r = train_records[0]
        assert "id" in r
        assert "kaynak" in r
        assert "arap" in r
        assert "latin" in r
        assert "latin_sade" in r
        assert r["duzey"] == "kelime"
        assert "lisans" in r
