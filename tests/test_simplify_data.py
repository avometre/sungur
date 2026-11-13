"""Görev 14 birim testleri: sadeleştirme veri seti derleyicisi."""

from pathlib import Path

from src.common.io import read_jsonl, write_jsonl
from src.simplify.build_simplify_data import build_simplify_dataset, clean_gloss


def test_clean_gloss():
    assert clean_gloss("  1. su.   2. deniz. ") == "1. su. 2. deniz."
    assert clean_gloss("") == ""


def test_build_simplify_dataset(tmp_path: Path):
    sade = tmp_path / "sade_sozluk.jsonl"
    tr_wikt = tmp_path / "tr_wikt.jsonl"
    wikt = tmp_path / "wikt.jsonl"
    vk_sade = tmp_path / "vk_sade.jsonl"

    write_jsonl(sade, [
        {"kelime_latin": "âb", "anlam_tr": "1. su. 2. deniz."},
        {"kelime_latin": "su", "anlam_tr": "su"},  # aynı olduğu için elenmeli
    ])
    write_jsonl(tr_wikt, [
        {"latin": "müşkil", "anlam_tr": "zor, güç"},
    ])
    write_jsonl(wikt, [
        {"latin": "cihad", "turkce": ["cihat"], "supheli": False},
        {"latin": "aya", "turkce": ["aya"], "supheli": True},
    ])
    write_jsonl(vk_sade, [
        {"osmanlica_latin": "Bu mes'ele gâyet müşkildir.", "turkce_sade": "Bu mesele çok zordur."},
    ])

    train_out = tmp_path / "simp_train.jsonl"
    val_out = tmp_path / "simp_val.jsonl"

    summary = build_simplify_dataset(
        sade_sozluk_path=sade,
        tr_wiktionary_path=tr_wikt,
        wiktionary_path=wikt,
        vk_sade_pairs_path=vk_sade,
        train_output_path=train_out,
        val_output_path=val_out,
        val_ratio=0.25,
        seed=0,
    )

    assert summary["toplam_kayit"] == 4
    assert summary["egitim_kayit_sayisi"] == 3
    assert summary["dogrulama_kayit_sayisi"] == 1
    assert summary["duzey_dagilimi"] == {"kelime": 3, "cumle": 1}
    assert len(list(read_jsonl(train_out))) == 3
    assert len(list(read_jsonl(val_out))) == 1
