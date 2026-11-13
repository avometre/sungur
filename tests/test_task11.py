"""Görev 11 birim testleri: içerik tabanlı eşleştirme, kelime verisi v2 ve rt_cer."""

from pathlib import Path

import pytest

from src.common.io import read_jsonl, write_jsonl
from src.data_collection.content_match import (
    CorpusDocument,
    CorpusNgramIndex,
    align_ocr_lines_to_document,
    extract_ngrams,
    match_printed_ocr_corpus,
    tokenize_for_ngram,
)
from src.translit.build_train import build_training_dataset_v2
from src.translit.roundtrip import (
    annotate_roundtrip_cer,
    compute_roundtrip_cer,
    filter_by_roundtrip_cer,
)


def test_ngram_tokenization_and_search():
    words = tokenize_for_ngram("Âlem-i fânîde bir hoş sadâ bıraktık.")
    assert words == ["alem", "fanide", "bir", "hos", "sada", "biraktik"]
    ngs = extract_ngrams(words, n=3)
    assert ("alem", "fanide", "bir") in ngs

    doc1 = CorpusDocument(
        doc_id="vk:1",
        kaynak="vikikaynak",
        baslik="Sefiller",
        satirlar=[
            "Âlem-i fânîde bir hoş sadâ bıraktık.",
            "İkinci satır burada devam eder.",
        ],
    )
    doc2 = CorpusDocument(
        doc_id="latoc:2",
        kaynak="latoc",
        baslik="Kanunname",
        satirlar=[
            "Devlet-i aliyye kanunları gereğince hüküm olunur.",
        ],
    )

    idx = CorpusNgramIndex(n=3)
    idx.add_document(doc1)
    idx.add_document(doc2)
    idx.finalize()

    hits = idx.search(["alem i fanide bir hos sada biraktik"], top_k=2, min_hits=2)
    assert len(hits) == 1
    assert hits[0][0].doc_id == "vk:1"


def test_align_ocr_lines_and_match_corpus(tmp_path: Path):
    doc = CorpusDocument(
        doc_id="vk:10",
        kaynak="vikikaynak",
        baslik="Deneme Eser",
        satirlar=[
            "Osmanlı Devleti büyük bir imparatorluk idi.",
            "İstanbul şehrinde çok güzel camiler vardır.",
        ],
    )
    other_doc = CorpusDocument(
        doc_id="latoc:99",
        kaynak="latoc",
        baslik="Başka Eser",
        satirlar=["Tamamen farklı bir metin burada yer alır."],
    )

    printed = [
        {"kitap": "kitap_a", "sayfa": "p1.jpg", "metin": "عثمانلی دولتی بویوك بر ایمپراطورلق ایدی"},
        {"kitap": "kitap_a", "sayfa": "p1.jpg", "metin": "استانبول شهرنده چوق كوزل جامعلر واردر"},
    ]

    def fake_s1b(texts):
        return [
            "osmanli devleti buyuk bir imparatorluk idi",
            "istanbul sehrinde cok guzel camiler vardir",
        ]

    out_path = tmp_path / "ocr_pairs.jsonl"
    summary = match_printed_ocr_corpus(
        printed_records=printed,
        corpus_docs=[doc, other_doc],
        transliterate_fn=fake_s1b,
        ngram_size=3,
        min_ngram_hits=2,
        min_skeleton=0.75,
        output_path=out_path,
    )

    assert summary["eslesen_kitap"] == 1
    assert summary["toplam_cift"] == 2
    saved = list(read_jsonl(out_path))
    assert len(saved) == 2
    assert all(r["iskelet_benzerlik"] >= 0.75 for r in saved)
    assert saved[0]["eslesen_doc_id"] == "vk:10"


def test_roundtrip_cer_and_filtering():
    assert compute_roundtrip_cer("Hâkime dik bir sesle:", "hakime dik bir sesle") == 0.0
    assert compute_roundtrip_cer("", "") == 0.0
    assert compute_roundtrip_cer("merhaba", "") == 1.0

    recs = [
        {"id": "1", "arap": "مرحبا", "latin": "merhaba"},
        {"id": "2", "arap": "دولت", "latin": "devlet"},
    ]

    def fake_forward(texts):
        return ["merhaba", "tamamen yanlis"]

    annotated = annotate_roundtrip_cer(recs, fake_forward)
    assert annotated[0]["rt_cer"] == 0.0
    assert annotated[1]["rt_cer"] > 0.5

    filtered = filter_by_roundtrip_cer(annotated, max_rt_cer=0.25)
    assert [r["id"] for r in filtered] == ["1"]


def test_build_training_dataset_v2(tmp_path: Path):
    wikt = tmp_path / "wikt.jsonl"
    gaz = tmp_path / "gaz.jsonl"
    tr_wikt = tmp_path / "tr_wikt.jsonl"
    wikidata = tmp_path / "wikidata.jsonl"
    tr_wiki = tmp_path / "tr_wiki.jsonl"

    write_jsonl(wikt, [{"arap": "كتاب", "latin": "kitâb", "supheli": False}])
    write_jsonl(gaz, [{"arap": "ادرنه", "latin": "edirne", "guvenilir": True}])
    write_jsonl(tr_wikt, [{"arap": "قلم", "latin": "kalem", "kaynak": "tr_wiktionary", "lisans": "CC BY-SA 4.0"}])
    write_jsonl(wikidata, [{"arap": "سلطان", "latin": "sultan", "kaynak": "wikidata", "lisans": "CC0 1.0"}])
    # Tekrar eden çift tekilleştirilmeli
    write_jsonl(tr_wiki, [
        {"arap": "قلم", "latin": "kalem", "kaynak": "tr_wikipedia", "lisans": "CC BY-SA 4.0"},
        {"arap": "مكتب", "latin": "mekteb", "kaynak": "tr_wikipedia", "lisans": "CC BY-SA 4.0"},
    ])

    train_out = tmp_path / "train_v2.jsonl"
    val_out = tmp_path / "val_v2.jsonl"

    summary = build_training_dataset_v2(
        wiktionary_path=wikt,
        gazetteer_path=gaz,
        tr_wiktionary_path=tr_wikt,
        wikidata_path=wikidata,
        tr_wikipedia_path=tr_wiki,
        train_output_path=train_out,
        val_output_path=val_out,
        val_ratio=0.2,
        seed=0,
    )

    assert summary["toplam_kayit"] == 5
    assert summary["egitim_kayit_sayisi"] + summary["dogrulama_kayit_sayisi"] == 5
    assert "tr_wiktionary" in summary["kaynak_dagilimi"]
    assert "wikidata" in summary["kaynak_dagilimi"]
    assert "tr_wikipedia" in summary["kaynak_dagilimi"]
