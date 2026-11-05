"""Geniş veri taraması (GÖREV 10) veri toplama ve eşleştirme birim testleri.
"""

from pathlib import Path
import pytest

from src.data_collection.match_latoc_ia import calculate_match_score
from src.data_collection.collect_tr_wiktionary import extract_ottoman_section, clean_wikitext_meaning
from src.data_collection.collect_tr_wikipedia import extract_pairs_from_wikitext
from src.gold.leakage import LeakageChecker, load_heldout_latin


def test_calculate_match_score():
    s1 = "celili leyla vu mecnun"
    s2 = "Celili - Leyla vü Mecnun (Divan Edebiyatı)"
    score = calculate_match_score(s1, s2)
    assert score >= 0.5

    s3 = "tamamen alakasiz bir metin"
    score_low = calculate_match_score(s1, s3)
    assert score_low < 0.3


def test_wikitext_parsing():
    wikitext = """
==Osmanlı Türkçesi==
===Ad===
{{ota-ad|كتاب|tr=kitap}}
# [[okuma|Okumak]] için basılmış nesne.
    """
    sec = extract_ottoman_section(wikitext)
    assert sec != ""
    assert "كتاب" in sec
    assert clean_wikitext_meaning("[[okuma|Okumak]] için basılmış nesne.") == "Okumak için basılmış nesne."


def test_extract_pairs_from_wikitext():
    lead = "'''İstanbul''' (Osmanlıca: استانبول), Türkiye'nin en kalabalık şehridir."
    pairs = extract_pairs_from_wikitext("İstanbul", lead)
    assert len(pairs) >= 1
    assert pairs[0][0] == "استانبول"
    assert pairs[0][1] == "İstanbul"


def test_heldout_latin_leakage():
    checker = LeakageChecker()
    # Eğer heldout_latin.txt mevcutsa Akis setinden bilinen bir cümleyi sızıntı olarak yakalar
    if Path("data/gold/heldout_latin.txt").exists():
        assert checker.is_latin_heldout("Kahve iskemleleri atılmış tarafın mukabilindeki")
    assert not checker.is_latin_heldout("Bu kesinlikle hicbir yerde gecmeyen rastgele 12345 bir cumledir.")
