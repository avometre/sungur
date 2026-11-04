"""Ölçüm ve metrik modülü (src/eval/metrics.py) birim testleri."""

import pytest
from src.eval.metrics import (
    compute_metrics,
    evaluate_both_modes,
    evaluate_split,
    normalize_eval_text,
)


def test_normalize_eval_text_sade():
    """'sade' modunda diyakritik temizliği, noktalama ayıklama ve kelime içi tire/kesme korunmasını sınar."""
    # Verilen örnekler
    assert normalize_eval_text("Hâkime dik bir sesle:", mode="sade") == "hakime dik bir sesle"
    assert normalize_eval_text("Zülfikâr'in sırrı ondadır!..", mode="sade") == "zülfikar'in sırrı ondadır"

    # Kelime içi tire ve kesme korunmalı, başta/sonda silinmeli
    assert normalize_eval_text("- Devlet-i aliyye -", mode="sade") == "devlet-i aliyye"
    assert normalize_eval_text("İstanbul'un fethi...", mode="sade") == "istanbul'un fethi"

    # Şapkalı harfler ve transkripsiyon diyakritikleri temizlenmeli (Türkçe ı/i korunmalı)
    assert normalize_eval_text("ṣulṭān-ı ʿālem", mode="sade") == "sultan-ı alem"


def test_normalize_eval_text_kati():
    """'katı' modunda diyakritiklerin, büyük/küçük harflerin ve transkripsiyon harflerinin korunduğunu sınar."""
    sample = "Hâkime dik bir sesle ʿālem"
    res = normalize_eval_text(sample, mode="katı")
    assert "Hâkime" in res
    assert "ʿālem" in res


def test_compute_metrics_known_values():
    """Bilinen küçük örneklerde CER, WER, chrF ve tam eşleşme hesaplarını sınar."""
    # Birebir aynı cümleler
    refs = ["merhaba dünya", "osmanlıca çeviriyazı"]
    hyps = ["merhaba dünya", "osmanlıca çeviriyazı"]
    res = compute_metrics(refs, hyps)
    assert res["cer"] == 0.0
    assert res["wer"] == 0.0
    assert res["tam_eslesme"] == 1.0
    assert res["chrf"] == 100.0

    # Tek karakter farkı (merhaba -> merhabe)
    # 1 harf değiştirildi, toplam ref uzunluğu 7, CER = 1/7
    refs_single = ["merhaba"]
    hyps_single = ["merhabe"]
    res_single = compute_metrics(refs_single, hyps_single)
    assert res_single["cer"] == pytest.approx(1 / 7, abs=1e-3)
    assert res_single["wer"] == 1.0
    assert res_single["tam_eslesme"] == 0.0
    assert res_single["chrf"] < 100.0


def test_evaluate_split_and_macro_average():
    """Üslup kırılımı ve sayfa makro ortalamasının doğru hesaplandığını sınar."""
    gold_records = [
        {"id": "1", "latin": "cümle bir", "uslup": "birebir", "pageid": "p1"},
        {"id": "2", "latin": "cümle iki", "uslup": "birebir", "pageid": "p1"},
        {"id": "3", "latin": "cümle üç", "uslup": "serbest", "pageid": "p2"},
    ]
    predictions = {
        "1": "cümle bir",
        "2": "cümle iki",
        "3": "farklı metin",
    }

    res = evaluate_split(gold_records, predictions, mode="sade")
    assert "toplam" in res
    assert "uslup" in res
    assert "sayfa_makro" in res

    # p1'de 2'de 2 tam eşleşme (tam_eslesme=1.0), p2'de 1'de 0 (tam_eslesme=0.0)
    # Sayfa makro tam eşleşme ortalaması: (1.0 + 0.0) / 2 = 0.5
    assert res["sayfa_makro"]["tam_eslesme"] == 0.5
    assert res["sayfa_makro"]["sayfa_sayisi"] == 2

    # Üslup kırılımı
    assert res["uslup"]["birebir"]["tam_eslesme"] == 1.0
    assert res["uslup"]["serbest"]["tam_eslesme"] == 0.0


def test_evaluate_both_modes():
    """Hem 'sade' hem 'katı' modlarını aynı anda çalıştıran fonksiyonu sınar."""
    gold_records = [{"id": "1", "latin": "Âlem-i fâni", "uslup": "birebir", "pageid": "p1"}]
    predictions = {"1": "Alem-i fani"}

    res = evaluate_both_modes(gold_records, predictions)
    assert "sade" in res
    assert "kati" in res
    # Sade modda şapkalar kaldırıldığı için tam eşleşmeli
    assert res["sade"]["toplam"]["tam_eslesme"] == 1.0
    # Katı modda şapkalar korunur (Â vs A), bu yüzden CER > 0
    assert res["kati"]["toplam"]["cer"] > 0.0
