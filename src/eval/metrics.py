"""Osmanlıca → Türkçe çeviriyazı ve sadeleştirme modelleri için ölçüm ve metrik modülü.

Bu modül:
1. İki değerlendirme modunu uygular:
   - "katı": normalize_latin(metin) — şapkalar, ʿ ʾ ve büyük/küçük harf korunur.
   - "sade": tr_lower(normalize_latin(metin, strip_diacritics=True)) — noktalama silinir
             (harf, rakam, boşluk ve kelime içi tire/kesme kalır); boşluklar tekleşir.
2. Metrikler: CER (jiwer), WER (jiwer), chrF (sacrebleu chrF2), tam_eslesme (birebir eşleşme oranı).
3. Raporlama boyutları: toplam, üsluba göre (birebir / serbest), sayfa makro ortalaması.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import sacrebleu

from src.common.normalize import normalize_latin, tr_lower


def normalize_eval_text(text: str, mode: str = "sade") -> str:
    """Metni belirtilen değerlendirme moduna göre normalize eder.

    Modlar:
      - 'katı': normalize_latin(metin)
      - 'sade': tr_lower(normalize_latin(metin, strip_diacritics=True))
                Noktalama silinir (harf, rakam, boşluk ve kelime içi tire/kesme kalır).
    """
    if not text:
        return ""

    if mode == "katı":
        return normalize_latin(text).strip()

    if mode == "sade":
        # 1. Latin normalizasyonu ve şapka/diyakritik temizliği
        s = normalize_latin(text, strip_diacritics=True)
        # 2. Türkçe uyumlu küçük harfe dönüştürme
        s = tr_lower(s)
        # 3. Harf, rakam, boşluk, tire ve kesme haricindeki tüm karakterleri boşluğa çevir
        s = re.sub(r"[^a-zçğıöşü0-9\s\-\'\’]", " ", s)
        # 4. Kelime içi olmayan (başta, sonda veya tek başına) tire ve kesmeleri temizle
        s = re.sub(r"(?<![a-zçğıöşü0-9])[\-\'\’]+", " ", s)
        s = re.sub(r"[\-\'\’]+(?![a-zçğıöşü0-9])", " ", s)
        # 5. Kesme işaretini standartlaştır (' tek tırnak)
        s = s.replace("’", "'")
        # 6. Fazla boşlukları tekleştir ve kırp
        s = re.sub(r"\s+", " ", s).strip()
        return s

    raise ValueError(f"Bilinmeyen ölçüm modu: {mode}. 'katı' veya 'sade' olmalıdır.")


def compute_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Referans ve tahmin listesi üzerinden CER, WER, chrF ve tam eşleşme oranını hesaplar."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Referans sayısı ({len(references)}) ile tahmin sayısı ({len(hypotheses)}) eşit olmalıdır."
        )

    if not references:
        return {
            "cer": 0.0,
            "wer": 0.0,
            "chrf": 0.0,
            "tam_eslesme": 0.0,
            "adet": 0,
        }

    # Tam eşleşme oranı
    exact_matches = sum(1 for ref, hyp in zip(references, hypotheses) if ref == hyp)
    tam_eslesme = exact_matches / len(references)

    # CER ve WER (jiwer)
    # jiwer hem boş liste hem tekil eleman durumlarını yönetir
    try:
        cer_val = float(jiwer.cer(references, hypotheses))
    except Exception:
        cer_val = 1.0

    try:
        wer_val = float(jiwer.wer(references, hypotheses))
    except Exception:
        wer_val = 1.0

    # chrF2 (sacrebleu)
    try:
        chrf_score = float(
            sacrebleu.corpus_chrf(hypotheses, [references], beta=2).score
        )
    except Exception:
        chrf_score = 0.0

    return {
        "cer": round(cer_val, 4),
        "cer_pct": round(cer_val * 100, 2),
        "wer": round(wer_val, 4),
        "wer_pct": round(wer_val * 100, 2),
        "chrf": round(chrf_score, 2),
        "tam_eslesme": round(tam_eslesme, 4),
        "tam_eslesme_pct": round(tam_eslesme * 100, 2),
        "adet": len(references),
    }


def evaluate_split(
    gold_records: List[Dict[str, Any]],
    predictions: Dict[str, str],
    mode: str = "sade",
) -> Dict[str, Any]:
    """Altın veri seti kayıtları ile sistem tahminlerini verilen modda değerlendirir.

    Raporlama boyutları:
    - toplam: tüm çiftler
    - uslup: birebir ve serbest alt kümeleri
    - sayfa_makro: sayfa başına metriklerin eşit ağırlıklı aritmetik ortalaması
    """
    all_refs: List[str] = []
    all_hyps: List[str] = []

    uslup_refs: Dict[str, List[str]] = defaultdict(list)
    uslup_hyps: Dict[str, List[str]] = defaultdict(list)

    page_refs: Dict[str, List[str]] = defaultdict(list)
    page_hyps: Dict[str, List[str]] = defaultdict(list)

    for item in gold_records:
        rec_id = item["id"]
        ref_raw = item.get("latin", "")
        hyp_raw = predictions.get(rec_id, "")

        ref_norm = normalize_eval_text(ref_raw, mode=mode)
        hyp_norm = normalize_eval_text(hyp_raw, mode=mode)

        all_refs.append(ref_norm)
        all_hyps.append(hyp_norm)

        uslup = item.get("uslup", "diger")
        uslup_refs[uslup].append(ref_norm)
        uslup_hyps[uslup].append(hyp_norm)

        page_key = str(item.get("pageid") or item.get("tr_baslik") or "bilinmiyor")
        page_refs[page_key].append(ref_norm)
        page_hyps[page_key].append(hyp_norm)

    # 1. Toplam metrikler
    toplam_metrics = compute_metrics(all_refs, all_hyps)

    # 2. Üslup kırılımı
    uslup_metrics: Dict[str, Dict[str, float]] = {}
    for u_key in sorted(uslup_refs.keys()):
        uslup_metrics[u_key] = compute_metrics(uslup_refs[u_key], uslup_hyps[u_key])

    # 3. Sayfa makro ortalaması
    page_metrics_list = []
    for p_key, p_refs in page_refs.items():
        p_hyps = page_hyps[p_key]
        p_res = compute_metrics(p_refs, p_hyps)
        page_metrics_list.append(p_res)

    if page_metrics_list:
        macro_cer = sum(m["cer"] for m in page_metrics_list) / len(page_metrics_list)
        macro_wer = sum(m["wer"] for m in page_metrics_list) / len(page_metrics_list)
        macro_chrf = sum(m["chrf"] for m in page_metrics_list) / len(page_metrics_list)
        macro_em = sum(m["tam_eslesme"] for m in page_metrics_list) / len(page_metrics_list)

        sayfa_makro = {
            "cer": round(macro_cer, 4),
            "cer_pct": round(macro_cer * 100, 2),
            "wer": round(macro_wer, 4),
            "wer_pct": round(macro_wer * 100, 2),
            "chrf": round(macro_chrf, 2),
            "tam_eslesme": round(macro_em, 4),
            "tam_eslesme_pct": round(macro_em * 100, 2),
            "sayfa_sayisi": len(page_metrics_list),
        }
    else:
        sayfa_makro = {
            "cer": 0.0,
            "cer_pct": 0.0,
            "wer": 0.0,
            "wer_pct": 0.0,
            "chrf": 0.0,
            "tam_eslesme": 0.0,
            "tam_eslesme_pct": 0.0,
            "sayfa_sayisi": 0,
        }

    return {
        "mod": mode,
        "toplam": toplam_metrics,
        "uslup": uslup_metrics,
        "sayfa_makro": sayfa_makro,
    }


def evaluate_both_modes(
    gold_records: List[Dict[str, Any]],
    predictions: Dict[str, str],
) -> Dict[str, Any]:
    """Hem 'sade' (ana ölçü) hem de 'katı' modlarında tam değerlendirme çalıştırır."""
    return {
        "sade": evaluate_split(gold_records, predictions, mode="sade"),
        "kati": evaluate_split(gold_records, predictions, mode="katı"),
    }
