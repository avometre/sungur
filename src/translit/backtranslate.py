"""Kelime kelime geri çeviri modülü (backtranslation: Latin cümle -> Arap cümle).

Bu modül (GÖREV 07 ve 07b güncellemeleriyle):
1. Sık kelime sözlüğü (src/translit/resources/sik_kelimeler.tsv):
   - Arama sırası: sik_kelimeler.tsv -> mevcut ters sözlük -> ters model.
   - Anahtar: tr_lower(normalize_latin(kelime, strip_diacritics=True)).
2. Özel kuralları uygular:
   - İzafet: "X-i", "X-ı", "X-u", "X-ü" -> Arap'ta ek yazılmaz (sadece X).
     "X-yi", "X-yı": X'in Arap yazımı ه ile bitiyorsa son ه yerine ۀ (U+06C0); değilse X + ی.
   - Tek başına "ü" / "vü" / "u" -> و
   - Rakamlar ve yıllar -> olduğu gibi
   - Tireli bileşikler ("pür-cefâ") -> parçaları ayrı çevirip boşlukla birleştirir.
3. Kalan kelimeler için:
   - Önce sık kelimeler sözlüğü
   - Yoksa ters sözlük (translit_train_kelime + morph, latin_sade anahtarlı, en sık arap).
   - Yoksa sungur-ters-small-v1 (beam 4, num_return_sequences=4):
     * max_new_tokens = 3 * (girdi bayt uzunluğu) + 8
     * Çıktı filtresi: Arap betiği + ZWNJ + rakam; 4+ tekrar yok; Arap harf sayısı 0.3x - 2x+2 aralığında.
     * Geçersizse beam 4'ün diğer adayları sırayla denenir; hepsi geçersizse "çevrilemedi" sayılır.
4. "Çevrilemedi" kelime içeren cümleler sentetik derlemden elenir.
5. Altın DEV setinde BT_ters_v2 olarak değerlendirir (data/eval/BT_ters_v2/dev_skor.json).
6. Vikikaynak ve LATOC derlemlerini çevirip data/processed/translit_sentetik_{vk,latoc}.jsonl yazar.
"""

import argparse
import functools
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import jiwer
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_latin, normalize_ottoman, tr_lower
from src.gold.leakage import is_heldout, load_heldout_all

print = functools.partial(print, flush=True)

logger = logging.getLogger(__name__)

DEFAULT_SIK_KELIMELER = Path("src/translit/resources/sik_kelimeler.tsv")
DEFAULT_TRAIN_KELIME = Path("data/processed/translit_train_kelime.jsonl")
DEFAULT_TRAIN_MORPH = Path("data/processed/translit_train_morph.jsonl")
DEFAULT_DEV_GOLD = Path("data/gold/translit_dev.jsonl")
DEFAULT_TERS_MODEL = Path("models/sungur-ters-small-v1")

DEFAULT_VK_INPUT = Path("data/interim/vikikaynak_latin.jsonl")
DEFAULT_LATOC_INPUT = Path("data/interim/latoc_sentences.jsonl")
DEFAULT_VK_OUTPUT = Path("data/processed/translit_sentetik_vk.jsonl")
DEFAULT_LATOC_OUTPUT = Path("data/processed/translit_sentetik_latoc.jsonl")


def load_sik_kelimeler(tsv_path: Path = DEFAULT_SIK_KELIMELER) -> Dict[str, str]:
    """sik_kelimeler.tsv dosyasını okuyup 'latin_sade -> arap' sözlüğü döndürür."""
    mapping: Dict[str, str] = {}
    if tsv_path.exists():
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    lat_key = tr_lower(normalize_latin(parts[0], strip_diacritics=True).strip())
                    ar_val = normalize_ottoman(parts[1]).strip()
                    if lat_key and ar_val:
                        mapping[lat_key] = ar_val
    return mapping


def is_valid_arabic_prediction(pred: str, latin_word: str) -> bool:
    """Ters model çıktısının geçerliliğini sınar.

    Kurallar:
    1. Boş olamaz.
    2. Çıktıda Arap betiği harfi, ZWNJ (\u200c) ve ASCII rakam dışında karakter varsa -> GEÇERSİZ
    3. Aynı karakterin 4+ kez art arda tekrarı (ör. '....', 'ییییی') -> GEÇERSİZ
    4. Arap harf sayısı > 2 × (Latin harf sayısı) + 2 ya da < 0.3 × (Latin harf sayısı) -> GEÇERSİZ
    """
    if not pred or not pred.strip():
        return False

    pred_clean = pred.strip()

    # Kural 2: Çıktıda Arap betiği harfi, ZWNJ ve ASCII rakam dışında karakter varsa -> GEÇERSİZ
    for c in pred_clean:
        if c == "\u200c":
            continue
        if "0" <= c <= "9":
            continue
        cp = ord(c)
        if (
            (0x0600 <= cp <= 0x06FF)
            or (0x0750 <= cp <= 0x077F)
            or (0x08A0 <= cp <= 0x08FF)
            or (0xFB50 <= cp <= 0xFDFF)
            or (0xFE70 <= cp <= 0xFEFF)
        ):
            continue
        return False

    # Kural 3: Aynı karakterin 4+ kez art arda tekrarı -> GEÇERSİZ
    if re.search(r"(.)\1{3,}", pred_clean):
        return False

    # Kural 4: Arap harf sayısı > 2 × (Latin harf sayısı) + 2 ya da < 0.3 × (Latin harf sayısı) -> GEÇERSİZ
    latin_letters = sum(1 for c in latin_word if c.isalpha())
    if latin_letters == 0:
        return True

    # Arapça harf sayısı (işaretler, ZWNJ ve rakamlar hariç)
    arap_letters = sum(
        1 for c in pred_clean
        if c != "\u200c" and not c.isdigit() and not (0x064B <= ord(c) <= 0x065F or ord(c) == 0x0670)
    )

    max_allowed = 2 * latin_letters + 2
    min_allowed = 0.3 * latin_letters

    if arap_letters > max_allowed or arap_letters < min_allowed:
        return False

    return True


class ReverseDictionary:
    """translit_train_kelime ve morph verisinden en sık Arapça karşılığı veren ters sözlük."""

    def __init__(
        self,
        kelime_path: Path = DEFAULT_TRAIN_KELIME,
        morph_path: Path = DEFAULT_TRAIN_MORPH,
    ) -> None:
        self.dict_map: Dict[str, str] = {}
        self._load(kelime_path, morph_path)

    def _load(self, kelime_path: Path, morph_path: Path) -> None:
        freqs: Dict[str, Counter] = defaultdict(Counter)

        for p in [kelime_path, morph_path]:
            if p.exists():
                for r in read_jsonl(p):
                    ar = normalize_ottoman(r.get("arap", "")).strip()
                    lat = r.get("latin", "").strip()
                    lat_sade = r.get("latin_sade") or normalize_latin(lat, strip_diacritics=True).strip()
                    clean_key = tr_lower(lat_sade)

                    if clean_key and ar:
                        freqs[clean_key][ar] += 1

        for key, c_counter in freqs.items():
            best_ar = c_counter.most_common(1)[0][0]
            self.dict_map[key] = best_ar

    def get(self, word: str) -> Optional[str]:
        clean_key = tr_lower(normalize_latin(word, strip_diacritics=True).strip())
        return self.dict_map.get(clean_key)


class BackTransliterator:
    """Latin cümleleri sık kelimeler, ters sözlük ve ByT5 ters modeli ile filtre uygulayarak çevirir."""

    def __init__(
        self,
        model_path: Optional[Path] = DEFAULT_TERS_MODEL,
        device: Optional[torch.device] = None,
        reverse_dict: Optional[ReverseDictionary] = None,
        sik_kelimeler_path: Path = DEFAULT_SIK_KELIMELER,
    ) -> None:
        self.sik_kelimeler = load_sik_kelimeler(sik_kelimeler_path)
        if reverse_dict is None:
            self.reverse_dict = ReverseDictionary()
        else:
            self.reverse_dict = reverse_dict

        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.word_cache: Dict[str, Optional[str]] = {}
        self.invalid_words: Set[str] = set()

    def _load_model(self) -> None:
        if self.model is None and self.model_path and self.model_path.exists():
            print(f"Ters model yükleniyor: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.model.eval()

    def translate_words_batch(self, words: List[str], beam_size: int = 4) -> Dict[str, Optional[str]]:
        """Ters model ile kelimeleri toplu çevirir, adayları filtreler ve önbelleğe yazar."""
        to_translate = []
        for w in words:
            clean_w = w.strip()
            if not clean_w:
                continue
            if clean_w not in self.word_cache:
                to_translate.append(clean_w)

        if not to_translate:
            return self.word_cache

        self._load_model()
        if self.model is None:
            for w in to_translate:
                self.word_cache[w] = None
                self.invalid_words.add(w)
            return self.word_cache

        batch_size = 64
        for i in range(0, len(to_translate), batch_size):
            chunk = to_translate[i : i + batch_size]
            max_new = 3 * max(len(w.encode("utf-8")) for w in chunk) + 8

            inputs = self.tokenizer(
                chunk,
                max_length=64,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_new,
                        num_beams=beam_size,
                        num_return_sequences=beam_size,
                        do_sample=False,
                    )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for w_idx, w_orig in enumerate(chunk):
                candidates = decoded[w_idx * beam_size : (w_idx + 1) * beam_size]
                chosen = None
                for cand in candidates:
                    norm_cand = normalize_ottoman(cand).strip()
                    if is_valid_arabic_prediction(norm_cand, w_orig):
                        chosen = norm_cand
                        break

                if chosen is not None:
                    self.word_cache[w_orig] = chosen
                else:
                    self.word_cache[w_orig] = None
                    self.invalid_words.add(w_orig)

        return self.word_cache

    def translate_single_word(self, word: str) -> Optional[str]:
        """Tek bir kelimeyi hiyerarşi sırasıyla (sık kelimeler -> sözlük -> model) çevirir."""
        clean_w = word.strip()
        if not clean_w:
            return ""

        clean_key = tr_lower(normalize_latin(clean_w, strip_diacritics=True).strip())

        # 1. Sık kelimeler kontrolü
        sik_hit = self.sik_kelimeler.get(clean_key)
        if sik_hit:
            return sik_hit

        # 2. Ters sözlük kontrolü
        dict_hit = self.reverse_dict.get(clean_key)
        if dict_hit:
            return dict_hit

        # 3. Önbellek kontrolü
        if clean_w in self.word_cache:
            return self.word_cache[clean_w]

        # 4. Model ile çevir
        self.translate_words_batch([clean_w])
        return self.word_cache.get(clean_w)

    def translate_token(self, token: str) -> Optional[str]:
        """Özel kuralları (izafet, bağlaç, rakam, tireli bileşik) uygulayarak belirteci çevirir."""
        t = token.strip()
        if not t:
            return ""

        # 1. Rakamlar ve sayılar -> olduğu gibi
        if re.match(r"^\d+$", t):
            return t

        # 2. Tek başına bağlaç: "ü", "vü", "u" -> و
        if t.lower() in {"ü", "vü", "u"}:
            return "و"

        # 3. İzafet Kuralları:
        # (a) "-yi", "-yı", "-yu", "-yü":
        m_yi = re.match(r"^(.+)-(yi|yı|yu|yü)$", t, re.IGNORECASE)
        if m_yi:
            stem = m_yi.group(1)
            stem_ar = self.translate_single_word(stem)
            if not stem_ar:
                return None
            if stem_ar.endswith("ه"):
                # Son ه yerine ۀ (U+06C0)
                return stem_ar[:-1] + "\u06C0"
            else:
                return stem_ar + "ی"

        # (b) "-i", "-ı", "-u", "-ü":
        m_i = re.match(r"^(.+)-(i|ı|u|ü)$", t, re.IGNORECASE)
        if m_i:
            stem = m_i.group(1)
            stem_ar = self.translate_single_word(stem)
            if not stem_ar:
                return None
            return stem_ar

        # 4. Tireli bileşikler ("pür-cefâ", "melek-sîmâ")
        if "-" in t and not t.startswith("-") and not t.endswith("-"):
            parts = t.split("-")
            trans_parts = []
            for p in parts:
                p_ar = self.translate_single_word(p)
                if not p_ar:
                    return None
                trans_parts.append(p_ar)
            return " ".join(trans_parts)

        # 5. Normal kelime çevirisi
        return self.translate_single_word(t)

    def translate_sentence(self, sentence: str) -> Tuple[str, bool]:
        """Latin cümleyi belirteçlerine ayırıp çevirir.

        Returns:
            (pred_ar, is_valid): pred_ar üretilen metindir.
            is_valid ise cümlede çevrilemeyen kelime olup olmadığını belirtir.
        """
        if not sentence:
            return "", False

        tokens = re.split(r"(\s+|[.,!?:;\"'«»()\[\]—–])", sentence)
        result = []
        is_all_valid = True

        for tok in tokens:
            if not tok:
                continue
            if re.match(r"^(\s+|[.,!?:;\"'«»()\[\]—–])$", tok):
                result.append(tok)
            else:
                ar_trans = self.translate_token(tok)
                if ar_trans is None:
                    is_all_valid = False
                    ar_trans = ""
                result.append(ar_trans)

        full_ar = normalize_ottoman("".join(result)).strip()
        return full_ar, is_all_valid

    def precache_sentences(self, sentences: List[str]) -> None:
        """Verilen tüm cümlelerdeki bilinmeyen kelimeleri toplayıp tek seferde GPU'da çevirir."""
        needed_words: Set[str] = set()
        for sent in sentences:
            if not sent:
                continue
            tokens = re.split(r"(\s+|[.,!?:;\"'«»()\[\]—–])", sent)
            for tok in tokens:
                t = tok.strip()
                if not t or re.match(r"^(\s+|[.,!?:;\"'«»()\[\]—–]|\d+)$", t):
                    continue
                if t.lower() in {"ü", "vü", "u"}:
                    continue

                m_yi = re.match(r"^(.+)-(yi|yı|yu|yü)$", t, re.IGNORECASE)
                m_i = re.match(r"^(.+)-(i|ı|u|ü)$", t, re.IGNORECASE)
                if m_yi:
                    words_to_check = [m_yi.group(1)]
                elif m_i:
                    words_to_check = [m_i.group(1)]
                elif "-" in t and not t.startswith("-") and not t.endswith("-"):
                    words_to_check = t.split("-")
                else:
                    words_to_check = [t]

                for w in words_to_check:
                    clean_w = w.strip()
                    if not clean_w:
                        continue
                    clean_key = tr_lower(normalize_latin(clean_w, strip_diacritics=True).strip())
                    if clean_key in self.sik_kelimeler:
                        continue
                    if self.reverse_dict.get(clean_key):
                        continue
                    if clean_w in self.word_cache:
                        continue
                    needed_words.add(clean_w)

        if needed_words:
            print(f"  [Önbellek] {len(needed_words):,} benzersiz kelime ters modelle toplu çevriliyor...")
            self.translate_words_batch(sorted(list(needed_words)))
            print(f"  [Önbellek] Tamamlandı. (Şu ana kadar {len(self.invalid_words):,} kelime geçersiz çıktı)")


def evaluate_backtranslation_on_dev(
    bt: BackTransliterator,
    dev_gold_path: Path = DEFAULT_DEV_GOLD,
    output_dir: Path = Path("data/eval/BT_ters_v2"),
) -> Dict[str, Any]:
    """Altın DEV setinin Latin cümlelerini geri çevirip altın Arapça ile kıyaslar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    skor_path = output_dir / "dev_skor.json"
    pred_path = output_dir / "dev.jsonl"

    gold_records = list(read_jsonl(dev_gold_path))

    # Önce DEV setindeki tüm kelimeleri GPU'da toplu önbellekle
    bt.precache_sentences([r.get("latin", "") for r in gold_records])

    predictions = []
    all_refs = []
    all_hyps = []
    sample_15 = []

    # 'o' kelimesi sayımı
    o_total_count = 0
    o_correct_count = 0
    o_translations: Counter = Counter()

    for idx, r in enumerate(gold_records):
        rec_id = r["id"]
        lat = r.get("latin", "")
        gold_ar = normalize_ottoman(r.get("arap", "")).strip()

        pred_ar, is_valid = bt.translate_sentence(lat)
        predictions.append({"id": rec_id, "tahmin": pred_ar, "gecerli": is_valid})

        all_refs.append(gold_ar)
        all_hyps.append(pred_ar)

        # 'o' kelimesini say
        tokens = re.split(r"(\s+|[.,!?:;\"'«»()\[\]—–])", lat)
        for tok in tokens:
            t = tok.strip()
            if t.lower() == "o":
                o_total_count += 1
                trans = bt.translate_token(t)
                o_translations[str(trans)] += 1
                if trans == "او":
                    o_correct_count += 1

        if idx < 15:
            sample_15.append({
                "id": rec_id,
                "latin": lat,
                "uretilen_arap": pred_ar,
                "gercek_arap": gold_ar,
            })

    write_jsonl(pred_path, predictions)

    cer_val = float(jiwer.cer(all_refs, all_hyps))
    chrf_val = float(sacrebleu.corpus_chrf(all_hyps, [all_refs], beta=2).score)

    skor_data = {
        "system": "BT_ters_v2",
        "split": "dev",
        "adet": len(gold_records),
        "cer": round(cer_val, 4),
        "cer_pct": round(cer_val * 100, 2),
        "chrf": round(chrf_val, 2),
        "o_kelimesi": {
            "toplam": o_total_count,
            "dogru_au": o_correct_count,
            "orani_pct": round(o_correct_count / max(o_total_count, 1) * 100, 2),
            "tum_ceviriler": dict(o_translations),
        },
        "ornek_15": sample_15,
    }

    with open(skor_path, "w", encoding="utf-8") as f:
        json.dump(skor_data, f, ensure_ascii=False, indent=2)

    return skor_data


def backtranslate_corpus(
    bt: BackTransliterator,
    input_path: Path,
    output_path: Path,
    kaynak_etiketi: str,
    uslup_etiketi: str,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Bir Latin derlemini (Vikikaynak veya LATOC) geri çevirerek sentetik veri setini üretir."""
    heldout_pages, heldout_arap, heldout_titles, heldout_latin = load_heldout_all()

    input_records = list(read_jsonl(input_path))
    if max_records and len(input_records) > max_records:
        input_records = input_records[:max_records]

    print(f"[{kaynak_etiketi.upper()}] {len(input_records):,} Latin cümle geri çevriliyor...")

    # 1. Tüm derlemdeki bilinmeyen kelimeleri tek seferde topluca GPU'da çevir
    bt.precache_sentences([r.get("latin", "") for r in input_records])

    output_records = []
    leaked_count = 0
    invalid_word_sentence_count = 0
    low_quality_count = 0

    for idx, r in enumerate(input_records, 1):
        lat = r.get("latin", "").strip()
        if not lat:
            continue

        # Sızıntı kontrolü
        if is_heldout(
            latin=lat,
            heldout_pages=heldout_pages,
            heldout_arap=heldout_arap,
            heldout_titles=heldout_titles,
            heldout_latin=heldout_latin,
        ):
            leaked_count += 1
            continue

        pred_ar, is_valid = bt.translate_sentence(lat)

        # Kural: "Çevrilemedi" kelime içeren cümleyi sentetik derleme KOYMA
        if not is_valid:
            invalid_word_sentence_count += 1
            continue

        # Kalite kontrolü: çıkarılamayan (boş veya çok kısa) kelime oranı > %20 ise at
        orig_words = [w for w in re.split(r"\s+", lat) if w]
        ar_words = [w for w in re.split(r"\s+", pred_ar) if w]

        clean_lat_words = [re.sub(r"[^\w]", "", w) for w in orig_words]
        clean_lat_words = [w for w in clean_lat_words if w]
        clean_ar_words = [re.sub(r"[^\w\u0600-\u06FF]", "", w) for w in ar_words]
        clean_ar_words = [w for w in clean_ar_words if w]

        if not clean_lat_words:
            continue

        # Arapça kelimelerden tek harfli olup 'و' veya rakam olmayanlar ya da boş kalanlar
        missing_or_short = sum(1 for w in clean_ar_words if len(w) <= 1 and w != "\u0648" and not w.isdigit())
        missing_words = max(0, len(clean_lat_words) - len(clean_ar_words))
        bad_ratio = (missing_or_short + missing_words) / max(len(clean_lat_words), 1)

        if len(clean_ar_words) < len(clean_lat_words) * 0.5 or bad_ratio > 0.20:
            low_quality_count += 1
            continue

        output_records.append({
            "id": f"{kaynak_etiketi}:{idx:06d}",
            "kaynak": kaynak_etiketi,
            "arap": pred_ar,
            "latin": lat,
            "uslup": uslup_etiketi,
            "duzey": "cumle",
            "ters_model": "sungur-ters-small-v1",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, output_records)

    return {
        "kaynak": kaynak_etiketi,
        "toplam_girdi": len(input_records),
        "kabul_edilen": len(output_records),
        "sizinti_atilan": leaked_count,
        "gecersiz_kelime_cumle_atilan": invalid_word_sentence_count,
        "kalite_atilan": low_quality_count,
        "toplam_gecersiz_kelime_sayisi": len(bt.invalid_words),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Kelime kelime geri çeviri (BT_ters_v2).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_TERS_MODEL, help="Ters model klasörü")
    parser.add_argument("--eval-dev", action="store_true", help="Altın DEV setinde BT_ters_v2 değerlendirmesi yap")
    parser.add_argument("--corpora", action="store_true", help="Vikikaynak ve LATOC derlemlerini geri çevir")
    parser.add_argument("--all", action="store_true", help="Hem dev değerlendirmesini hem derlem geri çevirisini yap")
    args = parser.parse_args()

    do_eval = args.eval_dev or args.all
    do_corpora = args.corpora or args.all

    if not do_eval and not do_corpora:
        do_eval = True
        do_corpora = True

    print(f"Geri çevirici başlatılıyor (Model: {args.model_dir})...")
    bt = BackTransliterator(model_path=args.model_dir)

    if do_eval:
        print("\n--- ALTIN DEV KÜMESİNDE BT_TERS_V2 DEĞERLENDİRMESİ ---")
        dev_res = evaluate_backtranslation_on_dev(bt)
        print(f"DEV CER: %{dev_res['cer_pct']:.2f} | chrF: {dev_res['chrf']:.2f} ({dev_res['adet']} kayıt)")
        o_info = dev_res.get("o_kelimesi", {})
        print(f"'o' kelimesi doğru çeviri: {o_info.get('dogru_au')}/{o_info.get('toplam')} (%{o_info.get('orani_pct', 0.0):.1f})")
        print("\nÖrnek Çıktılar (ilk 3):")
        for ex in dev_res["ornek_15"][:3]:
            print(f"  [Latin]  {ex['latin']}")
            print(f"  [Üretim] {ex['uretilen_arap']}")
            print(f"  [Gerçek] {ex['gercek_arap']}")
            print()

    if do_corpora:
        print("\n--- SENTETİK DERLEMLER OLUŞTURULUYOR (v2 Düzeltmeleriyle) ---")
        # 1. Vikikaynak
        vk_res = backtranslate_corpus(
            bt=bt,
            input_path=DEFAULT_VK_INPUT,
            output_path=DEFAULT_VK_OUTPUT,
            kaynak_etiketi="bt_vk",
            uslup_etiketi="vikikaynak",
        )
        print(
            f"Vikikaynak tamamlandı: {vk_res['kabul_edilen']:,} kabul / {vk_res['toplam_girdi']:,} girdi "
            f"({vk_res['sizinti_atilan']} sızıntı, {vk_res['gecersiz_kelime_cumle_atilan']} geçersiz kelime, "
            f"{vk_res['kalite_atilan']} kalite atıldı)."
        )

        # 2. LATOC
        latoc_res = backtranslate_corpus(
            bt=bt,
            input_path=DEFAULT_LATOC_INPUT,
            output_path=DEFAULT_LATOC_OUTPUT,
            kaynak_etiketi="bt_latoc",
            uslup_etiketi="ijmes",
        )
        print(
            f"LATOC tamamlandı: {latoc_res['kabul_edilen']:,} kabul / {latoc_res['toplam_girdi']:,} girdi "
            f"({latoc_res['sizinti_atilan']} sızıntı, {latoc_res['gecersiz_kelime_cumle_atilan']} geçersiz kelime, "
            f"{latoc_res['kalite_atilan']} kalite atıldı)."
        )
        print(f"\nToplam benzersiz geçersiz kelime sayısı: {len(bt.invalid_words):,}")


if __name__ == "__main__":
    main()
