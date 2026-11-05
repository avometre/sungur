"""OCR değerlendirme ve kıyaslama modülü (src/ocr/eval_ocr.py).

Bu modül:
1. Azra-1-Mini modelini kullanarak satır bazlı OCR çıkarımı yapar.
2. Sabit max_new_tokens=512 ile dinamik (en/boy oranına bağlı) sınırları karşılaştırır.
3. normalize_ottoman sonrası:
   - CER (boşluklu)
   - CER (boşluksuz)
   - Font bazında ve temiz/bozuk alt kırılımlarında ölçer.
4. Çıktıları data/eval/O0_azra_mini/{sentetik_dev,hf}_skor.json dosyalarına yazar.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from src.common.normalize import normalize_ottoman


def compute_dynamic_max_tokens(w: int, h: int, min_tokens: int = 32, max_tokens: int = 512) -> int:
    """Satır görselinin en/boy oranına göre dinamik max_new_tokens hesaplar.

    Formül: int(2 * (w / h) * 8 + 32)
    """
    aspect = max(0.5, float(w) / max(1, h))
    tokens = int(2.0 * aspect * 8.0 + 32.0)
    return max(min_tokens, min(max_tokens, tokens))


def resize_to_multiple_of_64(image: Image.Image) -> Image.Image:
    """Görüntüyü kenarları 64'ün katı olacak şekilde LANCZOS ile boyutlandırır."""
    w, h = image.size
    new_w = ((w + 63) // 64) * 64
    new_h = ((h + 63) // 64) * 64
    if (new_w, new_h) != (w, h):
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image


class AzraOCRRunner:
    """Azra-1-Mini OCR modeli için çıkarım sınıfı."""

    def __init__(
        self,
        model_id: str = "OttomanNLP/Azra-1-Mini-0.8b",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

        print(f"[OCR] Model yükleniyor: {model_id} ({self.device}, {self.dtype})...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()
        print("[OCR] Model başarıyla yüklendi!")

    def predict_line(
        self,
        image: Image.Image,
        prompt: str = "Görseldeki Osmanlıca metni transkribe et:",
        max_new_tokens: int = 512,
    ) -> str:
        """Tek bir satır görüntüsünden metin çıkarır."""
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        processed_img = resize_to_multiple_of_64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[:, prompt_len:]
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        return text

    def predict_batch(
        self,
        images: List[Image.Image],
        prompt: str = "Görseldeki Osmanlıca metni transkribe et:",
        max_new_tokens: int = 512,
        use_dynamic_tokens: bool = False,
        backend: str = "sirali",
        batch_size: int = 16,
    ) -> List[str]:
        """Görüntü listesi üzerinde çıkarım yapar.

        backend='sirali' (varsayılan): Satır satır sıralı çıkarım yapar.
        backend='batch': Toplu (batch_size=16) çıkarım yapar (padding_side='left').
        """
        if not images:
            return []

        if backend == "batch" and len(images) > 1:
            orig_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left"
            results: List[str] = []
            try:
                for i in range(0, len(images), batch_size):
                    chunk_imgs = images[i : i + batch_size]
                    batch_messages = []
                    chunk_tokens = []
                    for img in chunk_imgs:
                        p_img = resize_to_multiple_of_64(img.convert("RGB"))
                        batch_messages.append([
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": p_img},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ])
                        w, h = img.size
                        t = (
                            compute_dynamic_max_tokens(w, h)
                            if use_dynamic_tokens
                            else max_new_tokens
                        )
                        chunk_tokens.append(t)

                    max_chunk_tok = max(chunk_tokens)
                    texts = [
                        self.processor.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True
                        )
                        for msg in batch_messages
                    ]
                    image_inputs, _ = process_vision_info(batch_messages)
                    inputs = self.processor(
                        text=texts,
                        images=image_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)

                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_chunk_tok,
                            do_sample=False,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )

                    prompt_len = inputs["input_ids"].shape[1]
                    out_ids = generated_ids[:, prompt_len:]
                    decoded = self.processor.batch_decode(
                        out_ids, skip_special_tokens=True
                    )
                    results.extend([d.strip() for d in decoded])
            finally:
                self.processor.tokenizer.padding_side = orig_padding_side

            return results

        # Varsayılan: sıralı çıkarım
        results = []
        for img in images:
            w, h = img.size
            tokens = compute_dynamic_max_tokens(w, h) if use_dynamic_tokens else max_new_tokens
            pred = self.predict_line(img, prompt=prompt, max_new_tokens=tokens)
            results.append(pred)
        return results


def calculate_ocr_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """normalize_ottoman uygulayarak boşluklu ve boşluksuz CER hesaplar."""
    if not references:
        return {"cer": 0.0, "cer_bosluksuz": 0.0, "adet": 0}

    norm_refs: List[str] = []
    norm_hyps: List[str] = []
    refs_nospace: List[str] = []
    hyps_nospace: List[str] = []

    for r, h in zip(references, hypotheses):
        nr = normalize_ottoman(r, keep_harakat=False)
        nh = normalize_ottoman(h, keep_harakat=False)
        norm_refs.append(nr)
        norm_hyps.append(nh)
        refs_nospace.append(nr.replace(" ", ""))
        hyps_nospace.append(nh.replace(" ", ""))

    try:
        cer_val = float(jiwer.cer(norm_refs, norm_hyps))
    except Exception:
        cer_val = 1.0

    try:
        cer_nospace_val = float(jiwer.cer(refs_nospace, hyps_nospace))
    except Exception:
        cer_nospace_val = 1.0

    return {
        "cer": round(cer_val, 4),
        "cer_pct": round(cer_val * 100, 2),
        "cer_bosluksuz": round(cer_nospace_val, 4),
        "cer_bosluksuz_pct": round(cer_nospace_val * 100, 2),
        "adet": len(references),
    }


def evaluate_synthetic_dev(
    runner: AzraOCRRunner,
    labels_file: Path = Path("data/ocr/sentetik_dev/etiketler.jsonl"),
    output_score_file: Path = Path("data/eval/O0_azra_mini/sentetik_dev_skor.json"),
) -> Dict[str, Any]:
    """data/ocr/sentetik_dev/ kümesini iki ayarda (sabit 512 vs dinamik) değerlendirir."""
    records = [
        json.loads(line)
        for line in labels_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"[EVAL] {len(records)} adet yapay satır değerlendiriliyor...")

    images = [Image.open(r["goruntu"]) for r in records]
    references = [r["metin"] for r in records]

    # 1. Ayar A: Sabit 512
    t0 = time.time()
    hyps_fixed = runner.predict_batch(images, max_new_tokens=512, use_dynamic_tokens=False)
    time_fixed = time.time() - t0

    # 2. Ayar B: Dinamik sınır
    t1 = time.time()
    hyps_dynamic = runner.predict_batch(images, use_dynamic_tokens=True)
    time_dynamic = time.time() - t1

    # Metrikleri hesapla (Ayar B - dinamik standart olarak raporlanır, A ile karşılaştırılır)
    metrics_fixed = calculate_ocr_metrics(references, hyps_fixed)
    metrics_dynamic = calculate_ocr_metrics(references, hyps_dynamic)

    # Kırılımlar (Dinamik ayar üzerinde)
    font_subsets: Dict[str, Tuple[List[str], List[str]]] = {}
    cond_subsets: Dict[str, Tuple[List[str], List[str]]] = {}

    for r, h, meta in zip(references, hyps_dynamic, records):
        f = meta["font"]
        cond = "bozuk" if meta["bozuk"] else "temiz"

        font_subsets.setdefault(f, ([], []))[0].append(r)
        font_subsets[f][1].append(h)

        cond_subsets.setdefault(cond, ([], []))[0].append(r)
        cond_subsets[cond][1].append(h)

    font_scores = {
        f: calculate_ocr_metrics(sub_refs, sub_hyps)
        for f, (sub_refs, sub_hyps) in font_subsets.items()
    }

    cond_scores = {
        c: calculate_ocr_metrics(sub_refs, sub_hyps)
        for c, (sub_refs, sub_hyps) in cond_subsets.items()
    }

    speed_fixed = len(images) / max(0.1, time_fixed)
    speed_dynamic = len(images) / max(0.1, time_dynamic)

    output = {
        "sistem": "O0_azra_mini",
        "kume": "sentetik_dev",
        "toplam_ornek": len(records),
        "ayar_karsilastirmasi": {
            "sabit_512": {
                "toplam": metrics_fixed,
                "sure_saniye": round(time_fixed, 2),
                "satir_bolu_saniye": round(speed_fixed, 2),
            },
            "dinamik_sinir": {
                "toplam": metrics_dynamic,
                "sure_saniye": round(time_dynamic, 2),
                "satir_bolu_saniye": round(speed_dynamic, 2),
            },
        },
        "font_kirilimi": font_scores,
        "durum_kirilimi": cond_scores,
    }

    output_score_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_score_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[EVAL] Değerlendirme tamamlandı. Skorlar -> {output_score_file}")
    return output


def record_hf_dataset_status(
    output_score_file: Path = Path("data/eval/O0_azra_mini/hf_skor.json"),
) -> Dict[str, Any]:
    """Hugging Face veri seti durumunu (giriş gerekli) kaydeder."""
    output = {
        "sistem": "O0_azra_mini",
        "kume": "xgboostgod/ottoman-newdata-ocr",
        "durum": "giriş gerekli",
        "aciklama": (
            "xgboostgod/ottoman-newdata-ocr veri seti kapalı/gated veya özel (private) bir havuzdur. "
            "Erişim için huggingface-cli login ile yetkili bir HF_TOKEN tanımlanmalıdır."
        ),
        "lisans": "Bilinmiyor (özel/gated havuz)",
        "sizinti_riski": (
            "Yüksek/Bilinmiyor: Azra-1-Mini modeli aynı yazar (OttomanNLP / xgboostgod) ve "
            "ekosistem tarafından yayınlandığından, eğitim setinin bir parçası olma ihtimali çok yüksektir."
        ),
    }
    output_score_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_score_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


if __name__ == "__main__":
    record_hf_dataset_status()
    runner = AzraOCRRunner()
    evaluate_synthetic_dev(runner)
