"""Sungur kod çözme, uzunluk sınırı ve tekrar döngüsü koruma modülü.

Bu modül, ByT5 çeviriyazı çıkarımında döngüye giren (dejenere) ve gereksiz uzayan
çıktıları tespit eder, uzunluk sınırına göre kırpar, no_repeat_ngram_size ile
yeniden dener ve gerekirse son çare temizlemesi uygular.
"""

import math
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.common.normalize import normalize_ottoman

# İzin verilen ek karakterler: boşluk, noktalama, tırnak ve Osmanlıca çeviriyazı kesme/ayn/hemze işaretleri
ALLOWED_PUNCT_AND_EXTRA = set(" .,;:!?\'\"«»“”‘’—–-()…/[]\t\nʻʼʿʾ`")

# Varsayılan DEV setinde optimize edilmiş N (GÖREV 08'de DEV ölçümüyle seçilecek)
DEFAULT_NO_REPEAT_NGRAM_SIZE = 12


def compute_output_limit(arap_text: str) -> int:
    """Girdi metninin uzunluğuna göre azami Latin UTF-8 bayt sınırını hesaplar.
    
    Kural: limit = ceil(1.3 * arap_bayt) + 16 (ön ek HARİÇ girdi metninin UTF-8 uzunluğu).
    """
    # Varsa model ön eklerini (<vk> , <ijmes> , <kelime> ) temizle
    clean = re.sub(r"^<(vk|ijmes|kelime)>\s*", "", arap_text.strip())
    clean = normalize_ottoman(clean).strip()
    arap_bayt = len(clean.encode("utf-8"))
    return math.ceil(1.3 * arap_bayt) + 16


def truncate_to_limit(text: str, limit_bytes: int) -> str:
    """Metni en fazla limit_bytes kadar UTF-8 baytına güvenle keser."""
    b = text.encode("utf-8")
    if len(b) <= limit_bytes:
        return text
    return b[:limit_bytes].decode("utf-8", errors="ignore")


def is_allowed_char(c: str) -> bool:
    """Karakterin izin verilen Latin harf, rakam veya noktalama olup olmadığını denetler."""
    if c in ALLOWED_PUNCT_AND_EXTRA or c.isdigit():
        return True
    try:
        # Latin harfleri (şapkalı, aksanlı ve Türkçe karakterler dahil)
        name = unicodedata.name(c)
        return "LATIN" in name or "MODIFIER LETTER" in name
    except ValueError:
        return False


def is_degenerate(cikti: str, limit_bytes: int) -> bool:
    """Çıktının döngüye girip girmediğini veya bozuk olup olmadığını denetler.
    
    Kriterler:
      a) Çıktı limite ulaştıysa (len(cikti bayt) >= limit_bytes).
      b) En az 3 karakterlik bir birim art arda 3+ kez tekrar ediyorsa (ör. 'Belter:Belter:Belter:').
         (DİKKAT: 'hemen hemen', 'aman, aman' gibi 2 kez tekrar meşrudur, bozuk SAYILMAZ).
      c) Aynı karakter art arda 5+ kez geçiyorsa ('00000', '±±±±±').
      d) İzin verilen Latin harf/rakam/noktalama dışı karakter içeriyorsa (ör. '±').
    """
    if not cikti:
        return False

    # a) Çıktı limite ulaştıysa
    b_len = len(cikti.encode("utf-8"))
    if b_len >= limit_bytes:
        return True

    # b) En az 3 karakterlik bir birim art arda 3+ kez tekrar ediyorsa
    if re.search(r"(.{3,})\1{2,}", cikti):
        return True

    # c) Aynı karakter art arda 5+ kez geçiyorsa
    if re.search(r"(.)\1{4,}", cikti):
        return True

    # d) İzin verilen karakterler dışı yabancı simge/karakter denetimi
    for c in cikti:
        if not is_allowed_char(c):
            return True

    return False


def clean_fallback(cikti: str, limit_bytes: int) -> str:
    """Yeniden denemeden sonra da bozuk kalan çıktılar için son çare temizlemesi.
    
    1. Tekrar eden birimi (3+ karakterlik 3+ tekrar) teke indirir.
    2. 5+ aynı karakter tekrarını teke indirir.
    3. Geçersiz yabancı karakterleri eler.
    4. Limite göre keser.
    """
    # 1. 3+ karakterlik tekrarları teke indir (ör. 'Belter:Belter:Belter:' -> 'Belter:')
    res = re.sub(r"(.{3,})\1{2,}", r"\1", cikti)

    # 2. 5+ aynı karakter tekrarını teke indir
    res = re.sub(r"(.)\1{4,}", r"\1", res)

    # 3. İzin verilmeyen yabancı simgeleri temizle
    res = "".join(c for c in res if is_allowed_char(c))

    # 4. Limite göre kes
    res = truncate_to_limit(res, limit_bytes)
    return res.strip()


def generate_with_guard(
    model: Any,
    tokenizer: Any,
    sentences: List[str],
    prefix: str = "<vk> ",
    beam_size: int = 4,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    apply_retry: bool = True,
    apply_limit: bool = True,
) -> Tuple[List[str], int]:
    """Cümle listesini uzunluk sınırı ve tekrar korumasıyla güvenli bir şekilde çevirir.
    
    Döndürdüğü:
      (temizlenmiş_çıktılar, başlangıçta_bozuk_çıkan_satır_sayısı)
    """
    if device is None:
        device = next(model.parameters()).device

    results: List[str] = [""] * len(sentences)
    limits: List[int] = [compute_output_limit(s) for s in sentences]
    degen_count = 0

    # 1. Aşama: Standart üretim (veya uzunluk sınırlı üretim)
    for start_idx in range(0, len(sentences), batch_size):
        end_idx = min(start_idx + batch_size, len(sentences))
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_limits = limits[start_idx:end_idx]
        chunk_prompts = [f"{prefix}{normalize_ottoman(s).strip()}" for s in chunk_sentences]

        inputs = tokenizer(
            chunk_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        max_new_tok = max(chunk_limits) if apply_limit else 512

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tok,
                    num_beams=beam_size,
                    do_sample=False,
                )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, (dec, lim) in enumerate(zip(decoded, chunk_limits)):
            global_idx = start_idx + i
            text = dec.strip()
            if apply_limit:
                text = truncate_to_limit(text, lim)
            results[global_idx] = text

    # 2. Aşama: Bozukluk tespiti ve yeniden deneme (sadece bozuk satırlar)
    degen_indices = []
    for idx, (res, lim) in enumerate(zip(results, limits)):
        if is_degenerate(res, lim):
            degen_indices.append(idx)

    degen_count = len(degen_indices)

    if apply_retry and degen_indices and no_repeat_ngram_size > 0:
        # Sadece bozuk çıkan satırları yeniden üret
        for chunk_start in range(0, len(degen_indices), batch_size):
            chunk_sub_indices = degen_indices[chunk_start : chunk_start + batch_size]
            sub_prompts = [f"{prefix}{normalize_ottoman(sentences[idx]).strip()}" for idx in chunk_sub_indices]
            sub_limits = [limits[idx] for idx in chunk_sub_indices]

            inputs = tokenizer(
                sub_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            max_new_tok = max(sub_limits) if apply_limit else 512

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    retry_outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_new_tok,
                        num_beams=beam_size,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        do_sample=False,
                    )

            retry_decoded = tokenizer.batch_decode(retry_outputs, skip_special_tokens=True)

            for idx, retry_text, lim in zip(chunk_sub_indices, retry_decoded, sub_limits):
                r_text = retry_text.strip()
                if apply_limit:
                    r_text = truncate_to_limit(r_text, lim)

                # Hâlâ bozuk mu?
                if is_degenerate(r_text, lim):
                    # Son çare temizliği
                    r_text = clean_fallback(r_text, lim)

                results[idx] = r_text

    return results, degen_count
