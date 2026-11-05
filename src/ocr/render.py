"""Yapay Osmanlıca satır seti üretme modülü (src/ocr/render.py).

Altın DEV setinden 300 satırlık temiz baskı üst sınırı veri kümesi üretir:
- Fontlar: Amiri, Scheherazade New, Noto Naskh Arabic (OFL)
- Şekillendirme: arabic-reshaper + python-bidi
- Yazı boyutu: 28-40 px
- Yarısı temiz, yarısı bozulmuş (hafif bulanıklık, Gauss gürültüsü, ±1° eğiklik)
- Çıktı: data/ocr/sentetik_dev/{id}.png + data/ocr/sentetik_dev/etiketler.jsonl
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def split_text_at_word_boundary(text: str, max_len: int = 60) -> List[str]:
    """60 karakterden uzun metinleri kelime sınırlarından böler."""
    text = text.strip()
    if len(text) <= max_len:
        return [text]

    words = text.split()
    chunks: List[str] = []
    curr: List[str] = []
    curr_len = 0

    for w in words:
        w_len = len(w)
        # Kelime tek başına max_len'den uzunsa mecburen ekle
        if curr and (curr_len + 1 + w_len) > max_len:
            chunks.append(" ".join(curr))
            curr = [w]
            curr_len = w_len
        else:
            curr.append(w)
            curr_len += (1 + w_len) if curr_len > 0 else w_len

    if curr:
        chunks.append(" ".join(curr))

    return chunks


def shape_arabic_text(text: str) -> str:
    """Arap harfli metni libraqm olmadığında doğru bitişik görüntülemek için şekillendirir."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def apply_degradation(
    img: Image.Image,
    rng: random.Random,
    max_angle: float = 1.0,
) -> Image.Image:
    """Hafif bozulma uygular: ±1° dönme, hafif Gauss bulanıklığı ve tuz-biber/Gauss gürültüsü."""
    # 1. Hafif dönme (±1.0°)
    angle = rng.uniform(-max_angle, max_angle)
    img = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )

    # 2. Hafif Gauss bulanıklığı (sigma 0.5 - 0.9)
    radius = rng.uniform(0.4, 0.8)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 3. Hafif Gauss gürültüsü
    arr = np.array(img, dtype=np.float32)
    noise_gen = np.random.RandomState(rng.randint(0, 1000000))
    noise = noise_gen.normal(0, 7.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def render_line_image(
    text: str,
    font_path: Path,
    font_size: int = 32,
    pad_x: int = 24,
    pad_y: int = 16,
    is_degraded: bool = False,
    seed: int = 42,
) -> Image.Image:
    """Tek bir satır metni verilen font ve boyutta PIL görüntüsü olarak çizer."""
    shaped = shape_arabic_text(text)
    font = ImageFont.truetype(str(font_path), font_size)

    # Metin sınırlarını ölç
    dummy_img = Image.new("RGB", (10, 10), color="white")
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), shaped, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    img_w = text_w + 2 * pad_x
    img_h = text_h + 2 * pad_y

    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Metni ortalayarak veya sağa/sola hizalayarak çiz
    # textbbox sol üst ofsetini hesaba kat
    draw_x = pad_x - bbox[0]
    draw_y = pad_y - bbox[1]
    draw.text((draw_x, draw_y), shaped, font=font, fill=(0, 0, 0))

    if is_degraded:
        rng = random.Random(seed)
        img = apply_degradation(img, rng)

    return img


def build_synthetic_dev_set(
    dev_path: Path = Path("data/gold/translit_dev.jsonl"),
    output_dir: Path = Path("data/ocr/sentetik_dev"),
    num_samples: int = 300,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Altın DEV setinden 300 satırlık yapay OCR veri setini üretir ve kaydeder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Font yolları
    fonts_dir = Path("data/fonts")
    font_configs = [
        ("Amiri", fonts_dir / "Amiri-Regular.ttf"),
        ("ScheherazadeNew", fonts_dir / "ScheherazadeNew-Regular.ttf"),
        ("NotoNaskhArabic", fonts_dir / "NotoNaskhArabic-Regular.ttf"),
    ]

    for name, p in font_configs:
        if not p.exists():
            raise FileNotFoundError(f"Font dosyası bulunamadı: {p}")

    # DEV setini oku (TEST'ten DEĞİL)
    dev_records = [
        json.loads(line)
        for line in dev_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Arapça satırları topla ve >60 karakter olanları böl
    candidate_lines: List[Tuple[str, str]] = []  # (metin, orijinal_id)
    for r in dev_records:
        orig_id = r["id"]
        raw_arap = r.get("arap", "").strip()
        if not raw_arap:
            continue
        chunks = split_text_at_word_boundary(raw_arap, max_len=60)
        for chunk in chunks:
            if len(chunk) >= 5:  # En az 5 karakter
                candidate_lines.append((chunk, orig_id))

    # Sabit tohumla karıştır ve 300 satır seç
    rng = random.Random(seed)
    rng.shuffle(candidate_lines)
    selected_lines = candidate_lines[:num_samples]

    metadata: List[Dict[str, Any]] = []
    font_sizes = [28, 32, 36, 40]

    for i, (text, orig_id) in enumerate(selected_lines):
        line_id = f"sentetik_dev_{i+1:03d}"
        img_name = f"{line_id}.png"
        img_path = output_dir / img_name

        # Font seçimi (eşit dağılım)
        font_name, font_path = font_configs[i % len(font_configs)]
        font_size = font_sizes[i % len(font_sizes)]

        # Yarısı temiz (i < 150), yarısı bozulmuş (i >= 150)
        is_degraded = i >= (num_samples // 2)

        # Görseli üret
        img = render_line_image(
            text=text,
            font_path=font_path,
            font_size=font_size,
            is_degraded=is_degraded,
            seed=seed + i,
        )
        img.save(img_path)

        record = {
            "id": line_id,
            "goruntu": str(img_path).replace("\\", "/"),
            "metin": text,
            "font": font_name,
            "bozuk": is_degraded,
            "font_size": font_size,
            "orijinal_id": orig_id,
        }
        metadata.append(record)

    # etiketler.jsonl kaydet
    labels_path = output_dir / "etiketler.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    return metadata


if __name__ == "__main__":
    meta = build_synthetic_dev_set()
    print(f"Toplam {len(meta)} adet yapay satır üretildi -> data/ocr/sentetik_dev/")
