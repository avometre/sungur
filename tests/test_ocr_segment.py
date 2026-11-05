"""Satır bölütleme modülü (src/ocr/segment.py) için birim testleri.

Ağsız ve GPU'suz çalışır.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.ocr.segment import (
    deskew_image,
    estimate_skew_angle,
    segment_page_classical,
)


def test_classical_segmentation_5_lines():
    """Yapay 5 satırlı sayfada klasik yöntemin tam 5 satır bulduğunu ve sırasını doğrular."""
    # Beyaz sayfa (600x800)
    page = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)

    # 5 adet belirgin siyah dikdörtgen satır (y=100, 200, 300, 400, 500)
    expected_ys = [100, 200, 300, 400, 500]
    for y in expected_ys:
        draw.rectangle([50, y, 550, y + 25], fill=(0, 0, 0))

    results = segment_page_classical(page)

    assert len(results) == 5, f"Beklenen 5 satır, bulunan: {len(results)}"

    # Sıralamanın yukarıdan aşağıya olduğunu doğrula
    prev_y = -1
    for i, (box, crop) in enumerate(results):
        x0, y0, x1, y1 = box
        assert y0 > prev_y, f"Satır {i} yukarıdan aşağı sıralı değil: {y0} <= {prev_y}"
        prev_y = y0
        assert crop.size[0] > 0 and crop.size[1] > 0


def test_deskew_angle_detection():
    """Eğiklik açısı tespiti fonksiyonunu test eder."""
    # 0 derece düz yatay çizgiler
    binary = np.zeros((200, 200), dtype=np.uint8)
    binary[40:50, 20:180] = 255
    binary[90:100, 20:180] = 255
    binary[140:150, 20:180] = 255

    angle = estimate_skew_angle(binary)
    assert abs(angle) <= 0.5, f"Beklenen ~0 derece, tespit edilen: {angle}"


def test_empty_page_returns_empty():
    """Tamamen beyaz sayfada 0 satır dönmelidir."""
    page = Image.new("RGB", (400, 400), color=(255, 255, 255))
    results = segment_page_classical(page)
    assert len(results) == 0


def test_two_column_segmentation_right_column_first():
    """Yapay 2 sütunlu sayfada (her sütunda 4 satır) tam 8 satır bulduğunu
    ve Osmanlıca sağdan sola okuma sırasına göre SAĞ sütunun önce geldiğini doğrular.
    """
    # Beyaz sayfa (800x600)
    page = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)

    ys = [80, 180, 280, 380]

    # Sağ sütun (x=460..740)
    for y in ys:
        draw.rectangle([460, y, 740, y + 25], fill=(0, 0, 0))

    # Sol sütun (x=60..340)
    for y in ys:
        draw.rectangle([60, y, 340, y + 25], fill=(0, 0, 0))

    results = segment_page_classical(page)

    assert len(results) == 8, f"Beklenen 8 satır, bulunan: {len(results)}"

    # İlk 4 satır SAĞ sütundan olmalıdır (x0 > 400)
    for i in range(4):
        box, _ = results[i]
        assert box[0] >= 400, f"Satır {i} sağ sütunda olmalıydı fakat x0={box[0]}"

    # Sonraki 4 satır SOL sütundan olmalıdır (x1 < 400)
    for i in range(4, 8):
        box, _ = results[i]
        assert box[2] <= 400, f"Satır {i} sol sütunda olmalıydı fakat x1={box[2]}"
