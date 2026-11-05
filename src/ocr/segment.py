"""Osmanlıca basılı sayfalar için satır bölütleme modülü.

İki yöntem desteklenir:
1. "klasik": Gri ton -> Otsu/Sauvola eşikleme -> Eğiklik düzeltme (±5°) ->
   Yatay izdüşüm profili -> Harekeler/noktalar için %25 satır genişletme ->
   Kenar/sütun kırpma -> Yukarıdan aşağıya sıralı satırlar.
2. "kraken": Kraken blla varsayılan modeli (ayrı .venv-ocr ortamı üzerinden izole çalışır).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def estimate_skew_angle(binary: np.ndarray, angles: Optional[List[float]] = None) -> float:
    """Yatay izdüşüm profili varyansını maksimize ederek eğiklik açısını (±5°) tahmin eder."""
    if angles is None:
        angles = [a * 0.5 for a in range(-10, 11)]  # -5.0 .. +5.0 derece (0.5 adım)

    h, w = binary.shape
    scale = 1.0
    if h > 1000 or w > 1000:
        scale = 800.0 / max(h, w)
        small_bin = cv2.resize(
            binary, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST
        )
    else:
        small_bin = binary

    sh, sw = small_bin.shape
    center = (sw // 2, sh // 2)

    best_angle = 0.0
    max_variance = -1.0

    for angle in angles:
        if abs(angle) < 1e-4:
            proj = np.sum(small_bin > 0, axis=1)
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                small_bin,
                M,
                (sw, sh),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            proj = np.sum(rotated > 0, axis=1)

        var = float(np.var(proj))
        if var > max_variance:
            max_variance = var
            best_angle = angle

    return best_angle


def deskew_image(image: Image.Image, angle: float) -> Image.Image:
    """Görüntüyü tahmin edilen açı kadar ters yönde döndürür (beyaz arkaplan)."""
    if abs(angle) < 0.2:
        return image
    # PIL rotate: pozitif açı saat yönünün tersine döndürür
    return image.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )


def find_text_margin_bounds(binary: np.ndarray, thresh_ratio: float = 0.01) -> Tuple[int, int, int, int]:
    """Sayfa kenarındaki tarama gölgeleri ve lekeleri atıp ana metin bloğu sınırlarını bulur."""
    h, w = binary.shape
    col_proj = np.sum(binary > 0, axis=0)
    col_thresh = max(float(h) * thresh_ratio, 2.0)
    valid_cols = np.where(col_proj > col_thresh)[0]
    if len(valid_cols) == 0:
        return 0, 0, w, h
    x0, x1 = int(valid_cols[0]), int(valid_cols[-1])

    row_proj = np.sum(binary > 0, axis=1)
    row_thresh = max(float(w) * thresh_ratio, 2.0)
    valid_rows = np.where(row_proj > row_thresh)[0]
    if len(valid_rows) == 0:
        return 0, 0, w, h
    y0, y1 = int(valid_rows[0]), int(valid_rows[-1])

    pad = 8
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def segment_page_classical(
    image: Image.Image,
    min_line_height: int = 8,
    merge_gap: int = 5,
    expansion_ratio: float = 0.25,
) -> List[Tuple[Tuple[int, int, int, int], Image.Image]]:
    """Klasik yöntemle (eğiklik düzeltme + Otsu + yatay izdüşüm + %25 genişletme) satırları bölütler.

    Döndürülen liste: [(kutu=(x0, y0, x1, y1), kirpilmis_satir_goruntusu), ...]
    Satır sırası YUKARIDAN AŞAĞIYA doğrudur.
    """
    # 1. Gri ton
    gray_pil = image.convert("L")
    gray_arr = np.array(gray_pil)

    # 2. Otsu eşikleme (siyah metin = 255)
    _, binary = cv2.threshold(gray_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Eğiklik açısı tespiti ve düzeltme (±5°)
    angle = estimate_skew_angle(binary)
    if abs(angle) >= 0.2:
        image = deskew_image(image, angle)
        gray_pil = image.convert("L")
        gray_arr = np.array(gray_pil)
        _, binary = cv2.threshold(gray_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape

    # 4. Kenar gölge ve boşluklarını buda
    mx0, my0, mx1, my1 = find_text_margin_bounds(binary)
    cropped_bin = binary[my0:my1, mx0:mx1]
    ch, cw = cropped_bin.shape
    if ch < min_line_height or cw < 10:
        return []

    # 5. Yatay izdüşüm profili (row projection)
    row_counts = np.sum(cropped_bin > 0, axis=1)
    max_c = float(np.max(row_counts)) if len(row_counts) > 0 else 0.0
    thresh = max(max_c * 0.015, 2.0)

    is_text = row_counts > thresh
    raw_lines: List[Tuple[int, int]] = []
    in_line = False
    start_y = 0

    for y in range(ch):
        if is_text[y]:
            if not in_line:
                in_line = True
                start_y = y
        else:
            if in_line:
                in_line = False
                if (y - start_y) >= min_line_height:
                    raw_lines.append((start_y, y))

    if in_line and (ch - start_y) >= min_line_height:
        raw_lines.append((start_y, ch))

    # Çok yakın parçaları birleştir (harekeler / noktalar)
    merged_lines: List[Tuple[int, int]] = []
    for y0, y1 in raw_lines:
        if not merged_lines:
            merged_lines.append((y0, y1))
        else:
            prev_y0, prev_y1 = merged_lines[-1]
            if y0 - prev_y1 <= merge_gap:
                merged_lines[-1] = (prev_y0, y1)
            else:
                merged_lines.append((y0, y1))

    # 6. Harekeler ve noktalar için satır kutusunu %25 genişlet
    # (Komşu satırların çekirdek sınırlarını aşmayacak şekilde)
    expanded_lines: List[Tuple[int, int]] = []
    num_l = len(merged_lines)
    for i, (y0, y1) in enumerate(merged_lines):
        lh = y1 - y0
        dy = max(int(lh * expansion_ratio), 2)
        ey0 = max(0, y0 - dy)
        ey1 = min(ch, y1 + dy)

        if i > 0:
            ey0 = max(ey0, merged_lines[i - 1][1])
        if i < num_l - 1:
            ey1 = min(ey1, merged_lines[i + 1][0])

        expanded_lines.append((ey0, ey1))

    # 7. Her satırın yatay (x0, x1) sınırlarını ve nihai koordinatlarını hesapla
    results: List[Tuple[Tuple[int, int, int, int], Image.Image]] = []
    for ey0, ey1 in expanded_lines:
        line_strip = cropped_bin[ey0:ey1, :]
        col_proj = np.sum(line_strip > 0, axis=0)
        active_cols = np.where(col_proj > 0)[0]
        if len(active_cols) == 0:
            lx0 = 0
            lx1 = cw
        else:
            lx0 = max(0, int(active_cols[0]) - 8)
            lx1 = min(cw, int(active_cols[-1]) + 8)

        # Orijinal sayfa koordinatlarına dönüştür
        orig_x0 = mx0 + lx0
        orig_y0 = my0 + ey0
        orig_x1 = mx0 + lx1
        orig_y1 = my0 + ey1

        # Kırp ve ekle
        box = (orig_x0, orig_y0, orig_x1, orig_y1)
        line_crop = image.crop(box)
        results.append((box, line_crop))

    # 8. Yukarıdan aşağıya sırala (y0'a göre)
    results.sort(key=lambda item: item[0][1])
    return results


def _run_kraken_subprocess(image_path: Path) -> List[Tuple[int, int, int, int]]:
    """Ayrı .venv-ocr ortamında Kraken blla çalıştırarak satır sınırlarını alır."""
    project_root = Path(__file__).resolve().parent.parent.parent
    ocr_python = project_root / ".venv-ocr" / "Scripts" / "python.exe"

    script = """
import sys
import json
from PIL import Image
from kraken import blla

img_path = sys.argv[1]
im = Image.open(img_path)
res = blla.segment(im)
lines = res.lines if hasattr(res, 'lines') else res.get('lines', [])

boxes = []
for l in lines:
    poly = getattr(l, 'boundary', None)
    if poly is None and isinstance(l, dict):
        poly = l.get('boundary')
    if poly:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])

print(json.dumps(boxes))
"""
    cmd = [str(ocr_python), "-c", script, str(image_path)]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    boxes_raw = json.loads(p.stdout.strip())
    return [tuple(b) for b in boxes_raw]


def segment_page_kraken(
    image: Image.Image,
    temp_dir: Optional[Path] = None,
) -> List[Tuple[Tuple[int, int, int, int], Image.Image]]:
    """Kraken blla ile satır bölütleme yapar.

    .venv-ocr ortamını çağırır. Yukarıdan aşağıya sıralı satırlar döndürür.
    """
    # Görüntüyü geçici olarak kaydet
    if temp_dir is None:
        temp_dir = Path("data/ocr/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_img = temp_dir / "_kraken_tmp_page.png"
    image.save(temp_img)

    try:
        boxes = _run_kraken_subprocess(temp_img)
    finally:
        if temp_img.exists():
            temp_img.unlink()

    # Yukarıdan aşağıya sırala
    boxes.sort(key=lambda b: b[1])

    results: List[Tuple[Tuple[int, int, int, int], Image.Image]] = []
    w, h = image.size
    for b in boxes:
        x0, y0, x1, y1 = b
        x0 = max(0, min(w, x0))
        y0 = max(0, min(h, y0))
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        if (x1 - x0) > 5 and (y1 - y0) > 5:
            box = (x0, y0, x1, y1)
            results.append((box, image.crop(box)))

    return results


def segment_page(
    image: Union[Image.Image, str, Path],
    method: str = "klasik",
) -> List[Tuple[Tuple[int, int, int, int], Image.Image]]:
    """Belirtilen yöntemle ('klasik' veya 'kraken') sayfadaki satırları bölütler."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    method = method.lower().strip()
    if method == "klasik":
        return segment_page_classical(image)
    elif method == "kraken":
        return segment_page_kraken(image)
    else:
        raise ValueError(f"Bilinmeyen bölütleme yöntemi: {method}. 'klasik' veya 'kraken' olmalıdır.")
