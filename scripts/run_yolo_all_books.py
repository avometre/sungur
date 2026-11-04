"""Tüm 17 LATOC kitabı için DocLayout-YOLO ile düzen çıkarma scripti."""

import json
import os
import sys
import time
from pathlib import Path

import fitz
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

from src.scrapers.latoc import load_latoc_metadata, parse_exclusion_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PDF_DIR = Path("data/raw/latoc/pdf")
LAYOUT_DIR = Path("data/raw/latoc/layout")
LAYOUT_DIR.mkdir(parents=True, exist_ok=True)

all_books, _ = load_latoc_metadata()
exclusions = parse_exclusion_file(Path("third_party/latoc/exclude_pages.txt"))

print("DocLayout-YOLO modeli yükleniyor...")
ckpt = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
)
model = YOLOv10(ckpt)
print("DocLayout-YOLO RTX 3060 CUDA üzerinde hazır!\n")

start_time = time.time()
processed_books = 0
total_active_pages = 0
total_abandon_blocks = 0
total_kept_blocks = 0

for b_idx, book in enumerate(all_books, 1):
    stem = book["stem"]
    pdf_path = PDF_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        continue

    layout_file = LAYOUT_DIR / f"{stem}_layout.json"

    # exclude_pages eşleştir
    clean_stem = stem.replace("_", "").replace(" ", "").lower()
    b_excl = set()
    for k, p_set in exclusions.items():
        k_clean = k.replace(".xml", "").replace("_", "").replace(" ", "").lower()
        if k_clean == clean_stem or k_clean in clean_stem:
            b_excl = p_set
            break

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    book_t0 = time.time()
    results = []
    book_abandon = 0
    book_kept = 0

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if page_num in b_excl:
            continue

        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )[:, :, :3]

        det_res = model.predict(
            img, imgsz=1024, conf=0.25, device="cuda", verbose=False
        )
        img_h, img_w = img.shape[:2]
        page_w, page_h = page.rect.width, page.rect.height

        page_segments = {}
        for box in det_res[0].boxes:
            label_id = int(box.cls[0].item())
            label = det_res[0].names[label_id]
            if label in ("abandon", "footnote"):
                book_abandon += 1
                continue
            if label not in ("plain text", "title"):
                continue

            x0, y0, x1, y1 = box.xyxy[0].tolist()
            rect = fitz.Rect(
                x0 / img_w * page_w,
                y0 / img_h * page_h,
                x1 / img_w * page_w,
                y1 / img_h * page_h,
            )
            region_text = page.get_text("text", clip=rect).strip()
            if region_text:
                page_segments.setdefault(label, []).append({
                    "bbox": [x0, y0, x1, y1],
                    "pdf_rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "text": region_text,
                })
                book_kept += 1

        results.append({
            "page_num": page_num,
            "segments": page_segments,
        })

    doc.close()

    # JSON dosyasına yaz (DocLayout-YOLO temiz düzeni)
    with open(layout_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    book_dur = time.time() - book_t0
    processed_books += 1
    total_active_pages += len(results)
    total_abandon_blocks += book_abandon
    total_kept_blocks += book_kept

    print(
        f"[{processed_books}/17] {book['kitap_id']} ({stem[:30]}): "
        f"{len(results)} sayfa, {book_abandon} abandon atıldı, {book_kept} blok alındı ({book_dur:.1f} sn)"
    )

elapsed = time.time() - start_time
print("=" * 80)
print(
    f"YOLO Düzen Çıkarma Tamamlandı! Toplam Süre: {elapsed / 60:.1f} dakika\n"
    f"İşlenen Kitap: {processed_books}, Toplam Aktif Sayfa: {total_active_pages}\n"
    f"Atılan Abandon/Footnote: {total_abandon_blocks}, Alınan Plain Text/Title: {total_kept_blocks}"
)
print("=" * 80)
