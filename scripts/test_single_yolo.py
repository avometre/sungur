"""1 kitap üzerinde DocLayout-YOLO test scripti."""

import json
import time
import sys
from pathlib import Path
import fitz
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from src.scrapers.latoc import parse_exclusion_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pdf_path = Path("data/raw/latoc/pdf/taşlıcalı yahya_şah u geda.pdf")
excl_file = Path("third_party/latoc/exclude_pages.txt")
exclusions = parse_exclusion_file(excl_file)

# Eşleşen dışlanan sayfaları bul
excl_pages = set()
for k, p_set in exclusions.items():
    if "şah u geda" in k or "sah u geda" in k:
        excl_pages = p_set
        break

print(f"PDF: {pdf_path.name}")
print(f"Dışlanan sayfalar ({len(excl_pages)}): {sorted(list(excl_pages))}")

ckpt = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
model = YOLOv10(ckpt)

doc = fitz.open(pdf_path)
total_pages = len(doc)
print(f"Toplam sayfa: {total_pages}")

results = []
t0 = time.time()
abandon_count = 0
kept_count = 0

for page_idx in range(total_pages):
    page_num = page_idx + 1
    if page_num in excl_pages:
        continue
        
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    
    det_res = model.predict(img, imgsz=1024, conf=0.25, device="cuda", verbose=False)
    img_h, img_w = img.shape[:2]
    page_w, page_h = page.rect.width, page.rect.height
    
    page_segments = {}
    for box in det_res[0].boxes:
        label_id = int(box.cls[0].item())
        label = det_res[0].names[label_id]
        if label in ("abandon", "footnote"):
            abandon_count += 1
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
            kept_count += 1
            
    results.append({
        "page_num": page_num,
        "segments": page_segments,
    })

t1 = time.time()
print(f"Tamamlandı! Süre: {t1 - t0:.2f} sn ({len(results)} aktif sayfa)")
print(f"Atılan abandon/footnote blok sayısı: {abandon_count}")
print(f"Saklanan plain text / title blok sayısı: {kept_count}")
