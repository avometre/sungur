"""Sungur Gradio web arayüzü (Görev 17)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from app.pipeline import SungurPipeline


def create_demo(
    translit_model_dir: Path = Path("models/sungur-ceviriyazi-small-v1"),
):
    """Gradio Blocks arayüzünü oluşturur."""
    import gradio as gr

    pipe = SungurPipeline(translit_model_dir=translit_model_dir)

    def run_text(arap_metin: str) -> Tuple[str, str]:
        if not arap_metin or not arap_metin.strip():
            return "", ""
        out = pipe.process_text(arap_metin)
        return out["latin"], out["sade"]

    def run_image(img, segmenter: str, backend: str) -> Tuple[str, str, str]:
        if img is None:
            return "", "", ""
        out = pipe.process_image(img, segmenter=segmenter, backend=backend)
        return out["arap"], out["latin"], out["sade"]

    with gr.Blocks(title="Sungur — Osmanlıca → Türkçe") as demo:
        gr.Markdown("# Sungur — Osmanlıca → Türkçe Uçtan Uca Boru Hattı")

        with gr.Tab("Metin Çeviriyazı"):
            inp_text = gr.Textbox(label="Arap Harfli Osmanlıca Metin", lines=4)
            btn_text = gr.Button("Çevir")
            out_lat = gr.Textbox(label="Latin Çeviriyazı", lines=4)
            out_sad = gr.Textbox(label="Sadeleştirilmiş Türkçe", lines=4)
            btn_text.click(run_text, inputs=[inp_text], outputs=[out_lat, out_sad])

        with gr.Tab("Sayfa Görüntüsü (OCR)"):
            inp_img = gr.Image(type="pil", label="Osmanlıca Sayfa Görüntüsü")
            with gr.Row():
                seg = gr.Radio(["klasik", "kraken"], value="klasik", label="Satır Bölütleme")
                bck = gr.Radio(["sirali", "batch"], value="sirali", label="OCR Motoru")
            btn_img = gr.Button("Sayfayı İşle")
            img_arap = gr.Textbox(label="OCR (Arap Harfli)", lines=5)
            img_lat = gr.Textbox(label="Latin Çeviriyazı", lines=5)
            img_sad = gr.Textbox(label="Sadeleştirilmiş Türkçe", lines=5)
            btn_img.click(run_image, inputs=[inp_img, seg, bck], outputs=[img_arap, img_lat, img_sad])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
