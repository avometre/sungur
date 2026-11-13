"""Görev 17 birim testleri: SungurPipeline uçtan uca akış."""

from PIL import Image

from app.pipeline import SungurPipeline


class _FakeOCRRunner:
    def predict_batch(self, images, use_dynamic_tokens=True, backend="sirali"):
        return ["عثمانلی توركجه‌سی" for _ in images]


def test_pipeline_process_text_with_custom_fns():
    pipe = SungurPipeline(
        translit_fn=lambda texts: ["osmanlı türkçesi" for _ in texts],
        simplify_fn=lambda texts: ["Osmanlı Türkçesi (sade)" for _ in texts],
    )
    out = pipe.process_text("عثمانلی توركجه‌سی\n")
    assert out["satir_sayisi"] == 1
    assert out["latin"] == "osmanlı türkçesi"
    assert out["sade"] == "Osmanlı Türkçesi (sade)"


def test_pipeline_process_text_fallback_rules():
    # Model klasörü yoksa kural tabanlı yedek devreye girmeli
    pipe = SungurPipeline(translit_model_dir="models/olmayan-klasor")
    out = pipe.process_text("كتاب")
    assert out["satir_sayisi"] == 1
    assert len(out["latin"]) > 0


def test_pipeline_process_image(monkeypatch):
    # segment_page'i sahte 1 kutu döndürecek şekilde taklit et
    import src.ocr.segment as seg_mod

    dummy_img = Image.new("RGB", (200, 80), color="white")
    monkeypatch.setattr(seg_mod, "segment_page", lambda img, method="klasik": [((0, 0, 200, 40), dummy_img)])

    pipe = SungurPipeline(
        ocr_runner=_FakeOCRRunner(),
        translit_fn=lambda texts: ["osmanlı türkçesi" for _ in texts],
        simplify_fn=lambda texts: ["Osmanlı Türkçesi" for _ in texts],
    )
    out = pipe.process_image(dummy_img)
    assert out["satir_sayisi"] == 1
    assert out["satirlar"][0].box == (0, 0, 200, 40)
    assert out["latin"] == "osmanlı türkçesi"
    assert out["sade"] == "Osmanlı Türkçesi"
