"""Sungur Modal.com GPU çalıştırma betikleri (Görev 11 & 12).

Yerel makineden kullanım:
  1. Verileri/modelleri Modal Volume'a yükle:
     modal volume put sungur-data data /data
     modal volume put sungur-data models /models

  2. İşleri çalıştır:
     modal run scripts/modal_jobs.py --task content_match
     modal run scripts/modal_jobs.py --task roundtrip
     modal run scripts/modal_jobs.py --task train_ters
     modal run scripts/modal_jobs.py --task train_ileri
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("sungur-gpu")
volume = modal.Volume.from_name("sungur-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "datasets>=2.19.0",
        "jiwer>=3.0.0",
        "sacrebleu>=2.4.0",
        "regex>=2024.0.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
        "tqdm>=4.66.0",
    )
    .add_local_dir("src", remote_path="/root/src")
)

VOL_PATH = Path("/vol")


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": volume},
    timeout=3600 * 4,
)
def run_content_match(
    model_subpath: str = "models/sungur-ceviriyazi-small-v1",
    printed_subpath: str = "data/interim/ottoman_printed_all.jsonl",
    vk_subpath: str = "data/interim/vikikaynak_latin.jsonl",
    latoc_subpath: str = "data/interim/latoc_sentences.jsonl",
    output_subpath: str = "data/processed/translit_ocr_pairs.jsonl",
) -> dict:
    """Görev 11: printed_ocr satırlarını S1b ile çevirip Vikikaynak/LATOC ile eşleştirir."""
    os.chdir("/vol")
    import sys

    sys.path.insert(0, "/root")

    from src.common.io import read_jsonl
    from src.data_collection.content_match import (
        load_corpus_documents,
        match_printed_ocr_corpus,
    )
    from src.translit.infer import SungurTransliterator

    engine = SungurTransliterator(model_dir=VOL_PATH / model_subpath)
    printed_records = list(read_jsonl(VOL_PATH / printed_subpath))
    corpus_docs = load_corpus_documents(
        vk_path=VOL_PATH / vk_subpath,
        latoc_path=VOL_PATH / latoc_subpath,
    )

    summary = match_printed_ocr_corpus(
        printed_records=printed_records,
        corpus_docs=corpus_docs,
        transliterate_fn=lambda texts: engine.transliterate_batch(texts, prefix="<vk> ", batch_size=32),
        output_path=VOL_PATH / output_subpath,
    )
    volume.commit()
    return {
        "toplam_kitap": summary["toplam_kitap"],
        "eslesen_kitap": summary["eslesen_kitap"],
        "toplam_cift": summary["toplam_cift"],
        "elendi_sizinti": summary["elendi_sizinti"],
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": volume},
    timeout=3600 * 2,
)
def run_roundtrip_score(
    input_subpath: str = "data/processed/translit_sentetik_vk.jsonl",
    output_subpath: str = "data/processed/translit_sentetik_vk_rt.jsonl",
    model_subpath: str = "models/sungur-ceviriyazi-small-v1",
    max_rt_cer: float = 0.25,
) -> dict:
    """Görev 11: Sentetik veri için S1b üzerinden rt_cer hesaplar ve filtreler."""
    os.chdir("/vol")
    import sys

    sys.path.insert(0, "/root")

    from src.translit.roundtrip import score_synthetic_file

    res = score_synthetic_file(
        input_path=VOL_PATH / input_subpath,
        output_path=VOL_PATH / output_subpath,
        model_dir=VOL_PATH / model_subpath,
        max_rt_cer=max_rt_cer,
        batch_size=32,
    )
    volume.commit()
    return res


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=3600 * 6,
)
def run_train_byt5(
    direction: str = "ileri",
    model_name: str = "google/byt5-base",
    epochs: int = 5,
    batch_size: int = 16,
) -> None:
    """Görev 12: ByT5 eğitimini A100 GPU üzerinde çalıştırır."""
    os.chdir("/vol")
    import sys

    sys.path.insert(0, "/root")

    from src.translit.train_byt5 import train_byt5

    train_byt5(
        direction=direction,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        gradient_checkpointing=False,
    )
    volume.commit()


@app.local_entrypoint()
def main(task: str = "content_match") -> None:
    if task == "content_match":
        print(run_content_match.remote())
    elif task == "roundtrip":
        print(run_roundtrip_score.remote())
    elif task == "train_ters":
        run_train_byt5.remote(direction="ters", model_name="google/byt5-small", epochs=10, batch_size=32)
    elif task == "train_ileri":
        run_train_byt5.remote(direction="ileri", model_name="google/byt5-base", epochs=5, batch_size=16)
    else:
        raise ValueError(f"Bilinmeyen görev: {task}")
