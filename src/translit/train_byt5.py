"""ByT5-small model eğitim betiği (ters model ve ileri model).

Bu modül iki yönü de destekler:
1. Ters Model (--direction ters):
   - Latin kelime -> Arap kelime (sungur-ters-small-v1)
   - Veri: translit_train_kelime + translit_train_morph (hem latin hem latin_sade çiftleri)
   - Doğrulama: translit_val_kelime (normalize_ottoman sonrası CER)
   - Çıktı: models/sungur-ters-small-v1/
2. İleri Model (--direction ileri):
   - Arap cümle/kelime -> Latin çeviriyazı (sungur-ceviriyazi-small-v1)
   - Girdi üslup ön eki: "<vk> ", "<ijmes> ", "<kelime> "
   - Veri karışımı: sentetik_vk x2, sentetik_latoc x1, kelime+morph x1
   - Doğrulama: Altın DEV seti (sade CER)
   - Çıktı: models/sungur-ceviriyazi-small-v1/
"""

import argparse
import functools
import json
import logging
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.common.io import read_jsonl
from src.common.normalize import normalize_latin, normalize_ottoman
from src.eval.metrics import normalize_eval_text
from src.translit.decode import (
    clean_fallback,
    compute_output_limit,
    is_degenerate,
    truncate_to_limit,
)

print = functools.partial(print, flush=True)

logger = logging.getLogger(__name__)


class Seq2SeqTextDataset(Dataset):
    """Metin çiftlerini tutan ve dinamik belirteçleme yapan veri kümesi."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        tokenizer: Any,
        max_source_length: int = 64,
        max_target_length: int = 64,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.samples[idx]


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Benzer uzunluktaki örnekleri aynı batch'e getiren örnekleyici (dolgu israfını önler)."""

    def __init__(
        self,
        dataset: Seq2SeqTextDataset,
        batch_size: int,
        mega_batch_mult: int = 50,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.mega_batch_size = batch_size * mega_batch_mult
        self.lengths = [len(s[0].encode("utf-8")) + len(s[1].encode("utf-8")) for s in dataset.samples]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)

        batches = []
        for i in range(0, len(indices), self.mega_batch_size):
            mega_batch = indices[i : i + self.mega_batch_size]
            mega_batch.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(mega_batch), self.batch_size):
                b = mega_batch[j : j + self.batch_size]
                if len(b) == self.batch_size:
                    batches.append(b)

        rng.shuffle(batches)
        for b in batches:
            yield from b

    def __len__(self) -> int:
        return (len(self.dataset) // self.batch_size) * self.batch_size


def collate_fn_builder(tokenizer: Any, max_source_len: int, max_target_len: int):
    """Dinamik padding yapan collate fonksiyonu üretir."""
    def collate_fn(batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        sources = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        model_inputs = tokenizer(
            sources,
            max_length=max_source_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenizer(
            targets,
            max_length=max_target_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels_ids = labels["input_ids"]
        # PyTorch cross-entropy kaybında pad token'ları (-100) yok sayılır
        labels_ids[labels_ids == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels_ids

        return model_inputs

    return collate_fn


def load_reverse_data(
    train_kelime_path: Path = Path("data/processed/translit_train_kelime.jsonl"),
    train_morph_path: Path = Path("data/processed/translit_train_morph.jsonl"),
    val_kelime_path: Path = Path("data/processed/translit_val_kelime.jsonl"),
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Ters model (Latin -> Arap) için eğitim ve doğrulama çiftlerini hazırlar."""
    train_samples: List[Tuple[str, str]] = []
    seen_pairs = set()

    for p in [train_kelime_path, train_morph_path]:
        if p.exists():
            for r in read_jsonl(p):
                ar = normalize_ottoman(r.get("arap", "")).strip()
                lat = r.get("latin", "").strip()
                lat_sade = r.get("latin_sade") or normalize_latin(lat, strip_diacritics=True).strip()

                if not ar or not lat:
                    continue

                # 1. Şapkalı biçim
                pair1 = (lat, ar)
                if pair1 not in seen_pairs:
                    seen_pairs.add(pair1)
                    train_samples.append(pair1)

                # 2. Şapkasız sade biçim (varsa ve farklıysa)
                if lat_sade and lat_sade != lat:
                    pair2 = (lat_sade, ar)
                    if pair2 not in seen_pairs:
                        seen_pairs.add(pair2)
                        train_samples.append(pair2)

    val_samples: List[Tuple[str, str]] = []
    if val_kelime_path.exists():
        for r in read_jsonl(val_kelime_path):
            ar = normalize_ottoman(r.get("arap", "")).strip()
            lat = r.get("latin", "").strip()
            if ar and lat:
                val_samples.append((lat, ar))

    return train_samples, val_samples


def load_forward_data(
    sentetik_vk_path: Path = Path("data/processed/translit_sentetik_vk.jsonl"),
    sentetik_latoc_path: Path = Path("data/processed/translit_sentetik_latoc.jsonl"),
    train_kelime_path: Path = Path("data/processed/translit_train_kelime.jsonl"),
    train_morph_path: Path = Path("data/processed/translit_train_morph.jsonl"),
    dev_gold_path: Path = Path("data/gold/translit_dev.jsonl"),
    seed: int = 0,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """İleri model (Arap -> Latin) için ağırlıklı karışım eğitim ve dev çiftlerini hazırlar."""
    # Karışım: sentetik_vk x2, sentetik_latoc x1, kelime+morph x1
    vk_samples: List[Tuple[str, str]] = []
    if sentetik_vk_path.exists():
        for r in read_jsonl(sentetik_vk_path):
            ar = normalize_ottoman(r.get("arap", "")).strip()
            lat = normalize_latin(r.get("latin", "")).strip()
            if ar and lat:
                vk_samples.append((f"<vk> {ar}", lat))

    latoc_samples: List[Tuple[str, str]] = []
    if sentetik_latoc_path.exists():
        for r in read_jsonl(sentetik_latoc_path):
            ar = normalize_ottoman(r.get("arap", "")).strip()
            lat = normalize_latin(r.get("latin", "")).strip()
            if ar and lat:
                latoc_samples.append((f"<ijmes> {ar}", lat))

    word_samples: List[Tuple[str, str]] = []
    for p in [train_kelime_path, train_morph_path]:
        if p.exists():
            for r in read_jsonl(p):
                ar = normalize_ottoman(r.get("arap", "")).strip()
                lat = normalize_latin(r.get("latin", "")).strip()
                if ar and lat:
                    word_samples.append((f"<kelime> {ar}", lat))

    # Ağırlıklı birleştirme: 2x vk + 1x latoc + 1x word
    train_samples = (vk_samples * 2) + latoc_samples + word_samples
    rng = random.Random(seed)
    rng.shuffle(train_samples)

    # Dev seti (doğrulama için)
    dev_samples: List[Tuple[str, str]] = []
    if dev_gold_path.exists():
        for r in read_jsonl(dev_gold_path):
            ar = normalize_ottoman(r.get("arap", "")).strip()
            lat = r.get("latin", "").strip()
            if ar and lat:
                # Varsayılan çıkarım ön eki: <vk>
                dev_samples.append((f"<vk> {ar}", lat))

    return train_samples, dev_samples


def evaluate_model(
    model: Any,
    tokenizer: Any,
    eval_samples: List[Tuple[str, str]],
    device: torch.device,
    direction: str = "ters",
    max_len: int = 64,
    batch_size: int = 32,
) -> float:
    """Modeli doğrulama kümesinde değerlendirir ve ortalama CER döndürür."""
    was_training = model.training
    model.eval()
    prev_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    all_refs = []
    all_hyps = []

    with torch.no_grad():
        for i in range(0, len(eval_samples), batch_size):
            batch = eval_samples[i : i + batch_size]
            srcs = [item[0] for item in batch]
            refs = [item[1] for item in batch]

            inputs = tokenizer(
                srcs,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            if direction == "ileri":
                limits = [compute_output_limit(s) for s in srcs]
                max_gen_len = max(limits)
            else:
                limits = [max_len] * len(srcs)
                max_gen_len = max_len

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_gen_len,
                    num_beams=1,
                    do_sample=False,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for ref, hyp, lim in zip(refs, decoded, limits):
                if direction == "ters":
                    # Arapça için normalize_ottoman ile kıyasla
                    ref_norm = normalize_ottoman(ref).strip()
                    hyp_norm = normalize_ottoman(hyp).strip()
                else:
                    # Uzunluk sınırı ve bozulma koruması
                    hyp_clean = truncate_to_limit(hyp.strip(), lim)
                    if is_degenerate(hyp_clean, lim):
                        hyp_clean = clean_fallback(hyp_clean, lim)
                    ref_norm = normalize_eval_text(ref, mode="sade")
                    hyp_norm = normalize_eval_text(hyp_clean, mode="sade")

                all_refs.append(ref_norm)
                all_hyps.append(hyp_norm)

    model.config.use_cache = prev_use_cache
    if was_training:
        model.train()

    if not all_refs:
        return 1.0

    try:
        cer_score = float(jiwer.cer(all_refs, all_hyps))
    except Exception:
        cer_score = 1.0

    return cer_score


def train_byt5(
    direction: str = "ters",
    model_name: str = "google/byt5-small",
    output_dir: Optional[Path] = None,
    epochs: int = 10,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    max_source_length: int = 64,
    max_target_length: int = 64,
    trial_steps: int = 200,
    resume: bool = False,
    patience: int = 2,
    num_workers: int = 0,
    group_by_length: bool = True,
    gradient_checkpointing: bool = False,
) -> Dict[str, Any]:
    """ByT5-small modelini verilen yönde eğitir."""
    if output_dir is None:
        if direction == "ters":
            output_dir = Path("models/sungur-ters-small-v1")
        else:
            output_dir = Path("models/sungur-ceviriyazi-small-v1")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[{direction.upper()}] Model: {model_name} | Cihaz: {device} | Çıktı: {output_dir}")

    # 1. Veri yükleme
    if direction == "ters":
        train_samples, val_samples = load_reverse_data()
    else:
        train_samples, val_samples = load_forward_data()

    print(f"Eğitim çifti sayısı: {len(train_samples):,} | Doğrulama çifti sayısı: {len(val_samples):,}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Model başlatma veya checkpoint'ten yükleme
    resume_checkpoint = output_dir / "latest_checkpoint.pt"
    if resume and resume_checkpoint.exists():
        print(f"Checkpoint bulundu, eğitim kaldığı yerden devam ediyor: {resume_checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(output_dir), torch_dtype=torch.bfloat16)
        ckpt_data = torch.load(resume_checkpoint, map_location="cpu")
        start_epoch = ckpt_data.get("epoch", 0)
        best_cer = ckpt_data.get("best_cer", 999.0)
        history = ckpt_data.get("history", [])
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        start_epoch = 0
        best_cer = 999.0
        history = []

    model.to(device)

    if gradient_checkpointing:
        print("Gradient checkpointing etkinleştirildi.")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataset = Seq2SeqTextDataset(
        train_samples,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    collate_fn = collate_fn_builder(tokenizer, max_source_length, max_target_length)
    sampler = LengthGroupedSampler(dataset, batch_size=batch_size) if (group_by_length and direction == "ileri") else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    total_training_steps = steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # 3. 200 Adımlık Deneme Koşusu ve Süre Tahmini
    if trial_steps > 0 and start_epoch == 0:
        print(f"\n--- {trial_steps} ADIMLIK DENEME KOŞUSU BAŞLIYOR ---")
        model.train()
        trial_start_t = time.time()
        trial_loss = 0.0
        trial_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        step_done = 0
        for b_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            loss.backward()

            if (b_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                trial_optimizer.step()
                trial_optimizer.zero_grad()
                step_done += 1
                trial_loss += loss.item() * gradient_accumulation_steps

                if step_done >= trial_steps:
                    break

        trial_duration = time.time() - trial_start_t
        steps_per_sec = step_done / trial_duration
        est_total_sec = total_training_steps / max(steps_per_sec, 0.01)
        est_total_hours = est_total_sec / 3600.0

        print(f"Deneme tamamlandı: {step_done} adım, {trial_duration:.2f} sn ({steps_per_sec:.2f} adım/sn).")
        print(f"Ortalama deneme kaybı: {trial_loss / step_done:.4f}")
        print(f"Tahmini toplam eğitim süresi ({epochs} epoch, {total_training_steps:,} adım): {est_total_hours:.2f} saat.")

        # 8 saat sınırını denetle
        if est_total_hours > 8.0:
            adjusted_epochs = max(1, int(epochs * (7.5 / est_total_hours)))
            print(f"UYARI: Tahmini süre ({est_total_hours:.2f} saat) 8 saati aşıyor! Epoch sayısı {epochs} -> {adjusted_epochs} olarak güncelleniyor.")
            epochs = adjusted_epochs
            total_training_steps = steps_per_epoch * epochs
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
            )

        print("-" * 60)

    # 4. Tam Eğitim Döngüsü
    print(f"\n--- TAM EĞİTİM BAŞLIYOR ({direction.upper()}: {epochs} Epoch) ---")
    training_start_time = time.time()
    no_improve_epochs = 0
    best_epoch = start_epoch

    for ep in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        ep_start_t = time.time()
        model.train()
        running_loss = 0.0
        global_step = ep * steps_per_epoch

        optimizer.zero_grad()
        for b_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            loss.backward()

            if (b_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                running_loss += loss.item() * gradient_accumulation_steps

                # Her 500 adımda bir durum bas
                if global_step % 500 == 0:
                    lr_curr = lr_scheduler.get_last_lr()[0]
                    print(f"  [Epoch {ep+1}/{epochs} | Adım {global_step}/{total_training_steps}] Kayıp: {loss.item() * gradient_accumulation_steps:.4f} | LR: {lr_curr:.2e}")

        ep_duration = time.time() - ep_start_t
        ep_loss = running_loss / max(steps_per_epoch, 1)

        # Epoch sonu doğrulama
        print(f"Epoch {ep+1} tamamlandı ({ep_duration:.1f} sn, Ortalama Kayıp: {ep_loss:.4f}). Doğrulama yapılıyor...")
        val_cer = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_samples=val_samples,
            device=device,
            direction=direction,
            max_len=max_target_length,
        )
        print(f"  --> Epoch {ep+1} Doğrulama CER: %{val_cer * 100:.2f} (Önceki En İyi: %{best_cer * 100:.2f})")

        history.append({
            "epoch": ep + 1,
            "train_loss": round(ep_loss, 4),
            "val_cer": round(val_cer, 4),
            "duration_sec": round(ep_duration, 1),
        })

        # En iyi checkpoint'i kaydet
        if val_cer < best_cer:
            best_cer = val_cer
            best_epoch = ep + 1
            no_improve_epochs = 0
            print(f"  [+] YENİ EN İYİ MODEL! Kaydediliyor -> {output_dir}")
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
        else:
            no_improve_epochs += 1

        # Checkpoint durumunu kaydet (resume için)
        torch.save({
            "epoch": ep + 1,
            "best_cer": best_cer,
            "history": history,
        }, resume_checkpoint)

        # Erken durdurma (Early Stopping)
        if no_improve_epochs >= patience:
            print(f"Erken durdurma tetiklendi: {patience} epoch boyunca doğrulama CER iyileşmedi.")
            break

    total_time = time.time() - training_start_time
    print(f"\nEğitim tamamlandı! Toplam süre: {total_time / 60.0:.2f} dk. En iyi Epoch: {best_epoch} (CER: %{best_cer * 100:.2f})")

    # Eğitim geçmişini kaydet
    history_file = output_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "direction": direction,
            "model_name": model_name,
            "total_time_seconds": round(total_time, 2),
            "best_epoch": best_epoch,
            "best_val_cer": round(best_cer, 4),
            "history": history,
        }, f, ensure_ascii=False, indent=2)

    return {
        "direction": direction,
        "total_time_sec": total_time,
        "best_epoch": best_epoch,
        "best_val_cer": best_cer,
        "history": history,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="ByT5-small model eğitimi.")
    parser.add_argument("--direction", choices=["ters", "ileri"], required=True, help="Eğitim yönü")
    parser.add_argument("--model-name", default="google/byt5-small", help="Temel model adı")
    parser.add_argument("--output-dir", type=Path, default=None, help="Çıktı klasörü")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch boyutu")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Öğrenme oranı")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup adımları")
    parser.add_argument("--trial-steps", type=int, default=200, help="Deneme adım sayısı")
    parser.add_argument("--resume", action="store_true", help="Checkpoint'ten devam et")
    default_workers = 4 if os.name != "nt" else 0
    parser.add_argument("--num-workers", type=int, default=default_workers, help="Dataloader worker sayısı")
    parser.add_argument("--no-group-by-length", action="store_true", help="Uzunluğa göre gruplamayı kapat")
    parser.add_argument("--max-len", type=int, default=None, help="Maksimum token uzunluğu")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Gradient checkpointing etkinleştir")
    args = parser.parse_args()

    # Varsayılan epoch sayıları
    default_epochs = 10 if args.direction == "ters" else 5
    epochs = args.epochs if args.epochs is not None else default_epochs

    default_max_len = 64 if args.direction == "ters" else 512
    max_len = args.max_len if args.max_len is not None else default_max_len

    train_byt5(
        direction=args.direction,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_source_length=max_len,
        max_target_length=max_len,
        trial_steps=args.trial_steps,
        resume=args.resume,
        num_workers=args.num_workers,
        group_by_length=not args.no_group_by_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
