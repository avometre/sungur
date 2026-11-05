"""ByT5 eğitim veri hazırlığı, ön ek ekleme ve belirteçleme birim testleri."""

import pytest
import torch
from unittest.mock import MagicMock

from src.translit.train_byt5 import collate_fn_builder


def test_collate_fn_tokenization_and_labels():
    """collate_fn_builder'ın doğru padding, tensör tipi ve -100 maskesi ürettiğini sınar."""
    # Sahte bir tokenizer oluştur
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    def mock_tokenize(texts, max_length, padding, truncation, return_tensors):
        # Sahte input_ids: metin uzunluğuna göre tensör üret
        max_t_len = max(len(t) for t in texts)
        ids = []
        for t in texts:
            row = [ord(c) % 250 + 1 for c in t] + [0] * (max_t_len - len(t))
            ids.append(row[:max_length])
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}

    tokenizer.side_effect = mock_tokenize

    collate_fn = collate_fn_builder(tokenizer, max_source_len=16, max_target_len=16)
    batch = [
        ("<vk> \u06a9\u062a\u0627\u0628", "kitab"),
        ("<kelime> \u062f\u0648\u0644\u062a", "devlet"),
    ]

    res = collate_fn(batch)
    assert "input_ids" in res
    assert "labels" in res
    assert res["input_ids"].shape[0] == 2
    assert res["labels"].shape[0] == 2

    # Pad olan yerler labels'ta -100 olmalı
    # "kitab" (5) vs "devlet" (6) -> "kitab" sonunda 1 pad token (0) var
    assert -100 in res["labels"].tolist()[0]


def test_prefix_and_mixture_logic():
    """Ön ek ekleme (<vk>, <ijmes>, <kelime>) ve ağırlıklı karışım oranını sınar."""
    # vk: 2x, latoc: 1x, kelime: 1x
    vk_raw = [{"arap": "\u0627\u0644\u0641", "latin": "elif"}]
    latoc_raw = [{"arap": "\u0628\u0627", "latin": "ba"}]
    word_raw = [{"arap": "\u062a\u0627", "latin": "ta"}]

    vk_samples = [(f"<vk> {r['arap']}", r["latin"]) for r in vk_raw]
    latoc_samples = [(f"<ijmes> {r['arap']}", r["latin"]) for r in latoc_raw]
    word_samples = [(f"<kelime> {r['arap']}", r["latin"]) for r in word_raw]

    # Karışım 2x vk + 1x latoc + 1x word
    mixture = (vk_samples * 2) + latoc_samples + word_samples

    assert len(mixture) == 4
    # 2 adet vk
    vk_count = sum(1 for s, _ in mixture if s.startswith("<vk> "))
    latoc_count = sum(1 for s, _ in mixture if s.startswith("<ijmes> "))
    word_count = sum(1 for s, _ in mixture if s.startswith("<kelime> "))

    assert vk_count == 2
    assert latoc_count == 1
    assert word_count == 1
