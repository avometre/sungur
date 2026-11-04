import json
from pathlib import Path
from src.common.io import append_jsonl, read_jsonl, write_jsonl


def test_write_and_read_jsonl(tmp_path: Path) -> None:
    target_file = tmp_path / "subfolder" / "data.jsonl"
    records = [
        {"id": 1, "text": "پادشاه‌لر", "latin": "padişahlar"},
        {"id": 2, "text": "كتاب", "latin": "kitab"},
    ]

    count = write_jsonl(target_file, records)
    assert count == 2
    assert target_file.exists()

    # UTF-8 karakterlerin kaçışsız (ensure_ascii=False) yazıldığını doğrula
    raw_content = target_file.read_text(encoding="utf-8")
    assert "پادشاه‌لر" in raw_content
    assert "\\u" not in raw_content

    # Okuma testi
    loaded = list(read_jsonl(target_file))
    assert loaded == records


def test_append_jsonl_and_blank_lines(tmp_path: Path) -> None:
    target_file = tmp_path / "append_test.jsonl"
    write_jsonl(target_file, [{"id": 1, "text": "ilk"}])

    # Araya boş satır ekleyelim
    with target_file.open("a", encoding="utf-8") as f:
        f.write("\n   \n\t\n")

    append_jsonl(target_file, {"id": 2, "text": "ikinci"})

    loaded = list(read_jsonl(target_file))
    assert len(loaded) == 2
    assert loaded[0]["id"] == 1
    assert loaded[1]["id"] == 2
