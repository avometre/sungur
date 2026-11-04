import json
from pathlib import Path
from typing import Iterable, Iterator, Union


def read_jsonl(path: Union[str, Path]) -> Iterator[dict]:
    """UTF-8 formatında JSONL dosyasını okur. Boş satırları atlar."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def write_jsonl(path: Union[str, Path], records: Iterable[dict]) -> int:
    """UTF-8 formatında JSONL dosyası yazar. Klasörü yoksa oluşturur.

    Yazılan kayıt sayısını döndürür.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Union[str, Path], record: dict) -> None:
    """UTF-8 formatında JSONL dosyasına tek bir kayıt ekler.

    Klasörü yoksa oluşturur.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
