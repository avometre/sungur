"""Altın test ve geliştirme veri seti üretim betiği (Gold Set Builder).

Bu modül; `data/interim/ws_parallel.jsonl` içerisindeki güvenilir (guvenilir == True)
satır çiftlerini alır, tekrarları ayıklar ve sayfa bazında deterministik SHA-1 karmasıyla
yaklaşık %30 dev ve %70 test olarak ikiye böler.
Ayrıca model eğitiminde veri sızıntısını önlemek için `heldout_pages.txt` ve
`heldout_arap.txt` kütüklerini oluşturur.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_latin

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL_PATH = Path("data/interim/ws_parallel.jsonl")
DEFAULT_GOLD_DIR = Path("data/gold")

README_CONTENT = """# Osmanlıca → Türkçe Çeviriyazı Altın Değerlendirme Seti (Gold Set)

Bu dizin, Osmanlıca (Arap harfli) metinler ile günümüz Türkçesi (Latin harfli) karşılıklarını
içeren yüksek kaliteli ve filtrelenmiş altın değerlendirme setini barındırır.

## Kaynaklar ve Lisans
- **Kaynaklar**: Osmanlıca Vikikaynak (ota.wikisource.org) ve Türkçe Vikikaynak (tr.wikisource.org)
- **Lisans**: CC BY-SA 4.0 / Kamu Malı (orijinal eserlere göre)
- **Üretim Yöntemi**: Gale-Church benzeri dinamik programlama satır hizalama v2 (cümle bölme ve içerik doğrulama eşikleriyle)

## Veri Seti Bölümleri
1. `translit_dev.jsonl`: Model geliştirme, hiperparametre ayarlama ve ara doğrulama seti (~%30).
2. `translit_test.jsonl`: Modelin nihai başarısını ölçen kör test seti (~%70).
3. `heldout_pages.txt`: Değerlendirme setinde yer alan tüm sayfa kimlikleri (`pageid`).
4. `heldout_arap.txt`: Değerlendirme setinde yer alan tüm Arapça satır metinleri.
5. `ws_title_map.csv`: Sayfa başlıkları eşleme ve aday arama takip tablosu.

## KESİNLİKLE EĞİTİMDE KULLANILMAZ (VERİ SIZINTISI KORUMASI)
> [!IMPORTANT]
> Bu veri setindeki sayfalar ve satırlar yalnızca **model değerlendirme ve test** amaçlıdır.
> `src/gold/leakage.py` modülü aracılığıyla sonraki tüm eğitim verisi üretim boru hatlarında
> bu verilerin eğitime karışması engellenmektedir.
"""


def partition_dev_test(
    records: List[Dict[str, Any]], dev_ratio: float = 0.3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Güvenilir ve tekrarsız çiftleri sayfa bazında deterministik SHA-1 ile dev ve test olarak ayırır.

    Aynı sayfanın satırları yalnızca tek bir bölmede yer alır.
    """
    dev_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    # Sayfa içi sıra takipçisi
    page_sira: Dict[int, int] = {}

    for r in records:
        pid = int(r["pageid"])
        sira = page_sira.get(pid, 0)
        page_sira[pid] = sira + 1

        # Deterministik SHA-1 karması
        h = int(hashlib.sha1(str(pid).encode("utf-8")).hexdigest(), 16) % 100
        bolum = "dev" if h < int(dev_ratio * 100) else "test"

        latin_str = str(r.get("latin", ""))
        latin_sade = r.get("latin_sade") or normalize_latin(
            latin_str, strip_diacritics=True
        )

        item = {
            "id": f"gold:{pid}:{sira}",
            "arap": r["arap"],
            "latin": latin_str,
            "latin_sade": latin_sade,
            "pageid": pid,
            "tr_baslik": r.get("tr_baslik", ""),
            "skor": r.get("skor", 1.0),
            "bolum": bolum,
        }

        if bolum == "dev":
            dev_records.append(item)
        else:
            test_records.append(item)

    return dev_records, test_records


def build_gold_dataset(
    parallel_path: Path = DEFAULT_PARALLEL_PATH,
    gold_dir: Path = DEFAULT_GOLD_DIR,
    dev_ratio: float = 0.3,
) -> Dict[str, Any]:
    """ws_parallel.jsonl dosyasından altın dev/test setlerini ve heldout dosyalarını üretir."""
    if not parallel_path.exists():
        raise FileNotFoundError(f"Paralel veri dosyası bulunamadı: {parallel_path}")

    gold_dir.mkdir(parents=True, exist_ok=True)

    # 1. Güvenilir çiftleri filtrele ve (arap, latin) tekrarlarını ayıkla
    all_raw = list(read_jsonl(parallel_path))
    reliable_pairs = [p for p in all_raw if p.get("guvenilir") is True]

    seen_pairs = set()
    deduped_pairs: List[Dict[str, Any]] = []
    for p in reliable_pairs:
        pair_key = (p["arap"], p["latin"])
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            deduped_pairs.append(p)

    # 2. Dev ve Test olarak ayır
    dev_records, test_records = partition_dev_test(deduped_pairs, dev_ratio=dev_ratio)

    # 3. Dosyaları yaz
    dev_path = gold_dir / "translit_dev.jsonl"
    test_path = gold_dir / "translit_test.jsonl"
    write_jsonl(dev_path, dev_records)
    write_jsonl(test_path, test_records)

    # 4. Heldout listelerini yaz
    heldout_pids = sorted(list({r["pageid"] for r in (dev_records + test_records)}))
    heldout_araps = sorted(list({r["arap"].strip() for r in (dev_records + test_records)}))

    (gold_dir / "heldout_pages.txt").write_text(
        "\n".join(str(p) for p in heldout_pids) + "\n", encoding="utf-8"
    )
    (gold_dir / "heldout_arap.txt").write_text(
        "\n".join(heldout_araps) + "\n", encoding="utf-8"
    )

    # 5. README.md yaz
    (gold_dir / "README.md").write_text(README_CONTENT, encoding="utf-8")

    dev_pids = {r["pageid"] for r in dev_records}
    test_pids = {r["pageid"] for r in test_records}

    summary = {
        "toplam_paralel_kayit": len(all_raw),
        "guvenilir_kayit": len(reliable_pairs),
        "tekrarsiz_guvenilir": len(deduped_pairs),
        "toplam_sayfa": len(heldout_pids),
        "dev_sayfa_sayisi": len(dev_pids),
        "dev_cift_sayisi": len(dev_records),
        "test_sayfa_sayisi": len(test_pids),
        "test_cift_sayisi": len(test_records),
        "dev_cift_orani": (len(dev_records) / len(deduped_pairs)) if deduped_pairs else 0.0,
        "test_cift_orani": (len(test_records) / len(deduped_pairs)) if deduped_pairs else 0.0,
    }
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Altın değerlendirme seti (dev/test) üret")
    parser.add_argument(
        "--parallel-file",
        type=Path,
        default=DEFAULT_PARALLEL_PATH,
        help="Girdi paralel hizalama dosyası",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=DEFAULT_GOLD_DIR,
        help="Altın veri setlerinin kaydedileceği dizin",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.3,
        help="Geliştirme (dev) oranı (varsayılan: 0.3)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ALTIN VERİ SETİ ÜRETİMİ (build_gold)")
    print("=" * 80)

    summary = build_gold_dataset(
        parallel_path=args.parallel_file,
        gold_dir=args.gold_dir,
        dev_ratio=args.dev_ratio,
    )

    print(f"Toplam Giren Paralel Çift    : {summary['toplam_paralel_kayit']}")
    print(f"Güvenilir Çift (skor >= 0.55): {summary['guvenilir_kayit']}")
    print(f"Tekrarsız Güvenilir Çift    : {summary['tekrarsiz_guvenilir']}")
    print(f"Toplam Sayfa Sayısı         : {summary['toplam_sayfa']}")
    print("-" * 80)
    print(
        f"Geliştirme (Dev) Seti       : {summary['dev_sayfa_sayisi']} sayfa, {summary['dev_cift_sayisi']} çift (%{summary['dev_cift_orani']*100:.1f})"
    )
    print(
        f"Test Seti                   : {summary['test_sayfa_sayisi']} sayfa, {summary['test_cift_sayisi']} çift (%{summary['test_cift_orani']*100:.1f})"
    )
    print("-" * 80)
    print("Üretilen Dosyalar:")
    print("  - data/gold/translit_dev.jsonl")
    print("  - data/gold/translit_test.jsonl")
    print("  - data/gold/heldout_pages.txt")
    print("  - data/gold/heldout_arap.txt")
    print("  - data/gold/README.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
