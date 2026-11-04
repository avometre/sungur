"""Altın test ve geliştirme veri seti üretim betiği (Gold Set Builder).

Bu modül; `data/interim/ws_parallel.jsonl` içerisindeki güvenilir satır çiftlerini alır,
eşik filtresi (--min-score, varsayılan 0.75) ve kısa satır kuralı uygular, tekrarları ayıklar
ve sayfa bazında deterministik SHA-1 karmasıyla yaklaşık %30 dev ve %70 test olarak ikiye böler.
Her kayda sayfanın altın setteki çiftlerinin ortalama skoru (sayfa_ort_skor) eklenir.
Ayrıca model eğitiminde veri sızıntısını önlemek için `heldout_pages.txt` ve
`heldout_arap.txt` kütüklerini oluşturur.
"""

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_latin
from src.translit.skeleton import arabic_skeleton, latin_skeleton

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL_PATH = Path("data/interim/ws_parallel.jsonl")
DEFAULT_GOLD_DIR = Path("data/gold")
DEFAULT_PAGE_STYLE_PATH = Path("data/gold/page_style.csv")
DEFAULT_MIN_SCORE = 0.75

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
6. `page_style.csv`: Sayfa bazında üslup ve çeviri karakteri tablosu (birebir vs. serbest).

## Üslup Ayrımı (birebir vs. serbest)
Altın setteki sayfalar kaynak metnin ve çevirinin doğasına göre iki gruba ayrılmıştır:
- `birebir`: Harfi harfine veya mısrası mısrasına tam uyuşan klasik metinler (şiirler, kanunlar, fermanlar vb.).
- `serbest`: Günümüz Türkçesine sadeleştirilmiş baskılardan alınan metinler (ör. Ömer Seyfettin hikâyeleri). Bu metinlerde "صاحیدن" (sahiden) yerine "gerçekten" gibi kelime değişiklikleri bulunabilir. Model değerlendirmesinde CER/WER/chrF skorları bu iki grupta ayrı olarak da ölçülür.

## KESİNLİKLE EĞİTİMDE KULLANILMAZ (VERİ SIZINTISI KORUMASI)
> [!IMPORTANT]
> Bu veri setindeki sayfalar ve satırlar yalnızca **model değerlendirme ve test** amaçlıdır.
> `src/gold/leakage.py` modülü aracılığıyla sonraki tüm eğitim verisi üretim boru hatlarında
> bu verilerin eğitime karışması engellenmektedir.
"""


def load_page_style_map(page_style_path: Path = DEFAULT_PAGE_STYLE_PATH) -> Dict[int, str]:
    """page_style.csv dosyasından sayfa kimliği -> üslup (birebir|serbest) eşlemesini yükler."""
    if not page_style_path.exists():
        raise FileNotFoundError(f"page_style.csv dosyası bulunamadı: {page_style_path}")

    style_map: Dict[int, str] = {}
    with open(page_style_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid_raw = row.get("pageid", "").strip()
            if not pid_raw:
                continue
            pid = int(pid_raw)
            uslup = row.get("uslup", "").strip()
            if not uslup:
                raise ValueError(f"Geçersiz veya boş üslup: pageid={pid}")
            style_map[pid] = uslup
    return style_map


def is_reliable_pair(
    pair: Dict[str, Any],
    min_score: float = DEFAULT_MIN_SCORE,
) -> bool:
    """Bir satır çiftinin altın set kalite kriterlerini sağlayıp sağlamadığını denetler.

    Kurallar:
    - skor >= min_score (varsayılan: 0.75)
    - Kısa satır kuralı: min(len(arabic_skeleton(arap)), len(latin_skeleton(latin))) < 4
      ise sadece tam eşleşmeler (skor >= 1.0) kabul edilir.
    """
    skor = float(pair.get("skor", 0.0))
    if skor < min_score:
        return False

    arap = str(pair.get("arap", ""))
    latin = str(pair.get("latin", ""))

    skel_a_len = len(arabic_skeleton(arap))
    skel_l_len = len(latin_skeleton(latin))
    min_skel_len = min(skel_a_len, skel_l_len)

    if min_skel_len < 4 and skor < 0.9999:
        return False

    return True


def partition_dev_test(
    records: List[Dict[str, Any]],
    dev_ratio: float = 0.3,
    page_avg_scores: Dict[int, float] = None,
    page_style_map: Optional[Dict[int, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Güvenilir ve tekrarsız çiftleri sayfa bazında deterministik SHA-1 ile dev ve test olarak ayırır.

    Aynı sayfanın satırları yalnızca tek bir bölmede yer alır.
    Her kayda 'sayfa_ort_skor' ve 'uslup' (birebir|serbest) eklenir.
    Eğer sayfaya ait üslup page_style_map içinde bulunamazsa ValueError fırlatır.
    """
    if page_avg_scores is None:
        page_avg_scores = {}

    dev_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    # Sayfa içi sıra takipçisi
    page_sira: Dict[int, int] = {}

    for r in records:
        pid = int(r["pageid"])
        sira = page_sira.get(pid, 0)
        page_sira[pid] = sira + 1

        # Üslup denetimi
        if page_style_map is not None:
            if pid not in page_style_map:
                raise ValueError(
                    f"Eşleşmeyen sayfa kimliği: {pid} (page_style.csv içinde bulunamadı)"
                )
            uslup = page_style_map[pid]
        else:
            uslup = r.get("uslup", "birebir")

        # Deterministik SHA-1 karması
        h = int(hashlib.sha1(str(pid).encode("utf-8")).hexdigest(), 16) % 100
        bolum = "dev" if h < int(dev_ratio * 100) else "test"

        latin_str = str(r.get("latin", ""))
        latin_sade = r.get("latin_sade") or normalize_latin(
            latin_str, strip_diacritics=True
        )

        sayfa_ort = page_avg_scores.get(pid, round(float(r.get("skor", 1.0)), 4))

        item = {
            "id": f"gold:{pid}:{sira}",
            "arap": r["arap"],
            "latin": latin_str,
            "latin_sade": latin_sade,
            "pageid": pid,
            "tr_baslik": r.get("tr_baslik", ""),
            "skor": r.get("skor", 1.0),
            "sayfa_ort_skor": sayfa_ort,
            "bolum": bolum,
            "uslup": uslup,
        }

        if bolum == "dev":
            dev_records.append(item)
        else:
            test_records.append(item)

    return dev_records, test_records


def build_gold_dataset(
    parallel_path: Path = DEFAULT_PARALLEL_PATH,
    gold_dir: Path = DEFAULT_GOLD_DIR,
    page_style_path: Optional[Path] = None,
    dev_ratio: float = 0.3,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Dict[str, Any]:
    """ws_parallel.jsonl dosyasından altın dev/test setlerini ve heldout dosyalarını üretir."""
    if not parallel_path.exists():
        raise FileNotFoundError(f"Paralel veri dosyası bulunamadı: {parallel_path}")

    gold_dir.mkdir(parents=True, exist_ok=True)

    if page_style_path is None:
        page_style_path = gold_dir / "page_style.csv"

    # Üslup haritasını yükle
    style_map = load_page_style_map(page_style_path)

    # 1. Ham veriyi oku
    all_raw = list(read_jsonl(parallel_path))

    # Eski havuzdaki tüm sayfa kimliklerini belirle (heldout_pages değişmesin)
    old_heldout_pids = set()
    heldout_pages_file = gold_dir / "heldout_pages.txt"
    if heldout_pages_file.exists():
        for line in heldout_pages_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    old_heldout_pids.add(int(line))
                except ValueError:
                    pass

    # 2. Kalite kriterlerine göre filtrele ve tekrarları ayıkla
    filtered_pairs = [p for p in all_raw if is_reliable_pair(p, min_score=min_score)]

    seen_pairs = set()
    deduped_pairs: List[Dict[str, Any]] = []
    for p in filtered_pairs:
        pair_key = (p["arap"], p["latin"])
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            deduped_pairs.append(p)

    # 3. Sayfa bazında ortalama skorları hesapla
    page_scores_map: Dict[int, List[float]] = {}
    for p in deduped_pairs:
        pid = int(p["pageid"])
        page_scores_map.setdefault(pid, []).append(float(p.get("skor", 1.0)))

    page_avg_scores: Dict[int, float] = {
        pid: round(sum(scores) / len(scores), 4)
        for pid, scores in page_scores_map.items()
    }

    # 4. Dev ve Test olarak ayır (üslup eşlemesiyle)
    dev_records, test_records = partition_dev_test(
        deduped_pairs,
        dev_ratio=dev_ratio,
        page_avg_scores=page_avg_scores,
        page_style_map=style_map,
    )

    # 5. Dosyaları yaz
    dev_path = gold_dir / "translit_dev.jsonl"
    test_path = gold_dir / "translit_test.jsonl"
    write_jsonl(dev_path, dev_records)
    write_jsonl(test_path, test_records)

    # 6. Heldout listelerini yaz
    current_pids = {r["pageid"] for r in (dev_records + test_records)}
    final_heldout_pids = sorted(list(old_heldout_pids.union(current_pids)))

    # heldout_arap.txt yeniden üretilsin
    heldout_araps = sorted(list({r["arap"].strip() for r in (dev_records + test_records)}))

    (gold_dir / "heldout_pages.txt").write_text(
        "\n".join(str(p) for p in final_heldout_pids) + "\n", encoding="utf-8"
    )
    (gold_dir / "heldout_arap.txt").write_text(
        "\n".join(heldout_araps) + "\n", encoding="utf-8"
    )

    # 7. README.md yaz
    (gold_dir / "README.md").write_text(README_CONTENT, encoding="utf-8")

    dev_pids = {r["pageid"] for r in dev_records}
    test_pids = {r["pageid"] for r in test_records}

    all_gold = dev_records + test_records
    birebir_ciftler = [r for r in all_gold if r.get("uslup") == "birebir"]
    serbest_ciftler = [r for r in all_gold if r.get("uslup") == "serbest"]

    summary = {
        "toplam_paralel_kayit": len(all_raw),
        "filtrelenen_kayit": len(filtered_pairs),
        "tekrarsiz_guvenilir": len(deduped_pairs),
        "toplam_sayfa": len(current_pids),
        "heldout_sayfa_sayisi": len(final_heldout_pids),
        "dev_sayfa_sayisi": len(dev_pids),
        "dev_cift_sayisi": len(dev_records),
        "test_sayfa_sayisi": len(test_pids),
        "test_cift_sayisi": len(test_records),
        "dev_cift_orani": (len(dev_records) / len(deduped_pairs)) if deduped_pairs else 0.0,
        "test_cift_orani": (len(test_records) / len(deduped_pairs)) if deduped_pairs else 0.0,
        "min_score": min_score,
        "birebir_sayfa_sayisi": len({r["pageid"] for r in birebir_ciftler}),
        "birebir_cift_sayisi": len(birebir_ciftler),
        "serbest_sayfa_sayisi": len({r["pageid"] for r in serbest_ciftler}),
        "serbest_cift_sayisi": len(serbest_ciftler),
        "dev_birebir_cift": sum(1 for r in dev_records if r.get("uslup") == "birebir"),
        "dev_serbest_cift": sum(1 for r in dev_records if r.get("uslup") == "serbest"),
        "test_birebir_cift": sum(1 for r in test_records if r.get("uslup") == "birebir"),
        "test_serbest_cift": sum(1 for r in test_records if r.get("uslup") == "serbest"),
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
    parser.add_argument(
        "--page-style-file",
        type=Path,
        default=DEFAULT_PAGE_STYLE_PATH,
        help="Sayfa üslup eşleme dosyası (varsayılan: data/gold/page_style.csv)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Altın sete girme minimum benzerlik skoru (varsayılan: 0.75)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ALTIN VERİ SETİ ÜRETİMİ (build_gold)")
    print("=" * 80)

    summary = build_gold_dataset(
        parallel_path=args.parallel_file,
        gold_dir=args.gold_dir,
        page_style_path=args.page_style_file,
        dev_ratio=args.dev_ratio,
        min_score=args.min_score,
    )

    print(f"Minimum Skor Eşiği           : {summary['min_score']}")
    print(f"Toplam Giren Paralel Çift    : {summary['toplam_paralel_kayit']}")
    print(f"Filtreyi Geçen Çift Sayısı   : {summary['filtrelenen_kayit']}")
    print(f"Tekrarsız Güvenilir Çift     : {summary['tekrarsiz_guvenilir']}")
    print(f"Toplam Sayfa Sayısı          : {summary['toplam_sayfa']}")
    print(f"Heldout Sayfa Sayısı         : {summary['heldout_sayfa_sayisi']}")
    print("-" * 80)
    print(
        f"Geliştirme (Dev) Seti        : {summary['dev_sayfa_sayisi']} sayfa, {summary['dev_cift_sayisi']} çift (%{summary['dev_cift_orani']*100:.1f})"
    )
    print(
        f"  - Birebir Çift             : {summary['dev_birebir_cift']}"
    )
    print(
        f"  - Serbest Çift             : {summary['dev_serbest_cift']}"
    )
    print(
        f"Test Seti                    : {summary['test_sayfa_sayisi']} sayfa, {summary['test_cift_sayisi']} çift (%{summary['test_cift_orani']*100:.1f})"
    )
    print(
        f"  - Birebir Çift             : {summary['test_birebir_cift']}"
    )
    print(
        f"  - Serbest Çift             : {summary['test_serbest_cift']}"
    )
    print("-" * 80)
    print(f"Toplam Birebir               : {summary['birebir_sayfa_sayisi']} sayfa, {summary['birebir_cift_sayisi']} çift")
    print(f"Toplam Serbest               : {summary['serbest_sayfa_sayisi']} sayfa, {summary['serbest_cift_sayisi']} çift")
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
