"""Ham veri setlerini temizleme, filtreleme ve standartlaştırma betiği.

GÖREV 02 kapsamında:
- Wikisource: Kürtçe sayfaları/satırları, kısa satırları ve sayfa içi tekrarları ayıklar.
- Wiktionary: Latin okunuşu olmayanları eler, mükerrer kayıtları birleştirir,
  düşük iskelet benzerlikli kayıtları 'supheli' olarak işaretler.
- Gazetteer: Çoklu okunuş adaylarını ayırır, Arapça imlaya en uygun olanı iskelet
  benzerliğiyle seçer, güvenilirlik ve belirsizlik alanlarını ekler.

Tüm temiz veriler `data/interim/`, elenen kayıtlar `data/interim/excluded/` dizinine kaydedilir.
"""

import argparse
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import (
    arabic_letter_count,
    arabic_ratio,
    has_kurdish_letters,
    normalize_latin,
    normalize_ottoman,
)
from src.translit.skeleton import fine_skeleton_similarity, skeleton_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Wikisource Temizleme Fonksiyonları
# ---------------------------------------------------------------------------


def clean_wikisource(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Wikisource ham verisini temizler.

    Kurallar:
    - Sayfa başlığı Kürtçe harf içeriyorsa VEYA sayfadaki satırların %20'sinden fazlası
      Kürtçe harf içeriyorsa sayfanın TÜMÜ elenir (neden: 'kurtce_sayfa').
    - Kalan sayfalarda:
      * Kürtçe harf içeren satırlar elenir (neden: 'kurtce_satir').
      * arap alanı normalize_ottoman ile arap_ham'dan yeniden üretilir.
      * arabic_letter_count(arap) < 3 olanlar elenir (neden: 'cok_kisa').
      * arabic_ratio(arap_ham) < 0.6 olanlar elenir (neden: 'latin_agirlikli').
      * Aynı sayfada daha önce geçmiş aynı 'arap' metni elenir (neden: 'tekrar').
      * Geçerli satırlara sayfa içi artan 'sira' alanı atanır.
    """
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    # 1. Kayıtları sayfa bazında grupla (pageid veya id öneki)
    pages: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        rec_id = str(r.get("id", ""))
        pageid = rec_id.split(":")[1] if ":" in rec_id else str(r.get("baslik", "unknown"))
        pages.setdefault(pageid, []).append(r)

    kurdish_page_titles: list[tuple[str, str, int]] = []
    cross_page_duplicates_count = 0
    seen_across_all_pages: set[str] = set()

    for pid, page_lines in pages.items():
        if not page_lines:
            continue

        title = str(page_lines[0].get("baslik", ""))
        k_in_title = has_kurdish_letters(title)
        k_lines_count = sum(
            1 for line in page_lines if has_kurdish_letters(str(line.get("arap_ham", "")))
        )
        k_ratio = k_lines_count / len(page_lines)

        # Sayfa eleniyor mu?
        if k_in_title or k_ratio > 0.20:
            kurdish_page_titles.append((pid, title, len(page_lines)))
            for line in page_lines:
                ex = dict(line)
                ex["neden"] = "kurtce_sayfa"
                excluded.append(ex)
            continue

        # Sayfa geçerli, satırları incele
        page_sira = 0
        seen_in_this_page: set[str] = set()

        for line in page_lines:
            ham = str(line.get("arap_ham", ""))

            # Kürtçe harf kontrolü
            if has_kurdish_letters(ham):
                ex = dict(line)
                ex["neden"] = "kurtce_satir"
                excluded.append(ex)
                continue

            # arap alanını güncellenmiş normalize_ottoman ile yeniden üret
            arap = normalize_ottoman(ham)
            line_copy = dict(line)
            line_copy["arap"] = arap

            # Kısa satır kontrolü
            if arabic_letter_count(arap) < 3:
                line_copy["neden"] = "cok_kisa"
                excluded.append(line_copy)
                continue

            # Latin ağırlıklı satır kontrolü
            if arabic_ratio(ham) < 0.6:
                line_copy["neden"] = "latin_agirlikli"
                excluded.append(line_copy)
                continue

            # Sayfa içi tekrar kontrolü
            if arap in seen_in_this_page:
                line_copy["neden"] = "tekrar"
                excluded.append(line_copy)
                continue

            # Sayfalar arası tekrar sayımı (elenmez, sadece sayılır)
            if arap in seen_across_all_pages:
                cross_page_duplicates_count += 1
            else:
                seen_across_all_pages.add(arap)

            seen_in_this_page.add(arap)
            line_copy["sira"] = page_sira
            page_sira += 1
            kept.append(line_copy)

    stats = {
        "giren": len(records),
        "kalan": len(kept),
        "atilan": len(excluded),
        "nedenler": Counter(ex["neden"] for ex in excluded),
        "kurtce_sayfalar": kurdish_page_titles,
        "farkli_sayfalarda_tekrar_sayisi": cross_page_duplicates_count,
    }
    return kept, excluded, stats


# ---------------------------------------------------------------------------
# 2. Wiktionary Temizleme Fonksiyonları
# ---------------------------------------------------------------------------


def clean_wiktionary(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Wiktionary verisini temizler ve şüpheli kayıtları belirler.

    Kurallar:
    - arap alanı normalize_ottoman ile yeniden üretilir.
    - arap harfi içermeyenler elenir (neden: 'arap_harfi_yok').
    - latin alanı boş olanlar elenir (neden: 'latin_yok').
    - (arap, latin_sade, tur) üçlüsüne göre tekrarlar birleştirilir.
    - skeleton_similarity(arap, latin) < 0.3 olanlar SİLİNMEZ, 'supheli': True işaretlenir.
    """
    excluded: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []

    for r in records:
        r_copy = dict(r)
        ham = str(r_copy.get("arap_ham", ""))
        r_copy["arap"] = normalize_ottoman(ham)

        # Arap harfi içermeyen kayıtları ele (rakam girdileri vb.)
        if arabic_letter_count(ham) == 0:
            r_copy["neden"] = "arap_harfi_yok"
            excluded.append(r_copy)
            continue

        latin = r_copy.get("latin")
        if not latin or not str(latin).strip():
            r_copy["neden"] = "latin_yok"
            excluded.append(r_copy)
            continue

        r_copy["latin"] = str(latin).strip()
        r_copy["latin_sade"] = normalize_latin(r_copy["latin"], strip_diacritics=True)
        valid_records.append(r_copy)

    # (arap, latin_sade, tur) üçlüsüne göre birleştirme
    grouped: dict[tuple[str, Optional[str], str], list[dict[str, Any]]] = {}
    for r in valid_records:
        key = (r["arap"], r["latin_sade"], str(r.get("tur", "")))
        grouped.setdefault(key, []).append(r)

    kept: list[dict[str, Any]] = []
    supheli_list: list[dict[str, Any]] = []

    for key, group in grouped.items():
        base = dict(group[0])

        # Anlam, turkce ve form_of listelerini birleştir
        all_anlam: list[str] = []
        all_turkce: list[str] = []
        all_form_of: list[dict[str, Any]] = []

        for item in group:
            for a in item.get("anlam", []):
                if a and a not in all_anlam:
                    all_anlam.append(a)
            for t in item.get("turkce", []):
                if t and t not in all_turkce:
                    all_turkce.append(t)
            for fo in item.get("form_of", []):
                if fo and fo not in all_form_of:
                    all_form_of.append(fo)

        base["anlam"] = all_anlam
        base["turkce"] = all_turkce
        base["form_of"] = all_form_of
        base["cekim_mi"] = bool(all_form_of)

        # İskelet benzerliği denetimi
        sim = skeleton_similarity(base["arap"], base["latin"])
        base["eslesme_skoru"] = round(sim, 4)
        if sim < 0.3:
            base["supheli"] = True
            supheli_list.append(base)
        else:
            base["supheli"] = False

        kept.append(base)

    stats = {
        "giren": len(records),
        "kalan": len(kept),
        "atilan": len(excluded),
        "birlestirilen_tekrar": len(valid_records) - len(kept),
        "nedenler": Counter(ex["neden"] for ex in excluded),
        "supheli_sayisi": len(supheli_list),
        "supheli_ornekler": supheli_list[:10],
    }
    return kept, excluded, stats


# ---------------------------------------------------------------------------
# 3. Gazetteer Temizleme ve Okunuş Adayı Seçimi
# ---------------------------------------------------------------------------


def extract_gazetteer_candidates(latin_str: str) -> tuple[list[str], bool]:
    """Gazetteer Latin sütununu parçalayarak okunuş adaylarını ve belirsizlik durumunu çıkarır.

    Örnek: 'Abadan/Hazar (el-Hazra)' -> (['Abadan', 'Hazar', 'el-Hazra'], False)
    Örnek: 'İpsili (?) / Kanlıada' -> (['İpsili', 'Kanlıada'], True)
    """
    if not latin_str:
        return [], False

    belirsiz = "(?)" in latin_str
    cleaned_str = latin_str.replace("(?)", "")

    # Parantez içlerini bul ve ana metinden çıkar
    parens = re.findall(r"\((.*?)\)", cleaned_str)
    main_text = re.sub(r"\(.*?\)", "", cleaned_str)

    candidates: list[str] = []

    def _add(cand: str) -> None:
        c = cand.strip(" \t\r\n/,.?")
        if c and c not in candidates:
            candidates.append(c)

    for part in re.split(r"[/,]", main_text):
        _add(part)
    for p in parens:
        for part in re.split(r"[/,]", p):
            _add(part)

    return candidates, belirsiz


def clean_gazetteer(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Gazetteer verisinden okunuş adaylarını seçer ve mükerrerleri eler.

    Kurallar:
    - arap alanı normalize_ottoman ile yeniden üretilir.
    - latin alanından adaylar çıkarılır. Aday yoksa elenir ('latin_yok').
    - Her aday için skeleton_similarity(arap, aday) hesaplanır; en yükseği 'latin' seçilir.
    - En yüksek iki adayın kaba skor farkı <= 0.02 ise ince iskelet (fine_skeleton_similarity)
      ile B/P ve C/Ç sınıfları ayrılarak eşitlik bozulur.
    - 'latin_adaylar', 'eslesme_skoru', 'guvenilir': (skor >= 0.6) ve 'belirsiz' alanları eklenir.
    - 'latin_sade' seçilen adaydan yeniden üretilir.
    - (arap, latin) tekrarları elenir (neden: 'tekrar').
    """
    excluded: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []

    for r in records:
        r_copy = dict(r)
        ham = str(r_copy.get("arap_ham", ""))
        arap = normalize_ottoman(ham)
        r_copy["arap"] = arap

        raw_latin = str(r_copy.get("latin", "") or "")
        candidates, belirsiz = extract_gazetteer_candidates(raw_latin)

        if not candidates:
            r_copy["neden"] = "latin_yok"
            excluded.append(r_copy)
            continue

        # En iyi adayı seç (kaba iskelet benzerliği)
        scored_candidates = [(cand, skeleton_similarity(arap, cand)) for cand in candidates]
        # Skora göre azalan sırala (kararlı sıralama: eşit skorda adayların geliş sırası korunur)
        sorted_cands = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

        best_cand, best_score = sorted_cands[0]

        # En yüksek iki adayın skor farkı <= 0.02 ise ince iskelet ile eşitlik boz
        if len(sorted_cands) >= 2:
            cand1, score1 = sorted_cands[0]
            cand2, score2 = sorted_cands[1]
            if round(score1 - score2, 4) <= 0.02:
                fine1 = fine_skeleton_similarity(arap, cand1)
                fine2 = fine_skeleton_similarity(arap, cand2)
                if fine2 > fine1:
                    best_cand = cand2
                    best_score = score2
                else:
                    best_cand = cand1
                    best_score = score1

        r_copy["latin_adaylar"] = candidates
        r_copy["latin"] = best_cand
        r_copy["eslesme_skoru"] = round(best_score, 4)
        r_copy["guvenilir"] = best_score >= 0.6
        if belirsiz:
            r_copy["belirsiz"] = True
        r_copy["latin_sade"] = normalize_latin(best_cand, strip_diacritics=True)
        processed.append(r_copy)

    # (arap, latin) tekrarlarını ayıkla
    kept: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for r in processed:
        pair_key = (r["arap"], r["latin"])
        if pair_key in seen_pairs:
            r["neden"] = "tekrar"
            excluded.append(r)
        else:
            seen_pairs.add(pair_key)
            kept.append(r)

    guvenilir_count = sum(1 for r in kept if r.get("guvenilir"))
    guvenilir_oran = (guvenilir_count / len(kept)) if kept else 0.0

    multi_cand_examples = [
        {
            "arap": r["arap"],
            "adaylar": r["latin_adaylar"],
            "secilen": r["latin"],
            "skor": r["eslesme_skoru"],
        }
        for r in kept
        if len(r.get("latin_adaylar", [])) > 1
    ]

    rng = random.Random(42)
    sample_15 = (
        rng.sample(multi_cand_examples, min(15, len(multi_cand_examples)))
        if multi_cand_examples
        else []
    )

    stats = {
        "giren": len(records),
        "kalan": len(kept),
        "atilan": len(excluded),
        "nedenler": Counter(ex["neden"] for ex in excluded),
        "guvenilir_sayisi": guvenilir_count,
        "guvenilir_oran": guvenilir_oran,
        "cok_adayli_sayisi": len(multi_cand_examples),
        "cok_adayli_ornekler": sample_15,
    }
    return kept, excluded, stats


# ---------------------------------------------------------------------------
# CLI ve Ana Akış
# ---------------------------------------------------------------------------


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Ham Osmanlıca veri setlerini temizle")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Ham JSONL dosyalarının bulunduğu dizin",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Temizlenmiş JSONL dosyalarının kaydedileceği dizin",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    interim_dir: Path = args.interim_dir
    excluded_dir: Path = interim_dir / "excluded"

    interim_dir.mkdir(parents=True, exist_ok=True)
    excluded_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VERİ TEMİZLİĞİ BAŞLIYOR (clean_raw)")
    print("=" * 80)

    # 1. Wikisource Temizliği
    ws_raw_file = raw_dir / "wikisource_ota.jsonl"
    if ws_raw_file.exists():
        print(f"\n1) Wikisource Temizleniyor: {ws_raw_file}")
        ws_records = list(read_jsonl(ws_raw_file))
        ws_kept, ws_excluded, ws_stats = clean_wikisource(ws_records)

        write_jsonl(interim_dir / "wikisource_clean.jsonl", ws_kept)
        write_jsonl(excluded_dir / "wikisource.jsonl", ws_excluded)

        print(f"   Giren Satır   : {ws_stats['giren']}")
        print(f"   Kalan Satır   : {ws_stats['kalan']}")
        print(f"   Atılan Satır  : {ws_stats['atilan']}")
        print("   Atılma Nedenleri Dağılımı:")
        for reason, cnt in ws_stats["nedenler"].most_common():
            print(f"     - {reason:<16}: {cnt:>5}")
        print(f"   Elenen Kürtçe Sayfa Sayısı: {len(ws_stats['kurtce_sayfalar'])}")
        for pid, title, count in ws_stats["kurtce_sayfalar"]:
            print(f"     * [{pid}] {title} ({count} satır)")
        print(
            f"   Farklı sayfalarda mükerrer satır sayısı (korunan): {ws_stats['farkli_sayfalarda_tekrar_sayisi']}"
        )
    else:
        print(f"UYARI: {ws_raw_file} bulunamadı, Wikisource temizliği atlandı.")

    # 2. Wiktionary Temizliği
    wikt_raw_file = raw_dir / "wiktionary_ota.jsonl"
    if wikt_raw_file.exists():
        print(f"\n2) Wiktionary Temizleniyor: {wikt_raw_file}")
        wikt_records = list(read_jsonl(wikt_raw_file))
        wikt_kept, wikt_excluded, wikt_stats = clean_wiktionary(wikt_records)

        write_jsonl(interim_dir / "wiktionary_clean.jsonl", wikt_kept)
        write_jsonl(excluded_dir / "wiktionary.jsonl", wikt_excluded)

        print(f"   Giren Kayıt   : {wikt_stats['giren']}")
        print(f"   Kalan Kayıt   : {wikt_stats['kalan']}")
        print(f"   Atılan Kayıt  : {wikt_stats['atilan']}")
        print(f"   Birleştirilen : {wikt_stats['birlestirilen_tekrar']}")
        print("   Atılma Nedenleri Dağılımı:")
        for reason, cnt in wikt_stats["nedenler"].most_common():
            print(f"     - {reason:<16}: {cnt:>5}")
        print(f"   Şüpheli Kayıt Sayısı (sim < 0.3): {wikt_stats['supheli_sayisi']}")
        print("   Şüpheli Kayıtlardan 10 Örnek:")
        for idx, ex in enumerate(wikt_stats["supheli_ornekler"], 1):
            print(
                f"     {idx:>2}. {ex.get('arap')} | {ex.get('latin')} | skor={ex.get('eslesme_skoru')}"
            )
    else:
        print(f"UYARI: {wikt_raw_file} bulunamadı, Wiktionary temizliği atlandı.")

    # 3. Gazetteer Temizliği
    gaz_raw_file = raw_dir / "gazetteer.jsonl"
    if gaz_raw_file.exists():
        print(f"\n3) Gazetteer Temizleniyor ve Adaylar Seçiliyor: {gaz_raw_file}")
        gaz_records = list(read_jsonl(gaz_raw_file))
        gaz_kept, gaz_excluded, gaz_stats = clean_gazetteer(gaz_records)

        write_jsonl(interim_dir / "gazetteer_clean.jsonl", gaz_kept)
        write_jsonl(excluded_dir / "gazetteer.jsonl", gaz_excluded)

        print(f"   Giren Kayıt   : {gaz_stats['giren']}")
        print(f"   Kalan Kayıt   : {gaz_stats['kalan']}")
        print(f"   Atılan Kayıt  : {gaz_stats['atilan']}")
        print("   Atılma Nedenleri Dağılımı:")
        for reason, cnt in gaz_stats["nedenler"].most_common():
            print(f"     - {reason:<16}: {cnt:>5}")
        print(
            f"   Güvenilir Kayıt Oranı (skor >= 0.6): %{gaz_stats['guvenilir_oran']*100:.2f} ({gaz_stats['guvenilir_sayisi']}/{gaz_stats['kalan']})"
        )
        print(f"   Çok Adaylı Kayıt Sayısı: {gaz_stats['cok_adayli_sayisi']}")
        print("   Çok Adaylı Kayıtlardan Rastgele 15 Örnek:")
        for idx, ex in enumerate(gaz_stats["cok_adayli_ornekler"], 1):
            print(
                f"     {idx:>2}. Arap: {ex['arap']:<15} | Seçilen: {ex['secilen']:<15} | Skor: {ex['skor']:.3f} | Adaylar: {ex['adaylar']}"
            )
    else:
        print(f"UYARI: {gaz_raw_file} bulunamadı, Gazetteer temizliği atlandı.")

    print("\n" + "=" * 80)
    print("VERİ TEMİZLİĞİ TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    main()
