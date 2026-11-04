"""Osmanlıca Vikikaynak ve Türkçe Vikikaynak satır hizalama modülü.

Bu modül; temizlenmiş Osmanlıca Vikikaynak sayfalarını (wikisource_clean.jsonl)
Türkçe Vikikaynak'taki (tr.wikisource.org) Latin harfli çağdaş karşılıklarıyla
MediaWiki langlinks, Wikidata sitelinks veya elle eşleme tablosu (ws_title_map.csv)
üzerinden eşleştirir. Dinamik programlama (DP) tabanlı satır hizalayıcı ile
Arapça ve Latin satırları 1-1, 1-2 ve 2-1 eşleştirerek altın test setini (ws_parallel.jsonl)
üretir.
"""

import argparse
import csv
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import regex
from bs4 import BeautifulSoup

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import arabic_ratio
from src.scrapers.http import get_default_session, polite_get
from src.translit.skeleton import skeleton_similarity

logger = logging.getLogger(__name__)

TR_WIKISOURCE_API = "https://tr.wikisource.org/w/api.php"
WIKISOURCE_API = "https://wikisource.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

LATIN_LETTER_PATTERN = regex.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ]")


# ---------------------------------------------------------------------------
# 1. HTML Temizleyici (Türkçe Vikikaynak)
# ---------------------------------------------------------------------------


def clean_latin_html(html_content: str) -> list[str]:
    """Türkçe Vikikaynak HTML içeriğinden Latin harfli metin satırlarını ayıklar.

    Kurallar:
    - script, style, noscript, table vb. etiketleri temizler.
    - Başlık şablonları, kaynak/yazar kutuları, navigasyon ve referans seçicilerini siler:
      .mw-editsection, .ws-header, .headertemplate, .ws-noexport, .navbox,
      .vertical-navbox, .metadata, sup.reference, ol.references, .reflist,
      .catlinks, .noprint, .sisterproject, .interproject, .licenseContainer,
      .plainlinks, .ws-summary
    - <br> etiketlerini yeni satıra dönüştürür.
    - Metin satırlarına bölünür; arabic_ratio < 0.1 ve en az 3 Latin harfi
      ([a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ]) içeren satırları döndürür.
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup.find_all(["script", "style", "noscript", "table"]):
        tag.decompose()

    selectors_to_remove = [
        ".mw-editsection",
        ".ws-header",
        ".headertemplate",
        ".ws-noexport",
        ".navbox",
        ".vertical-navbox",
        ".metadata",
        "sup.reference",
        "ol.references",
        ".reflist",
        ".catlinks",
        ".noprint",
        ".sisterproject",
        ".interproject",
        ".licenseContainer",
        ".plainlinks",
        ".ws-summary",
    ]
    for sel in selectors_to_remove:
        for el in soup.select(sel):
            el.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")

    lines: list[str] = []
    for raw in soup.get_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if arabic_ratio(line) < 0.1 and len(LATIN_LETTER_PATTERN.findall(line)) >= 3:
            lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# 2. Dinamik Programlama Tabanlı Satır Hizalama (align_lines)
# ---------------------------------------------------------------------------


def align_lines(
    arap_lines: list[str], latin_lines: list[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Arapça ve Latin satırları dinamik programlama (Gale-Church benzeri) ile hizalar.

    İzin verilen hamleler ve maliyetleri:
      1-1: 1 - sim(a, l)
      2-1 / 1-2: 1 - sim(birleşik, birleşik) + 0.05
      1-0 / 0-1 (atlama): 0.75
    sim = skeleton_similarity.

    Çıktı çiftleri:
      {"arap": ..., "latin": ..., "hizalama": "1-1"|"2-1"|"1-2", "skor": sim}
    Atlanan satırlar çıktıya eklenmez, sadece sayılır.
    """
    N = len(arap_lines)
    M = len(latin_lines)
    if N == 0 or M == 0:
        return [], {"atlanan_arap": N, "atlanan_latin": M}

    # DP[i][j]: arap_lines[:i] ile latin_lines[:j] arasındaki minimum maliyet
    DP: list[list[float]] = [[0.0] * (M + 1) for _ in range(N + 1)]
    # back[i][j]: (hamle_tipi, prev_i, prev_j, benzerlik_skoru)
    back: list[list[Optional[tuple[str, int, int, float]]]] = [
        [None] * (M + 1) for _ in range(N + 1)
    ]

    for i in range(1, N + 1):
        DP[i][0] = DP[i - 1][0] + 0.75
        back[i][0] = ("1-0", i - 1, 0, 0.0)

    for j in range(1, M + 1):
        DP[0][j] = DP[0][j - 1] + 0.75
        back[0][j] = ("0-1", 0, j - 1, 0.0)

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            candidates: list[tuple[float, str, int, int, float]] = []

            # 1. Hamle: 1-1
            sim_11 = skeleton_similarity(arap_lines[i - 1], latin_lines[j - 1])
            cost_11 = 1.0 - sim_11
            candidates.append((DP[i - 1][j - 1] + cost_11, "1-1", i - 1, j - 1, sim_11))

            # 2. Hamle: 1-0 (Arapça satır atlama)
            candidates.append((DP[i - 1][j] + 0.75, "1-0", i - 1, j, 0.0))

            # 3. Hamle: 0-1 (Latin satır atlama)
            candidates.append((DP[i][j - 1] + 0.75, "0-1", i, j - 1, 0.0))

            # 4. Hamle: 2-1 (2 Arapça satır, 1 Latin satır)
            # Birleşimin skoru tek tek parçaların skorundan yüksek olmalıdır
            if i >= 2:
                comb_a = arap_lines[i - 2] + " " + arap_lines[i - 1]
                sim_21 = skeleton_similarity(comb_a, latin_lines[j - 1])
                s_a1 = skeleton_similarity(arap_lines[i - 2], latin_lines[j - 1])
                s_a2 = skeleton_similarity(arap_lines[i - 1], latin_lines[j - 1])
                if sim_21 > max(s_a1, s_a2):
                    cost_21 = 1.0 - sim_21 + 0.05
                    candidates.append(
                        (DP[i - 2][j - 1] + cost_21, "2-1", i - 2, j - 1, sim_21)
                    )

            # 5. Hamle: 1-2 (1 Arapça satır, 2 Latin satır)
            if j >= 2:
                comb_l = latin_lines[j - 2] + " " + latin_lines[j - 1]
                sim_12 = skeleton_similarity(arap_lines[i - 1], comb_l)
                s_l1 = skeleton_similarity(arap_lines[i - 1], latin_lines[j - 2])
                s_l2 = skeleton_similarity(arap_lines[i - 1], latin_lines[j - 1])
                if sim_12 > max(s_l1, s_l2):
                    cost_12 = 1.0 - sim_12 + 0.05
                    candidates.append(
                        (DP[i - 1][j - 2] + cost_12, "1-2", i - 1, j - 2, sim_12)
                    )

            best_cost, best_move, pi, pj, best_sim = min(candidates, key=lambda c: c[0])
            DP[i][j] = best_cost
            back[i][j] = (best_move, pi, pj, best_sim)

    # Geriye doğru iz sürerek (backtracking) eşleşen çiftleri oluştur
    i, j = N, M
    pairs: list[dict[str, Any]] = []
    skipped_arap = 0
    skipped_latin = 0

    while i > 0 or j > 0:
        entry = back[i][j]
        if entry is None:
            break
        move, pi, pj, sim = entry

        if move == "1-1":
            pairs.append(
                {
                    "arap": arap_lines[i - 1],
                    "latin": latin_lines[j - 1],
                    "hizalama": "1-1",
                    "skor": round(sim, 4),
                }
            )
        elif move == "2-1":
            pairs.append(
                {
                    "arap": arap_lines[i - 2] + " " + arap_lines[i - 1],
                    "latin": latin_lines[j - 1],
                    "hizalama": "2-1",
                    "skor": round(sim, 4),
                }
            )
        elif move == "1-2":
            pairs.append(
                {
                    "arap": arap_lines[i - 1],
                    "latin": latin_lines[j - 2] + " " + latin_lines[j - 1],
                    "hizalama": "1-2",
                    "skor": round(sim, 4),
                }
            )
        elif move == "1-0":
            skipped_arap += 1
        elif move == "0-1":
            skipped_latin += 1

        i, j = pi, pj

    pairs.reverse()
    stats = {"atlanan_arap": skipped_arap, "atlanan_latin": skipped_latin}
    return pairs, stats


# ---------------------------------------------------------------------------
# 3. Sayfa Eşleme (Page Linking)
# ---------------------------------------------------------------------------


def find_page_links(
    pages: dict[int, str],
    gold_csv_path: Path,
    session: Optional[Any] = None,
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Osmanlıca sayfaların Türkçe Vikikaynak karşılıklarını bulur.

    Sırayla:
      1. wikisource.org langlinks (lllang=tr)
      2. wikisource.org pageprops wikibase_item -> Wikidata sitelinks trwikisource
      3. Elle eşleme tablosu data/gold/ws_title_map.csv

    Dönüş:
      (matched_dict: {pageid: {pageid, baslik_arap, tr_baslik, yontem}}, unmatched_ids)
    """
    matched: dict[int, dict[str, Any]] = {}
    unmatched: list[int] = []

    pids = list(pages.keys())
    if not pids:
        return matched, unmatched

    # 1. Adım: MediaWiki API langlinks sorgusu (50'şerli paketler)
    logger.info("Wikisource langlinks sorgulanıyor (%d sayfa)...", len(pids))
    for i in range(0, len(pids), 50):
        chunk = pids[i : i + 50]
        ids_str = "|".join(str(x) for x in chunk)
        try:
            resp = polite_get(
                WIKISOURCE_API,
                params={
                    "action": "query",
                    "pageids": ids_str,
                    "prop": "langlinks",
                    "lllang": "tr",
                    "format": "json",
                },
                session=session,
            ).json()
            for pid_s, pdata in resp.get("query", {}).get("pages", {}).items():
                pid = int(pid_s)
                ll = pdata.get("langlinks", [])
                if ll and ll[0].get("*"):
                    matched[pid] = {
                        "pageid": pid,
                        "baslik_arap": pages[pid],
                        "tr_baslik": ll[0]["*"],
                        "yontem": "langlinks",
                    }
        except Exception as e:
            logger.warning("Langlinks sorgulamasında hata: %s", e)

    # 2. Adım: wikibase_item kontrolü (langlinks ile bulunamayanlar)
    remaining_step1 = [pid for pid in pids if pid not in matched]
    if remaining_step1:
        logger.info(
            "Langlinks ile bulunamayan %d sayfa için Wikidata kontrol ediliyor...",
            len(remaining_step1),
        )
        for i in range(0, len(remaining_step1), 50):
            chunk = remaining_step1[i : i + 50]
            ids_str = "|".join(str(x) for x in chunk)
            try:
                resp = polite_get(
                    WIKISOURCE_API,
                    params={
                        "action": "query",
                        "pageids": ids_str,
                        "prop": "pageprops",
                        "ppprop": "wikibase_item",
                        "format": "json",
                    },
                    session=session,
                ).json()
                for pid_s, pdata in resp.get("query", {}).get("pages", {}).items():
                    pid = int(pid_s)
                    wb_id = pdata.get("pageprops", {}).get("wikibase_item")
                    if wb_id:
                        # Wikidata üzerinden trwikisource bağlantısına bak
                        wd_resp = polite_get(
                            WIKIDATA_API,
                            params={
                                "action": "wbgetentities",
                                "ids": wb_id,
                                "props": "sitelinks",
                                "format": "json",
                            },
                            session=session,
                        ).json()
                        trws = (
                            wd_resp.get("entities", {})
                            .get(wb_id, {})
                            .get("sitelinks", {})
                            .get("trwikisource", {})
                            .get("title")
                        )
                        if trws:
                            matched[pid] = {
                                "pageid": pid,
                                "baslik_arap": pages[pid],
                                "tr_baslik": trws,
                                "yontem": "wikidata",
                            }
            except Exception as e:
                logger.warning("Wikidata kontrolünde hata: %s", e)

    # 3. Adım: data/gold/ws_title_map.csv elle eşleme dosyası
    existing_gold_rows: dict[int, dict[str, str]] = {}
    if gold_csv_path.exists():
        try:
            with gold_csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        pid = int(row.get("pageid", 0))
                        existing_gold_rows[pid] = row
                        # Eğer tr_baslik doluysa ve henüz eşleşmemişse elle eşleme olarak al
                        if pid not in matched and row.get("tr_baslik", "").strip():
                            matched[pid] = {
                                "pageid": pid,
                                "baslik_arap": pages.get(pid, row.get("baslik_arap", "")),
                                "tr_baslik": row["tr_baslik"].strip(),
                                "yontem": "elle",
                            }
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning("ws_title_map.csv okunurken hata: %s", e)

    # Eşleşmeyen tüm sayfaları belirle
    for pid in pids:
        if pid not in matched:
            unmatched.append(pid)

    # ws_title_map.csv yoksa veya yeni eşleşmeyen sayfalar varsa dosyayı güncelle/oluştur
    gold_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows_to_write: list[dict[str, Any]] = []

    for pid in pids:
        if pid in existing_gold_rows:
            csv_rows_to_write.append(existing_gold_rows[pid])
        else:
            tr_b = matched[pid]["tr_baslik"] if pid in matched else ""
            not_val = matched[pid]["yontem"] if pid in matched else ""
            csv_rows_to_write.append(
                {
                    "pageid": pid,
                    "baslik_arap": pages[pid],
                    "tr_baslik": tr_b,
                    "not": not_val,
                }
            )

    try:
        with gold_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["pageid", "baslik_arap", "tr_baslik", "not"]
            )
            writer.writeheader()
            for r in csv_rows_to_write:
                writer.writerow(r)
    except Exception as e:
        logger.warning("ws_title_map.csv yazılırken hata: %s", e)

    return matched, unmatched


# ---------------------------------------------------------------------------
# 4. Latin Metni Çekme ve Önbellekleme
# ---------------------------------------------------------------------------


def sanitize_filename(title: str) -> str:
    """Başlığı dosya sistemi için güvenli bir dosya adına dönüştürür."""
    return re.sub(r'[\\/*?:"<>|]', "_", title).strip()


def fetch_latin_page_html(
    title: str, cache_dir: Path, session: Optional[Any] = None
) -> Optional[str]:
    """Türkçe Vikikaynak'tan sayfa HTML içeriğini çeker ve önbelleğe alır."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{sanitize_filename(title)}.html"

    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8")
        except Exception:
            pass

    try:
        resp = polite_get(
            TR_WIKISOURCE_API,
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json",
            },
            session=session,
        ).json()

        if "error" in resp:
            logger.warning(
                "tr.wikisource.org parse hatası (%s): %s",
                title,
                resp["error"].get("info", ""),
            )
            return None

        html = resp.get("parse", {}).get("text", {}).get("*", "")
        if html:
            cache_file.write_text(html, encoding="utf-8")
            return html
    except Exception as e:
        logger.warning("tr.wikisource.org isteği başarısız oldu (%s): %s", title, e)

    return None


# ---------------------------------------------------------------------------
# 5. CLI ve Ana Akış
# ---------------------------------------------------------------------------


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Osmanlıca ve Türkçe Vikikaynak metinlerini satır satır hizala"
    )
    parser.add_argument(
        "--clean-file",
        type=Path,
        default=Path("data/interim/wikisource_clean.jsonl"),
        help="Temizlenmiş Osmanlıca satırlar dosyası",
    )
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=Path("data/gold/ws_title_map.csv"),
        help="Elle başlık eşleme CSV dosyası",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/_cache/tr_wikisource"),
        help="Türkçe Vikikaynak HTML önbellek dizini",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Çıktıların kaydedileceği dizin",
    )
    args = parser.parse_args()

    clean_file: Path = args.clean_file
    gold_csv: Path = args.gold_csv
    cache_dir: Path = args.cache_dir
    interim_dir: Path = args.interim_dir

    interim_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VİKİKAYNAK SATIR HİZALAMA BAŞLIYOR (vikikaynak_align)")
    print("=" * 80)

    if not clean_file.exists():
        print(f"HATA: {clean_file} bulunamadı! Önce clean_raw çalıştırılmalıdır.")
        sys.exit(1)

    # 1. Temizlenmiş satırları oku ve sayfa bazında topla
    clean_records = list(read_jsonl(clean_file))
    pages_lines: dict[int, list[dict[str, Any]]] = {}
    pages_titles: dict[int, str] = {}

    for r in clean_records:
        rec_id = str(r.get("id", ""))
        try:
            pid = int(rec_id.split(":")[1])
        except (IndexError, ValueError):
            continue
        pages_lines.setdefault(pid, []).append(r)
        if pid not in pages_titles:
            pages_titles[pid] = str(r.get("baslik", ""))

    print(
        f"Toplam {len(clean_records)} satır, {len(pages_titles)} benzersiz sayfa bulundu."
    )

    # 2. Sayfa Eşleme (Page Linking)
    matched_pages, unmatched_ids = find_page_links(
        pages_titles, gold_csv_path=gold_csv
    )

    # Sayfa bağlantılarını ws_page_links.jsonl dosyasına kaydet
    page_links_list = list(matched_pages.values())
    write_jsonl(interim_dir / "ws_page_links.jsonl", page_links_list)

    yontem_sayilari = Counter(p["yontem"] for p in matched_pages.values())

    print("\n--- Sayfa Eşleme Özeti ---")
    print(f"Eşleşen Sayfa Sayısı   : {len(matched_pages)}")
    for ym, cnt in yontem_sayilari.items():
        print(f"  - Yöntem: {ym:<12} : {cnt:>4} sayfa")
    print(f"Eşleşmeyen Sayfa Sayısı : {len(unmatched_ids)}")
    print(f"Elle eşleme CSV dosyası : {gold_csv}")

    # 3. Latin HTML Metinlerini Çek ve Hizala
    all_parallel_pairs: list[dict[str, Any]] = []
    total_skipped_arap = 0
    total_skipped_latin = 0

    print("\nSayfalar Türkçe Vikikaynak ile satır satır hizalanıyor...")

    for pid, link_info in matched_pages.items():
        tr_title = link_info["tr_baslik"]
        arap_title = link_info["baslik_arap"]
        page_records = pages_lines.get(pid, [])
        arap_lines = [r["arap"] for r in page_records]

        html = fetch_latin_page_html(tr_title, cache_dir=cache_dir)
        if not html:
            logger.info("Metin içeriği alınamadı (%s), atlanıyor.", tr_title)
            continue

        latin_lines = clean_latin_html(html)
        if not latin_lines:
            logger.info("Latin satır bulunamadı (%s), atlanıyor.", tr_title)
            continue

        pairs, skip_stats = align_lines(arap_lines, latin_lines)
        total_skipped_arap += skip_stats["atlanan_arap"]
        total_skipped_latin += skip_stats["atlanan_latin"]

        for pair in pairs:
            skor = pair["skor"]
            all_parallel_pairs.append(
                {
                    "kaynak": "wikisource+vikikaynak",
                    "pageid": pid,
                    "baslik_arap": arap_title,
                    "tr_baslik": tr_title,
                    "arap": pair["arap"],
                    "latin": pair["latin"],
                    "hizalama": pair["hizalama"],
                    "skor": skor,
                    "guvenilir": skor >= 0.55,
                    "lisans": "CC BY-SA 4.0",
                }
            )

    # 4. Çıktıyı Kaydet
    output_parallel_file = interim_dir / "ws_parallel.jsonl"
    write_jsonl(output_parallel_file, all_parallel_pairs)

    total_pairs = len(all_parallel_pairs)
    guvenilir_pairs = sum(1 for p in all_parallel_pairs if p.get("guvenilir"))

    # Skor dağılımı (0.1'lik kutular)
    score_buckets: dict[str, int] = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
    for p in all_parallel_pairs:
        s = p["skor"]
        idx = min(int(s * 10), 9)
        bucket_key = f"{idx/10:.1f}-{(idx+1)/10:.1f}"
        score_buckets[bucket_key] += 1

    print("\n" + "=" * 80)
    print("HİZALAMA İSTATİSTİKLERİ VE SONUÇ")
    print("=" * 80)
    print(f"Toplam Üretilen Çift     : {total_pairs}")
    print(
        f"Güvenilir Çift (≥ 0.55)  : {guvenilir_pairs} (%{(guvenilir_pairs/total_pairs*100.0) if total_pairs else 0.0:.2f})"
    )
    print(
        f"Atlanan Satır Sayısı     : Arapça={total_skipped_arap}, Latin={total_skipped_latin}"
    )
    print(f"Çıktı Dosyası            : {output_parallel_file}")

    print("\nSkor Dağılımı (0.1'lik Kutular):")
    for b_range, cnt in score_buckets.items():
        pct = (cnt / total_pairs * 100.0) if total_pairs else 0.0
        bar = "#" * int(pct / 2)
        print(f"  [{b_range}] : {cnt:>5} (%{pct:5.1f}) {bar}")

    # En yüksek 10 çift
    sorted_pairs = sorted(all_parallel_pairs, key=lambda x: x["skor"], reverse=True)
    print("\nEn Yüksek Skorlu 10 Çift:")
    for idx, p in enumerate(sorted_pairs[:10], 1):
        print(f"  {idx:>2}. [{p['hizalama']}] Skor: {p['skor']:.4f}")
        print(f"      Arap : {p['arap']}")
        print(f"      Latin: {p['latin']}")

    # 0.4 - 0.6 arası rastgele 10 çift
    mid_pairs = [p for p in all_parallel_pairs if 0.4 <= p["skor"] <= 0.6]
    rng = random.Random(42)
    sample_mid = rng.sample(mid_pairs, min(10, len(mid_pairs))) if mid_pairs else []

    print(f"\n0.4 - 0.6 Arası Rastgele {len(sample_mid)} Çift (seed=42):")
    for idx, p in enumerate(sample_mid, 1):
        print(f"  {idx:>2}. [{p['hizalama']}] Skor: {p['skor']:.4f}")
        print(f"      Arap : {p['arap']}")
        print(f"      Latin: {p['latin']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
