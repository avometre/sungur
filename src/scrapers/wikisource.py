"""Osmanlıca Vikikaynak (wikisource.org) metin toplayıcısı.

MediaWiki API üzerinden Osmanlıca kategorilerindeki sayfaları gezinir,
HTML içeriğindeki gezinme/başlık şablonlarını, dipnotları ve tabloları ayıklar,
satır satır ayrıştırıp Arapça oranı yüksek satırları standart JSONL şemasıyla kaydeder.
"""

import argparse
import collections
import logging
import sys
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import quote

from bs4 import BeautifulSoup

from src.common.io import write_jsonl
from src.common.normalize import arabic_ratio, normalize_ottoman
from src.scrapers.http import polite_get

logger = logging.getLogger(__name__)

WIKISOURCE_API_URL = "https://wikisource.org/w/api.php"
DEFAULT_CATEGORY = "Category:Ottoman"


def clean_and_extract_lines(html_content: str, min_arabic_ratio: float = 0.6) -> list[str]:
    """Sayfa HTML içeriğini temizler ve Arapça oranı eşik değerin üzerinde olan satırları döndürür."""
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")

    # Gereksiz etiket ve şablonları kaldır
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
    ]
    for selector in selectors_to_remove:
        for el in soup.select(selector):
            el.decompose()

    # Şiir ve mısra satır sonlarını yakalayabilmek için <br> etiketlerini yeni satıra çevir
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.get_text()
    cleaned_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Yalnızca Arap harfli metin satırlarını tut (Latin dipnot, açıklama vs. atla)
        if arabic_ratio(line) >= min_arabic_ratio:
            cleaned_lines.append(line)

    return cleaned_lines


def fetch_category_pages(
    category_title: str = DEFAULT_CATEGORY,
    max_depth: int = 3,
    min_interval: float = 1.0,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """MediaWiki API list=categorymembers kullanarak alt kategoriler dahil sayfaları toplar."""
    visited_categories: set[str] = set()
    queue = collections.deque([(category_title, 1)])
    discovered_pages: dict[int, dict[str, Any]] = {}

    while queue:
        if limit is not None and len(discovered_pages) >= limit:
            break

        current_cat, depth = queue.popleft()
        if current_cat in visited_categories:
            continue
        visited_categories.add(current_cat)

        logger.info("Kategori taranıyor (Derinlik %d): %s", depth, current_cat)
        cmcontinue = None

        while True:
            params: dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": current_cat,
                "cmlimit": 500,
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            resp = polite_get(
                WIKISOURCE_API_URL,
                params=params,
                min_interval=min_interval,
                maxlag=5,
            )
            data = resp.json()
            query_data = data.get("query", {})
            members = query_data.get("categorymembers", [])

            for member in members:
                ns = member.get("ns")
                pageid = member.get("pageid")
                title = member.get("title")

                # ns == 0: Ana madde / sayfa
                if ns == 0 and pageid and title:
                    discovered_pages[pageid] = {"pageid": pageid, "title": title}
                    if limit is not None and len(discovered_pages) >= limit:
                        break
                # ns == 14: Alt kategori
                elif ns == 14 and title and depth < max_depth:
                    if title not in visited_categories:
                        queue.append((title, depth + 1))

            if limit is not None and len(discovered_pages) >= limit:
                break

            if "continue" in data and "cmcontinue" in data["continue"]:
                cmcontinue = data["continue"]["cmcontinue"]
            else:
                break

    return list(discovered_pages.values())


def fetch_page_html(pageid: int, min_interval: float = 1.0) -> str:
    """Belirtilen pageid için action=parse ile HTML metnini alır."""
    params = {
        "action": "parse",
        "pageid": pageid,
        "prop": "text",
        "format": "json",
    }
    resp = polite_get(
        WIKISOURCE_API_URL,
        params=params,
        min_interval=min_interval,
        maxlag=5,
    )
    data = resp.json()
    return data.get("parse", {}).get("text", {}).get("*", "")


def parse_page_records(
    pageid: int,
    title: str,
    html_content: str,
    min_arabic_ratio: float = 0.6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sayfa HTML'ini ayrıştırarak satır kayıtlarını ve sayfa özet kaydını oluşturur."""
    lines = clean_and_extract_lines(html_content, min_arabic_ratio=min_arabic_ratio)
    url = f"https://wikisource.org/wiki/{quote(title)}"

    line_records: list[dict[str, Any]] = []
    for line_idx, line in enumerate(lines):
        line_records.append(
            {
                "kaynak": "wikisource",
                "id": f"ws:{pageid}:{line_idx}",
                "baslik": title,
                "arap_ham": line,
                "arap": normalize_ottoman(line),
                "lisans": "CC BY-SA 4.0 / kamu malı (sayfaya göre)",
                "url": url,
            }
        )

    page_summary = {
        "pageid": pageid,
        "baslik": title,
        "url": url,
        "satir_sayisi": len(lines),
    }

    return line_records, page_summary


def scrape_wikisource(
    category_title: str = DEFAULT_CATEGORY,
    max_depth: int = 3,
    limit: Optional[int] = None,
    min_arabic_ratio: float = 0.6,
    min_interval: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tüm Vikikaynak Osmanlıca sayfalarını çeker ve ayrıştırır."""
    pages = fetch_category_pages(
        category_title=category_title,
        max_depth=max_depth,
        min_interval=min_interval,
        limit=limit,
    )
    logger.info("Toplam %d sayfa tespit edildi.", len(pages))

    if limit is not None:
        pages = pages[:limit]
        logger.info("Limit uygulandı: %d sayfa çekilecek.", len(pages))

    all_line_records: list[dict[str, Any]] = []
    all_page_summaries: list[dict[str, Any]] = []

    for i, pinfo in enumerate(pages, 1):
        pid = pinfo["pageid"]
        title = pinfo["title"]
        logger.info("[%d/%d] Sayfa içeriği alınıyor: %s (pageid=%d)", i, len(pages), title, pid)

        try:
            html = fetch_page_html(pid, min_interval=min_interval)
            lines, summary = parse_page_records(
                pid, title, html, min_arabic_ratio=min_arabic_ratio
            )
            all_line_records.extend(lines)
            all_page_summaries.append(summary)
        except Exception as e:
            logger.error("Sayfa %s (id=%d) işlenirken hata: %s", title, pid, e)
            continue

    return all_line_records, all_page_summaries


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Osmanlıca Vikikaynak (wikisource.org) toplayıcı")
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_CATEGORY,
        help="Başlangıç kategorisi (varsayılan: Category:Ottoman)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Alt kategori derinliği",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksimum işlenecek sayfa sayısı (test amaçlı)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/wikisource_ota.jsonl",
        help="Satır bazlı çıktı JSONL dosyası",
    )
    parser.add_argument(
        "--pages-out",
        type=str,
        default="data/raw/wikisource_pages.jsonl",
        help="Sayfa özetleri çıktı JSONL dosyası",
    )
    parser.add_argument(
        "--min-arabic-ratio",
        type=float,
        default=0.6,
        help="Filtrelenecek minimum Arapça karakter oranı",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Vikikaynak taraması başlatılıyor (Kategori: %s)", args.category)

    lines, pages = scrape_wikisource(
        category_title=args.category,
        max_depth=args.max_depth,
        limit=args.limit,
        min_arabic_ratio=args.min_arabic_ratio,
    )

    out_lines_path = Path(args.out)
    out_pages_path = Path(args.pages_out)

    saved_lines = write_jsonl(out_lines_path, lines)
    saved_pages = write_jsonl(out_pages_path, pages)

    print("\n--- Vikikaynak Özeti ---")
    print(f"İşlenen sayfa sayısı  : {saved_pages}")
    print(f"Toplanan satır sayısı : {saved_lines}")
    print(f"Satır çıktı dosyası   : {out_lines_path.resolve()}")
    print(f"Sayfa özet dosyası    : {out_pages_path.resolve()}\n")


if __name__ == "__main__":
    main()
