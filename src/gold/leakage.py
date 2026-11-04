"""Veri sızıntısı koruma modülü (Data Leakage Protection).

Bu modül; altın dev ve test setlerinde yer alan sayfaların (pageid, tr_baslik),
Arapça metinlerin (arap) ve Latin cümlelerin (latin) sonraki eğitim veri
setlerine karışmasını kesin olarak önlemek için kullanılır.
"""

import csv
import json
from pathlib import Path
from typing import Optional, Set, Tuple

DEFAULT_HELDOUT_PAGES = Path("data/gold/heldout_pages.txt")
DEFAULT_HELDOUT_ARAP = Path("data/gold/heldout_arap.txt")
DEFAULT_PAGE_STYLE = Path("data/gold/page_style.csv")
DEFAULT_GOLD_DEV = Path("data/gold/translit_dev.jsonl")
DEFAULT_GOLD_TEST = Path("data/gold/translit_test.jsonl")

_CACHED_HELDOUT_PAGES: Optional[Set[int]] = None
_CACHED_HELDOUT_ARAP: Optional[Set[str]] = None
_CACHED_HELDOUT_TITLES: Optional[Set[str]] = None
_CACHED_HELDOUT_LATIN: Optional[Set[str]] = None


def load_heldout_titles(page_style_path: Optional[Path] = None) -> Set[str]:
    """page_style.csv dosyasından korunan Türkçe başlıkları yükler."""
    path = page_style_path or DEFAULT_PAGE_STYLE
    titles: Set[str] = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("tr_baslik", "").strip()
                if t:
                    titles.add(t)
    return titles


def load_heldout_latin(
    dev_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Set[str]:
    """Altın dev ve test setlerindeki tüm birebir Latin cümlelerini yükler."""
    d_path = dev_path or DEFAULT_GOLD_DEV
    t_path = test_path or DEFAULT_GOLD_TEST
    sentences: Set[str] = set()

    for p in (d_path, t_path):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        lat = record.get("latin", "").strip()
                        if lat:
                            sentences.add(lat)
                    except json.JSONDecodeError:
                        continue
    return sentences


def load_heldout(
    heldout_pages_path: Optional[Path] = None,
    heldout_arap_path: Optional[Path] = None,
) -> Tuple[Set[int], Set[str]]:
    """Dışarıda tutulan (heldout) sayfa kimliklerini ve Arapça satırları yükler.

    Dosyalar bulunamazsa boş kümeler döndürür.
    """
    pages_path = heldout_pages_path or DEFAULT_HELDOUT_PAGES
    arap_path = heldout_arap_path or DEFAULT_HELDOUT_ARAP

    heldout_pages: Set[int] = set()
    if pages_path.exists():
        for line in pages_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    heldout_pages.add(int(line))
                except ValueError:
                    continue

    heldout_arap: Set[str] = set()
    if arap_path.exists():
        for line in arap_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                heldout_arap.add(line)

    return heldout_pages, heldout_arap


def load_heldout_all(
    heldout_pages_path: Optional[Path] = None,
    heldout_arap_path: Optional[Path] = None,
    page_style_path: Optional[Path] = None,
    dev_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[Set[int], Set[str], Set[str], Set[str]]:
    """Tüm boyutlardaki dışarıda tutulan (heldout) kümeleri (sayfa, arap, başlık, latin) yükler."""
    heldout_pages, heldout_arap = load_heldout(
        heldout_pages_path=heldout_pages_path,
        heldout_arap_path=heldout_arap_path,
    )
    heldout_titles = load_heldout_titles(page_style_path)
    heldout_latin = load_heldout_latin(dev_path, test_path)
    return heldout_pages, heldout_arap, heldout_titles, heldout_latin


def is_title_heldout(title: str, heldout_titles: Set[str]) -> bool:
    """Verilen başlığın korunan bir başlık veya onun alt sayfası olup olmadığını denetler.

    Örnek: 'Diyet' veya 'Nutuk/19. bölüm' tam eşleşir.
    'Diyet/1. Bölüm' -> 'Diyet'in alt sayfası olduğu için korumaya takılır.
    """
    clean = title.strip()
    if not clean or not heldout_titles:
        return False

    if clean in heldout_titles:
        return True

    # Alt sayfa kontrolü (<başlık>/...)
    for ht in heldout_titles:
        if clean.startswith(f"{ht}/"):
            return True

    return False


def is_heldout(
    arap: Optional[str] = None,
    pageid: Optional[int] = None,
    tr_title: Optional[str] = None,
    latin: Optional[str] = None,
    heldout_pages: Optional[Set[int]] = None,
    heldout_arap: Optional[Set[str]] = None,
    heldout_titles: Optional[Set[str]] = None,
    heldout_latin: Optional[Set[str]] = None,
) -> bool:
    """Verilen metnin, sayfa kimliğinin, Türkçe başlığın veya Latin cümlenin altın sette (heldout) yer alıp almadığını denetler.

    Herhangi bir eşleşme durumunda True döner; aksi takdirde False döner.
    """
    if arap is None and pageid is None and tr_title is None and latin is None:
        return False

    global _CACHED_HELDOUT_PAGES, _CACHED_HELDOUT_ARAP, _CACHED_HELDOUT_TITLES, _CACHED_HELDOUT_LATIN

    if (
        heldout_pages is None
        and heldout_arap is None
        and heldout_titles is None
        and heldout_latin is None
    ):
        if (
            _CACHED_HELDOUT_PAGES is None
            or _CACHED_HELDOUT_ARAP is None
            or _CACHED_HELDOUT_TITLES is None
            or _CACHED_HELDOUT_LATIN is None
        ):
            (
                _CACHED_HELDOUT_PAGES,
                _CACHED_HELDOUT_ARAP,
                _CACHED_HELDOUT_TITLES,
                _CACHED_HELDOUT_LATIN,
            ) = load_heldout_all()
        heldout_pages = _CACHED_HELDOUT_PAGES
        heldout_arap = _CACHED_HELDOUT_ARAP
        heldout_titles = _CACHED_HELDOUT_TITLES
        heldout_latin = _CACHED_HELDOUT_LATIN
    else:
        heldout_pages = heldout_pages if heldout_pages is not None else set()
        heldout_arap = heldout_arap if heldout_arap is not None else set()
        heldout_titles = heldout_titles if heldout_titles is not None else set()
        heldout_latin = heldout_latin if heldout_latin is not None else set()

    if pageid is not None and heldout_pages and pageid in heldout_pages:
        return True

    if tr_title is not None and heldout_titles:
        if is_title_heldout(tr_title, heldout_titles):
            return True

    if arap is not None and heldout_arap:
        cleaned_arap = arap.strip()
        if cleaned_arap in heldout_arap:
            return True

    if latin is not None and heldout_latin:
        cleaned_latin = latin.strip()
        if cleaned_latin in heldout_latin:
            return True

    return False


class LeakageChecker:
    """Veri sızıntısını yüksek performansla kontrol etmek için durum tutucu sınıf."""

    def __init__(
        self,
        heldout_pages_path: Optional[Path] = None,
        heldout_arap_path: Optional[Path] = None,
        page_style_path: Optional[Path] = None,
        dev_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
    ):
        (
            self.heldout_pages,
            self.heldout_arap,
            self.heldout_titles,
            self.heldout_latin,
        ) = load_heldout_all(
            heldout_pages_path=heldout_pages_path,
            heldout_arap_path=heldout_arap_path,
            page_style_path=page_style_path,
            dev_path=dev_path,
            test_path=test_path,
        )

    def is_title_heldout(self, title: str) -> bool:
        return is_title_heldout(title, self.heldout_titles)

    def is_pageid_heldout(self, pageid: int) -> bool:
        return pageid in self.heldout_pages

    def is_arap_heldout(self, arap: str) -> bool:
        return arap.strip() in self.heldout_arap

    def is_latin_heldout(self, latin: str) -> bool:
        return latin.strip() in self.heldout_latin
