"""Veri sızıntısı koruma modülü (Data Leakage Protection).

Bu modül; altın dev ve test setlerinde yer alan sayfaların (pageid)
ve Arapça metinlerin (arap) sonraki eğitim veri setlerine karışmasını
kesin olarak önlemek için kullanılır.
"""

from pathlib import Path
from typing import Optional, Set, Tuple

DEFAULT_HELDOUT_PAGES = Path("data/gold/heldout_pages.txt")
DEFAULT_HELDOUT_ARAP = Path("data/gold/heldout_arap.txt")

_CACHED_HELDOUT_PAGES: Optional[Set[int]] = None
_CACHED_HELDOUT_ARAP: Optional[Set[str]] = None


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


def is_heldout(
    arap: Optional[str] = None,
    pageid: Optional[int] = None,
    heldout_pages: Optional[Set[int]] = None,
    heldout_arap: Optional[Set[str]] = None,
) -> bool:
    """Verilen metnin veya sayfa kimliğinin altın sette (heldout) yer alıp almadığını denetler.

    Eğer arap veya pageid altın sette varsa True döner; aksi takdirde False döner.
    Her iki parametre de None ise False döner.
    """
    if arap is None and pageid is None:
        return False

    global _CACHED_HELDOUT_PAGES, _CACHED_HELDOUT_ARAP

    if heldout_pages is None and heldout_arap is None:
        if _CACHED_HELDOUT_PAGES is None or _CACHED_HELDOUT_ARAP is None:
            _CACHED_HELDOUT_PAGES, _CACHED_HELDOUT_ARAP = load_heldout()
        heldout_pages = _CACHED_HELDOUT_PAGES
        heldout_arap = _CACHED_HELDOUT_ARAP
    else:
        heldout_pages = heldout_pages if heldout_pages is not None else set()
        heldout_arap = heldout_arap if heldout_arap is not None else set()

    if pageid is not None and heldout_pages and pageid in heldout_pages:
        return True

    if arap is not None and heldout_arap:
        cleaned_arap = arap.strip()
        if cleaned_arap in heldout_arap:
            return True

    return False
