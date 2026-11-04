"""Snapshot sayfalarını ve PDF indirme bağlantılarını incele."""

import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sample_urls = [
    "http://web.archive.org/web/20261006184909/https://ekitap.yek.gov.tr/urun/hatayi-divani_624.aspx",
    "http://web.archive.org/web/20220720115557/http://ekitap.yek.gov.tr/urun/serh-i-m%C3%BBcez-fi%E2%80%99t-tibb_776.aspx",
    "http://web.archive.org/web/20260305075115/https://ekitap.yek.gov.tr/urun/muradi-divani_591.aspx",
    "http://web.archive.org/web/20240712214823/http://ekitap.yek.gov.tr/urun/tercume-i-aca%E2%80%99ibu%E2%80%99l-mahl%C3%BBkat_694.aspx",
]

for url in sample_urls:
    print(f"Kontrol: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            print(f"  Sayfa boyutu: {len(html)} bayt")
            matches = re.findall(r'href=["\']([^"\']*(?:\.pdf|indir|download|Uploads)[^"\']*)["\']', html, re.I)
            print(f"  Bulunan olası bağlantılar: {matches}")
            # E-kitap indirme düğmesi var mı?
            btn_matches = re.findall(r'(?:indir|pdf|e-kitap|oku)[^<]*', html, re.I)
            sample_btns = [b.strip() for b in btn_matches if len(b.strip()) > 2][:5]
            print(f"  Metin eşleşmeleri: {sample_btns}")
    except Exception as e:
        print(f"  Hata: {e}")
