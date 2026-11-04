"""Ortak HTTP istemcisi ve indirme yardımcıları.

Nezaketli tarama (polite scraping) kurallarına uygun olarak:
- Sabit User-Agent başlığı
- İstekler arası zaman sınırlaması (rate limiting)
- Üstel geri çekilme (exponential backoff) ve 429/503 Retry-After desteği
- Önbelleğe alma (cached download)
sağlar.
"""

import os
import time
from pathlib import Path
from typing import Any, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "osmanlica-ai/0.1 (personal research; non-commercial)"
_last_request_timestamp: float = 0.0


def get_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    """Yeniden deneme (retry) ve standart User-Agent yapılandırmasına sahip requests oturumu döndürür."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    retry_strategy = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_default_session: Optional[requests.Session] = None


def get_default_session() -> requests.Session:
    """Varsayılan tekil (singleton) oturumu döndürür."""
    global _default_session
    if _default_session is None:
        _default_session = get_session()
    return _default_session


def polite_get(
    url: str,
    params: Optional[dict[str, Any]] = None,
    min_interval: float = 1.0,
    session: Optional[requests.Session] = None,
    maxlag: Optional[int] = None,
    **kwargs: Any,
) -> requests.Response:
    """İstekler arasında en az min_interval saniye bekleyerek GET isteği gönderir.

    HTTP 429 ve 503 durumlarında Retry-After başlığına uyar.
    MediaWiki isteklerinde maxlag parametresini destekler.
    """
    global _last_request_timestamp

    if session is None:
        session = get_default_session()

    if params is None:
        params = {}
    else:
        params = dict(params)

    if maxlag is not None:
        params["maxlag"] = maxlag

    # İstekler arası bekleme süresi kontrolü
    elapsed = time.time() - _last_request_timestamp
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    if "timeout" not in kwargs:
        kwargs["timeout"] = 30.0

    max_attempts = 4
    for attempt in range(max_attempts):
        _last_request_timestamp = time.time()
        resp = session.get(url, params=params, **kwargs)

        # 429 Too Many Requests veya 503 Service Unavailable durumunda Retry-After kontrolü
        if resp.status_code in (429, 503) and attempt < max_attempts - 1:
            retry_after = resp.headers.get("Retry-After")
            wait_time = float(retry_after) if (retry_after and retry_after.isdigit()) else (2.0 ** attempt)
            time.sleep(max(wait_time, min_interval))
            continue

        # MediaWiki maxlag kontrolü
        if "X-Database-Lag" in resp.headers and attempt < max_attempts - 1:
            retry_after = resp.headers.get("Retry-After", "5")
            wait_time = float(retry_after) if retry_after.isdigit() else 5.0
            time.sleep(wait_time)
            continue

        return resp

    return resp


def download_cached(
    url: str,
    cache_dir: Union[str, Path] = "data/raw/_cache",
    filename: Optional[str] = None,
    min_interval: float = 1.0,
    session: Optional[requests.Session] = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Verilen URL'deki dosyayı akış halinde indirir ve önbelleğe alır.

    Dosya zaten önbellekte mevcutsa ve boş değilse doğrudan mevcut dosya yolunu döndürür.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        # URL'den dosya adını çıkar
        clean_url = url.split("?")[0].rstrip("/")
        filename = clean_url.split("/")[-1]
        if not filename:
            filename = "downloaded_file"

    target_file = cache_path / filename

    if target_file.exists() and target_file.stat().st_size > 0:
        return target_file

    part_file = target_file.with_suffix(target_file.suffix + ".part")

    if session is None:
        session = get_default_session()

    resp = polite_get(url, min_interval=min_interval, session=session, stream=True)
    resp.raise_for_status()

    with part_file.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    if part_file.exists():
        part_file.replace(target_file)

    return target_file
