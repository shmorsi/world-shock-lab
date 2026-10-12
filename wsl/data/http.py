"""Polite HTTP client: per-host rate limits, bounded retries, secrets redacted from logs.

The client never tries to defeat bot detection. If a host answers with a challenge page or
keeps refusing, the download fails and the source must be reconsidered (data/SOURCES.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

USER_AGENT = "Mozilla/5.0 (compatible; world-shock-lab/0.1; academic research)"
SECRET_PARAMS = frozenset({"api_key", "apikey", "token", "key"})
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# Minimum seconds between requests to the same host. Conservative by design.
HOST_MIN_INTERVAL: dict[str, float] = {
    "www.bankofcanada.ca": 1.0,
    "query1.finance.yahoo.com": 1.5,
    "api.nasdaq.com": 1.5,
    "api.stlouisfed.org": 0.6,  # FRED allows 120 requests/minute
    "sdmx.oecd.org": 3.0,  # OECD rate-limits its public API
    "www150.statcan.gc.ca": 0.5,
}
DEFAULT_MIN_INTERVAL = 0.5


class FetchError(RuntimeError):
    pass


def redact_url(url: str) -> str:
    """Remove credentials from a URL before it is logged or written to the manifest."""
    parts = urlsplit(url)
    query = [
        (k, "REDACTED" if k.lower() in SECRET_PARAMS else v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(parts._replace(query=urlencode(query)))


@dataclass(frozen=True)
class Fetched:
    url: str  # redacted, final URL
    content: bytes
    content_type: str | None


class Http:
    def __init__(self, timeout: float = 90.0, retries: int = 4) -> None:
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last: dict[str, float] = {}

    def _wait(self, host: str) -> None:
        gap = HOST_MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
        elapsed = time.monotonic() - self._last.get(host, 0.0)
        if elapsed < gap:
            time.sleep(gap - elapsed)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Fetched:
        host = urlsplit(url).netloc
        last_error = ""
        for attempt in range(self.retries + 1):
            self._wait(host)
            try:
                resp = self.session.request(
                    method, url, params=params, json=json_body, headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                self._last[host] = time.monotonic()
                if resp.status_code == 200:
                    return Fetched(
                        url=redact_url(resp.url),
                        content=resp.content,
                        content_type=resp.headers.get("Content-Type"),
                    )
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRY_STATUS:
                    break
            self._last[host] = time.monotonic()
            if attempt < self.retries:
                time.sleep(min(60.0, 2.0 * 2**attempt))
        full = url + ("?" + urlencode(params) if params else "")
        raise FetchError(f"{method} {redact_url(full)} failed: {last_error}")

    def get(self, url: str, **kw: Any) -> Fetched:
        return self.request("GET", url, **kw)

    def post_json(self, url: str, payload: Any) -> Fetched:
        return self.request("POST", url, json_body=payload)
