"""Bounded, credential-free HTTP image reads for explicitly enabled URL sources."""

from dataclasses import dataclass
import math


class URLImageError(OSError):
    """A classified external payload failure, without exposing the source URL."""

    def __init__(self, code: str, http_status: int | None = None):
        self.code = code
        self.http_status = http_status
        super().__init__(code + (f" (HTTP {http_status})" if http_status else ""))


@dataclass(frozen=True)
class URLImageOptions:
    """Timeout bounds each HTTP attempt including its response body."""

    timeout: float = 15.0
    retries: int = 1
    max_bytes: int = 32 * 1024 * 1024

    def __post_init__(self):
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or not 0 < self.timeout <= 86400
        ):
            raise ValueError("timeout must be finite and positive")
        for name, minimum in (("retries", 0), ("max_bytes", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class URLImageClient:
    """Reusable native connection pool; HTTP reads release the Python GIL."""

    def __init__(self, options: URLImageOptions | None = None):
        from ._webshart import _URLImageClient

        options = options or URLImageOptions()
        self._native = _URLImageClient(
            options.timeout, options.retries, options.max_bytes
        )

    def fetch(self, url: str) -> bytes:
        data, code, status = self._native.fetch(url if isinstance(url, str) else None)
        if code is not None:
            raise URLImageError(code, status)
        return data



def fetch_image(url: str, options: URLImageOptions) -> bytes:
    """Fetch raw HTTP(S) bytes with native timeout, retry and size bounds."""
    return URLImageClient(options).fetch(url)
