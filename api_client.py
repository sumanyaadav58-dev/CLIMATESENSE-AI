"""
ClimateSense AI - resilient HTTP layer for the Open-Meteo APIs.

This module has NO Streamlit dependency, so it can be unit-tested on its own.

Features
--------
* Separate connect timeout (20 s) and read timeout (90 s).
* Automatic retries (max 3 attempts) with exponential backoff (2 s, 4 s, ...).
* Handles connection errors, connect/read timeouts, HTTP errors (4xx / 5xx / 429)
  and invalid / unexpected JSON.
* Every failure is converted into an ``APIError`` whose ``kind`` tells the user
  whether the problem is:
      - "connectivity"  -> internet / DNS / VPN / firewall problem
      - "availability"  -> API server slow, overloaded or down
      - "request"       -> the API rejected the request (bad parameters, etc.)
      - "data"          -> the API answered with invalid / unexpected data
* Optional IPv4 fallback: many home / mobile networks have a broken IPv6 route,
  which makes ``api.open-meteo.com`` hang until the connect timeout. After the
  first connectivity failure the client automatically retries over IPv4 only.
  Set the environment variable ``CLIMATESENSE_FORCE_IPV4=1`` to force IPv4 from
  the very first attempt.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from datetime import date, timedelta
from typing import Any, Callable
from urllib.parse import urlparse

import requests

logger = logging.getLogger("climatesense.api")

# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# --------------------------------------------------------------------------- #
# Network settings
# --------------------------------------------------------------------------- #
CONNECT_TIMEOUT = 20          # seconds to establish the TCP/TLS connection
READ_TIMEOUT = 90             # seconds to wait for the API to send its response
TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)
MAX_ATTEMPTS = 3              # total attempts per request (1 try + 2 retries)
BACKOFF_BASE_SECONDS = 2      # wait before retry #1
BACKOFF_FACTOR = 2            # exponential growth: 2 s, 4 s, 8 s ...
MAX_BACKOFF_SECONDS = 30      # never sleep longer than this between attempts

USER_AGENT = "ClimateSenseAI/1.1 (MCA academic project; python-requests)"

CATEGORY_LABELS = {
    "connectivity": "Internet connectivity problem",
    "availability": "API availability problem",
    "request": "Request failure",
    "data": "Invalid API response",
}


# --------------------------------------------------------------------------- #
# Error type
# --------------------------------------------------------------------------- #
class APIError(Exception):
    """Raised when an Open-Meteo request ultimately fails."""

    def __init__(self, api: str, kind: str, message: str, detail: str = "",
                 attempts: int = 1, status: int | None = None):
        super().__init__(message)
        self.api = api
        self.kind = kind
        self.message = message
        self.detail = detail
        self.attempts = attempts
        self.status = status

    @property
    def label(self) -> str:
        return CATEGORY_LABELS.get(self.kind, "Request failure")

    def __str__(self) -> str:  # readable in logs and st.error()
        return f"{self.api}: {self.label}. {self.message}"


# --------------------------------------------------------------------------- #
# IPv4 fallback helper (patches urllib3 address-family selection safely)
# --------------------------------------------------------------------------- #
_force_ipv4 = os.environ.get("CLIMATESENSE_FORCE_IPV4", "").strip().lower() in ("1", "true", "yes")

try:  # pragma: no cover - depends on urllib3 internals
    import urllib3.util.connection as _u3conn

    _orig_allowed_gai_family = _u3conn.allowed_gai_family

    def _allowed_gai_family():
        if _force_ipv4:
            return socket.AF_INET
        return _orig_allowed_gai_family()

    _u3conn.allowed_gai_family = _allowed_gai_family
except Exception:  # pragma: no cover
    _u3conn = None


def _enable_ipv4_fallback() -> bool:
    """Switch to IPv4-only. Returns True if the mode actually changed."""
    global _force_ipv4
    if _u3conn is None or _force_ipv4:
        return False
    _force_ipv4 = True
    logger.warning("Connectivity problem detected - retrying over IPv4 only.")
    return True


# --------------------------------------------------------------------------- #
# Core request function
# --------------------------------------------------------------------------- #
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})


def _host(url: str) -> str:
    return urlparse(url).netloc or url


def _api_reason(resp: requests.Response) -> str:
    """Extract the human-readable reason Open-Meteo puts in error bodies."""
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("reason"):
            return str(body["reason"])
    except ValueError:
        pass
    text = (resp.text or "").strip().replace("\n", " ")
    return text[:200] if text else (resp.reason or "no details provided")


def _backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    delay = BACKOFF_BASE_SECONDS * (BACKOFF_FACTOR ** (attempt - 1))
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, MAX_BACKOFF_SECONDS)


class _Failure(Exception):
    """Internal: one failed attempt, with retry hints."""

    def __init__(self, error: APIError, retryable: bool = True, retry_after: float | None = None):
        super().__init__(str(error))
        self.error = error
        self.retryable = retryable
        self.retry_after = retry_after


def _single_attempt(sess, url, params, timeout, api_name, validator, attempt):
    """Perform one request. Returns parsed JSON or raises _Failure."""
    host = _host(url)
    try:
        resp = sess.get(url, params=params, timeout=timeout)
    except requests.exceptions.ConnectTimeout as exc:
        fell_back = _enable_ipv4_fallback() if attempt == 1 else False
        raise _Failure(APIError(
            api_name, "connectivity",
            f"Could not connect to {host} within {CONNECT_TIMEOUT} seconds. Check your internet "
            f"connection, VPN/proxy, firewall or DNS settings."
            + (" Retrying over IPv4." if fell_back else ""),
            f"{type(exc).__name__}: {exc}"))
    except requests.exceptions.SSLError as exc:
        raise _Failure(APIError(
            api_name, "connectivity",
            f"A secure (SSL/TLS) connection to {host} could not be established. Check the system "
            f"date/time, VPN/proxy or antivirus HTTPS scanning. On macOS with python.org Python, "
            f"run 'Install Certificates.command'.",
            f"{type(exc).__name__}: {exc}"))
    except requests.exceptions.ReadTimeout as exc:
        raise _Failure(APIError(
            api_name, "availability",
            f"{host} accepted the connection but did not respond within {READ_TIMEOUT} seconds. "
            f"The API server is slow or overloaded.",
            f"{type(exc).__name__}: {exc}"))
    except requests.exceptions.ConnectionError as exc:
        fell_back = _enable_ipv4_fallback() if attempt == 1 else False
        raise _Failure(APIError(
            api_name, "connectivity",
            f"Could not reach {host}. You appear to be offline, or DNS / VPN / firewall is blocking "
            f"the connection." + (" Retrying over IPv4." if fell_back else ""),
            f"{type(exc).__name__}: {exc}"))
    except requests.exceptions.Timeout as exc:
        raise _Failure(APIError(
            api_name, "availability", f"The request to {host} timed out.",
            f"{type(exc).__name__}: {exc}"))
    except requests.exceptions.RequestException as exc:
        raise _Failure(APIError(
            api_name, "request", f"The request to {host} failed: {exc}",
            f"{type(exc).__name__}: {exc}"), retryable=False)

    status = resp.status_code
    if status >= 400:
        reason = _api_reason(resp)
        if status == 429:
            try:
                retry_after = float(resp.headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                retry_after = None
            raise _Failure(APIError(
                api_name, "availability",
                f"{host} is rate-limiting requests (HTTP 429). Wait a minute, then press 'Refresh Data'.",
                reason, status=status), retry_after=retry_after)
        if status >= 500:
            raise _Failure(APIError(
                api_name, "availability",
                f"{host} is temporarily unavailable (HTTP {status}). This is a server-side problem, "
                f"not an internet problem.", reason, status=status))
        raise _Failure(APIError(
            api_name, "request",
            f"The API rejected the request (HTTP {status}): {reason}", reason, status=status),
            retryable=False)

    try:
        data = resp.json()
    except ValueError as exc:
        raise _Failure(APIError(
            api_name, "data", f"{host} returned a response that is not valid JSON.",
            f"{type(exc).__name__}: {exc}", status=status))

    if isinstance(data, dict) and data.get("error"):
        reason = str(data.get("reason", "unknown error"))
        raise _Failure(APIError(
            api_name, "request", f"The API reported an error: {reason}", reason, status=status),
            retryable=False)

    if validator is not None:
        try:
            validator(data)
        except ValueError as exc:
            raise _Failure(APIError(
                api_name, "data", f"{host} returned data in an unexpected format: {exc}",
                str(exc), status=status), retryable=False)
    return data


def get_json(url: str, params: dict | None = None, *, api_name: str = "API",
             validator: Callable[[Any], None] | None = None,
             max_attempts: int = MAX_ATTEMPTS, timeout=TIMEOUT,
             sleep: Callable[[float], None] = time.sleep,
             session: requests.Session | None = None) -> Any:
    """
    GET ``url`` and return parsed JSON, retrying transient failures with
    exponential backoff.

    Raises ``APIError`` (never a raw requests exception) when all attempts fail
    or when a non-retryable problem (e.g. HTTP 400) is encountered.
    """
    sess = session or _session
    for attempt in range(1, max_attempts + 1):
        try:
            return _single_attempt(sess, url, params, timeout, api_name, validator, attempt)
        except _Failure as fail:
            fail.error.attempts = attempt
            logger.warning("[%s] attempt %d/%d failed: %s", api_name, attempt, max_attempts, fail.error)
            if not fail.retryable or attempt >= max_attempts:
                raise fail.error from None
            sleep(_backoff_delay(attempt, fail.retry_after))
    raise APIError(api_name, "request", "Request failed for an unknown reason.")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Response validators (raise ValueError when the payload is unusable)
# --------------------------------------------------------------------------- #
def _require_daily(data: Any, name: str = "daily") -> None:
    if not isinstance(data, dict) or not isinstance(data.get(name), dict) or "time" not in data[name]:
        raise ValueError(f"missing '{name}' section")


def _validate_weather(data: Any) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("current"), dict):
        raise ValueError("missing 'current' section")
    _require_daily(data)


def _validate_air(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")


def _validate_history(data: Any) -> None:
    _require_daily(data)


def _validate_geocode(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")


# --------------------------------------------------------------------------- #
# Public API functions
# --------------------------------------------------------------------------- #
def geocode(place: str) -> dict:
    data = get_json(GEOCODING_URL,
                    {"name": place, "count": 1, "language": "en", "format": "json"},
                    api_name="Geocoding API", validator=_validate_geocode)
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Location '{place}' was not found. Try a nearby larger city.")
    return results[0]


def fetch_weather(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
        "timezone": "auto", "forecast_days": 7,
    }
    return get_json(FORECAST_URL, params, api_name="Weather API", validator=_validate_weather)


def fetch_air(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,us_aqi",
        "hourly": "pm10,pm2_5,us_aqi",
        "timezone": "auto", "forecast_days": 3,
    }
    return get_json(AIR_URL, params, api_name="Air-quality API", validator=_validate_air)


def fetch_history(lat: float, lon: float, end_date: str | None = None) -> dict:
    end = date.fromisoformat(end_date) if end_date else date.today() - timedelta(days=5)
    start = end - timedelta(days=89)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
    }
    return get_json(ARCHIVE_URL, params, api_name="Historical archive API", validator=_validate_history)
