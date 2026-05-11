"""
collect_kystverket.py
=====================
Fetch historic AIS data from the Kystverket historic AIS portal
(hais.kystverket.no) by replicating the HTTP calls the web portal makes.

SETUP — Finding the real API endpoints
---------------------------------------
The portal has no documented public API, so you need to capture its internal
HTTP calls once with browser devtools, then fill in the constants below.

Steps:
  1. Open https://hais.kystverket.no in Chrome/Firefox
  2. Open DevTools → Network tab (F12)
  3. Fill in a small test order (1 vessel, short date range, CSV format)
  4. Submit the form and watch the Network tab
  5. Look for XHR/Fetch requests — you'll see:
       - A POST/GET to something like /api/order or /api/request
       - Possibly a follow-up GET to poll order status
       - A final GET to download the file
  6. Right-click each relevant request → "Copy as cURL"
  7. Translate the URL, headers, and body into the constants below

Fill in these constants before running:
  BASE_URL, AUTH_HEADER / AUTH_TOKEN, ORDER_ENDPOINT, STATUS_ENDPOINT,
  DOWNLOAD_ENDPOINT

Usage
-----
    # Fetch all 998 BW fishing vessels, split into two 12-month chunks
    python collect_kystverket.py

Output
------
    data/raw/kystverket_ais.csv   (same format as bw_ais_historic.csv)
"""

import os
import sys
import csv
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

# =============================================================================
# CONFIG  —  fill these in from browser devtools (see SETUP above)
# =============================================================================

# Base URL of the internal API (e.g. "https://hais.kystverket.no/api")
BASE_URL = os.environ.get("KV_BASE_URL", "https://hais.kystverket.no")

# Authentication — try each in order until one works:
#   Option A: API key header  (e.g.  {"X-API-Key": "your-key"})
#   Option B: Bearer token    (e.g.  {"Authorization": "Bearer your-token"})
#   Option C: Basic auth      (requests.get(..., auth=("user", "pass")))
#   Option D: No auth (portal may use session cookies — see COOKIE below)
API_KEY    = os.environ.get("KV_API_KEY", "")         # Option A
AUTH_TOKEN = os.environ.get("KV_AUTH_TOKEN", "")      # Option B
KV_USER    = os.environ.get("KV_USER", "")            # Option C
KV_PASS    = os.environ.get("KV_PASS", "")            # Option C

# If the portal uses session cookies, paste the full Cookie header value here
# (copy from DevTools → Network → any request → Request Headers → Cookie)
COOKIE     = os.environ.get("KV_COOKIE", "")          # Option D

# Endpoint paths (relative to BASE_URL) — fill from DevTools
ORDER_ENDPOINT    = "/api/order"         # POST — submits a data request
STATUS_ENDPOINT   = "/api/order/{id}"   # GET  — polls until ready (if async)
DOWNLOAD_ENDPOINT = "/api/order/{id}/download"  # GET — download the file

# Output
OUTPUT_DIR    = Path("data/raw")
OUTPUT_CSV    = OUTPUT_DIR / "kystverket_ais.csv"
PROGRESS_FILE = OUTPUT_DIR / "kystverket_ais.progress"
MMSI_FILE     = OUTPUT_DIR / "bw_fishing_mmsi.csv"

# Date ranges — split into two requests (portal limit: 1 year per request)
DATE_RANGES = [
    ("2025-02-01", "2026-02-01"),
    ("2026-02-01", "2026-04-30"),
]

# Polling interval when waiting for async order to complete
POLL_INTERVAL_SEC = 30
POLL_TIMEOUT_SEC  = 3600   # 1 hour max wait per order

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# HTTP helper
# =============================================================================

def _auth_kwargs() -> dict:
    """Build auth-related kwargs for requests based on what's configured."""
    kwargs = {}
    headers = {"Accept": "application/json, text/csv, application/octet-stream"}

    if API_KEY:
        headers["X-API-Key"] = API_KEY
    elif AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    elif KV_USER and KV_PASS:
        kwargs["auth"] = (KV_USER, KV_PASS)

    if COOKIE:
        headers["Cookie"] = COOKIE

    kwargs["headers"] = headers
    return kwargs


def get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, timeout=120, **{**_auth_kwargs(), **kwargs})
    resp.raise_for_status()
    return resp


def post(url: str, payload: dict, **kwargs) -> requests.Response:
    kw = _auth_kwargs()
    kw["headers"]["Content-Type"] = "application/json"
    resp = requests.post(url, json=payload, timeout=120, **{**kw, **kwargs})
    resp.raise_for_status()
    return resp


# =============================================================================
# Load MMSI list
# =============================================================================

def load_mmsi_list() -> list[int]:
    if not MMSI_FILE.exists():
        log.error("MMSI file not found: %s", MMSI_FILE)
        sys.exit(1)
    mmsi_list = []
    with open(MMSI_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mmsi_list.append(int(row["MMSI"]))
    log.info("Loaded %d MMSIs from %s", len(mmsi_list), MMSI_FILE)
    return mmsi_list


# =============================================================================
# Order submission
# =============================================================================

def submit_order(mmsi_list: list[int], from_date: str, to_date: str) -> str:
    """
    Submit a data order to the portal.

    The payload structure below is a best-guess based on typical AIS portals.
    Adjust the field names to match what you saw in DevTools.

    Common alternative field names:
      mmsiList / mmsis / vessels / identifiers
      fromDate / startDate / dateFrom / start
      toDate   / endDate   / dateTo   / end
      format   / outputFormat / fileFormat
    """
    url = BASE_URL + ORDER_ENDPOINT

    payload = {
        "mmsiList":  mmsi_list,
        "fromDate":  from_date,
        "toDate":    to_date,
        "format":    "csv",          # or "parquet" — adjust if needed
        "vesselType": "fishing",     # optional filter — remove if not accepted
    }

    log.info("Submitting order: %d vessels, %s → %s", len(mmsi_list), from_date, to_date)
    log.debug("POST %s  payload=%s", url, json.dumps(payload)[:200])

    try:
        resp = post(url, payload)
    except requests.HTTPError as e:
        log.error("Order submission failed: %s\n%s", e, e.response.text[:500])
        raise

    data = resp.json()
    log.info("Order response: %s", json.dumps(data)[:300])

    # Extract order ID — adjust key name to match actual response
    order_id = (
        data.get("orderId") or
        data.get("id") or
        data.get("requestId") or
        data.get("jobId")
    )

    if order_id is None:
        # Portal might return the download URL directly (synchronous)
        download_url = data.get("downloadUrl") or data.get("url") or data.get("link")
        if download_url:
            log.info("Synchronous response — download URL: %s", download_url)
            return f"direct:{download_url}"
        log.error("Could not find order ID or download URL in response: %s", data)
        raise RuntimeError("Unexpected order response — check payload field names")

    log.info("Order ID: %s", order_id)
    return str(order_id)


# =============================================================================
# Status polling (for async portals)
# =============================================================================

def wait_for_order(order_id: str) -> str:
    """
    Poll the status endpoint until the order is ready.
    Returns the download URL.

    Adjust the status field names to match what DevTools showed.
    Common patterns:
      status: "PENDING" / "PROCESSING" / "COMPLETED" / "READY" / "DONE"
      downloadUrl / url / link / fileUrl
    """
    url = BASE_URL + STATUS_ENDPOINT.format(id=order_id)
    deadline = time.time() + POLL_TIMEOUT_SEC
    attempt  = 0

    while time.time() < deadline:
        attempt += 1
        try:
            resp = get(url)
            data = resp.json()
        except Exception as e:
            log.warning("Status poll %d failed: %s — retrying in %ds", attempt, e, POLL_INTERVAL_SEC)
            time.sleep(POLL_INTERVAL_SEC)
            continue

        status = (
            data.get("status") or
            data.get("state") or
            data.get("orderStatus") or
            ""
        ).upper()

        log.info("Poll %d: status=%s", attempt, status or repr(data))

        if status in ("COMPLETED", "READY", "DONE", "FINISHED", "SUCCESS"):
            download_url = (
                data.get("downloadUrl") or
                data.get("url") or
                data.get("link") or
                data.get("fileUrl")
            )
            if download_url:
                return download_url
            # Download might be at a predictable endpoint
            return BASE_URL + DOWNLOAD_ENDPOINT.format(id=order_id)

        if status in ("FAILED", "ERROR", "CANCELLED"):
            raise RuntimeError(f"Order {order_id} failed with status: {status}")

        time.sleep(POLL_INTERVAL_SEC)

    raise TimeoutError(f"Order {order_id} did not complete within {POLL_TIMEOUT_SEC}s")


# =============================================================================
# Download + parse
# =============================================================================

def download_and_parse(download_url: str) -> list[dict]:
    """Download the CSV file and parse it into pipeline-standard rows."""
    log.info("Downloading from: %s", download_url)
    resp = get(download_url, stream=True)

    # Write to temp file
    tmp_path = OUTPUT_DIR / "_kv_tmp.csv"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)

    log.info("Downloaded to %s (%d bytes)", tmp_path, tmp_path.stat().st_size)

    rows = parse_kystverket_csv(tmp_path)
    tmp_path.unlink()
    return rows


def parse_kystverket_csv(path: Path) -> list[dict]:
    """
    Normalise Kystverket CSV columns into the pipeline's standard format:
      MMSI, BaseDateTime, LAT, LON, SOG, COG

    The exact column names from Kystverket are unknown until you receive a
    sample file. Common Norwegian AIS CSV column names are listed below —
    uncomment / adjust once you have a real sample.
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")   # Kystverket often uses ;
        if reader.fieldnames:
            log.info("Kystverket CSV columns: %s", reader.fieldnames)

        for raw in reader:
            # Attempt to find each field under multiple possible names
            def _get(*keys):
                for k in keys:
                    v = raw.get(k) or raw.get(k.lower()) or raw.get(k.upper())
                    if v not in (None, ""):
                        return v
                return ""

            mmsi = _get("MMSI", "mmsi", "Mmsi")
            ts   = _get("timestamp", "datetime", "time", "msgtime",
                        "Timestamp", "DateTime", "Dato_tid")
            lat  = _get("LAT", "lat", "latitude", "Breddegrad", "Lat")
            lon  = _get("LON", "lon", "longitude", "Lengdegrad", "Lon")
            sog  = _get("SOG", "sog", "speedOverGround", "Fart", "Speed")
            cog  = _get("COG", "cog", "courseOverGround", "Kurs", "Course")

            if not all([mmsi, ts, lat, lon]):
                continue

            rows.append({
                "MMSI":         mmsi,
                "BaseDateTime": ts,
                "LAT":          lat,
                "LON":          lon,
                "SOG":          sog,
                "COG":          cog,
            })

    log.info("Parsed %d rows from %s", len(rows), path.name)
    return rows


# =============================================================================
# CSV output
# =============================================================================

FIELDNAMES = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"]


def append_rows(rows: list[dict]):
    write_header = not OUTPUT_CSV.exists()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    log.info("Appended %d rows → %s", len(rows), OUTPUT_CSV)


# =============================================================================
# Progress tracking
# =============================================================================

def load_progress() -> set:
    if not PROGRESS_FILE.exists():
        return set()
    done = set()
    with open(PROGRESS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)
    return done


def mark_done(key: str):
    with open(PROGRESS_FILE, "a") as f:
        f.write(key + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    # Basic config validation
    if not any([API_KEY, AUTH_TOKEN, KV_USER, COOKIE]):
        log.warning(
            "No authentication configured.\n"
            "Set one of: KV_API_KEY, KV_AUTH_TOKEN, KV_USER+KV_PASS, KV_COOKIE\n"
            "Or edit the constants at the top of this file.\n"
            "See SETUP instructions in the module docstring."
        )

    mmsi_list = load_mmsi_list()
    done      = load_progress()

    for from_date, to_date in DATE_RANGES:
        key = f"{from_date}__{to_date}"
        if key in done:
            log.info("Skipping already-fetched range %s → %s", from_date, to_date)
            continue

        log.info("=" * 60)
        log.info("Requesting range: %s → %s", from_date, to_date)
        log.info("=" * 60)

        try:
            order_id = submit_order(mmsi_list, from_date, to_date)

            if order_id.startswith("direct:"):
                download_url = order_id[len("direct:"):]
            else:
                log.info("Waiting for order %s to complete …", order_id)
                download_url = wait_for_order(order_id)

            rows = download_and_parse(download_url)
            append_rows(rows)
            mark_done(key)
            log.info("Range %s → %s complete: %d rows", from_date, to_date, len(rows))

        except Exception as e:
            log.error("Failed on range %s → %s: %s", from_date, to_date, e)
            log.error("Fix the issue and re-run — completed ranges will be skipped.")
            sys.exit(1)

    log.info("All ranges complete. Output: %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
