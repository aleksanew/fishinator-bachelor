"""
collect_barentswatch.py
=======================
Collect AIS data from the Barentswatch Historic and Live APIs and save it
in the same CSV format used by the rest of the pipeline:

    MMSI, BaseDateTime, LAT, LON, SOG, COG

Two collection modes
--------------------
  historic  – fetch tracks for a specific date range (up to 14 days back)
              Uses: GET /v1/historic/tracks/{mmsi}/{fromDate}/{toDate}
              or    GET /v1/historic/trackslast24hours/{mmsi}

  live      – stream the current snapshot of all vessel positions
              Uses: GET /v1/combined  (NDJSON stream)

Rate limiting
-------------
Barentswatch explicitly requests sequential, single-threaded batch jobs.
This script:
  * Uses a single thread with a configurable inter-request delay (default 1 s)
  * Retries failed requests with exponential back-off (max 3 retries)
  * Automatically refreshes the OAuth token before it expires

Setup
-----
1. Register at https://www.barentswatch.no/minside/ and create an API client
2. Fill in CLIENT_ID and CLIENT_SECRET below (or set as env vars)
3. Install deps:  pip install requests
4. Run:
      python collect_barentswatch.py --mode historic --days 7
      python collect_barentswatch.py --mode live
      python collect_barentswatch.py --mode historic --mmsi-file mmsi_list.txt --days 14

Output
------
  data/raw/bw_ais_historic.csv   (historic mode)
  data/raw/bw_ais_live.csv       (live mode)
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import json
import requests

# =============================================================================
# CONFIG  –  fill these in or set as environment variables
# =============================================================================
CLIENT_ID     = os.environ.get("BW_CLIENT_ID",     "YOUR_CLIENT_ID")
CLIENT_SECRET = os.environ.get("BW_CLIENT_SECRET", "YOUR_CLIENT_SECRET")

# API endpoints
TOKEN_URL    = "https://id.barentswatch.no/connect/token"
HISTORIC_URL = "https://historic.ais.barentswatch.no/v1/historic"
LIVE_URL     = "https://live.ais.barentswatch.no/v1/combined"

# Rate limiting  (Barentswatch asks for sequential single-threaded requests)
REQUEST_DELAY_SEC  = 1.0    # seconds to wait between vessel requests
MAX_RETRIES        = 3      # retry attempts per request
BACKOFF_BASE_SEC   = 2.0    # exponential back-off base

OUTPUT_DIR = Path("data/raw")

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
# Authentication
# =============================================================================

class TokenManager:
    """Fetches and caches an OAuth2 access token, refreshing before expiry."""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self._token        = None
        self._expires_at   = 0.0

    def get_token(self) -> str:
        # Refresh if expired or within 60 s of expiry
        if time.time() >= self._expires_at - 60:
            self._refresh()
        return self._token

    def _refresh(self):
        log.info("Fetching new access token …")
        resp = requests.post(
            TOKEN_URL,
            data={
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
                "scope":         "ais",
                "grant_type":    "client_credentials",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token      = data["access_token"]
        self._expires_at = time.time() + data.get("expires_in", 3600)
        log.info("Token refreshed, valid for %d s", data.get("expires_in", 3600))

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}


# =============================================================================
# Request helper with retry / back-off
# =============================================================================

def get_with_retry(
    url: str,
    token_mgr: TokenManager,
    params: dict = None,
    stream: bool = False,
    timeout: int = 60,
) -> requests.Response:
    """GET a URL with automatic retry and exponential back-off."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers=token_mgr.headers,
                params=params,
                stream=stream,
                timeout=timeout,
            )
            if resp.status_code == 429:
                wait = BACKOFF_BASE_SEC ** attempt
                log.warning("Rate limited (429). Waiting %.1f s …", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                # Token may have expired mid-batch – force refresh and retry
                log.warning("401 Unauthorised – refreshing token")
                token_mgr._refresh()
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            wait = BACKOFF_BASE_SEC ** attempt
            log.warning("Request error (attempt %d/%d): %s – retrying in %.1f s",
                        attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed to GET {url} after {MAX_RETRIES} attempts")


# =============================================================================
# AIS response → normalised rows
# =============================================================================

def parse_track_response(mmsi: int, data: list) -> list[dict]:
    """
    Convert a list of AIS position dicts (Barentswatch format) into
    normalised rows matching the pipeline's expected columns.
    """
    rows = []
    for pt in data:
        lat = pt.get("latitude")  or pt.get("lat")
        lon = pt.get("longitude") or pt.get("lon")
        sog = pt.get("speedOverGround") or pt.get("sog")
        cog = pt.get("courseOverGround") or pt.get("cog")
        ts  = pt.get("msgtime") or pt.get("time") or pt.get("timestamp")

        # Skip points missing essential fields
        if any(v is None for v in [lat, lon, ts]):
            continue

        rows.append({
            "MMSI":         mmsi,
            "BaseDateTime": ts,
            "LAT":          lat,
            "LON":          lon,
            "SOG":          sog if sog is not None else "",
            "COG":          cog if cog is not None else "",
        })
    return rows


# =============================================================================
# Historic mode
# =============================================================================

def collect_historic(
    token_mgr: TokenManager,
    mmsi_list: list[int],
    from_date: datetime,
    to_date:   datetime,
    progress_path: Path = None,
) -> list[dict]:
    """
    Fetch historic tracks for a list of MMSIs between two dates.
    Requests are made one vessel at a time (per Barentswatch guidance).
    Pass progress_path to skip already-fetched MMSIs on resume.
    """
    from_str = from_date.strftime("%Y-%m-%dT%H:%M:%S")
    to_str   = to_date.strftime("%Y-%m-%dT%H:%M:%S")

    done     = load_progress(progress_path) if progress_path else set()
    all_rows = []
    total    = len(mmsi_list)
    skipped  = 0

    for i, mmsi in enumerate(mmsi_list, 1):
        if mmsi in done:
            skipped += 1
            continue

        log.info("[%d/%d] Fetching historic track for MMSI %s  (%s → %s)",
                 i, total, mmsi, from_str[:10], to_str[:10])

        url  = f"{HISTORIC_URL}/tracks/{mmsi}/{from_str}/{to_str}"
        try:
            resp = get_with_retry(url, token_mgr, timeout=60)
            data = resp.json()
            if isinstance(data, list):
                rows = parse_track_response(mmsi, data)
            else:
                rows = parse_track_response(mmsi, data.get("positions", []))

            log.info("  → %d positions", len(rows))
            all_rows.extend(rows)

            if progress_path:
                mark_done(progress_path, mmsi)
        except RuntimeError as exc:
            log.error("  Skipping MMSI %s: %s", mmsi, exc)

        if i < total:
            time.sleep(REQUEST_DELAY_SEC)

    if skipped:
        log.info("Skipped %d already-fetched MMSIs", skipped)

    return all_rows


def collect_last24h(
    token_mgr: TokenManager,
    mmsi_list: list[int],
) -> list[dict]:
    """Fetch last-24-hours tracks for a list of MMSIs."""
    all_rows = []
    total    = len(mmsi_list)

    for i, mmsi in enumerate(mmsi_list, 1):
        log.info("[%d/%d] Fetching last-24h track for MMSI %s", i, total, mmsi)
        url = f"{HISTORIC_URL}/trackslast24hours/{mmsi}"
        try:
            resp = get_with_retry(url, token_mgr, timeout=60)
            data = resp.json()
            rows = parse_track_response(mmsi, data if isinstance(data, list)
                                        else data.get("positions", []))
            log.info("  → %d positions", len(rows))
            all_rows.extend(rows)
        except RuntimeError as exc:
            log.error("  Skipping MMSI %s: %s", mmsi, exc)

        if i < total:
            time.sleep(REQUEST_DELAY_SEC)

    return all_rows


# =============================================================================
# Live mode
# =============================================================================

def collect_live(token_mgr: TokenManager) -> list[dict]:
    """
    Stream the current live AIS snapshot from /v1/combined.
    Returns one row per vessel (latest position of each vessel).
    The stream closes automatically when Barentswatch has sent all vessels.
    """
    log.info("Streaming live AIS snapshot from %s …", LIVE_URL)
    log.info("(This may take a few minutes for the full stream to complete)")

    rows = []
    resp = get_with_retry(LIVE_URL, token_mgr, stream=True, timeout=600)

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        try:
            import json
            pt = json.loads(raw_line)
        except ValueError:
            continue

        mmsi = pt.get("mmsi")
        lat  = pt.get("latitude")
        lon  = pt.get("longitude")
        ts   = pt.get("msgtime")

        if any(v is None for v in [mmsi, lat, lon, ts]):
            continue

        rows.append({
            "MMSI":         mmsi,
            "BaseDateTime": ts,
            "LAT":          lat,
            "LON":          lon,
            "SOG":          pt.get("speedOverGround", ""),
            "COG":          pt.get("courseOverGround", ""),
        })

    log.info("Live stream complete: %d vessel positions collected", len(rows))
    return rows


# =============================================================================
# Save to CSV  –  merge with any existing file and deduplicate
# =============================================================================

def save_csv(rows: list[dict], path: Path):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"]

    # Load existing rows if the file already exists
    existing = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
        log.info("Existing file has %d rows – merging …", len(existing))

    # Combine and deduplicate on (MMSI, BaseDateTime) – the natural unique key
    combined = {(str(r["MMSI"]), str(r["BaseDateTime"])): r for r in existing}
    new_count = 0
    for r in rows:
        key = (str(r["MMSI"]), str(r["BaseDateTime"]))
        if key not in combined:
            new_count += 1
        combined[key] = r

    all_rows = sorted(combined.values(), key=lambda r: (r["MMSI"], r["BaseDateTime"]))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    log.info(
        "Saved %d total rows (%d new, %d already existed) → %s",
        len(all_rows), new_count, len(existing), path,
    )


# =============================================================================
# MMSI list helpers
# =============================================================================

def load_mmsi_file(path: str) -> list[int]:
    """Load a plain text file with one MMSI per line."""
    mmsi_list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                mmsi_list.append(int(line))
    log.info("Loaded %d MMSIs from %s", len(mmsi_list), path)
    return mmsi_list


def fetch_mmsi_list_from_live(token_mgr: TokenManager) -> list[int]:
    """
    Get a list of all currently visible MMSIs by hitting /v1/latest/combined.
    Useful when you want to then fetch full historic tracks for those vessels.
    """
    url  = "https://live.ais.barentswatch.no/v1/latest/combined"
    log.info("Fetching vessel list from %s …", url)
    resp = get_with_retry(url, token_mgr, timeout=120)
    data = resp.json()
    mmsi_list = [int(v["mmsi"]) for v in data if "mmsi" in v]
    log.info("Found %d vessels currently visible", len(mmsi_list))
    return mmsi_list


# =============================================================================
# Progress tracking  –  resume interrupted runs without re-fetching
# =============================================================================

def load_progress(progress_path: Path) -> set:
    """Return the set of MMSIs already successfully fetched."""
    if not progress_path.exists():
        return set()
    done = set()
    with open(progress_path) as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                done.add(int(line))
    log.info("Resume: %d MMSIs already fetched (from %s)", len(done), progress_path)
    return done


def mark_done(progress_path: Path, mmsi: int):
    """Append an MMSI to the progress file after a successful fetch."""
    with open(progress_path, "a") as f:
        f.write(str(mmsi) + "\n")



# =============================================================================
# Session window  –  persist the date range so resumed runs use identical dates
# =============================================================================

def load_window(progress_path: Path) -> tuple:
    """
    Load a previously saved (from_date, to_date) window.
    Returns (from_date, to_date) as datetime objects, or (None, None) if not found.
    """
    window_path = progress_path.with_suffix(".window")
    if not window_path.exists():
        return None, None
    data = json.loads(window_path.read_text())
    from_date = datetime.fromisoformat(data["from_date"])
    to_date   = datetime.fromisoformat(data["to_date"])
    log.info(
        "Resuming existing window: %s → %s  (loaded from %s)",
        from_date.strftime("%Y-%m-%d %H:%M"),
        to_date.strftime("%Y-%m-%d %H:%M"),
        window_path,
    )
    return from_date, to_date


def save_window(progress_path: Path, from_date: datetime, to_date: datetime):
    """Persist the date window so future resume runs use the same range."""
    window_path = progress_path.with_suffix(".window")
    window_path.write_text(json.dumps({
        "from_date": from_date.isoformat(),
        "to_date":   to_date.isoformat(),
    }))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect AIS data from Barentswatch APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream live snapshot (all vessels right now)
  python collect_barentswatch.py --mode live

  # Historic tracks for all currently visible vessels, last 7 days
  python collect_barentswatch.py --mode historic --days 7

  # Historic tracks for a specific list of MMSIs, last 14 days
  python collect_barentswatch.py --mode historic --mmsi-file mmsi_list.txt --days 14

  # Just the last 24 hours for all visible vessels
  python collect_barentswatch.py --mode last24h
        """,
    )
    p.add_argument(
        "--mode", choices=["live", "historic", "last24h"], default="historic",
        help="Collection mode (default: historic)",
    )
    p.add_argument(
        "--days", type=int, default=7,
        help="Number of days back for historic mode (max 14, default 7)",
    )
    p.add_argument(
        "--mmsi-file", type=str, default=None,
        help="Path to a text file with one MMSI per line. "
             "If omitted, fetches all currently visible vessels.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: data/raw/bw_ais_{mode}.csv)",
    )
    p.add_argument(
        "--delay", type=float, default=REQUEST_DELAY_SEC,
        help=f"Seconds between requests (default: {REQUEST_DELAY_SEC})",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Validate credentials
    if CLIENT_ID == "YOUR_CLIENT_ID" or CLIENT_SECRET == "YOUR_CLIENT_SECRET":
        log.error(
            "No credentials found.\n"
            "Set BW_CLIENT_ID and BW_CLIENT_SECRET as environment variables, or\n"
            "edit CLIENT_ID / CLIENT_SECRET at the top of this file.\n"
            "Register an API client at: https://www.barentswatch.no/minside/"
        )
        sys.exit(1)

    global REQUEST_DELAY_SEC
    REQUEST_DELAY_SEC = args.delay

    token_mgr = TokenManager(CLIENT_ID, CLIENT_SECRET)
    output    = Path(args.output) if args.output else OUTPUT_DIR / f"bw_ais_{args.mode}.csv"

    # ── LIVE ──────────────────────────────────────────────────────────────────
    if args.mode == "live":
        rows = collect_live(token_mgr)
        save_csv(rows, output)

    # ── LAST 24H ──────────────────────────────────────────────────────────────
    elif args.mode == "last24h":
        if args.mmsi_file:
            mmsi_list = load_mmsi_file(args.mmsi_file)
        else:
            mmsi_list = fetch_mmsi_list_from_live(token_mgr)
            time.sleep(REQUEST_DELAY_SEC)

        rows = collect_last24h(token_mgr, mmsi_list)
        save_csv(rows, output)

    # ── HISTORIC ──────────────────────────────────────────────────────────────
    elif args.mode == "historic":
        days = min(args.days, 14)   # API hard limit
        if args.days > 14:
            log.warning("Barentswatch only holds 14 days of data – clamping to 14")

        to_date   = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=days)

        if args.mmsi_file:
            mmsi_list = load_mmsi_file(args.mmsi_file)
        else:
            mmsi_list = fetch_mmsi_list_from_live(token_mgr)
            time.sleep(REQUEST_DELAY_SEC)

        # Progress file: tracks which MMSIs are done so runs can be resumed
        progress_path = output.with_suffix(".progress")
        log.info("Progress file: %s", progress_path)

        # Load existing window if resuming, otherwise create and save a new one.
        # This ensures all vessels in a dataset share the exact same date range
        # even if collection is spread across multiple days.
        saved_from, saved_to = load_window(progress_path)
        if saved_from is not None:
            from_date = saved_from
            to_date   = saved_to
            log.info("Using saved date window (consistent resume)")
        else:
            save_window(progress_path, from_date, to_date)
            log.info(
                "New session: window %s → %s saved",
                from_date.strftime("%Y-%m-%d"),
                to_date.strftime("%Y-%m-%d"),
            )

        rows = collect_historic(token_mgr, mmsi_list, from_date, to_date,
                                progress_path=progress_path)
        save_csv(rows, output)

        done_count = len(load_progress(progress_path))
        remaining  = len(mmsi_list) - done_count
        if remaining > 0:
            log.info(
                "%d/%d MMSIs fetched. %d remaining – re-run the same command to continue.",
                done_count, len(mmsi_list), remaining,
            )
        else:
            log.info(
                "All %d MMSIs fetched. Dataset complete for window %s → %s",
                done_count,
                from_date.strftime("%Y-%m-%d"),
                to_date.strftime("%Y-%m-%d"),
            )
            log.info(
                "To start a fresh collection: delete %s and %s",
                progress_path, progress_path.with_suffix(".window"),
            )

    log.info("Done.")


if __name__ == "__main__":
    main()
