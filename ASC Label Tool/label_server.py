#!/usr/bin/env python3
"""ASC Training Data labeler — Clear/Cloudy + rain + snow (web UI)."""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

TOOL_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Volumes/James' SSD/ASC Training Data")
CSV_PATH = TOOL_DIR / "labels.csv"
LABEL_BACKUP_DIR = TOOL_DIR / "label_backups"
# Local SSD journal (Mac internal disk) — Enter must not wait on ExFAT full CSV rewrite
LOCAL_STATE_DIR = Path.home() / ".asc_label_tool"
JOURNAL_PATH = LOCAL_STATE_DIR / "labels_journal.jsonl"
UI_PATH = TOOL_DIR / "label_ui.html"
INGEST_MAP = TOOL_DIR / "ingested_source_names.txt"
SOLAR_DIR = Path("/Users/tianqiming/Desktop/Pomfret Astro/website/observatory")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CSV_COLUMNS = ["filename", "sky", "rain", "snow", "day_night", "notes"]
MARKED_DIR = DATA_DIR / "Marked For Training"
TEST_DIR = DATA_DIR / "Test"
PORT = 8765
MAX_LABEL_BACKUPS = 30
BACKUP_EVERY_N_SAVES = 25

_save_count = 0
_cached_usable = 0
_marked_name_cache: list[str] | None = None
_pool_cache: list[str] | None = None
_labels_lock = threading.RLock()
_flush_event = threading.Event()
_flush_thread: threading.Thread | None = None
_stop_flush = threading.Event()
_dirty = False
_pool_labeled_count = 0


_hh_re = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.tiff$", re.I)
_nm_re = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_", re.I)
_sky_re = re.compile(r"^sky_(\d+)\.jpg$", re.I)
_src_re = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(\d{2})\.tiff$")

_sky_source_times: dict[int, tuple[int, int, int, int, int, int]] | None = None


def _load_sky_source_times():
    global _sky_source_times
    if _sky_source_times is not None:
        return _sky_source_times
    out = {}
    if INGEST_MAP.is_file():
        lines = INGEST_MAP.read_text(encoding="utf-8").strip().splitlines()
        for i, line in enumerate(lines, start=1):
            m = _src_re.match(line.strip())
            if m:
                y, mo, d = map(int, m.group(1).split("-"))
                out[i] = (y, mo, d, int(m.group(2)), int(m.group(3)), int(m.group(4)))
    _sky_source_times = out
    return out


def infer_day_night(filename: str) -> str:
    """Production rule: nautical dawn→dusk = day (same as ASC on Pi)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    sys.path.insert(0, str(SOLAR_DIR))
    from observatory_solar import is_asc_model_daytime

    tz = ZoneInfo("America/New_York")
    dt = None
    m = _hh_re.match(filename)
    if m:
        y, mo, d = map(int, m.group(1).split("-"))
        dt = datetime(y, mo, d, int(m.group(2)), int(m.group(3)), tzinfo=tz)
    else:
        m = _nm_re.match(filename)
        if m:
            y, mo, d = map(int, m.group(1).split("-"))
            dt = datetime(
                y, mo, d, int(m.group(2)), int(m.group(3)), int(m.group(4)), tzinfo=tz
            )
        else:
            m = _sky_re.match(filename)
            if m:
                t = _load_sky_source_times().get(int(m.group(1)))
                if t:
                    dt = datetime(*t, tzinfo=tz)
    if dt is None:
        return ""
    return "day" if is_asc_model_daytime(dt.astimezone()) else "night"


def collect_images(force: bool = False) -> list[str]:
    """Unlabeled pool = files in Training Data root. Cached — ExFAT directory scans are slow."""
    global _pool_cache
    if _pool_cache is not None and not force:
        return _pool_cache
    _pool_cache = sorted(
        p.name
        for p in DATA_DIR.iterdir()
        if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return _pool_cache


def resolve_image(filename: str) -> Path | None:
    for base in (DATA_DIR, MARKED_DIR, TEST_DIR):
        path = (base / filename).resolve()
        try:
            path.relative_to(DATA_DIR.resolve())
        except ValueError:
            continue
        if path.is_file():
            return path
    return None


def is_complete(row: dict[str, str]) -> bool:
    return bool(row.get("sky") and row.get("rain") and row.get("snow") and row.get("day_night"))


def read_csv_labels() -> dict[str, dict[str, str]]:
    """Read every row from labels.csv. Never invent blank rows here."""
    labels: dict[str, dict[str, str]] = {}
    if not CSV_PATH.exists():
        return labels
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("filename") or "").strip()
            if not name:
                continue
            labels[name] = {col: (row.get(col) or "").strip() for col in CSV_COLUMNS}
            labels[name]["filename"] = name
    return labels


def _merge_protect(
    memory: dict[str, dict[str, str]], disk: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Merge disk→memory so a blank in-memory row can never erase a labeled disk row."""
    out = {k: dict(v) for k, v in disk.items()}
    for name, row in memory.items():
        prev = out.get(name)
        if prev is None:
            out[name] = dict(row)
            out[name]["filename"] = name
            continue
        # If memory blanks a previously labeled sky/rain, keep disk values.
        merged = dict(prev)
        for col in CSV_COLUMNS:
            if col == "filename":
                merged["filename"] = name
                continue
            new_v = (row.get(col) or "").strip()
            old_v = (prev.get(col) or "").strip()
            if col in {"sky", "rain", "snow", "day_night"} and old_v and not new_v:
                continue  # protect
            if new_v or col == "notes":
                # notes may be intentionally cleared; sky/rain/snow/day_night only overwrite if non-empty
                if col == "notes":
                    merged[col] = new_v
                elif new_v:
                    merged[col] = new_v
        out[name] = merged
    return out


def load_labels() -> dict[str, dict[str, str]]:
    """Load CSV (including Marked rows) + ensure root unlabeled images exist as empty rows."""
    global _cached_usable, _marked_name_cache, _pool_labeled_count
    labels = read_csv_labels()
    n_journal = replay_journal(labels)
    if n_journal:
        print(f"Replayed {n_journal} journal entries from {JOURNAL_PATH}", flush=True)
    root = collect_images(force=True)
    added = 0
    for name in root:
        if name not in labels:
            labels[name] = {
                "filename": name,
                "sky": "",
                "rain": "",
                "snow": "",
                "day_night": infer_day_night(name),
                "notes": "",
            }
            added += 1
        else:
            labels[name].setdefault("notes", "")
            if not labels[name].get("day_night"):
                labels[name]["day_night"] = infer_day_night(name)

    _marked_name_cache = list_marked_names(force=True)
    _cached_usable = sum(1 for r in labels.values() if r.get("sky") and r.get("rain"))
    _pool_labeled_count = _count_pool_labeled(labels)
    # Compact journal into CSV once at startup if needed
    if added or n_journal:
        save_labels(labels, backup=True)
        clear_journal()
    return labels


def _rotate_label_backups() -> None:
    """Keep the newest MAX_LABEL_BACKUPS timestamped copies of labels.csv."""
    if not CSV_PATH.is_file():
        return
    LABEL_BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = LABEL_BACKUP_DIR / f"labels_{stamp}.csv"
    shutil.copy2(CSV_PATH, dst)
    latest = LABEL_BACKUP_DIR / "labels_latest.csv"
    if latest.exists():
        try:
            latest.chmod(0o644)
        except OSError:
            pass
    shutil.copy2(CSV_PATH, latest)
    backups = sorted(
        LABEL_BACKUP_DIR.glob("labels_2*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in backups[MAX_LABEL_BACKUPS:]:
        if "FROZEN" in old.name:
            continue
        try:
            old.unlink()
        except OSError:
            pass


def list_marked_names(force: bool = False) -> list[str]:
    global _marked_name_cache
    if _marked_name_cache is not None and not force:
        return _marked_name_cache
    marked_dir = MARKED_DIR
    if not marked_dir.is_dir():
        _marked_name_cache = []
        return _marked_name_cache
    _marked_name_cache = sorted(
        p.name
        for p in marked_dir.iterdir()
        if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return _marked_name_cache


def save_labels(labels: dict[str, dict[str, str]], backup: bool | None = None) -> None:
    """Atomic full rewrite of labels.csv onto the ExFAT volume (slow — call from flusher)."""
    global _save_count, _cached_usable

    marked_names = list_marked_names()
    root_names = collect_images()
    all_names = list(dict.fromkeys(marked_names + root_names + sorted(labels.keys())))

    new_usable = sum(
        1
        for name in all_names
        if (labels.get(name) or {}).get("sky") and (labels.get(name) or {}).get("rain")
    )
    if _cached_usable >= 50 and new_usable < int(_cached_usable * 0.9):
        disk = read_csv_labels()
        protected = _merge_protect(labels, disk)
        labels.clear()
        labels.update(protected)
        new_usable = sum(1 for r in labels.values() if r.get("sky") and r.get("rain"))
        if new_usable < int(_cached_usable * 0.9):
            raise RuntimeError(
                f"Refusing to save labels.csv: usable would drop {_cached_usable} → {new_usable}"
            )
        all_names = list(dict.fromkeys(marked_names + root_names + sorted(labels.keys())))

    _save_count += 1
    do_backup = backup if backup is not None else (_save_count % BACKUP_EVERY_N_SAVES == 0)
    if do_backup and CSV_PATH.is_file():
        _rotate_label_backups()

    tmp_path = CSV_PATH.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for name in all_names:
            row = labels.get(name, {})
            writer.writerow(
                {
                    "filename": name,
                    "sky": row.get("sky", ""),
                    "rain": row.get("rain", ""),
                    "snow": row.get("snow", ""),
                    "day_night": row.get("day_night", ""),
                    "notes": row.get("notes", "") or "",
                }
            )
    tmp_path.replace(CSV_PATH)
    _cached_usable = new_usable


def append_journal(row: dict[str, str]) -> None:
    """Crash-safe tiny append on internal disk (milliseconds)."""
    LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def replay_journal(labels: dict[str, dict[str, str]]) -> int:
    if not JOURNAL_PATH.is_file():
        return 0
    n = 0
    with JOURNAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = (row.get("filename") or "").strip()
            if not name:
                continue
            labels[name] = {col: (row.get(col) or "").strip() for col in CSV_COLUMNS}
            labels[name]["filename"] = name
            n += 1
    return n


def clear_journal() -> None:
    if JOURNAL_PATH.is_file():
        JOURNAL_PATH.write_text("", encoding="utf-8")


def _count_pool_labeled(labels: dict[str, dict[str, str]]) -> int:
    return sum(1 for name in collect_images() if is_complete(labels.get(name, {})))


def schedule_flush() -> None:
    global _dirty
    _dirty = True
    _flush_event.set()


def _flush_worker(labels_ref: dict[str, dict[str, str]]) -> None:
    """Coalesce dirty flags and rewrite CSV in the background."""
    global _dirty
    while not _stop_flush.is_set():
        _flush_event.wait(timeout=0.5)
        _flush_event.clear()
        if _stop_flush.is_set():
            break
        if not _dirty:
            continue
        # small debounce so rapid Enter only triggers one ExFAT write
        time.sleep(0.15)
        if _stop_flush.is_set():
            break
        with _labels_lock:
            if not _dirty:
                continue
            snapshot = {k: dict(v) for k, v in labels_ref.items()}
            _dirty = False
        try:
            t0 = time.time()
            save_labels(snapshot, backup=None)
            clear_journal()
            print(f"[flush] labels.csv written in {time.time() - t0:.2f}s", flush=True)
        except Exception as e:
            # keep journal; mark dirty again
            with _labels_lock:
                _dirty = True
            print(f"[flush] ERROR {e}", flush=True)


def start_flusher(labels_ref: dict[str, dict[str, str]]) -> None:
    global _flush_thread
    _stop_flush.clear()
    _flush_thread = threading.Thread(
        target=_flush_worker, args=(labels_ref,), name="label-flush", daemon=True
    )
    _flush_thread.start()


def stop_flusher(labels_ref: dict[str, dict[str, str]]) -> None:
    """Final synchronous flush on shutdown."""
    global _dirty
    _stop_flush.set()
    _flush_event.set()
    if _flush_thread is not None:
        _flush_thread.join(timeout=30)
    with _labels_lock:
        need = _dirty or (JOURNAL_PATH.is_file() and JOURNAL_PATH.stat().st_size > 0)
        snapshot = {k: dict(v) for k, v in labels_ref.items()}
        _dirty = False
    if need:
        save_labels(snapshot, backup=True)
        clear_journal()
        print("[flush] final sync done", flush=True)


def jpeg_preview_bytes(img_path: Path, max_side: int = 1280) -> bytes:
    from PIL import Image

    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()


class LabelHandler(BaseHTTPRequestHandler):
    labels: dict[str, dict[str, str]] = {}

    def log_message(self, fmt: str, *args) -> None:
        print(f"[label] {args[0]}" if args else fmt, flush=True)

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            content = UI_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/state":
            images = collect_images()
            # Only send labels for the labeling pool (root). Marked rows stay in memory/CSV
            # but must not bloat this response or the UI will hang on load.
            pool_labels = {
                name: self.labels.get(
                    name,
                    {
                        "filename": name,
                        "sky": "",
                        "rain": "",
                        "snow": "",
                        "day_night": "",
                        "notes": "",
                    },
                )
                for name in images
            }
            labeled = sum(1 for name in images if is_complete(pool_labels.get(name, {})))
            marked_labeled = sum(
                1
                for name in list_marked_names()
                if is_complete(self.labels.get(name, {}))
            )
            self._send_json(
                {
                    "images": images,
                    "labels": pool_labels,
                    "total": len(images),
                    "labeled": labeled,
                    "marked_labeled": marked_labeled,
                    "data_dir": str(DATA_DIR),
                    "csv_path": str(CSV_PATH),
                }
            )
            return

        if path.startswith("/img/"):
            filename = unquote(path[5:])
            img_path = resolve_image(filename)
            if img_path is None:
                self.send_error(404)
                return
            suffix = img_path.suffix.lower()
            try:
                if suffix in {".tif", ".tiff"}:
                    data = jpeg_preview_bytes(img_path)
                    mime = "image/jpeg"
                else:
                    data = img_path.read_bytes()
                    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(404)
            return

        payload = self._read_json()
        filename = (payload.get("filename") or "").strip()
        if filename not in self.labels:
            self._send_json({"ok": False, "error": "unknown file"}, 400)
            return

        sky = str(payload.get("sky", "")).strip().lower()
        rain = str(payload.get("rain", "")).strip().lower()
        snow = str(payload.get("snow", "")).strip().lower()
        day_night = str(payload.get("day_night", "")).strip().lower()
        notes = str(payload.get("notes", self.labels[filename].get("notes", ""))).strip()

        if sky not in {"", "clear", "cloudy"}:
            self._send_json({"ok": False, "error": "sky must be clear|cloudy"}, 400)
            return
        if rain not in {"", "true", "false"}:
            self._send_json({"ok": False, "error": "rain must be true|false"}, 400)
            return
        if snow not in {"", "true", "false"}:
            self._send_json({"ok": False, "error": "snow must be true|false"}, 400)
            return
        if day_night not in {"", "day", "night"}:
            self._send_json({"ok": False, "error": "day_night must be day|night"}, 400)
            return

        old = dict(self.labels.get(filename, {}))
        self.labels[filename] = {
            "filename": filename,
            "sky": sky,
            "rain": rain,
            "snow": snow,
            "day_night": day_night or infer_day_night(filename),
            "notes": notes,
        }
        # Instant path: memory + local journal. ExFAT CSV rewrite happens in background.
        append_journal(self.labels[filename])
        global _pool_labeled_count, _cached_usable
        was_complete = is_complete(old)
        now_complete = is_complete(self.labels[filename])
        if not was_complete and now_complete:
            _pool_labeled_count += 1
        elif was_complete and not now_complete:
            _pool_labeled_count = max(0, _pool_labeled_count - 1)
        was_usable = bool(old.get("sky") and old.get("rain"))
        now_usable = bool(sky and rain)
        if not was_usable and now_usable:
            _cached_usable += 1
        elif was_usable and not now_usable:
            _cached_usable = max(0, _cached_usable - 1)
        schedule_flush()

        images = collect_images()
        self._send_json(
            {"ok": True, "labeled": _pool_labeled_count, "total": len(images)}
        )


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(f"DATA_DIR missing: {DATA_DIR}")
    print("Loading labels + scanning image pool (ExFAT may take ~30–60s once)...", flush=True)
    LabelHandler.labels = load_labels()
    start_flusher(LabelHandler.labels)
    n_labels = len(LabelHandler.labels)
    n_pool = len(collect_images())
    usable = sum(1 for r in LabelHandler.labels.values() if r.get("sky") and r.get("rain"))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), LabelHandler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"ASC Label Tool: {url}", flush=True)
    print(
        f"Pool (root): {n_pool}  |  labels in memory: {n_labels}  |  usable: {usable}",
        flush=True,
    )
    print(
        f"Marked labeled: {sum(1 for n in list_marked_names() if is_complete(LabelHandler.labels.get(n, {})))}",
        flush=True,
    )
    print(f"Labels file: {CSV_PATH}", flush=True)
    print(f"Local journal: {JOURNAL_PATH}", flush=True)
    print("Ready. Enter is instant; CSV flushes in background. Ctrl+C to stop.", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nflushing before exit...", flush=True)
        stop_flusher(LabelHandler.labels)
        print("stopped", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
