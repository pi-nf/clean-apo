"""
Offline SQLite database storage and multi-threaded sync worker for Clean APO.
Stores and compresses 43,000+ IEM/headphone measurements and 1,800+ target curves
into a high-performance, single-file SQLite database (~60 MB compressed).
"""

from __future__ import annotations
import os
import time
import zlib
import sqlite3
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Set, Any
from PyQt6 import QtCore

CACHE_DIR = os.path.expanduser(r"~/.clean_apo_cache")
SQLITE_DB_PATH = os.path.join(CACHE_DIR, "offline_library.sqlite3")

# Core priority databases for 'essential' sync mode
ESSENTIAL_DB_KEYWORDS = [
    "peqdb", "sharur", "diamond", "superreview", "jacstone", "jaraba",
    "timmy", "viento", "crinacle", "badguy", "hbb", "dan", "gizaudio",
    "oratory", "autoeq", "audioai", "arn", "elrics", "listener"
]


class OfflineCurveDatabase:
    """Thread-safe SQLite database for compressed frequency response curves and targets."""

    def __init__(self, db_path: str = SQLITE_DB_PATH):
        self.db_path = os.path.abspath(db_path)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._clear_callbacks: List[Callable[[], None]] = []
        self._init_db()

    def register_clear_callback(self, callback: Callable[[], None]) -> None:
        """Registers a callback invoked when clear_all() wipes the offline database."""
        with self._lock:
            if callback not in self._clear_callbacks:
                self._clear_callbacks.append(callback)

    def _get_connection(self) -> sqlite3.Connection:
        """Returns the thread-local SQLite connection, initialized once per thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Closes the current thread's connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS measurements (
                        db_id TEXT,
                        model_file TEXT,
                        channel TEXT,
                        point_count INTEGER,
                        data BLOB,
                        updated_at REAL,
                        PRIMARY KEY (db_id, model_file, channel)
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS targets (
                        db_id TEXT,
                        target_file TEXT,
                        target_name TEXT,
                        point_count INTEGER,
                        data BLOB,
                        updated_at REAL,
                        PRIMARY KEY (db_id, target_file)
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_meas_file ON measurements(model_file);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_meas_db ON measurements(db_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_target_name ON targets(target_name);")

    @staticmethod
    def pack_curve(freqs: List[float], spls: List[float]) -> bytes:
        """Packs and zlib-compresses (freqs, spls) into a compact binary blob."""
        n = min(len(freqs), len(spls))
        if n == 0:
            return b""
        raw = np.asarray(freqs[:n], dtype=np.float32).tobytes() + np.asarray(spls[:n], dtype=np.float32).tobytes()
        return zlib.compress(raw, level=6)

    @staticmethod
    def unpack_curve(blob: bytes) -> Tuple[List[float], List[float]]:
        """Decompresses and unpacks a binary blob into (freqs, spls)."""
        if not blob:
            return [], []
        try:
            raw = zlib.decompress(blob)
            total_floats = len(raw) // 4
            n = total_floats // 2
            arr = np.frombuffer(raw[:n * 8], dtype=np.float32)
            return arr[:n].tolist(), arr[n:2 * n].tolist()
        except Exception:
            return [], []

    def get_measurement(
        self,
        db_id: str,
        model_file: str,
        channel: str = "avg",
        allow_cross_db: bool = False,
        allow_channel_fallback: Optional[bool] = None,
    ) -> Optional[Tuple[List[float], List[float]]]:
        """Retrieves and unpacks a cached measurement curve, strictly enforcing db_id & channel isolation."""
        clean_db = (db_id or "").strip()
        clean_file = (model_file or "").strip()
        if not clean_file:
            return None
        if allow_channel_fallback is None:
            # Only fall back to another stored channel when requesting generic "avg";
            # explicit "L" or "R" requests must not silently return "avg" or the opposite ear.
            allow_channel_fallback = (channel == "avg")

        conn = self._get_connection()
        cur = conn.cursor()

        if clean_db:
            # 1. Exact match by db_id, file, and channel
            cur.execute(
                "SELECT data FROM measurements WHERE db_id = ? AND model_file = ? AND channel = ?",
                (clean_db, clean_file, channel),
            )
            row = cur.fetchone()
            if row and row[0]:
                return self.unpack_curve(row[0])

            # 2. Optional within-database channel fallback (default only when channel == "avg")
            if allow_channel_fallback:
                cur.execute(
                    "SELECT data FROM measurements WHERE db_id = ? AND model_file = ? ORDER BY updated_at DESC LIMIT 1",
                    (clean_db, clean_file),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return self.unpack_curve(row[0])

        # 3. Opt-in cross-database fallback ONLY (disabled by default to preserve coupler/rig isolation)
        if allow_cross_db:
            cur.execute(
                "SELECT data FROM measurements WHERE model_file = ? AND channel = ? ORDER BY updated_at DESC LIMIT 1",
                (clean_file, channel),
            )
            row = cur.fetchone()
            if row and row[0]:
                return self.unpack_curve(row[0])
            if allow_channel_fallback:
                cur.execute(
                    "SELECT data FROM measurements WHERE model_file = ? ORDER BY updated_at DESC LIMIT 1",
                    (clean_file,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return self.unpack_curve(row[0])
        return None

    def save_measurement(
        self, db_id: str, model_file: str, channel: str, freqs: List[float], spls: List[float]
    ) -> None:
        """Saves a measurement curve into SQLite."""
        if not freqs or not spls:
            return
        blob = self.pack_curve(freqs, spls)
        n = min(len(freqs), len(spls))
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO measurements (db_id, model_file, channel, point_count, data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (db_id, model_file.strip(), channel, n, blob, time.time()))

    def save_measurements_batch(
        self, items: List[Tuple[str, str, str, List[float], List[float]]]
    ) -> None:
        """Batch insert for high-performance syncing."""
        if not items:
            return
        rows = []
        now = time.time()
        for db_id, model_file, channel, freqs, spls in items:
            n = min(len(freqs), len(spls))
            if n > 0:
                blob = self.pack_curve(freqs, spls)
                rows.append((db_id, model_file.strip(), channel, n, blob, now))

        if not rows:
            return

        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO measurements (db_id, model_file, channel, point_count, data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, rows)

    @staticmethod
    def _target_file_variants(raw: str) -> List[str]:
        """Builds normalized target filename/stem variants without crossing databases."""
        clean = (raw or "").strip()
        if not clean:
            return []
        stem = clean
        for ext in (".txt", ".csv"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)].strip()
                break
        base = stem[:-7].strip() if stem.lower().endswith(" target") else stem
        candidates = [clean, stem, base, f"{base} Target", f"{stem}.txt", f"{base} Target.txt"]
        out: List[str] = []
        seen: Set[str] = set()
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def get_target(
        self,
        db_id: str,
        target_file: str,
        target_name: Optional[str] = None,
        allow_cross_db: bool = False,
    ) -> Optional[Tuple[List[float], List[float]]]:
        """Retrieves and unpacks a target curve, strictly enforcing coupler/db_id isolation."""
        clean_db = (db_id or "").strip()
        clean_file = (target_file or "").strip()
        clean_name = (target_name or "").strip()
        if not clean_file and not clean_name:
            return None

        conn = self._get_connection()
        cur = conn.cursor()

        file_variants = self._target_file_variants(clean_file or clean_name)

        if clean_db:
            # 1. Exact or stem-normalized match within the exact db_id
            for variant in file_variants:
                cur.execute(
                    "SELECT data FROM targets WHERE db_id = ? AND target_file = ?",
                    (clean_db, variant),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return self.unpack_curve(row[0])

            # 2. Match by target_name within the SAME db_id
            name_variants = self._target_file_variants(clean_name or clean_file)
            for nv in name_variants:
                cur.execute(
                    "SELECT data FROM targets WHERE db_id = ? AND LOWER(target_name) = LOWER(?) ORDER BY updated_at DESC LIMIT 1",
                    (clean_db, nv),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return self.unpack_curve(row[0])

        # 3. Opt-in cross-database fallback ONLY (disabled by default to preserve rig isolation)
        if allow_cross_db:
            if clean_name:
                cur.execute(
                    "SELECT data FROM targets WHERE LOWER(target_name) = LOWER(?) ORDER BY updated_at DESC LIMIT 1",
                    (clean_name,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return self.unpack_curve(row[0])
            for variant in file_variants:
                cur.execute(
                    "SELECT data FROM targets WHERE target_file = ? ORDER BY updated_at DESC LIMIT 1",
                    (variant,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return self.unpack_curve(row[0])
        return None

    def get_targets_for_db(self, db_id: str) -> List[Tuple[str, str]]:
        """Returns list of (target_file, target_name) cached for a specific db_id."""
        clean_db = (db_id or "").strip()
        if not clean_db:
            return []
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT target_file, target_name FROM targets WHERE db_id = ?",
            (clean_db,),
        )
        return list(cur.fetchall())

    def save_target(
        self, db_id: str, target_file: str, target_name: str, freqs: List[float], spls: List[float]
    ) -> None:
        """Saves a target curve into SQLite."""
        if not freqs or not spls:
            return
        blob = self.pack_curve(freqs, spls)
        n = min(len(freqs), len(spls))
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO targets (db_id, target_file, target_name, point_count, data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (db_id, target_file.strip(), target_name.strip(), n, blob, time.time()))

    def save_targets_batch(
        self, items: List[Tuple[str, str, str, List[float], List[float]]]
    ) -> None:
        """Batch insert for target curves."""
        if not items:
            return
        rows = []
        now = time.time()
        for db_id, target_file, target_name, freqs, spls in items:
            n = min(len(freqs), len(spls))
            if n > 0:
                blob = self.pack_curve(freqs, spls)
                rows.append((db_id, target_file.strip(), target_name.strip(), n, blob, now))

        if not rows:
            return

        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO targets (db_id, target_file, target_name, point_count, data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, rows)

    def get_cached_measurement_keys(self) -> Set[Tuple[str, str]]:
        """Returns set of (db_id, model_file) already present in SQLite."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT db_id, model_file FROM measurements")
        return set(cur.fetchall())

    def get_cached_target_keys(self) -> Set[Tuple[str, str]]:
        """Returns set of (db_id, target_file) already present in SQLite."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT db_id, target_file FROM targets")
        return set(cur.fetchall())

    def _get_total_size_bytes(self) -> int:
        size_bytes = 0
        for p in (self.db_path, self.db_path + "-wal", self.db_path + "-shm"):
            try:
                if os.path.exists(p):
                    size_bytes += os.path.getsize(p)
            except OSError:
                pass
        return size_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistical metrics on the offline SQLite library (including WAL/SHM storage)."""
        size_mb = self._get_total_size_bytes() / (1024 * 1024)

        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM measurements")
        meas_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM targets")
        target_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT db_id) FROM measurements")
        db_count = cur.fetchone()[0]

        return {
            "measurements": meas_count,
            "targets": target_count,
            "databases": db_count,
            "size_mb": round(size_mb, 2),
            "db_path": self.db_path,
        }

    def get_synced_databases_summary(self) -> List[Dict[str, Any]]:
        """
        Returns per-db_id measurement and target counts sorted by measurement count descending.
        Each item has keys: 'db_id', 'measurements', 'targets', 'last_updated'.
        """
        conn = self._get_connection()
        cur = conn.cursor()

        summary: Dict[str, Dict[str, Any]] = {}
        cur.execute(
            "SELECT db_id, COUNT(DISTINCT model_file), MAX(updated_at) FROM measurements GROUP BY db_id"
        )
        for db_id, meas_cnt, last_ts in cur.fetchall():
            key = str(db_id or "")
            summary[key] = {
                "db_id": key,
                "measurements": int(meas_cnt or 0),
                "targets": 0,
                "last_updated": float(last_ts or 0.0),
            }

        cur.execute(
            "SELECT db_id, COUNT(DISTINCT target_file), MAX(updated_at) FROM targets GROUP BY db_id"
        )
        for db_id, targ_cnt, last_ts in cur.fetchall():
            key = str(db_id or "")
            if key not in summary:
                summary[key] = {
                    "db_id": key,
                    "measurements": 0,
                    "targets": int(targ_cnt or 0),
                    "last_updated": float(last_ts or 0.0),
                }
            else:
                summary[key]["targets"] = int(targ_cnt or 0)
                if last_ts and float(last_ts) > summary[key]["last_updated"]:
                    summary[key]["last_updated"] = float(last_ts)

        items = list(summary.values())
        items.sort(key=lambda d: (-d["measurements"], -d["targets"], d["db_id"].lower()))
        return items

    def compact_database(self) -> Tuple[bool, str]:
        """
        Checkpoints WAL with TRUNCATE and safely executes VACUUM to reclaim unused space.
        Returns `(success, summary_message)` including reclaimed MB.
        """
        before_bytes = self._get_total_size_bytes()
        vacuum_err: Optional[str] = None
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.OperationalError as e:
                return False, f"WAL checkpoint failed: {e}"
            except Exception as e:
                return False, f"WAL checkpoint failed: {e}"

            try:
                conn.execute("VACUUM")
            except sqlite3.OperationalError as e:
                vacuum_err = str(e)
            except Exception as e:
                vacuum_err = str(e)

            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass

        after_bytes = self._get_total_size_bytes()
        freed_mb = max(0.0, (before_bytes - after_bytes) / (1024 * 1024))
        after_mb = after_bytes / (1024 * 1024)

        if vacuum_err:
            return (
                False,
                f"WAL checkpoint completed ({after_mb:.2f} MB, freed {freed_mb:.2f} MB), "
                f"but VACUUM could not run while database is busy ({vacuum_err}).",
            )
        return True, f"Compacted database to {after_mb:.2f} MB (freed {freed_mb:.2f} MB)."

    def clear_all(self) -> None:
        """Clears all cached measurements and targets, checkpoints/vacuums safely, and notifies caches."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("DELETE FROM measurements")
                conn.execute("DELETE FROM targets")
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass
            try:
                conn.execute("VACUUM")
            except sqlite3.OperationalError:
                # Another thread may briefly hold a read snapshot; rows are already cleared.
                pass
            callbacks = list(self._clear_callbacks)

        for cb in callbacks:
            try:
                cb()
            except Exception:
                pass


# Global singleton instance
_GLOBAL_OFFLINE_DB: Optional[OfflineCurveDatabase] = None
_DB_INIT_LOCK = threading.Lock()


def get_offline_db() -> OfflineCurveDatabase:
    global _GLOBAL_OFFLINE_DB
    if _GLOBAL_OFFLINE_DB is None:
        with _DB_INIT_LOCK:
            if _GLOBAL_OFFLINE_DB is None:
                _GLOBAL_OFFLINE_DB = OfflineCurveDatabase()
    return _GLOBAL_OFFLINE_DB


# -----------------------------------------------------------------------------
# Background Multi-Threaded Sync Worker (QThread)
# -----------------------------------------------------------------------------
class OfflineSyncWorker(QtCore.QThread):
    """
    High-throughput background worker that downloads and stores measurement
    curves and targets into the local SQLite database.
    Supports selective filtering by db_id, category, and favorite providers,
    plus real-time throughput and ETA estimation.
    """
    sigProgress = QtCore.pyqtSignal(int, int, str, str)          # current, total, status_text, item_name
    sigProgressDetail = QtCore.pyqtSignal(int, int, str, str, float)  # current, total, status_text, item_name, eta_seconds
    sigPhaseChanged = QtCore.pyqtSignal(str)                      # phase name
    sigStatsUpdated = QtCore.pyqtSignal(int, int, float)          # meas_count, target_count, size_mb
    sigFinished = QtCore.pyqtSignal(int, int, int)                # new_saved, total_meas, total_targets
    sigError = QtCore.pyqtSignal(str)

    def __init__(
        self,
        squig_client,
        scope: str = "essential",  # "targets_only", "essential", "full"
        max_workers: int = 24,
        parent: Optional[QtCore.QObject] = None,
        selected_db_ids: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[str]] = None,
        favorites_only: bool = False,
        favorite_providers: Optional[Sequence[str]] = None,
    ):
        super().__init__(parent)
        self.squig_client = squig_client
        self.scope = scope
        self.max_workers = max_workers
        self.selected_db_ids: Optional[List[str]] = (
            [str(x).strip() for x in selected_db_ids if str(x or "").strip()]
            if selected_db_ids is not None
            else None
        )
        self.categories: Optional[List[str]] = (
            [str(x).strip() for x in categories if str(x or "").strip()]
            if categories is not None
            else None
        )
        self.favorites_only: bool = bool(favorites_only)
        self.favorite_providers: List[str] = (
            [str(x).strip() for x in favorite_providers if str(x or "").strip()]
            if favorite_providers
            else []
        )
        self.eta_seconds: Optional[float] = None
        self.items_per_second: float = 0.0
        self.db = get_offline_db()
        self._is_paused = False
        self._abort = False
        self._pause_cond = threading.Condition()

    @staticmethod
    def format_eta(eta_seconds: Optional[float]) -> str:
        """Formats an ETA in seconds into a concise human-readable duration string."""
        if eta_seconds is None or eta_seconds < 0.0:
            return ""
        total_secs = int(round(eta_seconds))
        if total_secs < 60:
            return f"{total_secs}s"
        if total_secs < 3600:
            mins, secs = divmod(total_secs, 60)
            return f"{mins}m {secs:02d}s"
        hrs, rem = divmod(total_secs, 3600)
        mins = rem // 60
        return f"{hrs}h {mins:02d}m"

    def _compute_eta(
        self,
        done_count: int,
        total_count: int,
        phase_start: float,
        paused_seconds: float,
    ) -> Optional[float]:
        if total_count <= 0 or done_count <= 0:
            self.eta_seconds = None
            return None
        if done_count >= total_count:
            self.eta_seconds = 0.0
            return 0.0
        elapsed = max(0.001, (time.monotonic() - phase_start) - paused_seconds)
        rate = done_count / elapsed
        self.items_per_second = rate
        if rate <= 0.0:
            self.eta_seconds = None
            return None
        eta = (total_count - done_count) / rate
        self.eta_seconds = eta
        return eta

    def _emit_progress(
        self,
        current: int,
        total: int,
        base_status: str,
        item_name: str,
        eta: Optional[float],
    ) -> None:
        eta_str = self.format_eta(eta) if (eta is not None and current < total) else ""
        status_text = f"{base_status} - ETA {eta_str}" if eta_str else base_status
        self.sigProgress.emit(current, total, status_text, item_name)
        self.sigProgressDetail.emit(
            current,
            total,
            status_text,
            item_name,
            float(eta) if eta is not None else -1.0,
        )

    def filter_catalog_entries(
        self,
        catalog: Sequence[Dict[str, Any]],
        apply_scope: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Applies selective sync filters (`selected_db_ids`, `categories`, `favorites_only`,
        and `scope`) to catalog entries while preserving hangout.audio exclusion.
        """
        from squig_client import _normalize_base_url

        selected_set: Optional[Set[str]] = (
            {s.lower() for s in self.selected_db_ids} if self.selected_db_ids is not None else None
        )
        fav_set: Set[str] = {f.lower() for f in self.favorite_providers}

        active_categories: List[str] = []
        if self.categories:
            for cat in self.categories:
                cat_low = cat.lower()
                if cat_low in ("all", "all categories", "all types"):
                    active_categories = []
                    break
                active_categories.append(cat)

        filtered: List[Dict[str, Any]] = []
        for r in catalog:
            base_url = r.get("base_url") or ""
            if not base_url or "hangout.audio" in base_url.lower():
                continue

            p_name = str(r.get("provider") or "")
            u_name = str(r.get("username") or "")
            db_type = str(r.get("db_type") or "IEMs")
            rig = str(r.get("rig") or "")
            db_id = self.squig_client.get_db_id(base_url)

            p_low = p_name.strip().lower()
            u_low = u_name.strip().lower()
            db_id_low = db_id.strip().lower()
            disp_low = (f"{p_name} ({db_type})" if db_type else p_name).strip().lower()
            norm_url_low = _normalize_base_url(base_url)
            identity_tokens = {tok for tok in (db_id_low, u_low, p_low, disp_low, norm_url_low) if tok}

            if selected_set is not None:
                if not (identity_tokens & selected_set):
                    continue

            if active_categories:
                if not any(
                    self.squig_client._matches_category(cat, db_type, p_name, base_url, rig)
                    for cat in active_categories
                ):
                    continue

            if self.favorites_only:
                if not fav_set or not (identity_tokens & fav_set):
                    continue

            # Apply 'essential' scope only when user has not explicitly narrowed to specific DBs or favorites
            if (
                apply_scope
                and self.scope == "essential"
                and selected_set is None
                and not self.favorites_only
            ):
                is_essential = any(
                    k in p_low or k in u_low or k in base_url.lower()
                    for k in ESSENTIAL_DB_KEYWORDS
                )
                if not is_essential:
                    continue

            filtered.append(r)

        return filtered

    def pause(self) -> None:
        with self._pause_cond:
            self._is_paused = True

    def resume(self) -> None:
        with self._pause_cond:
            self._is_paused = False
            self._pause_cond.notify_all()

    def is_paused(self) -> bool:
        return self._is_paused

    def cancel(self) -> None:
        self._abort = True
        self.resume()

    def run(self) -> None:
        try:
            from squig_client import SquigTarget, SquigModel

            self.eta_seconds = None
            self.items_per_second = 0.0

            if self._abort:
                stats = self.db.get_stats()
                self.sigFinished.emit(0, stats["measurements"], stats["targets"])
                return

            catalog = self.squig_client.get_global_catalog()
            if not catalog:
                self.sigError.emit("Failed to load global database catalog.")
                return

            cached_targets = self.db.get_cached_target_keys()
            cached_models = self.db.get_cached_measurement_keys()

            new_saved = 0

            # -----------------------------------------------------------------
            # PHASE 1: Sync Target Curves across filtered reviewer databases
            # -----------------------------------------------------------------
            self.sigPhaseChanged.emit("Syncing Target Curves...")
            target_reviewers = self.filter_catalog_entries(catalog, apply_scope=False)
            target_tasks = []
            seen_target_tasks: Set[Tuple[str, str]] = set()
            for r in target_reviewers:
                base_url = r.get("base_url")
                if not base_url:
                    continue
                db_id = self.squig_client.get_db_id(base_url)

                for t_name in r.get("targets", []):
                    clean_t = str(t_name).strip()
                    if not clean_t:
                        continue
                    t_file = clean_t if clean_t.lower().endswith("target") else f"{clean_t} Target"
                    # Check normalized keys so previously synced rows aren't re-downloaded
                    is_cached = (
                        (db_id, t_file) in cached_targets
                        or (db_id, clean_t) in cached_targets
                        or (db_id, f"{clean_t}.txt") in cached_targets
                    )
                    if not is_cached and (db_id, t_file) not in seen_target_tasks:
                        seen_target_tasks.add((db_id, t_file))
                        target_tasks.append((db_id, base_url, clean_t, t_file))

            total_t_tasks = len(target_tasks)
            done_t_tasks = 0

            if target_tasks and not self._abort:
                batch_targets = []
                pool_workers = min(self.max_workers, 20)
                phase1_start = time.monotonic()
                phase1_paused = 0.0

                def _worker_fetch_target(task_info):
                    if self._abort:
                        return None
                    db_i, b_url, t_nm, t_fl = task_info
                    t_obj = SquigTarget(name=t_nm, file_name=t_fl)
                    f_p, s_p = self.squig_client.fetch_target_curve(
                        b_url,
                        t_obj,
                        force_refresh=False,
                        skip_offline_db=True,
                        save_to_db=False,
                    )
                    return db_i, t_fl, t_nm, f_p, s_p

                executor = ThreadPoolExecutor(max_workers=pool_workers)
                active_futures = set()
                try:
                    task_iter = iter(target_tasks)

                    # Initial continuous saturation (fill worker pipeline)
                    for _ in range(pool_workers * 3):
                        if self._abort:
                            break
                        try:
                            t = next(task_iter)
                            active_futures.add(executor.submit(_worker_fetch_target, t))
                        except StopIteration:
                            break

                    while active_futures and not self._abort:
                        with self._pause_cond:
                            if self._is_paused and not self._abort:
                                pause_t0 = time.monotonic()
                                while self._is_paused and not self._abort:
                                    self._pause_cond.wait(timeout=0.5)
                                phase1_paused += time.monotonic() - pause_t0
                        if self._abort:
                            break

                        done, _ = wait(active_futures, timeout=0.5, return_when=FIRST_COMPLETED)
                        if not done:
                            continue
                        for fut in done:
                            active_futures.remove(fut)
                            try:
                                res = fut.result()
                                if res is None:
                                    continue
                                db_i, t_fl, t_nm, f_p, s_p = res
                                done_t_tasks += 1
                                if f_p and s_p:
                                    batch_targets.append((db_i, t_fl, t_nm, f_p, s_p))
                                    new_saved += 1
                                if len(batch_targets) >= 30:
                                    self.db.save_targets_batch(batch_targets)
                                    batch_targets.clear()
                                    stats = self.db.get_stats()
                                    self.sigStatsUpdated.emit(stats["measurements"], stats["targets"], stats["size_mb"])
                                if done_t_tasks % 15 == 0 or done_t_tasks == total_t_tasks:
                                    eta = self._compute_eta(done_t_tasks, total_t_tasks, phase1_start, phase1_paused)
                                    self._emit_progress(
                                        done_t_tasks,
                                        total_t_tasks,
                                        f"Downloaded {done_t_tasks}/{total_t_tasks} targets",
                                        t_nm,
                                        eta,
                                    )
                            except Exception:
                                done_t_tasks += 1

                            if not self._abort:
                                try:
                                    next_task = next(task_iter)
                                    active_futures.add(executor.submit(_worker_fetch_target, next_task))
                                except StopIteration:
                                    pass
                finally:
                    if self._abort:
                        for fut in active_futures:
                            fut.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        executor.shutdown(wait=True)

                if batch_targets:
                    self.db.save_targets_batch(batch_targets)
            elif not self._abort and self.scope == "targets_only":
                self.eta_seconds = 0.0
                self._emit_progress(1, 1, "All targets already cached", "", 0.0)

            stats = self.db.get_stats()
            self.sigStatsUpdated.emit(stats["measurements"], stats["targets"], stats["size_mb"])

            if self.scope == "targets_only" or self._abort:
                self.eta_seconds = 0.0
                self.sigFinished.emit(new_saved, stats["measurements"], stats["targets"])
                return

            # -----------------------------------------------------------------
            # PHASE 2 & 3: Sync Measurement Curves
            # -----------------------------------------------------------------
            self.sigPhaseChanged.emit("Syncing Measurement Curves...")
            self.eta_seconds = None

            candidate_reviewers = self.filter_catalog_entries(catalog, apply_scope=True)

            meas_tasks = []
            seen_meas_tasks: Set[Tuple[str, str]] = set()
            for r in candidate_reviewers:
                base_url = r.get("base_url")
                db_id = self.squig_client.get_db_id(base_url)
                for p in r.get("phones", []):
                    fn = p.get("file")
                    if isinstance(fn, list):
                        fn = fn[0] if fn else ""
                    if not fn:
                        continue
                    clean_fn = str(fn).strip()
                    if (db_id, clean_fn) not in cached_models and (db_id, clean_fn) not in seen_meas_tasks:
                        seen_meas_tasks.add((db_id, clean_fn))
                        model_name = p.get("name") or clean_fn
                        meas_tasks.append((db_id, base_url, clean_fn, model_name))

            total_m_tasks = len(meas_tasks)
            done_m_tasks = 0

            if meas_tasks and not self._abort:
                batch_meas = []
                meas_pool_workers = min(self.max_workers, 24)
                phase2_start = time.monotonic()
                phase2_paused = 0.0

                # Concurrent multi-threaded downloads with streaming sliding window
                def _worker_fetch(task_info):
                    if self._abort:
                        return None
                    db_i, b_url, c_fn, m_nm = task_info
                    m_obj = SquigModel(brand="", name=m_nm, full_name=m_nm, file_name=c_fn, display_name=m_nm)
                    f_p, s_p = self.squig_client.fetch_iem_measurement(
                        b_url,
                        m_obj,
                        channel_mode="avg",
                        force_refresh=False,
                        skip_offline_db=True,
                        save_to_db=False,
                    )
                    return db_i, c_fn, m_nm, f_p, s_p

                executor = ThreadPoolExecutor(max_workers=meas_pool_workers)
                active_futures = set()
                try:
                    task_iter = iter(meas_tasks)

                    # Prime pipeline with saturated in-flight futures
                    for _ in range(meas_pool_workers * 3):
                        if self._abort:
                            break
                        try:
                            t = next(task_iter)
                            active_futures.add(executor.submit(_worker_fetch, t))
                        except StopIteration:
                            break

                    while active_futures and not self._abort:
                        with self._pause_cond:
                            if self._is_paused and not self._abort:
                                pause_t0 = time.monotonic()
                                while self._is_paused and not self._abort:
                                    self._pause_cond.wait(timeout=0.5)
                                phase2_paused += time.monotonic() - pause_t0
                        if self._abort:
                            break

                        done, _ = wait(active_futures, timeout=0.5, return_when=FIRST_COMPLETED)
                        if not done:
                            continue
                        for fut in done:
                            active_futures.remove(fut)
                            try:
                                res = fut.result()
                                if res is None:
                                    continue
                                db_i, c_fn, m_nm, f_p, s_p = res
                                done_m_tasks += 1
                                if f_p and s_p:
                                    batch_meas.append((db_i, c_fn, "avg", f_p, s_p))
                                    new_saved += 1

                                if len(batch_meas) >= 50:
                                    self.db.save_measurements_batch(batch_meas)
                                    batch_meas.clear()
                                    stats = self.db.get_stats()
                                    self.sigStatsUpdated.emit(stats["measurements"], stats["targets"], stats["size_mb"])

                                if done_m_tasks % 20 == 0 or done_m_tasks == total_m_tasks:
                                    eta = self._compute_eta(done_m_tasks, total_m_tasks, phase2_start, phase2_paused)
                                    self._emit_progress(
                                        done_m_tasks,
                                        total_m_tasks,
                                        f"Synced {done_m_tasks}/{total_m_tasks} curves ({stats['size_mb']} MB)",
                                        m_nm,
                                        eta,
                                    )
                            except Exception:
                                done_m_tasks += 1

                            if not self._abort:
                                try:
                                    next_task = next(task_iter)
                                    active_futures.add(executor.submit(_worker_fetch, next_task))
                                except StopIteration:
                                    pass
                finally:
                    if self._abort:
                        for fut in active_futures:
                            fut.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        executor.shutdown(wait=True)

                if batch_meas:
                    self.db.save_measurements_batch(batch_meas)
            elif not self._abort:
                self.eta_seconds = 0.0
                self._emit_progress(1, 1, "All curves already cached", "", 0.0)

            self.eta_seconds = 0.0
            stats = self.db.get_stats()
            self.sigStatsUpdated.emit(stats["measurements"], stats["targets"], stats["size_mb"])
            self.sigFinished.emit(new_saved, stats["measurements"], stats["targets"])

        except Exception as e:
            self.sigError.emit(str(e))
        finally:
            self.db.close()
