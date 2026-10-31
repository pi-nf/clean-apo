"""
Offline SQLite database storage and multi-threaded sync worker for Clean APO.
Stores and compresses 43,000+ IEM/headphone measurements and 1,800+ target curves
into a high-performance, single-file SQLite database (~60 MB compressed).
"""

from __future__ import annotations
import os
import sys
import time
import zlib
import struct
import sqlite3
import threading
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Dict, List, Optional, Tuple, Set, Any
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
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
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
            finally:
                conn.close()

    @staticmethod
    def pack_curve(freqs: List[float], spls: List[float]) -> bytes:
        """Packs and zlib-compresses (freqs, spls) into a compact binary blob."""
        n = min(len(freqs), len(spls))
        if n == 0:
            return b""
        raw = struct.pack(f"{n}f", *freqs[:n]) + struct.pack(f"{n}f", *spls[:n])
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
            f_vals = list(struct.unpack(f"{n}f", raw[:n * 4]))
            s_vals = list(struct.unpack(f"{n}f", raw[n * 4:n * 8]))
            return f_vals, s_vals
        except Exception:
            return [], []

    def get_measurement(
        self, db_id: str, model_file: str, channel: str = "avg"
    ) -> Optional[Tuple[List[float], List[float]]]:
        """Retrieves and unpacks a cached measurement curve."""
        clean_file = model_file.strip()
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            # 1. Exact match by db_id, file, and channel
            cur.execute(
                "SELECT data FROM measurements WHERE db_id = ? AND model_file = ? AND channel = ?",
                (db_id, clean_file, channel)
            )
            row = cur.fetchone()
            if row and row[0]:
                return self.unpack_curve(row[0])

            # 2. Fallback to 'avg' if looking for L/R or vice versa
            cur.execute(
                "SELECT data FROM measurements WHERE db_id = ? AND model_file = ?",
                (db_id, clean_file)
            )
            row = cur.fetchone()
            if row and row[0]:
                return self.unpack_curve(row[0])

            # 3. Global fallback across any database for this model_file
            cur.execute(
                "SELECT data FROM measurements WHERE model_file = ? ORDER BY updated_at DESC LIMIT 1",
                (clean_file,)
            )
            row = cur.fetchone()
            if row and row[0]:
                return self.unpack_curve(row[0])
            return None
        finally:
            conn.close()

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
            try:
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO measurements (db_id, model_file, channel, point_count, data, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (db_id, model_file.strip(), channel, n, blob, time.time()))
            finally:
                conn.close()

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
            try:
                with conn:
                    conn.executemany("""
                        INSERT OR REPLACE INTO measurements (db_id, model_file, channel, point_count, data, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, rows)
            finally:
                conn.close()

    def get_target(
        self, db_id: str, target_file: str, target_name: Optional[str] = None
    ) -> Optional[Tuple[List[float], List[float]]]:
        """Retrieves and unpacks a target curve."""
        clean_file = target_file.strip()
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            # 1. Exact match by db_id and file
            cur.execute(
                "SELECT data FROM targets WHERE db_id = ? AND target_file = ?",
                (db_id, clean_file)
            )
            row = cur.fetchone()
            if row and row[0]:
                return self.unpack_curve(row[0])

            # 2. Match by target_name
            if target_name:
                cur.execute(
                    "SELECT data FROM targets WHERE target_name = ? ORDER BY updated_at DESC LIMIT 1",
                    (target_name.strip(),)
                )
                row = cur.fetchone()
                if row and row[0]:
                    return self.unpack_curve(row[0])

            # 3. Global match by target_file across any reviewer
            cur.execute(
                "SELECT data FROM targets WHERE target_file = ? ORDER BY updated_at DESC LIMIT 1",
                (clean_file,)
            )
            row = cur.fetchone()
            if row and row[0]:
                return self.unpack_curve(row[0])
            return None
        finally:
            conn.close()

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
            try:
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO targets (db_id, target_file, target_name, point_count, data, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (db_id, target_file.strip(), target_name.strip(), n, blob, time.time()))
            finally:
                conn.close()

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
            try:
                with conn:
                    conn.executemany("""
                        INSERT OR REPLACE INTO targets (db_id, target_file, target_name, point_count, data, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, rows)
            finally:
                conn.close()

    def get_cached_measurement_keys(self) -> Set[Tuple[str, str]]:
        """Returns set of (db_id, model_file) already present in SQLite."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT db_id, model_file FROM measurements")
            return set(cur.fetchall())
        finally:
            conn.close()

    def get_cached_target_keys(self) -> Set[Tuple[str, str]]:
        """Returns set of (db_id, target_file) already present in SQLite."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT db_id, target_file FROM targets")
            return set(cur.fetchall())
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistical metrics on the offline SQLite library."""
        size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        size_mb = size_bytes / (1024 * 1024)

        conn = self._get_connection()
        try:
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
        finally:
            conn.close()

    def clear_all(self) -> None:
        """Clears all cached measurements and targets and vacuums the database to reclaim disk space."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM measurements")
                cur.execute("DELETE FROM targets")
                conn.commit()
                cur.execute("VACUUM")
                conn.commit()
            finally:
                conn.close()


# Global singleton instance
_GLOBAL_OFFLINE_DB: Optional[OfflineCurveDatabase] = None


def get_offline_db() -> OfflineCurveDatabase:
    global _GLOBAL_OFFLINE_DB
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
    """
    sigProgress = QtCore.pyqtSignal(int, int, str, str)  # current, total, status_text, item_name
    sigPhaseChanged = QtCore.pyqtSignal(str)              # phase name
    sigStatsUpdated = QtCore.pyqtSignal(int, int, float)  # meas_count, target_count, size_mb
    sigFinished = QtCore.pyqtSignal(int, int, int)        # new_saved, total_meas, total_targets
    sigError = QtCore.pyqtSignal(str)

    def __init__(
        self,
        squig_client,
        scope: str = "essential",  # "targets_only", "essential", "full"
        max_workers: int = 24,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self.squig_client = squig_client
        self.scope = scope
        self.max_workers = max_workers
        self.db = get_offline_db()
        self._is_paused = False
        self._abort = False
        self._pause_cond = threading.Condition()

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
            self._abort = False
            self._is_paused = False

            catalog = self.squig_client.get_global_catalog()
            if not catalog:
                self.sigError.emit("Failed to load global database catalog.")
                return

            cached_targets = self.db.get_cached_target_keys()
            cached_models = self.db.get_cached_measurement_keys()

            new_saved = 0

            # -----------------------------------------------------------------
            # PHASE 1: Sync Target Curves across all reviewer databases
            # -----------------------------------------------------------------
            self.sigPhaseChanged.emit("Syncing Target Curves...")
            target_tasks = []
            for r in catalog:
                db_id = r.get("username") or r.get("provider") or "default"
                base_url = r.get("base_url")
                if not base_url or "hangout.audio" in base_url:
                    continue

                for t_name in r.get("targets", []):
                    clean_t = t_name.strip()
                    if not clean_t:
                        continue
                    t_file = f"{clean_t}.txt"
                    if (db_id, t_file) not in cached_targets:
                        target_tasks.append((db_id, base_url, t_name, t_file))

            total_t_tasks = len(target_tasks)
            done_t_tasks = 0

            if target_tasks:
                batch_targets = []
                pool_workers = min(self.max_workers, 20)

                def _worker_fetch_target(task_info):
                    db_i, b_url, t_nm, t_fl = task_info
                    t_obj = SquigTarget(name=t_nm, file_name=t_fl)
                    f_p, s_p = self.squig_client.fetch_target_curve(b_url, t_obj, force_refresh=False)
                    return db_i, t_fl, t_nm, f_p, s_p

                with ThreadPoolExecutor(max_workers=pool_workers) as executor:
                    active_futures = set()
                    task_iter = iter(target_tasks)

                    # Initial continuous saturation (fill worker pipeline)
                    for _ in range(pool_workers * 3):
                        try:
                            t = next(task_iter)
                            active_futures.add(executor.submit(_worker_fetch_target, t))
                        except StopIteration:
                            break

                    while active_futures and not self._abort:
                        with self._pause_cond:
                            while self._is_paused and not self._abort:
                                self._pause_cond.wait()
                        if self._abort:
                            break

                        done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
                        for fut in done:
                            active_futures.remove(fut)
                            try:
                                db_i, t_fl, t_nm, f_p, s_p = fut.result()
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
                                    self.sigProgress.emit(
                                        done_t_tasks, total_t_tasks,
                                        f"Downloaded {done_t_tasks}/{total_t_tasks} targets",
                                        t_nm
                                    )
                            except Exception:
                                done_t_tasks += 1

                            # Replenish pipeline immediately so workers NEVER starve
                            try:
                                next_task = next(task_iter)
                                active_futures.add(executor.submit(_worker_fetch_target, next_task))
                            except StopIteration:
                                pass

                if batch_targets:
                    self.db.save_targets_batch(batch_targets)

            stats = self.db.get_stats()
            self.sigStatsUpdated.emit(stats["measurements"], stats["targets"], stats["size_mb"])

            if self.scope == "targets_only" or self._abort:
                self.sigFinished.emit(new_saved, stats["measurements"], stats["targets"])
                return

            # -----------------------------------------------------------------
            # PHASE 2 & 3: Sync Measurement Curves
            # -----------------------------------------------------------------
            self.sigPhaseChanged.emit("Syncing Measurement Curves...")

            # Filter reviewer databases according to scope
            candidate_reviewers = []
            for r in catalog:
                p_name = (r.get("provider") or "").lower()
                u_name = (r.get("username") or "").lower()
                base_url = r.get("base_url") or ""
                if not base_url or "hangout.audio" in base_url:
                    continue

                if self.scope == "essential":
                    is_essential = any(k in p_name or k in u_name or k in base_url.lower() for k in ESSENTIAL_DB_KEYWORDS)
                    if is_essential:
                        candidate_reviewers.append(r)
                else:
                    # Full catalog
                    candidate_reviewers.append(r)

            meas_tasks = []
            for r in candidate_reviewers:
                db_id = r.get("username") or r.get("provider") or "default"
                base_url = r.get("base_url")
                for p in r.get("phones", []):
                    fn = p.get("file")
                    if isinstance(fn, list):
                        fn = fn[0] if fn else ""
                    if not fn:
                        continue
                    clean_fn = fn.strip()
                    if (db_id, clean_fn) not in cached_models:
                        model_name = p.get("name") or clean_fn
                        meas_tasks.append((db_id, base_url, clean_fn, model_name))

            total_m_tasks = len(meas_tasks)
            done_m_tasks = 0

            batch_meas = []
            meas_pool_workers = min(self.max_workers, 24)

            # Concurrent multi-threaded downloads with streaming sliding window
            def _worker_fetch(task_info):
                db_i, b_url, c_fn, m_nm = task_info
                m_obj = SquigModel(brand="", name=m_nm, full_name=m_nm, file_name=c_fn, display_name=m_nm)
                f_p, s_p = self.squig_client.fetch_iem_measurement(b_url, m_obj, channel_mode="avg", force_refresh=False)
                return db_i, c_fn, m_nm, f_p, s_p

            with ThreadPoolExecutor(max_workers=meas_pool_workers) as executor:
                active_futures = set()
                task_iter = iter(meas_tasks)

                # Prime pipeline with saturated in-flight futures
                for _ in range(meas_pool_workers * 3):
                    try:
                        t = next(task_iter)
                        active_futures.add(executor.submit(_worker_fetch, t))
                    except StopIteration:
                        break

                while active_futures and not self._abort:
                    with self._pause_cond:
                        while self._is_paused and not self._abort:
                            self._pause_cond.wait()
                    if self._abort:
                        break

                    done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        active_futures.remove(fut)
                        try:
                            db_i, c_fn, m_nm, f_p, s_p = fut.result()
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
                                self.sigProgress.emit(
                                    done_m_tasks, total_m_tasks,
                                    f"Synced {done_m_tasks}/{total_m_tasks} curves ({stats['size_mb']} MB)",
                                    m_nm
                                )
                        except Exception:
                            done_m_tasks += 1

                        # Replenish pipeline immediately so 24 workers stay continuously saturated
                        try:
                            next_task = next(task_iter)
                            active_futures.add(executor.submit(_worker_fetch, next_task))
                        except StopIteration:
                            pass

            if batch_meas:
                self.db.save_measurements_batch(batch_meas)

            stats = self.db.get_stats()
            self.sigStatsUpdated.emit(stats["measurements"], stats["targets"], stats["size_mb"])
            self.sigFinished.emit(new_saved, stats["measurements"], stats["targets"])

        except Exception as e:
            self.sigError.emit(str(e))

    def _fetch_target_online(self, base_url: str, target_name: str) -> Tuple[List[float], List[float]]:
        """Direct online fetch helper for target curves."""
        candidates = [
            f"targets/{target_name}.txt",
            f"targets/{target_name}.csv",
            f"targets/{target_name}",
            f"{target_name}.txt",
        ]
        for c in candidates:
            res = self.squig_client.fetch_curve_file(base_url, c, force_refresh=False)
            if res and res != "__FORBIDDEN__":
                f, s = self.squig_client.parse_spl_text(res)
                if f and s:
                    return f, s
        return [], []

    def _fetch_meas_online(self, base_url: str, file_name: str) -> Tuple[List[float], List[float]]:
        """Direct online fetch helper for model measurement curves."""
        candidates = [
            f"data/{file_name}.txt",
            f"data/{file_name} 711.txt",
            f"data/{file_name} L1.txt",
            f"data/{file_name} L.txt",
            f"data_hp/{file_name}.txt",
        ]
        for c in candidates:
            res = self.squig_client.fetch_curve_file(base_url, c, force_refresh=False)
            if res and res != "__FORBIDDEN__":
                f, s = self.squig_client.parse_spl_text(res)
                if f and s:
                    return f, s
        return [], []
