"""
Squiglink API Client & Database Cache Engine.
Fetches reviewer database indices, measurement curves, and compensation targets
with local disk caching for full offline capability. Enforces strict rig isolation.
"""

from __future__ import annotations
import sys
import os
import re
import json
import math
import time
import shutil
import tempfile
import threading
import weakref
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Sequence, Set
import numpy as np
from offline_db import get_offline_db

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Default primary reviewer databases (pre-configured with coupler / rig details)
DEFAULT_DATABASES: List[Dict[str, str]] = [
    # Top Headphone Databases
    {
        "id": "daemonxar",
        "name": "DaemonXar (Headphones)",
        "base_url": "https://daemonxar.squig.link/",
        "rig": "miniDSP EARS Pro",
        "category": "Headphones",
    },
    {
        "id": "gadgetrytech_hp",
        "name": "Gadgetry Tech (Headphones)",
        "base_url": "https://gadgetrytech.squig.link/headsets/",
        "rig": "KB006x Pinna Clone",
        "category": "Headphones",
    },
    {
        "id": "superreview_hp",
        "name": "Super* Review (Headphones)",
        "base_url": "https://squig.link/headphones/",
        "rig": "clone IEC 711 + KB006x",
        "category": "Headphones",
    },
    {
        "id": "kuulokenurkka",
        "name": "Kuulokenurkka (Headphones)",
        "base_url": "https://kuulokenurkka.squig.link/",
        "rig": "IEC 60318-4 (KB50xx)",
        "category": "Headphones",
    },
    {
        "id": "audioamigo",
        "name": "Audio Amigo (Headphones)",
        "base_url": "https://audioamigo.squig.link/headphones/",
        "rig": "IEC 60318-4 (711)",
        "category": "Headphones",
    },
    {
        "id": "sidesalad_hp",
        "name": "Side Salad Audio (Headphones)",
        "base_url": "https://suporsalad.squig.link/headphones/",
        "rig": "IEC 60318-4 (711)",
        "category": "Headphones",
    },
    {
        "id": "listener_hp",
        "name": "listener (Headphones)",
        "base_url": "https://listener.squig.link/",
        "rig": "IEC 60318-4 (711)",
        "category": "Headphones",
    },
    {
        "id": "crinacle_hp",
        "name": "Hangout.Audio (Headphones)",
        "base_url": "https://graph.hangout.audio/headphones/",
        "rig": "GRAS 43AG",
        "category": "Headphones",
    },
    # Top IEM Databases
    {
        "id": "peqdb",
        "name": "PEQdB (Sharur / Diamond)",
        "base_url": "https://peqdb.com/",
        "rig": "IEC 60318-4 / 5128 / KB50XX",
        "category": "All-in-One",
    },
    {
        "id": "jaraba",
        "name": "JaRaBa (PEQdB Lab)",
        "base_url": "https://squig.link/lab/jaraba/",
        "rig": "IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "jacstone",
        "name": "Jacstone (Sharur)",
        "base_url": "https://jacstone.squig.link/",
        "rig": "IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "aden",
        "name": "aden (IEMs)",
        "base_url": "https://aden.squig.link/",
        "rig": "clone 711 coupler",
        "category": "IEMs",
    },
    {
        "id": "superreview",
        "name": "Super* Review (IEMs)",
        "base_url": "https://squig.link/",
        "rig": "clone IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "hbb",
        "name": "HBB / Bad Guy (IEMs)",
        "base_url": "https://hbb.squig.link/",
        "rig": "clone IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "timmyv",
        "name": "Timmy / TimTuned (IEMs)",
        "base_url": "https://timmyv.squig.link/",
        "rig": "clone IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "precog",
        "name": "Precogvision (IEMs)",
        "base_url": "https://precog.squig.link/",
        "rig": "clone IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "tonedeafmonk",
        "name": "ToneDeafMonk (IEMs)",
        "base_url": "https://tonedeafmonk.squig.link/",
        "rig": "clone IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "jaysaudio",
        "name": "Jay's Audio Lab (IEMs)",
        "base_url": "https://jaysaudio.squig.link/",
        "rig": "clone IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "akros",
        "name": "Akros (IEMs)",
        "base_url": "https://akros.squig.link/",
        "rig": "clone IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "eliseaudio",
        "name": "Elise Audio (IEMs)",
        "base_url": "https://eliseaudio.squig.link/",
        "rig": "clone IEC 60318-4 (711)",
        "category": "IEMs",
    },
    {
        "id": "crinacle_711",
        "name": "Hangout.Audio (711 IEMs)",
        "base_url": "https://graph.hangout.audio/iem/711/",
        "rig": "clone IEC 711",
        "category": "IEMs",
    },
    {
        "id": "crinacle_5128",
        "name": "Hangout.Audio (5128 IEMs)",
        "base_url": "https://graph.hangout.audio/iem/5128/",
        "rig": "B&K 5128",
        "category": "IEMs",
    },
]


@dataclass
class GlobalSearchResult:
    brand: str
    name: str
    full_name: str
    file_name: str
    provider_name: str
    db_type: str
    base_url: str
    rig: str
    targets: List[str]
    rig_badge: str = ""
    is_offline_cached: bool = False


@dataclass
class SquigModel:
    brand: str
    name: str
    full_name: str
    file_name: str
    display_name: str
    variant: str = ""
    review_score: str = ""
    price: str = ""
    base_url: Optional[str] = None
    provider_name: Optional[str] = None
    rig: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brand": self.brand,
            "name": self.name,
            "full_name": self.full_name,
            "file_name": self.file_name,
            "display_name": self.display_name,
            "variant": self.variant,
            "review_score": self.review_score,
            "price": self.price,
            "base_url": self.base_url,
            "provider_name": self.provider_name,
            "rig": self.rig,
        }


@dataclass
class SquigTarget:
    name: str
    file_name: str
    category: str = "Reference"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_name": self.file_name,
            "category": self.category,
        }


# -----------------------------------------------------------------------------
# Shared URL Normalization, Rig Classification & Canonical DB ID Helpers
# -----------------------------------------------------------------------------
_SHARED_CATALOG_LOCK = threading.RLock()
_SHARED_CATALOG_DATA: Optional[List[Dict[str, Any]]] = None
_SHARED_DB_ID_MAP: Optional[Dict[str, str]] = None


def _normalize_base_url(base_url: str) -> str:
    """Normalizes a database base_url for canonical comparisons (lowercase host, no trailing slash)."""
    raw = (base_url or "").strip()
    if not raw:
        return ""
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"
    return raw.rstrip("/").lower()


def _safe_db_key_from_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse((base_url or "").strip())
    netloc = parsed.netloc.lower().replace(":", "_")
    path = parsed.path.strip("/").replace("/", "_")
    key = f"{netloc}_{path}" if path else netloc
    return re.sub(r"[^a-zA-Z0-9_\.-]", "_", key).strip("_") or "default"


def _classify_rig_family(rig_desc: str = "", base_url: str = "") -> str:
    """Classifies a measurement coupler/rig into an isolation family to prevent cross-rig mixing."""
    r = (rig_desc or "").lower()
    u = (base_url or "").lower()
    if "peqdb.com" in u:
        return "peqdb"
    if "5128" in r or "/5128" in u:
        return "5128"
    if "43ag" in r or "45ca" in r or "gras" in r:
        return "gras"
    if "ears" in r or "daemonxar" in u:
        return "ears"
    if "kb006" in r or "kb50" in r or "pinna" in r or "/headphone" in u or "/headset" in u:
        return "pinna_711"
    if "711" in r or "60318" in r or "coupler" in r:
        return "711"
    return "unknown"


def get_rig_badge(rig_desc: str, base_url: str = "", category: str = "") -> str:
    """
    Returns a crisp short coupler/rig badge label:
    '5128', '711', '43AG', '45CA', 'KB006x', 'EARS', 'HMS', 'KEMAR', or ''.
    """
    r = (rig_desc or "").strip().lower()
    u = (base_url or "").strip().lower()
    c = (category or "").strip().lower()

    # Multi-rig aggregators (e.g., PEQdB) combine multiple couplers; avoid mislabeling as a single rig
    if "peqdb.com" in u or "multi-rig" in r or ("5128" in r and ("60318" in r or "711" in r)):
        return ""

    if "5128" in r or "/5128" in u or c == "5128":
        return "5128"
    if "43ag" in r or "43ag" in u:
        return "43AG"
    if "45ca" in r or "45ca" in u:
        return "45CA"
    if "kemar" in r or "kemar" in u:
        return "KEMAR"
    if re.search(r"\bhms(?:\.\d+|\d*)\b", r) or "head acoustics" in r:
        return "HMS"
    if re.search(r"\bears\b", r) or "minidsp" in r or "daemonxar" in u:
        return "EARS"
    if "kb006" in r or "kb006" in u:
        return "KB006x"
    if (
        "711" in r
        or "60318" in r
        or "coupler" in r
        or "occluded-ear" in r
        or "kb50" in r
        or "/711" in u
    ):
        return "711"
    return ""


def _atomic_write_text(path: str, content: str) -> None:
    """Atomically writes UTF-8 text to disk via a sibling temp file and os.replace."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def _load_shared_catalog(extra_cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    global _SHARED_CATALOG_DATA
    with _SHARED_CATALOG_LOCK:
        if _SHARED_CATALOG_DATA is not None:
            return _SHARED_CATALOG_DATA

        candidate_paths = [
            os.path.join(getattr(sys, "_MEIPASS", ""), "squig_catalog.json"),
            os.path.abspath("squig_catalog.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "squig_catalog.json"),
        ]
        if extra_cache_dir:
            candidate_paths.insert(2, os.path.join(extra_cache_dir, "squig_catalog.json"))

        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            _SHARED_CATALOG_DATA = data
                            return _SHARED_CATALOG_DATA
                except Exception:
                    pass

        _SHARED_CATALOG_DATA = []
        return _SHARED_CATALOG_DATA


def _build_canonical_db_id_map(catalog: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Builds a collision-free canonical map from normalized base_url -> db_id.
    Preserves all DEFAULT_DATABASES IDs for backward compatibility with existing presets.json,
    while guaranteeing every multi-rig catalog URL receives a distinct db_id.
    """
    url_to_id: Dict[str, str] = {}
    used_ids: Set[str] = set()

    # 1. Pin DEFAULT_DATABASES IDs first
    for d in DEFAULT_DATABASES:
        norm_u = _normalize_base_url(d.get("base_url", ""))
        d_id = (d.get("id") or "").strip()
        if norm_u and d_id:
            url_to_id[norm_u] = d_id
            used_ids.add(d_id.lower())

    # 2. Count username occurrences across catalog to detect multi-rig reviewers
    username_counts: Dict[str, int] = {}
    for c in catalog:
        raw_u = (c.get("username") or c.get("provider") or "").strip().lower()
        if raw_u:
            username_counts[raw_u] = username_counts.get(raw_u, 0) + 1

    # 3. Assign unique IDs for all remaining catalog URLs
    for c in catalog:
        raw_url = c.get("base_url", "")
        norm_u = _normalize_base_url(raw_url)
        if not norm_u or norm_u in url_to_id:
            continue

        base_name = (c.get("username") or c.get("provider") or "").strip()
        base_slug = re.sub(r"[^a-zA-Z0-9_]", "_", base_name).strip("_")
        raw_key = base_name.lower()

        # Use plain username ONLY if unique across catalog and unused by DEFAULT_DATABASES
        if base_slug and username_counts.get(raw_key, 0) == 1 and base_slug.lower() not in used_ids:
            cand = base_slug
        else:
            parsed = urllib.parse.urlparse(raw_url)
            path_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", parsed.path.strip("/")).strip("_")
            if base_slug and path_slug:
                cand = f"{base_slug}_{path_slug}"
            elif base_slug:
                db_type_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(c.get("db_type", "")).strip().lower()).strip("_")
                cand = f"{base_slug}_{db_type_slug}" if db_type_slug else f"{base_slug}_root"
            else:
                cand = _safe_db_key_from_url(raw_url)

            # Final collision guard
            if cand.lower() in used_ids:
                cand = _safe_db_key_from_url(raw_url)
            suffix = 2
            orig_cand = cand
            while cand.lower() in used_ids:
                cand = f"{orig_cand}_{suffix}"
                suffix += 1

        url_to_id[norm_u] = cand
        used_ids.add(cand.lower())

    return url_to_id


def get_db_id(base_url: str, catalog: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Returns a canonical, collision-free database ID for a Squiglink/AutoEQ base URL.
    Preserves DEFAULT_DATABASES IDs and isolates multi-rig reviewers cleanly.
    """
    global _SHARED_DB_ID_MAP
    norm_u = _normalize_base_url(base_url)
    if not norm_u:
        return "default"

    with _SHARED_CATALOG_LOCK:
        if catalog is not None:
            id_map = _build_canonical_db_id_map(catalog)
        else:
            if _SHARED_DB_ID_MAP is None:
                _SHARED_DB_ID_MAP = _build_canonical_db_id_map(_load_shared_catalog())
            id_map = _SHARED_DB_ID_MAP

        if norm_u in id_map:
            return id_map[norm_u]

    return _safe_db_key_from_url(base_url)


class SquigClient:
    """
    Manages HTTP fetching and persistent local disk caching for Squiglink databases.
    Enforces coupler isolation: models and targets are strictly bound to the active database.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        self.offline_db = get_offline_db()
        if cache_dir:
            self.cache_dir = os.path.abspath(cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            primary_cache = os.path.join(os.path.expanduser("~"), ".cache", "clean_apo", "squiglink")
            try:
                os.makedirs(primary_cache, exist_ok=True)
                self.cache_dir = primary_cache
            except Exception:
                fallback_cache = os.path.join(tempfile.gettempdir(), "clean_apo_squiglink")
                os.makedirs(fallback_cache, exist_ok=True)
                self.cache_dir = fallback_cache

        self._user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CleanAPO/1.0"
        self._lock = threading.RLock()
        self._thread_local = threading.local()
        self._in_memory_indices: Dict[str, Tuple[List[SquigModel], List[SquigTarget], str]] = {}
        self._global_catalog: Optional[List[Dict[str, Any]]] = None
        self._db_id_cache: Dict[str, str] = {}

        # Backward-compat attribute for callers inspecting `.session`
        self.session = self._get_session() if HAS_REQUESTS else None

        # Clear in-memory and disk JSON caches whenever offline SQLite DB is wiped
        self_ref = weakref.ref(self)

        def _on_clear() -> None:
            client = self_ref()
            if client is not None:
                client._on_offline_db_cleared()

        if hasattr(self.offline_db, "register_clear_callback"):
            self.offline_db.register_clear_callback(_on_clear)

    def _get_session(self):
        """Returns a thread-local requests.Session so 24+ worker threads never race on shared state."""
        if not HAS_REQUESTS:
            return None
        sess = getattr(self._thread_local, "session", None)
        if sess is None:
            sess = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=24, max_retries=1)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            self._thread_local.session = sess
        return sess

    def _on_offline_db_cleared(self) -> None:
        """Clears cached measurement JSON files and in-memory indices when offline DB is deleted."""
        with self._lock:
            self._in_memory_indices.clear()
        try:
            if os.path.isdir(self.cache_dir):
                for entry in os.listdir(self.cache_dir):
                    sub = os.path.join(self.cache_dir, entry)
                    if os.path.isdir(sub):
                        shutil.rmtree(sub, ignore_errors=True)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Network / Cache Low-Level Utilities
    # -------------------------------------------------------------------------
    def _safe_db_key(self, base_url: str) -> str:
        return _safe_db_key_from_url(base_url)

    def _get_db_cache_dir(self, base_url: str) -> str:
        key = self._safe_db_key(base_url)
        path = os.path.join(self.cache_dir, key)
        os.makedirs(path, exist_ok=True)
        return path

    def get_db_id(self, base_url: str) -> str:
        """Returns the canonical, collision-free database ID for base_url."""
        norm_url = _normalize_base_url(base_url)
        if not norm_url:
            return "default"
        with self._lock:
            cached = self._db_id_cache.get(norm_url)
            if cached is not None:
                return cached
        db_id = get_db_id(base_url, catalog=self.get_global_catalog())
        with self._lock:
            self._db_id_cache[norm_url] = db_id
        return db_id

    def _get_db_id(self, base_url: str) -> str:
        return self.get_db_id(base_url)

    def _get_known_rig(self, base_url: str) -> Optional[str]:
        """Looks up known rig description for base_url from DEFAULT_DATABASES or catalog."""
        norm_u = _normalize_base_url(base_url)
        for d in DEFAULT_DATABASES:
            if _normalize_base_url(d.get("base_url", "")) == norm_u and d.get("rig"):
                return str(d["rig"]).strip()
        for c in self.get_global_catalog():
            if _normalize_base_url(c.get("base_url", "")) == norm_u and c.get("rig"):
                return str(c["rig"]).strip()
        return None

    def _http_get_text(self, url: str, timeout: int = 8) -> Tuple[int, str]:
        """Performs HTTP GET with a thread-local requests session (or urllib fallback)."""
        if HAS_REQUESTS:
            sess = self._get_session()
            if sess is not None:
                try:
                    resp = sess.get(
                        url,
                        headers={"User-Agent": self._user_agent},
                        timeout=timeout,
                    )
                    resp.encoding = "utf-8"
                    return resp.status_code, resp.text
                except Exception as e:
                    return 0, str(e)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = resp.getcode()
                content = resp.read().decode("utf-8", errors="replace")
                return code, content
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(e)
            return int(e.code), body
        except Exception as e:
            return 0, str(e)

    # -------------------------------------------------------------------------
    # SPL Parsing
    # -------------------------------------------------------------------------
    @staticmethod
    def _looks_like_html(text: str) -> bool:
        head = (text or "").lstrip("\ufeff \t\r\n")[:256].lower()
        return head.startswith(("<!doctype", "<html", "<head", "<body", "<script", "<?xml"))

    @staticmethod
    def parse_spl_text(text: str) -> Tuple[List[float], List[float]]:
        """
        Parses standard two- or three-column space/tab-separated frequency response files (.txt/.csv).
        Strips UTF-8 BOM, rejects NaN/Inf/HTML, and ensures monotonically ascending frequency order.
        """
        if not text or SquigClient._looks_like_html(text):
            return [], []
        clean_text = text.lstrip("\ufeff")
        freqs: List[float] = []
        spls: List[float] = []
        for line in clean_text.splitlines():
            line = line.strip()
            if not line or line.startswith(("*", "#", ";", "//")):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    s = float(parts[1])
                    if math.isfinite(f) and math.isfinite(s) and f > 0.0:
                        freqs.append(f)
                        spls.append(s)
                except ValueError:
                    continue

        if len(freqs) > 1 and any(freqs[i] > freqs[i + 1] for i in range(len(freqs) - 1)):
            paired = sorted(zip(freqs, spls), key=lambda kv: kv[0])
            freqs = [kv[0] for kv in paired]
            spls = [kv[1] for kv in paired]
        return freqs, spls

    # -------------------------------------------------------------------------
    # Database Discovery (All Squiglink Sites)
    # -------------------------------------------------------------------------
    def fetch_all_squig_sites(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetches the master directory of all ~125 Squiglink reviewer databases from squigsites.json.
        Caches locally on disk.
        """
        cache_file = os.path.join(self.cache_dir, "squigsites.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        code, text = self._http_get_text("https://squig.link/squigsites.json", timeout=6)
        if code == 200:
            try:
                data = json.loads(text)
                _atomic_write_text(cache_file, text)
                return data
            except Exception:
                pass

        return []

    # -------------------------------------------------------------------------
    # Database Index Fetching (Phonebook & Targets)
    # -------------------------------------------------------------------------
    def fetch_database_index(
        self,
        base_url: str,
        force_refresh: bool = False,
    ) -> Tuple[List[SquigModel], List[SquigTarget], str]:
        """
        Fetches and parses models, targets, and coupler rig info for a Squiglink database URL.
        Returns: (models_list, targets_list, rig_description)
        """
        if not base_url.endswith("/"):
            base_url += "/"

        # 0. Check in-memory cache first (instant 0 ms return)
        with self._lock:
            if not force_refresh and base_url in self._in_memory_indices:
                return self._in_memory_indices[base_url]

        db_dir = self._get_db_cache_dir(base_url)
        meta_file = os.path.join(db_dir, "_index_meta.json")

        # Load from disk cache if fresh and not forced
        if not force_refresh and os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    models = [SquigModel(**m) for m in data.get("models", [])]
                    targets = [SquigTarget(**t) for t in data.get("targets", [])]
                    rig = str(data.get("rig", self._get_known_rig(base_url) or "IEC 60318-4 (711)"))
                    if models and targets:
                        with self._lock:
                            self._in_memory_indices[base_url] = (models, targets, rig)
                        return models, targets, rig
            except Exception:
                pass

        # Handle PEQdB API endpoint (short-circuit on error; never probe Squiglink phone_book.json)
        if "peqdb.com" in base_url.lower():
            targets = [
                SquigTarget("PEQdB Diamond", "PEQdB Diamond", "Sharur Diamond"),
                SquigTarget("PEQdB Diamond β", "PEQdB Diamond β", "Sharur Diamond"),
                SquigTarget("PEQdB Ultra", "PEQdB Ultra", "Sharur Diamond"),
                SquigTarget("Sharur", "Sharur", "Sharur Reference"),
                SquigTarget("Harman IE 2019", "Harman IE 2019", "Preference"),
                SquigTarget("Harman OE 2018", "Harman OE 2018", "Preference"),
                SquigTarget("Diffuse Field", "Diffuse Field", "Reference"),
            ]
            rig = "IEC 60318-4 / 5128 / KB50XX (PEQdB Multi-Rig)"
            code, text = self._http_get_text("https://peqdb.com/opt-v3/lists", timeout=10)
            if code == 200 and len(text) > 500:
                try:
                    peq_json = json.loads(text)
                    phones = peq_json.get("phones", {})
                    if isinstance(phones, dict) and phones:
                        models: List[SquigModel] = []
                        for p_name in sorted(phones.keys()):
                            p_clean = str(p_name).strip()
                            if not p_clean:
                                continue
                            parts = p_clean.split(" ", 1)
                            brand = parts[0] if parts else "Unknown"
                            models.append(
                                SquigModel(
                                    brand=brand,
                                    name=p_clean,
                                    full_name=p_clean,
                                    file_name=p_clean,
                                    display_name=p_clean,
                                )
                            )
                        try:
                            _atomic_write_text(
                                meta_file,
                                json.dumps(
                                    {
                                        "models": [m.to_dict() for m in models],
                                        "targets": [t.to_dict() for t in targets],
                                        "rig": rig,
                                    },
                                    indent=2,
                                ),
                            )
                        except Exception:
                            pass
                        peq_map_file = os.path.join(db_dir, "peqdb_phones.json")
                        try:
                            _atomic_write_text(peq_map_file, json.dumps(phones))
                        except Exception:
                            pass
                        with self._lock:
                            self._in_memory_indices[base_url] = (models, targets, rig)
                        return models, targets, rig
                except Exception as e:
                    print(f"[SquigClient] Error loading PEQdB lists: {e}")
            return [], targets, rig

        # 1. Fetch Phonebook (phone_book.json or data/phone_book.json) — validate real JSON list
        pb_candidates = [
            base_url + "phone_book.json",
            base_url + "data/phone_book.json",
            base_url + "data_hp/phone_book.json",
        ]
        pb_json: Optional[List[Any]] = None
        for pb_url in pb_candidates:
            code, text = self._http_get_text(pb_url, timeout=7)
            if code == 200 and len(text) > 20 and not self._looks_like_html(text):
                try:
                    parsed_pb = json.loads(text.lstrip("\ufeff"))
                    if isinstance(parsed_pb, list):
                        pb_json = parsed_pb
                        break
                except Exception:
                    continue

        # 2. Fetch Config (config.js or assets/js/config.js) — reject HTML 404 pages
        cfg_candidates = [
            base_url + "config.js",
            base_url + "assets/js/config.js",
        ]
        cfg_text = ""
        for cfg_url in cfg_candidates:
            code, text = self._http_get_text(cfg_url, timeout=7)
            if code == 200 and len(text) > 40 and not self._looks_like_html(text):
                cfg_text = text
                break

        # Parse rig description from config.js, falling back to known catalog/default rig
        rig_desc = self._get_known_rig(base_url) or "clone IEC 60318-4 (711)"
        if cfg_text:
            rig_match = re.search(r'rig_description\s*=\s*[\'"]([^\'"]+)[\'"]', cfg_text)
            if rig_match and rig_match.group(1).strip():
                rig_desc = rig_match.group(1).strip()

        # Parse models from phonebook
        models = []
        target_names_from_pb: List[str] = []

        if pb_json:
            for brand_obj in pb_json:
                if not isinstance(brand_obj, dict):
                    continue
                brand_name = str(brand_obj.get("name", "")).strip()
                phones = brand_obj.get("phones", [])
                if not isinstance(phones, list):
                    continue

                is_target_cat = "target" in brand_name.lower() or brand_name == "_EQ"
                for p in phones:
                    if not isinstance(p, dict):
                        continue
                    p_name = str(p.get("name", "")).strip()
                    p_file = p.get("file", "")
                    p_score = str(p.get("reviewScore", ""))
                    p_price = str(p.get("price", ""))

                    if is_target_cat:
                        if isinstance(p_file, str) and p_file.strip():
                            target_names_from_pb.append(p_file.strip())
                        elif isinstance(p_file, list):
                            target_names_from_pb.extend([str(x).strip() for x in p_file if x])
                        continue

                    # Handle multi-variant files or single file
                    if isinstance(p_file, list):
                        raw_suffixes = p.get("suffix", [])
                        suffixes = raw_suffixes if isinstance(raw_suffixes, list) else []
                        for idx, f_item in enumerate(p_file):
                            if not f_item:
                                continue
                            raw_var = suffixes[idx] if idx < len(suffixes) and suffixes[idx] is not None else f"Variant {idx + 1}"
                            var_str = str(raw_var).strip()
                            full_n = f"{brand_name} {p_name}".strip()
                            disp = f"{full_n} {var_str}".strip() if var_str else full_n
                            models.append(
                                SquigModel(
                                    brand=brand_name,
                                    name=p_name,
                                    full_name=full_n,
                                    file_name=str(f_item).strip(),
                                    display_name=disp,
                                    variant=var_str,
                                    review_score=p_score,
                                    price=p_price,
                                )
                            )
                    elif isinstance(p_file, str) and p_file.strip():
                        full_n = f"{brand_name} {p_name}".strip()
                        models.append(
                            SquigModel(
                                brand=brand_name,
                                name=p_name,
                                full_name=full_n,
                                file_name=p_file.strip(),
                                display_name=full_n,
                                review_score=p_score,
                                price=p_price,
                            )
                        )

        # Parse targets from config.js
        targets: List[SquigTarget] = []
        seen_targets: Set[str] = set()

        def _add_target(raw_name: str, raw_file: Optional[str], category: str) -> None:
            clean_n = re.sub(r"\s+target$", "", (raw_name or "").strip(), flags=re.IGNORECASE).strip()
            clean_n = re.sub(r"\.(?:txt|csv)$", "", clean_n, flags=re.IGNORECASE).strip()
            if not clean_n:
                return
            norm_key = clean_n.lower()
            if norm_key in seen_targets:
                return
            seen_targets.add(norm_key)
            f_str = (raw_file or "").strip()
            f_str = re.sub(r"\.(?:txt|csv)$", "", f_str, flags=re.IGNORECASE).strip()
            if not f_str:
                f_str = clean_n if clean_n.lower().endswith("target") else f"{clean_n} Target"
            targets.append(
                SquigTarget(
                    name=clean_n,
                    file_name=f_str,
                    category=category,
                )
            )

        if cfg_text:
            # Strip JS block comments /* ... */ and line comments // ...
            no_block = re.sub(r"/\*[\s\S]*?\*/", "", cfg_text)
            clean_lines = []
            for line in no_block.splitlines():
                l_str = line.strip()
                if l_str.startswith("//"):
                    continue
                c_idx = re.search(r"(?<!:)\/\/", l_str)
                if c_idx:
                    l_str = l_str[:c_idx.start()].strip()
                clean_lines.append(l_str)
            clean_js = "\n".join(clean_lines)

            # Look for targets = [ ... ] (trailing semicolon optional)
            m = re.search(r"(?:(?:const|var|let)\s+)?targets\s*=\s*(\[(?:[^\[\]]|\[[^\]]*\])*\])\s*;?", clean_js)
            if m:
                block = m.group(1)
                obj_blocks = re.findall(r"\{([^{}]*)\}", block)
                for obj_body in obj_blocks:
                    type_m = re.search(r"""['"]?type['"]?\s*:\s*['"]([^'"]+)['"]""", obj_body)
                    files_m = re.search(r"""['"]?files['"]?\s*:\s*\[([\s\S]*?)\]""", obj_body)
                    if not files_m:
                        continue
                    cat_type = type_m.group(1).strip() if type_m else "Reference"
                    matches = re.findall(r"['\"]([^'\"]+)['\"]", files_m.group(1))
                    for item in matches:
                        clean_item = item.strip()
                        if clean_item:
                            file_str = clean_item if clean_item.lower().endswith("target") else f"{clean_item} Target"
                            _add_target(clean_item, file_str, cat_type)

            # If targets array wasn't found or was empty, check init_phones
            if not targets:
                init_m = re.search(r"init_phones\s*=\s*\[([\s\S]*?)\]", clean_js)
                if init_m:
                    matches = re.findall(r"['\"]([^'\"]+)['\"]", init_m.group(1))
                    for item in matches:
                        clean_item = item.strip()
                        if "target" in clean_item.lower():
                            _add_target(clean_item, clean_item, "Reference")

        # Add any targets found in phonebook
        for t_name in target_names_from_pb:
            clean_t = t_name.strip()
            if clean_t:
                file_str = clean_t if clean_t.lower().endswith("target") else f"{clean_t} Target"
                _add_target(clean_t, file_str, "Database Target")

        # Fallback to catalog targets if targets array could not be parsed from config.js
        if not targets:
            norm_url = _normalize_base_url(base_url)
            cat = self.get_global_catalog()
            for c in cat:
                if _normalize_base_url(c.get("base_url", "")) == norm_url:
                    for t_name in c.get("targets", []):
                        clean_t = str(t_name).strip()
                        if clean_t:
                            file_str = clean_t if clean_t.lower().endswith("target") else f"{clean_t} Target"
                            _add_target(clean_t, file_str, "Reference")
                    break

        # Fallback to offline_db targets (via OfflineCurveDatabase helper — no leaked connections)
        if not targets:
            db_id = self.get_db_id(base_url)
            try:
                for tf, tn in self.offline_db.get_targets_for_db(db_id):
                    _add_target(tn or tf or "", tf or None, "Offline Reference")
            except Exception:
                pass

        # Final fallback to standard targets matching the coupler rig
        if not targets:
            is_hp = "headphone" in base_url.lower() or "ears" in rig_desc.lower() or "kb006" in rig_desc.lower()
            if is_hp:
                std = ["Harman OE 2018", "Diffuse Field", "Free Field", "KEMAR DF"]
            else:
                std = ["Harman IE 2019", "IEF Neutral 2023", "Diffuse Field", "AutoEQ"]
            for s in std:
                _add_target(s, f"{s} Target", "Standard Reference")

        # Only persist & cache in-memory if models were actually retrieved (avoid poisoning session on outage)
        if models:
            try:
                _atomic_write_text(
                    meta_file,
                    json.dumps(
                        {
                            "models": [m.to_dict() for m in models],
                            "targets": [t.to_dict() for t in targets],
                            "rig": rig_desc,
                        },
                        indent=2,
                    ),
                )
            except Exception:
                pass
            with self._lock:
                self._in_memory_indices[base_url] = (models, targets, rig_desc)

        return models, targets, rig_desc

    # -------------------------------------------------------------------------
    # Global Catalog Search (43k+ models across 148 Squiglink databases)
    # -------------------------------------------------------------------------
    def get_global_catalog(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._global_catalog is not None:
                return self._global_catalog

        loaded = _load_shared_catalog(extra_cache_dir=self.cache_dir)
        with self._lock:
            self._global_catalog = loaded
            return self._global_catalog

    @staticmethod
    def _matches_category(category: Optional[str], db_type: str, provider_name: str, base_url: str, rig: str) -> bool:
        if not category:
            return True
        cat_low = category.strip().lower()
        if not cat_low or cat_low in ("all", "all categories", "all types"):
            return True

        dt_low = (db_type or "").strip().lower()
        p_low = (provider_name or "").strip().lower()
        u_low = (base_url or "").strip().lower()
        r_low = (rig or "").strip().lower()

        is_5128_hp = (
            dt_low == "5128"
            and (
                "headphone" in p_low
                or "headphone" in u_low
                or "headset" in u_low
                or "listener.squig.link/5128" in u_low
            )
        )

        if cat_low in ("headphones", "headphone", "over-ear", "hp"):
            return dt_low == "headphones" or is_5128_hp
        if cat_low in ("iems", "iem", "in-ear"):
            return dt_low in ("iems", "earbuds") or (dt_low == "5128" and not is_5128_hp)
        return cat_low in dt_low or cat_low in r_low

    def search_global_catalog(
        self,
        query: str,
        target_query: str = "",
        limit: int = 150,
        category: Optional[str] = None,
        favorite_providers: Optional[Sequence[str]] = None,
        cached_keys: Optional[Set[Tuple[str, str]]] = None,
    ) -> List[GlobalSearchResult]:
        catalog = self.get_global_catalog()
        if not catalog:
            return []

        tokens = query.strip().lower().split()
        target_tokens = [t.strip().lower() for t in re.split(r'[,;]+', target_query) if t.strip()]

        if not tokens and not target_tokens:
            return []

        fav_set: Set[str] = set()
        if favorite_providers:
            for fp in favorite_providers:
                fp_s = str(fp or "").strip().lower()
                if fp_s:
                    fav_set.add(fp_s)

        cached_set: Set[Tuple[str, str]]
        if cached_keys is None:
            cached_set = set()
        elif isinstance(cached_keys, (set, frozenset)):
            cached_set = cached_keys  # type: ignore[assignment]
        else:
            cached_set = set(cached_keys)

        ranked_results: List[Tuple[Tuple[int, int, int, str, str, str], GlobalSearchResult]] = []
        seen: Set[Tuple[str, str]] = set()
        query_norm = " ".join(tokens)

        for db in catalog:
            p_name = db.get("provider", "Reviewer")
            db_type = db.get("db_type", "IEMs")
            base_url = db.get("base_url", "")
            rig = db.get("rig", "IEC 60318-4 (711)")
            targets = db.get("targets", [])

            if not self._matches_category(category, db_type, p_name, base_url, rig):
                continue

            # Target filter: if target_tokens specified, provider must provide at least one target matching any token
            if target_tokens:
                targets_lower = [str(t).lower() for t in targets]
                has_target_match = False
                for tt in target_tokens:
                    if any(tt in tl for tl in targets_lower):
                        has_target_match = True
                        break
                if not has_target_match:
                    continue

            db_id = self.get_db_id(base_url)
            rig_badge = get_rig_badge(rig, base_url, db_type)
            provider_disp = f"{p_name} ({db_type})" if db_type else p_name
            provider_disp_low = provider_disp.strip().lower()

            is_fav = False
            if fav_set:
                db_id_low = db_id.lower()
                u_low = str(db.get("username", "") or "").strip().lower()
                p_low = str(p_name or "").strip().lower()
                norm_url_low = _normalize_base_url(base_url)
                if (
                    db_id_low in fav_set
                    or u_low in fav_set
                    or p_low in fav_set
                    or provider_disp_low in fav_set
                    or norm_url_low in fav_set
                ):
                    is_fav = True

            for p in db.get("phones", []):
                full_name = str(p.get("full_name", "") or "")
                full_lower = full_name.lower().strip()
                if not tokens or all(t in full_lower for t in tokens):
                    raw_file = p.get("file", full_name)
                    if isinstance(raw_file, list):
                        file_str = str(raw_file[0]) if raw_file else full_name
                    else:
                        file_str = str(raw_file or full_name)
                    file_clean = file_str.strip()
                    file_key = (base_url, file_clean)
                    if file_key not in seen:
                        seen.add(file_key)
                        model_name = str(p.get("name", "") or "")
                        name_lower = model_name.lower().strip()
                        is_cached = bool(cached_set) and (
                            (db_id, file_clean) in cached_set or (db_id, file_str) in cached_set
                        )

                        # Strict match tier:
                        # 0 = exact model match, 1 = prefix match, 2 = exact token match, 3 = substring match
                        if not query_norm:
                            match_tier = 0
                        elif full_lower == query_norm or name_lower == query_norm:
                            match_tier = 0
                        elif full_lower.startswith(query_norm) or name_lower.startswith(query_norm):
                            match_tier = 1
                        else:
                            word_tokens = set(full_lower.split()) | set(re.findall(r"[a-z0-9]+", full_lower))
                            if all(t in word_tokens for t in tokens):
                                match_tier = 2
                            else:
                                match_tier = 3

                        item = GlobalSearchResult(
                            brand=str(p.get("brand", "") or ""),
                            name=model_name,
                            full_name=full_name,
                            file_name=file_str,
                            provider_name=provider_disp,
                            db_type=db_type,
                            base_url=base_url,
                            rig=rig,
                            targets=targets,
                            rig_badge=rig_badge,
                            is_offline_cached=is_cached,
                        )
                        sort_key = (
                            match_tier,
                            0 if is_fav else 1,
                            0 if is_cached else 1,
                            full_lower,
                            provider_disp_low,
                            file_clean.lower(),
                        )
                        ranked_results.append((sort_key, item))

        ranked_results.sort(key=lambda pair: pair[0])
        ordered = [pair[1] for pair in ranked_results]
        return ordered[:limit] if limit and limit > 0 else ordered

    # -------------------------------------------------------------------------
    # Measurement & Target Curve Fetching
    # -------------------------------------------------------------------------
    def fetch_curve_file(
        self,
        base_url: str,
        file_path_relative: str,
        force_refresh: bool = False,
    ) -> Optional[str]:
        """
        Fetches a raw curve text file (.txt/.csv) from the database with local disk caching.
        Validates SPL content to prevent caching HTTP 200 HTML error / Cloudflare pages.
        """
        if not base_url.endswith("/"):
            base_url += "/"

        rel_clean = (file_path_relative or "").lstrip("/")
        if not rel_clean:
            return None

        db_dir = self._get_db_cache_dir(base_url)
        # Preserve relative subdirectory in cache key so data/ vs data/targets/ vs data_hp/ never collide
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", rel_clean.replace("/", "__"))
        cache_path = os.path.join(db_dir, safe_name)

        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
                    cached_text = f.read()
                if not self._looks_like_html(cached_text):
                    cf, cs = self.parse_spl_text(cached_text)
                    if len(cf) >= 2 and len(cs) >= 2:
                        return cached_text
            except Exception:
                pass

        url = base_url + urllib.parse.quote(rel_clean, safe="/")
        max_retries = 2
        backoffs = [0.5, 1.0]

        for attempt in range(max_retries + 1):
            code, text = self._http_get_text(url, timeout=8)
            if code == 403:
                return "__FORBIDDEN__"
            if code == 200 and text and not self._looks_like_html(text):
                pf, ps = self.parse_spl_text(text)
                if len(pf) >= 2 and len(ps) >= 2:
                    try:
                        _atomic_write_text(cache_path, text)
                    except Exception:
                        pass
                    return text

            is_transient = (
                code in (429, 502, 503)
                or (code == 0 and any(err in text.lower() for err in ("reset", "connection", "timeout", "timed out", "forcibly closed")))
            )
            if is_transient and attempt < max_retries:
                time.sleep(backoffs[attempt])
                continue
            break

        return None

    def fetch_iem_measurement(
        self,
        base_url: str,
        model: SquigModel,
        channel_mode: str = "avg",  # "avg", "L", "R"
        force_refresh: bool = False,
        skip_offline_db: bool = False,
        save_to_db: bool = True,
        _visited_urls: Optional[Set[str]] = None,
        _allow_fallback: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Fetches raw IEM/headphone measurement curve from the database.
        Averages Left and Right channels if both exist and channel_mode == "avg".
        Enforces strict coupler/rig isolation and cycle-free fallback resolution.
        Returns: (frequencies, spl_values)
        """
        norm_url = _normalize_base_url(base_url)
        visited = set(_visited_urls) if _visited_urls is not None else set()
        if norm_url in visited:
            return [], []
        visited.add(norm_url)

        db_id = self.get_db_id(base_url)

        # 0. Check Offline SQLite Library first (0ms instant lookup)
        if not force_refresh and not skip_offline_db:
            offline_meas = self.offline_db.get_measurement(db_id, model.file_name, channel_mode)
            if offline_meas and offline_meas[0]:
                return offline_meas

        db_dir = self._get_db_cache_dir(base_url)
        cache_fn = re.sub(r'[\\/*?:"<>|]', "_", f"{model.file_name}_{channel_mode}.json")
        cached_json = os.path.join(db_dir, cache_fn)
        if not force_refresh and os.path.exists(cached_json):
            try:
                with open(cached_json, "r", encoding="utf-8") as f_c:
                    data = json.load(f_c)
                    if data.get("freqs") and data.get("spls"):
                        if save_to_db:
                            self.offline_db.save_measurement(db_id, model.file_name, channel_mode, data["freqs"], data["spls"])
                        return data["freqs"], data["spls"]
            except Exception:
                pass

        def _save_cache(f_pts: List[float], s_pts: List[float]) -> None:
            if f_pts and s_pts:
                if save_to_db:
                    self.offline_db.save_measurement(db_id, model.file_name, channel_mode, f_pts, s_pts)
                try:
                    _atomic_write_text(cached_json, json.dumps({"freqs": f_pts, "spls": s_pts}))
                except Exception:
                    pass

        # If fetching from PEQdB aggregator, resolve model against reviewer databases
        if "peqdb.com" in base_url.lower():
            cat = self.get_global_catalog()
            exact_candidates: List[Tuple[str, str]] = []
            fuzzy_candidates: List[Tuple[str, str]] = []
            m_name_clean = model.name.lower().strip()
            m_full_clean = model.full_name.lower().strip()
            m_file_clean = model.file_name.lower().strip()

            # 1. Prefer mapped reviewer databases from peqdb_phones.json if available
            peq_map_file = os.path.join(db_dir, "peqdb_phones.json")
            if os.path.exists(peq_map_file):
                try:
                    with open(peq_map_file, "r", encoding="utf-8") as f_pm:
                        phones_map = json.load(f_pm)
                    phone_provs = phones_map.get(model.file_name, {}) if isinstance(phones_map, dict) else {}
                    if isinstance(phone_provs, dict):
                        prov_urls: Dict[str, str] = {}
                        for c_entry in cat:
                            p_name = str(c_entry.get("provider", "") or "").strip()
                            u = str(c_entry.get("base_url", "") or "").strip()
                            if p_name and u and "hangout.audio" not in u.lower():
                                prov_urls[re.sub(r"[^a-z0-9]", "", p_name.lower())] = u
                                uname = str(c_entry.get("username", "") or "").strip()
                                if uname:
                                    prov_urls[re.sub(r"[^a-z0-9]", "", uname.lower())] = u
                        for d in DEFAULT_DATABASES:
                            du = d.get("base_url", "")
                            if du and "peqdb.com" not in du.lower() and "hangout.audio" not in du.lower():
                                prov_urls[re.sub(r"[^a-z0-9]", "", d.get("name", "").lower())] = du
                                prov_urls[re.sub(r"[^a-z0-9]", "", d.get("id", "").lower())] = du

                        for r_name, sample_list in phone_provs.items():
                            clean_r = re.sub(r"[^a-z0-9]", "", str(r_name).lower())
                            target_url = prov_urls.get(clean_r)
                            if target_url:
                                sample_name = sample_list[0] if isinstance(sample_list, list) and sample_list and sample_list[0] else model.file_name
                                exact_candidates.append((target_url, str(sample_name)))
                except Exception as e:
                    print(f"[SquigClient] PEQdB resolution error: {e}")

            # 2. Search global catalog (exact matches prioritized ahead of word-boundary matches)
            for c_entry in cat:
                p_url = c_entry.get("base_url")
                if not p_url or "hangout.audio" in p_url.lower():
                    continue
                for p in c_entry.get("phones", []):
                    pn = str(p.get("name", "")).lower().strip()
                    pfn = str(p.get("full_name", "")).lower().strip()
                    pf = p.get("file", "")
                    if isinstance(pf, list):
                        pf = pf[0] if pf else ""
                    if not pf:
                        continue
                    pf_str = str(pf)
                    if m_name_clean == pn or m_full_clean == pfn or m_file_clean == pf_str.lower():
                        exact_candidates.append((p_url, pf_str))
                        break
                    elif (
                        (len(m_full_clean) >= 5 and re.search(rf"\b{re.escape(m_full_clean)}\b", pfn))
                        or (len(m_file_clean) >= 5 and re.search(rf"\b{re.escape(m_file_clean)}\b", pfn))
                    ):
                        fuzzy_candidates.append((p_url, pf_str))
                        break

            # Deduplicate BEFORE slicing to top 4
            deduped_cands: List[Tuple[str, str]] = []
            seen_cand_urls: Set[Tuple[str, str]] = set()
            for cand_url, cand_fn in exact_candidates + fuzzy_candidates:
                norm_c = _normalize_base_url(cand_url)
                pair_key = (norm_c, cand_fn)
                if norm_c in visited or pair_key in seen_cand_urls:
                    continue
                seen_cand_urls.add(pair_key)
                deduped_cands.append((cand_url, cand_fn))
                if len(deduped_cands) >= 4:
                    break

            for cand_url, cand_fn in deduped_cands:
                temp_model = SquigModel(
                    brand=model.brand,
                    name=model.name,
                    full_name=model.full_name,
                    file_name=cand_fn,
                    display_name=model.display_name,
                )
                f_res, s_res = self.fetch_iem_measurement(
                    cand_url,
                    temp_model,
                    channel_mode=channel_mode,
                    force_refresh=force_refresh,
                    skip_offline_db=skip_offline_db,
                    save_to_db=save_to_db,
                    _visited_urls=visited,
                    _allow_fallback=False,
                )
                if f_res and s_res:
                    _save_cache(f_res, s_res)
                    return f_res, s_res
            return [], []

        data_dirs = ["data/", "data_hp/"]

        # Build candidate file names: L.txt/R.txt first (standard Squiglink naming), mono s_files kept separate
        fn = model.file_name
        l_files = [f"{fn} L.txt", f"{fn} L1.txt", f"{fn} [L].txt", f"{fn} (L).txt"]
        r_files = [f"{fn} R.txt", f"{fn} R1.txt", f"{fn} [R].txt", f"{fn} (R).txt"]
        s_files = [f"{fn}.txt", f"{fn} 711.txt", f"{fn}.csv"]

        text_l, text_r, text_single = None, None, None

        for d in data_dirs:
            # Try Left channel
            if not text_l and channel_mode in ("avg", "L"):
                res: Optional[str] = None
                for lf in l_files:
                    res = self.fetch_curve_file(base_url, f"{d}{lf}", force_refresh)
                    if res == "__FORBIDDEN__":
                        text_l, text_r, text_single = None, None, None
                        break
                    elif res:
                        text_l = res
                        break
                if res == "__FORBIDDEN__":
                    break

            # Try Right channel
            if not text_r and channel_mode in ("avg", "R"):
                res = None
                for rf in r_files:
                    res = self.fetch_curve_file(base_url, f"{d}{rf}", force_refresh)
                    if res == "__FORBIDDEN__":
                        text_l, text_r, text_single = None, None, None
                        break
                    elif res:
                        text_r = res
                        break
                if res == "__FORBIDDEN__":
                    break

            # Try Single/mono file only if neither L nor R was found
            if not text_single and not (text_l or text_r):
                res = None
                for sf in s_files:
                    res = self.fetch_curve_file(base_url, f"{d}{sf}", force_refresh)
                    if res == "__FORBIDDEN__":
                        text_l, text_r, text_single = None, None, None
                        break
                    elif res:
                        text_single = res
                        break
                if res == "__FORBIDDEN__":
                    break

            if text_l or text_r or text_single:
                break

        # Parse available curves
        freqs_l, spls_l = self.parse_spl_text(text_l) if text_l else ([], [])
        freqs_r, spls_r = self.parse_spl_text(text_r) if text_r else ([], [])
        freqs_s, spls_s = self.parse_spl_text(text_single) if text_single else ([], [])

        ret_f: List[float] = []
        ret_s: List[float] = []
        if channel_mode == "L" and freqs_l:
            ret_f, ret_s = freqs_l, spls_l
        elif channel_mode == "R" and freqs_r:
            ret_f, ret_s = freqs_r, spls_r
        elif channel_mode == "avg" and freqs_l and freqs_r:
            common_f = freqs_l if len(freqs_l) >= len(freqs_r) else freqs_r
            s_l_interp = np.interp(common_f, freqs_l, spls_l)
            s_r_interp = np.interp(common_f, freqs_r, spls_r)
            s_avg = (s_l_interp + s_r_interp) / 2.0
            ret_f, ret_s = list(common_f), s_avg.tolist()
        elif freqs_l:
            ret_f, ret_s = freqs_l, spls_l
        elif freqs_r:
            ret_f, ret_s = freqs_r, spls_r
        elif freqs_s:
            ret_f, ret_s = freqs_s, spls_s

        if ret_f and ret_s:
            _save_cache(ret_f, ret_s)
            return ret_f, ret_s

        # Rig-isolated fallback: ONLY exact model/file match within the SAME coupler rig family,
        # never recursive beyond depth 1, and NEVER poisons the original db_id's cache.
        if not _allow_fallback:
            return [], []

        source_rig = model.rig or self._get_known_rig(base_url) or ""
        source_rig_family = _classify_rig_family(source_rig, base_url)
        if source_rig_family == "unknown":
            return [], []

        cat = self.get_global_catalog()
        m_name_clean = model.name.lower().strip()
        m_full_clean = model.full_name.lower().strip()
        m_file_clean = model.file_name.lower().strip()
        alt_candidates: List[Tuple[str, str]] = []
        for c_entry in cat:
            p_url = c_entry.get("base_url")
            norm_p = _normalize_base_url(p_url or "")
            if not p_url or norm_p in visited or "hangout.audio" in p_url.lower():
                continue
            cand_rig_family = _classify_rig_family(c_entry.get("rig", ""), p_url)
            if cand_rig_family != source_rig_family:
                continue
            for p in c_entry.get("phones", []):
                pn = str(p.get("name", "")).lower().strip()
                pfn = str(p.get("full_name", "")).lower().strip()
                pf = p.get("file", "")
                if isinstance(pf, list):
                    pf = pf[0] if pf else ""
                if not pf:
                    continue
                pf_str = str(pf)
                if (
                    (m_full_clean and m_full_clean == pfn)
                    or (m_file_clean and m_file_clean == pf_str.lower())
                    or (m_name_clean and len(m_name_clean) >= 4 and m_name_clean == pn)
                ):
                    alt_candidates.append((p_url, pf_str))
                    break

        for alt_url, alt_fn in alt_candidates[:3]:
            temp_m = SquigModel(
                brand=model.brand,
                name=model.name,
                full_name=model.full_name,
                file_name=alt_fn,
                display_name=model.display_name,
                rig=source_rig,
            )
            alt_f, alt_s = self.fetch_iem_measurement(
                alt_url,
                temp_m,
                channel_mode=channel_mode,
                force_refresh=force_refresh,
                skip_offline_db=skip_offline_db,
                save_to_db=save_to_db,
                _visited_urls=visited,
                _allow_fallback=False,
            )
            if alt_f and alt_s:
                # Cached under alt_url's own db_id; do NOT overwrite original db_id's cache
                return alt_f, alt_s

        return [], []

    def _load_bundled_target(self, candidates: Sequence[str]) -> Tuple[List[float], List[float]]:
        """Loads a local bundled target curve from assets/targets/ safely (basename-only)."""
        base_assets = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        local_dir = os.path.join(base_assets, "assets", "targets")
        if not os.path.isdir(local_dir):
            return [], []

        seen_paths: Set[str] = set()
        for cand in candidates:
            safe_cand = os.path.basename((cand or "").strip())
            if not safe_cand:
                continue
            stem = re.sub(r"\.(?:txt|csv)$", "", safe_cand, flags=re.IGNORECASE).strip()
            for ext in (".txt", ".csv"):
                loc_path = os.path.join(local_dir, f"{stem}{ext}")
                if loc_path in seen_paths:
                    continue
                seen_paths.add(loc_path)
                if os.path.isfile(loc_path):
                    try:
                        with open(loc_path, "r", encoding="utf-8") as f:
                            freqs, spls = self.parse_spl_text(f.read())
                            if freqs and spls:
                                return freqs, spls
                    except Exception:
                        pass
        return [], []

    def fetch_target_curve(
        self,
        base_url: str,
        target: SquigTarget,
        force_refresh: bool = False,
        skip_offline_db: bool = False,
        save_to_db: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Fetches reviewer reference target curve, prioritizing the active reviewer's own
        rig-matched target files before falling back to compatible bundled curves.
        Returns: (frequencies, spl_values)
        """
        db_id = self.get_db_id(base_url)
        raw_name = (target.name or "").strip()
        raw_file = (target.file_name or "").strip()

        # Strip any accidental .txt/.csv extensions so probes never become *.txt.txt
        target_name_clean = re.sub(r"\.(?:txt|csv)$", "", raw_name, flags=re.IGNORECASE).strip()
        target_file_clean = re.sub(r"\.(?:txt|csv)$", "", raw_file, flags=re.IGNORECASE).strip() or target_name_clean
        unbracketed_name = re.sub(r"^\[[^\]]+\]\s*", "", target_name_clean).strip()
        unbracketed_file = re.sub(r"^\[[^\]]+\]\s*", "", target_file_clean).strip()
        base_stem = re.sub(r"\s+target$", "", unbracketed_file or unbracketed_name, flags=re.IGNORECASE).strip()
        with_target_stem = f"{base_stem} Target" if base_stem else target_file_clean

        # 0. Check Offline SQLite Library first (0ms instant lookup)
        if not force_refresh and not skip_offline_db:
            offline_t = self.offline_db.get_target(db_id, target_file_clean, target_name_clean)
            if offline_t and offline_t[0]:
                return offline_t

        bundled_cands = [
            target_file_clean,
            target_name_clean,
            unbracketed_file,
            unbracketed_name,
            base_stem,
            with_target_stem,
        ]

        # 1. PEQdB uses bundled Sharur/Diamond/Harman reference curves directly
        is_peqdb = "peqdb.com" in (base_url or "").lower()
        if is_peqdb:
            bf, bs = self._load_bundled_target(bundled_cands)
            if bf and bs:
                if save_to_db:
                    self.offline_db.save_target(db_id, target_file_clean, target_name_clean, bf, bs)
                return bf, bs

        # 2. Try remote reviewer database FIRST so rig-specific targets (5128, KB006x, EARS, 711) are never shadowed
        probe_stems: List[str] = []
        for s in (with_target_stem, target_file_clean, target_name_clean, unbracketed_file, unbracketed_name, base_stem):
            s_clean = (s or "").strip()
            if s_clean and s_clean not in probe_stems:
                probe_stems.append(s_clean)

        target_probes: List[str] = []
        seen_probes: Set[str] = set()
        for prefix in ("data/targets/", "data/", "targets/"):
            for stem in probe_stems:
                rel = f"{prefix}{stem}.txt"
                if rel not in seen_probes:
                    seen_probes.add(rel)
                    target_probes.append(rel)
        for stem in probe_stems[:2]:
            rel_csv = f"data/targets/{stem}.csv"
            if rel_csv not in seen_probes:
                seen_probes.add(rel_csv)
                target_probes.append(rel_csv)

        for rel_path in target_probes:
            text = self.fetch_curve_file(base_url, rel_path, force_refresh)
            if text == "__FORBIDDEN__":
                break
            if text:
                freqs, spls = self.parse_spl_text(text)
                if freqs and spls:
                    if save_to_db:
                        self.offline_db.save_target(db_id, target_file_clean, target_name_clean, freqs, spls)
                    return freqs, spls

        # 3. Fallback to bundled assets/targets/ ONLY for compatible rig families (never mix 711 onto 5128/EARS/GRAS)
        rig_family = _classify_rig_family(self._get_known_rig(base_url) or "", base_url)
        if not is_peqdb and rig_family not in ("5128", "ears", "gras"):
            bf, bs = self._load_bundled_target(bundled_cands)
            if bf and bs:
                if save_to_db:
                    self.offline_db.save_target(db_id, target_file_clean, target_name_clean, bf, bs)
                return bf, bs

        return [], []

    def get_targets_for_provider(self, base_url: str) -> List[SquigTarget]:
        """Returns the list of target curves for the given provider base URL."""
        if not base_url:
            return []
        url_with_slash = base_url if base_url.endswith("/") else (base_url + "/")
        # 1. In-memory index
        with self._lock:
            if url_with_slash in self._in_memory_indices:
                return self._in_memory_indices[url_with_slash][1]
            if base_url in self._in_memory_indices:
                return self._in_memory_indices[base_url][1]
        # 2. Disk cache
        db_dir = self._get_db_cache_dir(url_with_slash)
        meta_file = os.path.join(db_dir, "_index_meta.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    targets = [SquigTarget(**t) for t in data.get("targets", [])]
                    if targets:
                        return targets
            except Exception:
                pass
        # 3. Global catalog
        norm_url = _normalize_base_url(base_url)
        cat = self.get_global_catalog()
        for c in cat:
            if _normalize_base_url(c.get("base_url", "")) == norm_url:
                targets = []
                for t in c.get("targets", []):
                    t_str = str(t).strip()
                    if not t_str:
                        continue
                    targets.append(SquigTarget(
                        name=t_str,
                        file_name=f"{t_str} Target" if not t_str.lower().endswith("target") else t_str,
                        category="Reference",
                    ))
                if targets:
                    return targets
        # 4. Standard coupler fallbacks
        return [
            SquigTarget("Harman IE 2019 v2", "Harman IE 2019 v2 Target", "Standard"),
            SquigTarget("IEF Neutral", "IEF Neutral Target", "Reference"),
            SquigTarget("JM-1", "JM-1 Target", "Reference"),
        ]

    def clear_database_cache(self, base_url: str) -> None:
        """Clears local cached files and in-memory index for a specific database."""
        if not base_url:
            return
        url_with_slash = base_url if base_url.endswith("/") else (base_url + "/")
        url_no_slash = base_url.rstrip("/")
        with self._lock:
            self._in_memory_indices.pop(base_url, None)
            self._in_memory_indices.pop(url_with_slash, None)
            self._in_memory_indices.pop(url_no_slash, None)
        db_dir = self._get_db_cache_dir(url_with_slash)
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir, ignore_errors=True)
