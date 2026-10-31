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
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from offline_db import get_offline_db

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request


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


class SquigClient:
    """
    Manages HTTP fetching and persistent local disk caching for Squiglink databases.
    Enforces coupler isolation: models and targets are strictly bound to the active database.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        self.offline_db = get_offline_db()
        if cache_dir:
            self.cache_dir = os.path.abspath(cache_dir)
        else:
            primary_cache = os.path.join(os.path.expanduser("~"), ".cache", "clean_apo", "squiglink")
            try:
                os.makedirs(primary_cache, exist_ok=True)
                self.cache_dir = primary_cache
            except Exception:
                self.cache_dir = os.path.abspath(os.path.join("cache", "squiglink"))
                os.makedirs(self.cache_dir, exist_ok=True)

        self._user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CleanAPO/1.0"
        self._in_memory_indices: Dict[str, Tuple[List[SquigModel], List[SquigTarget], str]] = {}
        self._global_catalog: Optional[List[Dict[str, Any]]] = None

        if HAS_REQUESTS:
            self.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=60, pool_maxsize=80, max_retries=1)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        else:
            self.session = None

    # -------------------------------------------------------------------------
    # Network / Cache Low-Level Utilities
    # -------------------------------------------------------------------------
    def _safe_db_key(self, base_url: str) -> str:
        parsed = urllib.parse.urlparse(base_url)
        netloc = parsed.netloc.replace(":", "_")
        path = parsed.path.strip("/").replace("/", "_")
        key = f"{netloc}_{path}" if path else netloc
        return re.sub(r"[^a-zA-Z0-9_\.-]", "_", key).strip("_")

    def _get_db_cache_dir(self, base_url: str) -> str:
        key = self._safe_db_key(base_url)
        path = os.path.join(self.cache_dir, key)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_db_id(self, base_url: str) -> str:
        clean_url = base_url.rstrip("/")
        for d in DEFAULT_DATABASES:
            if d.get("base_url", "").rstrip("/") == clean_url:
                return d.get("id", "default")
        cat = self.get_global_catalog()
        for c in cat:
            if c.get("base_url", "").rstrip("/") == clean_url:
                return c.get("username") or c.get("provider") or "default"
        parsed = urllib.parse.urlparse(base_url)
        return re.sub(r"[^a-zA-Z0-9_]", "_", parsed.netloc + parsed.path).strip("_") or "default"

    def _http_get_text(self, url: str, timeout: int = 8) -> Tuple[int, str]:
        """Performs HTTP GET with requests session (or urllib fallback), returning (status_code, text)."""
        if HAS_REQUESTS and self.session is not None:
            try:
                resp = self.session.get(
                    url,
                    headers={"User-Agent": self._user_agent},
                    timeout=timeout,
                )
                resp.encoding = "utf-8"
                return resp.status_code, resp.text
            except Exception as e:
                return 0, str(e)
        elif HAS_REQUESTS:
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": self._user_agent},
                    timeout=timeout,
                )
                resp.encoding = "utf-8"
                return resp.status_code, resp.text
            except Exception as e:
                return 0, str(e)
        else:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    code = resp.getcode()
                    content = resp.read().decode("utf-8", errors="replace")
                    return code, content
            except Exception as e:
                return 0, str(e)

    # -------------------------------------------------------------------------
    # SPL Parsing
    # -------------------------------------------------------------------------
    @staticmethod
    def parse_spl_text(text: str) -> Tuple[List[float], List[float]]:
        """
        Parses standard two- or three-column space/tab-separated frequency response files (.txt/.csv).
        Handles REW comment headers, commas, tabs, and invalid data gracefully.
        """
        freqs: List[float] = []
        spls: List[float] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("*", "#", ";", "//")):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    s = float(parts[1])
                    if f > 0:
                        freqs.append(f)
                        spls.append(s)
                except ValueError:
                    continue
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
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)
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
                    rig = str(data.get("rig", "IEC 60318-4 (711)"))
                    if models and targets:
                        self._in_memory_indices[base_url] = (models, targets, rig)
                        return models, targets, rig
            except Exception:
                pass

        # Handle PEQdB API endpoint
        if "peqdb.com" in base_url:
            code, text = self._http_get_text("https://peqdb.com/opt-v3/lists", timeout=10)
            if code == 200 and len(text) > 500:
                try:
                    peq_json = json.loads(text)
                    phones = peq_json.get("phones", {})
                    models: List[SquigModel] = []
                    for p_name in sorted(phones.keys()):
                        p_clean = p_name.strip()
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
                    try:
                        with open(meta_file, "w", encoding="utf-8") as f_meta:
                            json.dump(
                                {
                                    "models": [m.to_dict() for m in models],
                                    "targets": [t.to_dict() for t in targets],
                                    "rig": rig,
                                },
                                f_meta,
                                indent=2,
                            )
                    except Exception:
                        pass
                    peq_map_file = os.path.join(db_dir, "peqdb_phones.json")
                    try:
                        with open(peq_map_file, "w", encoding="utf-8") as f_pm:
                            json.dump(phones, f_pm)
                    except Exception:
                        pass
                    self._in_memory_indices[base_url] = (models, targets, rig)
                    return models, targets, rig
                except Exception as e:
                    print(f"[SquigClient] Error loading PEQdB lists: {e}")

        # 1. Fetch Phonebook (phone_book.json or data/phone_book.json)
        pb_candidates = [
            base_url + "phone_book.json",
            base_url + "data/phone_book.json",
            base_url + "data_hp/phone_book.json",
        ]
        pb_text = ""
        for pb_url in pb_candidates:
            code, text = self._http_get_text(pb_url, timeout=7)
            if code == 200 and len(text) > 100:
                pb_text = text
                break

        # 2. Fetch Config (config.js or assets/js/config.js)
        cfg_candidates = [
            base_url + "config.js",
            base_url + "assets/js/config.js",
        ]
        cfg_text = ""
        for cfg_url in cfg_candidates:
            code, text = self._http_get_text(cfg_url, timeout=7)
            if code == 200 and len(text) > 100:
                cfg_text = text
                break

        # Parse rig description from config.js
        rig_desc = "clone IEC 60318-4 (711)"
        if cfg_text:
            rig_match = re.search(r'rig_description\s*=\s*[\'"]([^\'"]+)[\'"]', cfg_text)
            if rig_match:
                rig_desc = rig_match.group(1).strip()

        # Parse models from phonebook
        models: List[SquigModel] = []
        target_names_from_pb: List[str] = []

        if pb_text:
            try:
                pb_json = json.loads(pb_text)
                if isinstance(pb_json, list):
                    for brand_obj in pb_json:
                        if not isinstance(brand_obj, dict):
                            continue
                        brand_name = str(brand_obj.get("name", "")).strip()
                        phones = brand_obj.get("phones", [])
                        
                        is_target_cat = "target" in brand_name.lower() or brand_name == "_EQ"
                        for p in phones:
                            if not isinstance(p, dict):
                                continue
                            p_name = str(p.get("name", "")).strip()
                            p_file = p.get("file", "")
                            p_score = str(p.get("reviewScore", ""))
                            p_price = str(p.get("price", ""))
                            
                            if is_target_cat:
                                if isinstance(p_file, str) and p_file:
                                    target_names_from_pb.append(p_file)
                                elif isinstance(p_file, list):
                                    target_names_from_pb.extend([str(x) for x in p_file if x])
                                continue

                            # Handle multi-variant files or single file
                            if isinstance(p_file, list):
                                suffixes = p.get("suffix", [])
                                for idx, f_item in enumerate(p_file):
                                    var_str = suffixes[idx] if idx < len(suffixes) else f"Variant {idx+1}"
                                    var_str = var_str.strip()
                                    disp = f"{brand_name} {p_name} {var_str}".strip() if var_str else f"{brand_name} {p_name}"
                                    models.append(
                                        SquigModel(
                                            brand=brand_name,
                                            name=p_name,
                                            full_name=f"{brand_name} {p_name}",
                                            file_name=str(f_item),
                                            display_name=disp,
                                            variant=var_str,
                                            review_score=p_score,
                                            price=p_price,
                                        )
                                    )
                            elif isinstance(p_file, str) and p_file.strip():
                                models.append(
                                    SquigModel(
                                        brand=brand_name,
                                        name=p_name,
                                        full_name=f"{brand_name} {p_name}".strip(),
                                        file_name=p_file.strip(),
                                        display_name=f"{brand_name} {p_name}".strip(),
                                        review_score=p_score,
                                        price=p_price,
                                    )
                                )
            except Exception:
                pass

        # Parse targets from config.js
        targets: List[SquigTarget] = []
        seen_targets = set()

        if cfg_text:
            # Strip JS line comments
            clean_lines = []
            for line in cfg_text.splitlines():
                l_str = line.strip()
                if l_str.startswith("//"):
                    continue
                c_idx = re.search(r"(?<!:)\/\/", l_str)
                if c_idx:
                    l_str = l_str[:c_idx.start()].strip()
                clean_lines.append(l_str)
            clean_js = "\n".join(clean_lines)

            # Look for targets = [ ... ]
            m = re.search(r"(?:const|var|let)?\s*targets\s*=\s*(\[[\s\S]*?\]);", clean_js)
            if m:
                block = m.group(1)
                # Find type: "...", files: [ ... ]
                cat_blocks = re.findall(r"\{[^{}]*type\s*:\s*[\'\"]([^\'\"]+)[\'\"][^{}]*files\s*:\s*\[([\s\S]*?)\][^{}]*\}", block)
                for cat_type, files_content in cat_blocks:
                    matches = re.findall(r"['\"]([^'\"]+)['\"]", files_content)
                    for item in matches:
                        clean_item = item.strip()
                        if clean_item and clean_item not in seen_targets:
                            seen_targets.add(clean_item)
                            file_str = clean_item if clean_item.lower().endswith("target") else f"{clean_item} Target"
                            targets.append(
                                SquigTarget(
                                    name=clean_item,
                                    file_name=file_str,
                                    category=cat_type.strip(),
                                )
                            )

            # If targets array wasn't found or was empty, check init_phones
            if not targets:
                init_m = re.search(r"init_phones\s*=\s*\[([\s\S]*?)\]", clean_js)
                if init_m:
                    matches = re.findall(r"['\"]([^'\"]+)['\"]", init_m.group(1))
                    for item in matches:
                        clean_item = item.strip()
                        if "target" in clean_item.lower() and clean_item not in seen_targets:
                            seen_targets.add(clean_item)
                            targets.append(
                                SquigTarget(
                                    name=clean_item.replace(" Target", "").strip(),
                                    file_name=clean_item,
                                    category="Reference",
                                )
                            )

        # Add any targets found in phonebook
        for t_name in target_names_from_pb:
            clean_t = t_name.strip()
            if clean_t and clean_t not in seen_targets:
                seen_targets.add(clean_t)
                file_str = clean_t if clean_t.lower().endswith("target") else f"{clean_t} Target"
                targets.append(
                    SquigTarget(
                        name=clean_t.replace(" Target", "").strip(),
                        file_name=file_str,
                        category="Database Target",
                    )
                )

        # Fallback to catalog targets if targets array could not be parsed from config.js
        if not targets:
            clean_url = base_url.rstrip("/")
            cat = self.get_global_catalog()
            for c in cat:
                if c.get("base_url", "").rstrip("/") == clean_url:
                    for t_name in c.get("targets", []):
                        clean_t = str(t_name).strip()
                        if clean_t and clean_t not in seen_targets:
                            seen_targets.add(clean_t)
                            file_str = clean_t if clean_t.lower().endswith("target") else f"{clean_t} Target"
                            targets.append(
                                SquigTarget(
                                    name=clean_t.replace(" Target", "").strip(),
                                    file_name=file_str,
                                    category="Reference",
                                )
                            )
                    break

        # Fallback to offline_db targets
        if not targets:
            db_id = self._get_db_id(base_url)
            try:
                import sqlite3
                if os.path.exists(self.offline_db.db_path):
                    with sqlite3.connect(self.offline_db.db_path) as conn:
                        c = conn.cursor()
                        c.execute(
                            "SELECT DISTINCT target_file, target_name FROM targets WHERE db_id = ?",
                            (db_id,)
                        )
                        for tf, tn in c.fetchall():
                            t_clean = (tn or tf or "").replace(" Target", "").replace(".txt", "").strip()
                            if t_clean and t_clean not in seen_targets:
                                seen_targets.add(t_clean)
                                targets.append(
                                    SquigTarget(
                                        name=t_clean,
                                        file_name=tf or f"{t_clean} Target",
                                        category="Offline Reference",
                                    )
                                )
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
                if s not in seen_targets:
                    seen_targets.add(s)
                    targets.append(
                        SquigTarget(
                            name=s,
                            file_name=f"{s} Target",
                            category="Standard Reference",
                        )
                    )

        # Save metadata to cache
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "models": [m.to_dict() for m in models],
                        "targets": [t.to_dict() for t in targets],
                        "rig": rig_desc,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

        self._in_memory_indices[base_url] = (models, targets, rig_desc)
        return models, targets, rig_desc

    # -------------------------------------------------------------------------
    # Global Catalog Search (43k+ models across 148 Squiglink databases)
    # -------------------------------------------------------------------------
    def get_global_catalog(self) -> List[Dict[str, Any]]:
        if self._global_catalog is not None:
            return self._global_catalog

        candidate_paths = [
            os.path.join(getattr(sys, '_MEIPASS', ''), "squig_catalog.json"),
            os.path.abspath("squig_catalog.json"),
            os.path.join(self.cache_dir, "squig_catalog.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "squig_catalog.json"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self._global_catalog = json.load(f)
                        return self._global_catalog
                except Exception:
                    pass

        self._global_catalog = []
        return self._global_catalog

    def search_global_catalog(
        self,
        query: str,
        target_query: str = "",
        limit: int = 150,
    ) -> List[GlobalSearchResult]:
        catalog = self.get_global_catalog()
        if not catalog:
            return []

        tokens = query.strip().lower().split()
        target_tokens = [t.strip().lower() for t in re.split(r'[,;]+', target_query) if t.strip()]

        if not tokens and not target_tokens:
            return []

        results: List[GlobalSearchResult] = []
        seen = set()

        for db in catalog:
            p_name = db.get("provider", "Reviewer")
            db_type = db.get("db_type", "IEMs")
            base_url = db.get("base_url", "")
            rig = db.get("rig", "IEC 60318-4 (711)")
            targets = db.get("targets", [])

            # Target filter: if target_tokens specified, provider must provide at least one target matching any token
            if target_tokens:
                targets_lower = [t.lower() for t in targets]
                has_target_match = False
                for tt in target_tokens:
                    if any(tt in tl for tl in targets_lower):
                        has_target_match = True
                        break
                if not has_target_match:
                    continue

            for p in db.get("phones", []):
                full_name = p.get("full_name", "")
                full_lower = full_name.lower()
                if not tokens or all(t in full_lower for t in tokens):
                    file_key = (base_url, p.get("file", full_name))
                    if file_key not in seen:
                        seen.add(file_key)
                        results.append(
                            GlobalSearchResult(
                                brand=p.get("brand", ""),
                                name=p.get("name", ""),
                                full_name=full_name,
                                file_name=p.get("file", full_name),
                                provider_name=f"{p_name} ({db_type})" if db_type else p_name,
                                db_type=db_type,
                                base_url=base_url,
                                rig=rig,
                                targets=targets,
                            )
                        )
                        if len(results) >= limit:
                            return results
        return results

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
        """
        if not base_url.endswith("/"):
            base_url += "/"

        db_dir = self._get_db_cache_dir(base_url)
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", os.path.basename(file_path_relative))
        cache_path = os.path.join(db_dir, safe_name)

        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:
                pass

        url = base_url + urllib.parse.quote(file_path_relative)
        max_retries = 2
        backoffs = [0.5, 1.0]

        for attempt in range(max_retries + 1):
            code, text = self._http_get_text(url, timeout=8)
            if code == 403:
                return "__FORBIDDEN__"
            if code == 200 and len(text) > 20:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(text)
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
    ) -> Tuple[List[float], List[float]]:
        """
        Fetches raw IEM measurement curve from the database.
        Averages Left and Right channels if both exist and channel_mode == "avg".
        Returns: (frequencies, spl_values)
        """
        db_id = self._get_db_id(base_url)

        # 0. Check Offline SQLite Library first (0ms instant lookup)
        if not force_refresh:
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
                        self.offline_db.save_measurement(db_id, model.file_name, channel_mode, data["freqs"], data["spls"])
                        return data["freqs"], data["spls"]
            except Exception:
                pass

        def _save_cache(f_pts: List[float], s_pts: List[float]) -> None:
            if f_pts and s_pts:
                self.offline_db.save_measurement(db_id, model.file_name, channel_mode, f_pts, s_pts)
                try:
                    with open(cached_json, "w", encoding="utf-8") as f_w:
                        json.dump({"freqs": f_pts, "spls": s_pts}, f_w)
                except Exception:
                    pass

        # If fetching from PEQdB aggregator, resolve model against reviewer databases
        if "peqdb.com" in base_url:

            cat = self.get_global_catalog()
            candidates_to_try = []
            m_name_clean = model.name.lower().strip()
            m_full_clean = model.full_name.lower().strip()
            m_file_clean = model.file_name.lower().strip()

            # 1. Search global catalog for exact model or filename matches
            for c_entry in cat:
                p_url = c_entry.get("base_url")
                if not p_url:
                    continue
                for p in c_entry.get("phones", []):
                    pn = str(p.get("name", "")).lower().strip()
                    pfn = str(p.get("full_name", "")).lower().strip()
                    pf = p.get("file", "")
                    if isinstance(pf, list):
                        pf = pf[0] if pf else ""
                    if not pf:
                        continue
                    if m_name_clean == pn or m_full_clean == pfn or m_file_clean == pf.lower():
                        candidates_to_try.append((p_url, pf))
                        break
                    elif (len(m_name_clean) > 3 and m_name_clean in pn) or (len(m_file_clean) > 3 and m_file_clean in pfn):
                        candidates_to_try.append((p_url, pf))
                        break

            # 2. Also map reviewer databases from peqdb_phones.json if available
            peq_map_file = os.path.join(db_dir, "peqdb_phones.json")
            if os.path.exists(peq_map_file):
                try:
                    with open(peq_map_file, "r", encoding="utf-8") as f_pm:
                        phones_map = json.load(f_pm)
                    phone_provs = phones_map.get(model.file_name, {})
                    prov_urls = {}
                    for c_entry in cat:
                        p_name = c_entry.get("provider", "").strip()
                        u = c_entry.get("base_url", "").strip()
                        if p_name and u:
                            prov_urls[re.sub(r"[^a-z0-9]", "", p_name.lower())] = u
                            uname = c_entry.get("username", "").strip()
                            if uname:
                                prov_urls[re.sub(r"[^a-z0-9]", "", uname.lower())] = u
                    for d in DEFAULT_DATABASES:
                        prov_urls[re.sub(r"[^a-z0-9]", "", d.get("name", "").lower())] = d.get("base_url", "")
                        prov_urls[re.sub(r"[^a-z0-9]", "", d.get("id", "").lower())] = d.get("base_url", "")

                    for r_name, sample_list in phone_provs.items():
                        clean_r = re.sub(r"[^a-z0-9]", "", r_name.lower())
                        target_url = prov_urls.get(clean_r)
                        if target_url:
                            sample_name = sample_list[0] if sample_list and sample_list[0] else model.file_name
                            candidates_to_try.append((target_url, sample_name))
                except Exception as e:
                    print(f"[SquigClient] PEQdB resolution error: {e}")

            # Try top candidates (limit to 3 to guarantee fast response)
            seen_cand_urls = set()
            for cand_url, cand_fn in candidates_to_try[:4]:
                pair_key = (cand_url, cand_fn)
                if pair_key in seen_cand_urls:
                    continue
                seen_cand_urls.add(pair_key)
                temp_model = SquigModel(
                    brand=model.brand,
                    name=model.name,
                    full_name=model.full_name,
                    file_name=cand_fn,
                    display_name=model.display_name,
                )
                f_res, s_res = self.fetch_iem_measurement(cand_url, temp_model, channel_mode, force_refresh)
                if f_res and s_res:
                    try:
                        with open(cached_json, "w", encoding="utf-8") as f_c:
                            json.dump({"freqs": f_res, "spls": s_res}, f_c)
                    except Exception:
                        pass
                    return f_res, s_res

        data_dirs = ["data/", "data_hp/"]
        
        # Build candidate file names for Left, Right, and Single
        fn = model.file_name
        l_files = [f"{fn} L1.txt", f"{fn} L.txt", f"{fn} [L].txt", f"{fn} (L).txt", f"{fn}.txt", f"{fn} 711.txt"]
        r_files = [f"{fn} R1.txt", f"{fn} R.txt", f"{fn} [R].txt", f"{fn} (R).txt"]
        s_files = [f"{fn}.txt", f"{fn} 711.txt", f"{fn}.csv"]

        text_l, text_r, text_single = None, None, None

        for d in data_dirs:
            # Try Left
            if not text_l and channel_mode in ("avg", "L"):
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

            # Try Right
            if not text_r and channel_mode in ("avg", "R"):
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

            # Try Single (only if L or R not found)
            if not text_single and not (text_l or text_r):
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

            # If we successfully found files in this directory, break early
            if text_l or text_r or text_single:
                break

        # Parse available curves
        freqs_l, spls_l = self.parse_spl_text(text_l) if text_l else ([], [])
        freqs_r, spls_r = self.parse_spl_text(text_r) if text_r else ([], [])
        freqs_s, spls_s = self.parse_spl_text(text_single) if text_single else ([], [])

        ret_f, ret_s = [], []
        if channel_mode == "L" and freqs_l:
            ret_f, ret_s = freqs_l, spls_l
        elif channel_mode == "R" and freqs_r:
            ret_f, ret_s = freqs_r, spls_r
        elif channel_mode == "avg" and freqs_l and freqs_r:
            import numpy as np
            common_f = freqs_l if len(freqs_l) >= len(freqs_r) else freqs_r
            s_l_interp = np.interp(common_f, freqs_l, spls_l)
            s_r_interp = np.interp(common_f, freqs_r, spls_r)
            s_avg = (s_l_interp + s_r_interp) / 2.0
            ret_f, ret_s = list(common_f), list(s_avg)
        elif freqs_l:
            ret_f, ret_s = freqs_l, spls_l
        elif freqs_r:
            ret_f, ret_s = freqs_r, spls_r
        elif freqs_s:
            ret_f, ret_s = freqs_s, spls_s

        if ret_f and ret_s:
            _save_cache(ret_f, ret_s)
            return ret_f, ret_s

        # Fallback to alternative open reviewers from squig_catalog.json
        cat = self.get_global_catalog()
        m_name_clean = model.name.lower().strip()
        m_full_clean = model.full_name.lower().strip()
        m_file_clean = model.file_name.lower().strip()
        alt_candidates = []
        for c_entry in cat:
            p_url = c_entry.get("base_url")
            if not p_url or p_url.rstrip("/") == base_url.rstrip("/"):
                continue
            if "hangout.audio" in p_url:
                continue
            for p in c_entry.get("phones", []):
                pn = str(p.get("name", "")).lower().strip()
                pfn = str(p.get("full_name", "")).lower().strip()
                pf = p.get("file", "")
                if isinstance(pf, list):
                    pf = pf[0] if pf else ""
                if not pf:
                    continue
                if m_name_clean == pn or m_full_clean == pfn or m_file_clean == pf.lower():
                    alt_candidates.append((p_url, pf))
                    break
                elif (len(m_name_clean) > 3 and m_name_clean in pn) or (len(m_file_clean) > 3 and m_file_clean in pfn):
                    alt_candidates.append((p_url, pf))
                    break

        for alt_url, alt_fn in alt_candidates[:3]:
            temp_m = SquigModel(
                brand=model.brand,
                name=model.name,
                full_name=model.full_name,
                file_name=alt_fn,
                display_name=model.display_name,
            )
            alt_f, alt_s = self.fetch_iem_measurement(alt_url, temp_m, channel_mode, force_refresh)
            if alt_f and alt_s:
                _save_cache(alt_f, alt_s)
                return alt_f, alt_s

        return [], []

    def fetch_target_curve(
        self,
        base_url: str,
        target: SquigTarget,
        force_refresh: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """
        Fetches reviewer reference target curve from local bundled targets or database.
        Returns: (frequencies, spl_values)
        """
        db_id = self._get_db_id(base_url)
        target_name_clean = target.name.strip()
        target_file_clean = target.file_name.strip()
        unbracketed_name = re.sub(r"^\[[^\]]+\]\s*", "", target_name_clean).strip()
        unbracketed_file = re.sub(r"^\[[^\]]+\]\s*", "", target_file_clean).strip()

        # 0. Check Offline SQLite Library first (0ms instant lookup)
        if not force_refresh:
            offline_t = self.offline_db.get_target(db_id, target_file_clean, target_name_clean)
            if offline_t and offline_t[0]:
                return offline_t

        # 1. First check local bundled reference curves in assets/targets/
        base_assets = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        local_dir = os.path.join(base_assets, "assets", "targets")
        if os.path.exists(local_dir):
            test_cands = [
                target_file_clean,
                target_name_clean,
                unbracketed_file,
                unbracketed_name,
                f"{unbracketed_file} Target",
                f"{unbracketed_name} Target",
            ]
            for cand in test_cands:
                for ext in [".txt", ".csv"]:
                    loc_path = os.path.join(local_dir, f"{cand}{ext}")
                    if os.path.exists(loc_path):
                        try:
                            with open(loc_path, "r", encoding="utf-8") as f:
                                freqs, spls = self.parse_spl_text(f.read())
                                if freqs and spls:
                                    self.offline_db.save_target(db_id, target_file_clean, target_name_clean, freqs, spls)
                                    return freqs, spls
                        except Exception:
                            pass

        # 2. Try remote database with focused search paths (prioritize standard data/targets/)
        target_probes = [
            f"data/targets/{target_file_clean}.txt",
            f"data/targets/{target_name_clean}.txt",
            f"targets/{target_file_clean}.txt",
            f"data/{target_file_clean}.txt",
            f"data/targets/{target_file_clean}.csv",
        ]
        if unbracketed_name and unbracketed_name != target_name_clean:
            target_probes.append(f"data/targets/{unbracketed_name}.txt")

        for rel_path in target_probes:
            text = self.fetch_curve_file(base_url, rel_path, force_refresh)
            if text == "__FORBIDDEN__":
                break
            if text:
                freqs, spls = self.parse_spl_text(text)
                if freqs and spls:
                    self.offline_db.save_target(db_id, target_file_clean, target_name_clean, freqs, spls)
                    return freqs, spls

        return [], []

    def get_targets_for_provider(self, base_url: str) -> List[SquigTarget]:
        """Returns the list of target curves for the given provider base URL."""
        if not base_url:
            return []
        # 1. In-memory index
        if base_url in self._in_memory_indices:
            return self._in_memory_indices[base_url][1]
        # 2. Disk cache
        db_dir = self._get_db_cache_dir(base_url)
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
        clean_url = base_url.rstrip("/")
        cat = self.get_global_catalog()
        for c in cat:
            if c.get("base_url", "").rstrip("/") == clean_url:
                targets = []
                for t in c.get("targets", []):
                    targets.append(SquigTarget(
                        name=t,
                        file_name=f"{t} Target" if not t.lower().endswith("target") else t,
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
        """Clears local cached files for a specific database."""
        db_dir = self._get_db_cache_dir(base_url)
        if os.path.exists(db_dir):
            import shutil
            shutil.rmtree(db_dir, ignore_errors=True)
