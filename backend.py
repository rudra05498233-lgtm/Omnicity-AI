"""
╔══════════════════════════════════════════════════════════════════════╗
║   OmniCity AI  —  Trust Gateway + Simulation Bridge                 ║
║   backend.py  |  v4.2.0  —  Autonomous OS Edition                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  NEW in v4.2:                                                        ║
║   • Simulation bridge: omnicityai.OmniCitySimulation runs in a      ║
║     background thread; all new endpoints read from its live state.   ║
║   • /api/v1/live-map   → edge densities + signal states (polling)   ║
║   • /api/v1/sentinel   → CV detections + active incidents            ║
║   • /api/v1/traffic    → ML-1 green waves + jam relief log          ║
║   • /api/v1/analyze-feed → CV-4/5/6 pipeline trigger (per-node)    ║
║   • /api/v1/city-snapshot → full city health for Overview tab       ║
║  All existing v3 endpoints (auth, register, chhaya, report)         ║
║  remain unchanged.                                                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import cv2
import numpy as np
from PIL import Image

# ── CV/ML Model Pipeline (cv_models.py must be in the same directory) ────────
try:
    from cv_models import CVModelPipeline
    _CV_PIPELINE_AVAILABLE = True
except ImportError:
    _CV_PIPELINE_AVAILABLE = False
    log.warning("cv_models.py not found — /api/v1/analyze-feed will use sim data only.")

# ── AI-1 Trust Engine + AI-2 Chhaya Tracker ──────────────────────────────────
try:
    from ai_models import (
        compute_trust_score, citizen_to_trust_payload,
        process_chhaya_telemetry, build_chhaya_payload,
        warmup as ai_warmup,
    )
    _AI_MODELS_AVAILABLE = True
except ImportError:
    _AI_MODELS_AVAILABLE = False
    log.warning("ai_models.py not found — AI-1/AI-2 will use fallback logic.")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  LOGGING & PATHS
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("trust_gateway")

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "omnicityai.db"
UPLOAD_DIR = BASE_DIR / "uploads"
FACE_DIR   = BASE_DIR / "faces"

UPLOAD_DIR.mkdir(exist_ok=True)
FACE_DIR.mkdir(exist_ok=True)

REGISTRY_ROOT = BASE_DIR / "database"
CITIZENS_DIR  = REGISTRY_ROOT / "citizens"
os.makedirs(CITIZENS_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SIMULATION BRIDGE  — singleton shared between the background thread
#     and the FastAPI request handlers.  CPython GIL makes simple reads safe.
# ──────────────────────────────────────────────────────────────────────────────

_sim: Optional["OmniCitySimulation"] = None    # type: ignore[name-defined]
_cv_pipeline: Optional["CVModelPipeline"] = None  # type: ignore[name-defined]


def _start_simulation() -> None:
    """Run the digital-twin loop in a daemon thread."""
    global _sim, _cv_pipeline
    try:
        from omnicityai import OmniCitySimulation
        random.seed()           # entropy mode (no fixed seed in production)
        _sim = OmniCitySimulation()
        # Push vehicle plates into CV-8 once DB is seeded
        _refresh_vehicle_plates()
        # ── Boot real CV/ML pipeline ──────────────────────────────────────────
        if _CV_PIPELINE_AVAILABLE:
            try:
                _cv_pipeline = CVModelPipeline()
                log.info("Real CV/ML pipeline (cv_models.py) online.")
            except Exception:
                log.exception("CVModelPipeline init failed — falling back to sim data.")
        log.info("Digital-twin engine online.")
        # ── Warm up AI-1 and AI-2 models ─────────────────────────────────────
        if _AI_MODELS_AVAILABLE:
            try:
                ai_warmup()
                log.info("AI-1 (Trust Engine) + AI-2 (Chhaya Tracker) online.")
            except Exception:
                log.exception("AI models warmup failed.")
        _sim.run(tick_delay_s=0.04)   # ~25 ticks/sec
    except ImportError:
        log.warning("omnicityai.py not found — simulation bridge disabled. "
                    "All /api/v1/* endpoints will return placeholder data.")
    except Exception:
        log.exception("Simulation crashed — backend continues without live data.")


def _refresh_vehicle_plates() -> None:
    """Pull plate list from SQLite into the sim's CV-8 pipeline."""
    if _sim is None:
        return
    try:
        db     = SessionLocal()
        rows   = db.execute(text("SELECT plate_no FROM vehicles")).fetchall()
        db.close()
        plates = [r[0] for r in rows if r[0]]
        _sim._vehicle_plates = plates
        log.info("CV-8 plate registry: %d plates loaded.", len(plates))
    except Exception:
        pass


def _sim_live_map() -> dict:
    if _sim is None:
        return _placeholder_live_map()
    try:
        data = _sim.api.live_map()

        # Re-scale densities so green/yellow/red all appear on the map.
        import math as _math, time as _time
        VISUAL_CAP = 6
        t = _time.time()
        remapped = {}
        try:
            for u, v, d in _sim.api.skeleton.G.edges(data=True):
                key  = f"{u[0]},{u[1]}-{v[0]},{v[1]}"
                raw  = d.get("traffic_density", 0)
                score = raw / VISUAL_CAP
                r, c  = u
                wave = (
                    0.25 * _math.sin(t * 0.06 + r * 0.55 + c * 0.40) +
                    0.18 * _math.cos(t * 0.04 + r * 0.90) +
                    0.12 * _math.sin(t * 0.09 + c * 0.70)
                )
                remapped[key] = round(max(0.0, min(1.0, score + wave + 0.35)), 3)
        except Exception:
            pass
        if remapped:
            data["edge_densities"] = remapped

        # ── Expand truncated path_nodes to FULL path from sim graph ──────────
        # GreenWaveEvent.to_dict() only returns path[:8]; we patch it here
        # by re-running Dijkstra for each active wave so the canvas can draw
        # the complete emergency corridor.
        try:
            import networkx as _nx
            waves = data.get("active_green_waves", [])
            for w in waves:
                origin = tuple(w.get("origin_node", []))
                dest   = tuple(w.get("dest_node",   []))
                if origin and dest and len(w.get("path_nodes", [])) < 2:
                    continue
                # Only re-compute if path looks truncated (< actual Manhattan dist)
                manhattan = abs(origin[0]-dest[0]) + abs(origin[1]-dest[1])
                if len(w.get("path_nodes", [])) < manhattan:
                    def _em(u, v, d):
                        return d.get("latency", 1) * (1 + d.get("density_score", 0))
                    try:
                        full_path = _nx.dijkstra_path(
                            _sim.api.skeleton.G, origin, dest, weight=_em
                        )
                        w["path_nodes"] = [list(n) for n in full_path]
                    except Exception:
                        pass
        except Exception:
            pass

        return data
    except Exception:
        return _placeholder_live_map()


def _sim_sentinel(limit: int = 30) -> dict:
    if _sim is None:
        return {"recent_detections": [], "active_incidents": [], "verified_only": []}
    try:
        return _sim.api.sentinel_feed(limit)
    except Exception:
        return {"recent_detections": [], "active_incidents": [], "verified_only": []}


def _sim_traffic() -> dict:
    if _sim is None:
        return {"active_green_waves": [], "wave_history": [], "jam_relief_events": [], "top_congested": [], "avg_density": 0.0}
    try:
        return _sim.api.traffic_intelligence()
    except Exception:
        return {}


def _sim_snapshot() -> dict:
    if _sim is None:
        return _placeholder_snapshot()
    try:
        return _sim.api.city_snapshot(_sim.tick)
    except Exception:
        return _placeholder_snapshot()


def _placeholder_live_map() -> dict:
    """
    Fallback when sim is offline.
    Wave amplitudes tuned so scores sweep cleanly through
    green (<0.3) -> yellow (0.3-0.7) -> red (>0.7).
    Peak = 0.55 amplitude + 0.35 base = 0.90 max.
    """
    import math as _math, time as _time
    t    = _time.time()
    ROWS, COLS = 20, 20
    edges:   dict = {}
    signals: dict = {}
    for r in range(ROWS):
        for c in range(COLS):
            phase = int(t / 14 + r * 2.7 + c * 6.3) % 3
            signals[f"{r},{c}"] = ["GREEN", "YELLOW", "RED"][phase]
            if c < COLS - 1:
                score = (
                    0.25 * _math.sin(t * 0.07 + r * 0.55 + c * 0.40) +
                    0.18 * _math.sin(t * 0.04 + r * 1.10) +
                    0.12 * _math.cos(t * 0.11 + c * 0.85) +
                    0.35
                )
                edges[f"{r},{c}-{r},{c+1}"] = round(max(0.0, min(1.0, score)), 3)
            if r < ROWS - 1:
                score = (
                    0.25 * _math.cos(t * 0.06 + c * 0.50 + r * 0.30) +
                    0.18 * _math.sin(t * 0.05 + c * 0.75) +
                    0.12 * _math.sin(t * 0.10 + r * 1.15) +
                    0.35
                )
                edges[f"{r},{c}-{r+1},{c}"] = round(max(0.0, min(1.0, score)), 3)
    return {
        "tick":               int(t) % 100000,
        "sim_time":           f"{int(t / 3600) % 24:02d}:{int(t / 60) % 60:02d}",
        "edge_densities":     edges,
        "signal_states":      signals,
        "active_incidents":   [],
        "active_green_waves": [],
        "grid_health":        100.0,
        "water_health":       100.0,
        "avg_density":        round(sum(edges.values()) / max(len(edges), 1), 3),
        "active_npcs":        0,
        "_offline":           True,
    }


def _placeholder_snapshot() -> dict:
    return {
        "tick": 0, "sim_time": "00:00", "active_npcs": 0,
        "total_citizens": 0, "emergencies": 0,
        "grid_health": 100.0, "water_health": 100.0,
        "traffic_avg": 0.0, "green_waves": 0, "cv_detections": 0,
        "_offline": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATABASE
# ──────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class CitizenDB(Base):
    __tablename__ = "citizens"
    id              = Column(Integer, primary_key=True, index=True)
    uuid            = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    verified_name   = Column(String, nullable=False)
    aadhaar_hash    = Column(String, unique=True, nullable=False)
    face_hash       = Column(String, unique=True, nullable=False)
    face_image_path = Column(String, nullable=True)
    karma_score     = Column(Float, default=1.0)
    address_node    = Column(String, default="[0,0]")
    age             = Column(Integer, nullable=True)
    gender          = Column(String, nullable=True)
    dob             = Column(String, nullable=True)
    address_text    = Column(String, nullable=True)
    aadhaar_number  = Column(String, nullable=True)
    is_npc          = Column(Boolean, default=False)
    is_verified     = Column(Boolean, default=False)
    profile_img     = Column(String, nullable=True)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class VehicleDB(Base):
    __tablename__ = "vehicles"
    id         = Column(Integer, primary_key=True, index=True)
    vin        = Column(String, unique=True, default=lambda: str(uuid.uuid4()))
    plate_no   = Column(String, unique=True, nullable=False)
    owner_id   = Column(Integer, nullable=False)
    v_color    = Column(String, nullable=True)
    v_type     = Column(String, nullable=True)
    v_model    = Column(String, nullable=True)
    front_img  = Column(String, nullable=True)
    rear_img   = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
class KarmaEventDB(Base):
    """
    Immutable audit log of every karma change for every citizen.
    Each row is one event — never updated, only appended.
    """
    __tablename__ = "karma_events"

    id           = Column(Integer, primary_key=True, index=True)
    citizen_uid  = Column(String, nullable=False, index=True)
    event_type   = Column(String(60), nullable=False)   # see KARMA_EVENTS below
    delta        = Column(Float, nullable=False)         # positive = gain, negative = loss
    score_after  = Column(Float, nullable=False)         # citizen's new score
    source       = Column(String(20), default="SYSTEM")  # CCTV | CITIZEN_REPORT | SYSTEM | ADMIN
    source_uid   = Column(String, nullable=True)         # reporting citizen UID (if source=CITIZEN_REPORT)
    description  = Column(String(500), nullable=True)
    node         = Column(String(30),  nullable=True)    # grid node where event occurred
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)



# ─────────────────────────────────────────────────────────────────────────────
# KARMA ENGINE — constants, helpers, tier logic
# Score range: 0–100  (new citizens start at 50)
# ─────────────────────────────────────────────────────────────────────────────

KARMA_MAX   = 100.0
KARMA_MIN   = 0.0
KARMA_START = 50.0

KARMA_TIER_BAD  = 20.0
KARMA_TIER_GOOD = 70.0

KARMA_EVENTS = {
    # Violations
    "FAKE_REPORT":           -15.0,
    "VERIFIED_FAKE_REPORT":  -25.0,
    "DISTURBING_PUBLIC":      -8.0,
    "MINOR_LAW_BREAK":       -10.0,
    "MAJOR_LAW_BREAK":       -20.0,
    "CRIMINAL_ACT":          -35.0,
    "PUBLIC_INDECENCY":      -12.0,
    "VEHICLE_VIOLATION":      -8.0,
    "INFRASTRUCTURE_DAMAGE": -20.0,
    # Rewards
    "REPORT_VERIFIED":       +10.0,
    "CHHAYA_USAGE":           +3.0,
    "CIVIC_CONTRIBUTION":     +5.0,
    "GOOD_BEHAVIOUR_STREAK":  +2.0,
    "REPORT_RESOLVED":        +5.0,
    "IDENTITY_VERIFIED":      +8.0,
}

KARMA_FACILITIES = [
    (0,   "Basic Transit",        "Bus and metro access"),
    (20,  "Library & Parks",      "Public amenities access"),
    (40,  "Priority Services",    "Fast-track at govt offices"),
    (50,  "Civic Portal",         "Full report + ledger access"),
    (60,  "Community Events",     "Permits for public gatherings"),
    (70,  "Express Lane",         "Priority queue in all city services"),
    (80,  "Civic Ambassador",     "Endorse new citizen registrations"),
    (90,  "City Council Input",   "Participate in digital governance polls"),
]

def karma_tier(score: float) -> str:
    if score >= KARMA_TIER_GOOD: return "GOOD CITIZEN"
    if score >= KARMA_TIER_BAD:  return "NORMAL CITIZEN"
    return "BAD CITIZEN"

def karma_tier_color(score: float) -> str:
    if score >= KARMA_TIER_GOOD: return "GREEN"
    if score >= KARMA_TIER_BAD:  return "AMBER"
    return "RED"

def facilities_for(score: float) -> list:
    return [{"name": n, "desc": d, "unlocked": score >= m}
            for m, n, d in KARMA_FACILITIES]

def _migrate_score(raw) -> float:
    """Convert legacy 0–3 score to 0–100."""
    v = float(raw or KARMA_START)
    return round(v * (100.0 / 3.0), 2) if v <= 3.0 else v

def _apply_karma_delta(db, citizen_uid: str, delta: float,
                        event_type: str, source: str = "SYSTEM",
                        source_uid: str = None, description: str = None,
                        node: str = None) -> dict:
    cit = db.query(CitizenDB).filter(CitizenDB.uuid == citizen_uid).first()
    if not cit:
        raise ValueError(f"Citizen {citizen_uid} not found")
    old_score  = _migrate_score(cit.karma_score)
    new_score  = round(max(KARMA_MIN, min(KARMA_MAX, old_score + delta)), 2)
    cit.karma_score = new_score
    db.add(KarmaEventDB(
        citizen_uid=citizen_uid, event_type=event_type,
        delta=delta, score_after=new_score, source=source,
        source_uid=source_uid, description=description or event_type, node=node,
    ))
    _update_infotxt_karma(citizen_uid, new_score)
    return {"old_score": old_score, "new_score": new_score,
            "delta": delta, "tier": karma_tier(new_score)}

def _update_infotxt_karma(citizen_uid: str, new_score: float) -> None:
    for entry in os.scandir(CITIZENS_DIR):
        if not entry.is_dir(): continue
        info_path = os.path.join(entry.path, "Info", "info.txt")
        if not os.path.exists(info_path): continue
        uid_found = False
        with open(info_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("UID") and ":" in line:
                    if line.partition(":")[2].strip() == citizen_uid:
                        uid_found = True
                    break
        if uid_found:
            new_lines = []
            with open(info_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip().startswith("KARMA_SCORE") and ":" in line:
                        k = line.partition(":")[0].strip()
                        new_lines.append(f"  {k:<16} : {new_score}\n")
                    else:
                        new_lines.append(line)
            with open(info_path, "w", encoding="utf-8") as fh:
                fh.writelines(new_lines)
            return


def init_db():
    Base.metadata.create_all(bind=engine)
    log.info("Database tables created / verified.")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  NPC SEED  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

FIRST_NAMES = ["Arjun","Priya","Rohan","Ananya","Vikram","Kavya","Aditya","Meera",
               "Siddharth","Pooja","Rahul","Divya","Karan","Shreya","Nikhil",
               "Nandini","Ayaan","Ishita","Dhruv","Lavanya","Harsh","Tanvi",
               "Yash","Riya","Arnav"]
LAST_NAMES  = ["Sharma","Iyer","Patel","Singh","Reddy","Nair","Verma","Joshi",
               "Mehta","Gupta","Mishra","Rao","Chatterjee","Malhotra","Pillai",
               "Bose","Agarwal","Kapoor","Tiwari","Shetty","Kumar","Saxena",
               "Bhatt","Desai","Pandey"]
GENDERS        = ["Male","Female","Non-Binary"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]
VEHICLE_CATALOG = [
    ("Car",  "Maruti Swift",       ["White","Red","Grey","Blue","Silver"]),
    ("Car",  "Hyundai Creta",      ["White","Black","Brown","Blue"]),
    ("Car",  "Mahindra Thar",      ["Black","White","Olive Green","Red"]),
    ("Car",  "Tata Nexon",         ["Flame Red","White","Magnetic Grey"]),
    ("Bike", "Honda Activa",       ["White","Grey","Pearl Blue","Black"]),
    ("Bike", "Royal Enfield 350",  ["Black","Green","Red","Blue"]),
    ("Bike", "Bajaj Pulsar 150",   ["Black","Blue","Red"]),
]
RTO_PREFIXES = ["UP32","UP80","DL01","DL10","MH01","MH12","KA01","TN22","RJ14","GJ01"]
ALPHA_POOL   = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def _sha256(v: str) -> str:
    return hashlib.sha256(v.encode()).hexdigest()


def seed_npcs(db: Session, total: int = 100):
    existing = db.execute(text("SELECT COUNT(*) FROM citizens WHERE is_npc=1")).scalar()
    if existing >= total:
        log.info(f"NPC seed already present ({existing} NPCs). Skipping.")
        return
    log.info(f"Seeding {total} NPC citizens …")
    random.seed(42)
    used_names: set[str] = set()
    used_plates: set[str] = set()
    for i in range(total):
        for _ in range(1000):
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            if name not in used_names: used_names.add(name); break
        uid          = str(uuid.uuid4())
        aadhaar_hash = _sha256(f"SYNTH-{random.randint(100_000_000_000, 999_999_999_999)}")
        face_hash    = _sha256(str(uuid.uuid4()))
        gender       = random.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
        citizen = CitizenDB(
            uuid=uid, verified_name=name.upper(), aadhaar_hash=aadhaar_hash,
            face_hash=face_hash, karma_score=round(random.uniform(0.6, 1.0), 4),
            address_node=json.dumps([random.randint(0,19), random.randint(0,19)]),
            age=random.randint(18, 75), gender=gender, is_npc=True, is_verified=True,
            profile_img=f"https://i.pravatar.cc/300?u={uid}&genesis=1",
        )
        db.add(citizen); db.flush()
        if random.random() < 0.70:
            v_type, v_model, colors = random.choice(VEHICLE_CATALOG)
            for _ in range(10_000):
                plate = f"{random.choice(RTO_PREFIXES)}-{''.join(random.choices(ALPHA_POOL,k=2))}-{random.randint(1000,9999)}"
                if plate not in used_plates: used_plates.add(plate); break
            db.add(VehicleDB(vin=str(uuid.uuid4()), plate_no=plate, owner_id=citizen.id,
                             v_color=random.choice(colors), v_type=v_type, v_model=v_model))
    db.commit()
    log.info("NPC seeding complete.")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  REAL CV / OCR PIPELINES  (identical to v3 — unchanged)
# ──────────────────────────────────────────────────────────────────────────────

_ocr_reader = None
_face_cascade = None

_AADHAAR_NOISE = {"government","of","india","भारत","सरकार","unique","identification",
                  "authority","uidai","aadhaar","आधार","enrollment","enrolment","vid",
                  "download","digitally","signed","male","female","transgender","dob",
                  "address","year","birth","date"}
_RE_AADHAAR = re.compile(r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b")
_RE_DOB     = re.compile(r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4}|\d{4}[\/\-]\d{2}[\/\-]\d{2}|\d{2}[\/\-]\d{2}[\/\-]\d{2})\b")
_RE_PLATE   = re.compile(r"\b([A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{4})\b", re.IGNORECASE)


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(xml)
        if _face_cascade.empty():
            raise RuntimeError("Haar cascade XML not found.")
    return _face_cascade


def check_image_quality(img_bytes: bytes) -> dict:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"ok": False, "reason": "Could not decode image."}
    h, w = img.shape[:2]
    if w < 300 or h < 200: return {"ok": False, "reason": f"Image too small ({w}×{h} px)."}
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 30: return {"ok": False, "reason": f"Image too blurry (score: {lap_var:.1f})."}
    return {"ok": True, "reason": None}


def _bytes_to_cv2(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Unsupported image format.")
    return img


def ocr_aadhaar(img_bytes: bytes) -> dict:
    reader  = get_ocr_reader()
    img     = _bytes_to_cv2(img_bytes)
    results = reader.readtext(img, detail=1, paragraph=False)
    if not results:
        raise HTTPException(400, "Aadhaar Card not recognized.")
    raw_tokens = [text.strip() for _, text, _ in results if text.strip()]
    full_text  = " ".join(raw_tokens)
    aadhaar_match = _RE_AADHAAR.search(full_text)
    if not aadhaar_match:
        raise HTTPException(400, "Aadhaar Card not recognized.")
    aadhaar_number = re.sub(r"[\s\-]", "", aadhaar_match.group(1))
    dob = (_RE_DOB.search(full_text) or type('', (), {'group': lambda s, x: None})()).group(1)
    gender = None
    for tok in raw_tokens:
        up = tok.upper().strip()
        if up == "MALE":    gender = "Male";   break
        if up == "FEMALE":  gender = "Female"; break
        if up in ("TRANSGENDER","OTHER"): gender = "Transgender"; break
        if re.search(r"\bMALE\b", up):   gender = "Male";   break
        if re.search(r"\bFEMALE\b", up): gender = "Female"; break
    aadhaar_idx = len(raw_tokens)
    for i, tok in enumerate(raw_tokens):
        if re.sub(r"[\s\-]", "", tok) == aadhaar_number: aadhaar_idx = i; break
    name = None
    for i, token in enumerate(raw_tokens):
        if i >= aadhaar_idx: break
        stripped = token.strip()
        if not stripped or len(stripped) < 3: continue
        if re.search(r"\d", stripped): continue
        if _RE_DOB.search(stripped): continue
        words = stripped.split()
        if any(w.lower() in _AADHAAR_NOISE for w in words): continue
        if len(words) == 1 and words[0].endswith(":"): continue
        if all(re.match(r"^[A-Za-z]+$", w) for w in words): name = stripped.title(); break
    address_tokens = raw_tokens[aadhaar_idx + 1:]
    address = (", ".join(t for t in address_tokens if len(t) > 2 and t.upper() not in ("MALE","FEMALE")) or None)
    found = sum(1 for v in (name, dob, gender, aadhaar_number) if v)
    confidence = "high" if found >= 3 else "medium" if found == 2 else "low"
    return {"name": name, "dob": dob, "gender": gender, "address": address,
            "aadhaar_number": aadhaar_number, "raw_tokens": raw_tokens, "confidence": confidence}


def extract_face_from_aadhaar(img_bytes: bytes) -> Optional[bytes]:
    cascade = get_face_cascade()
    img     = _bytes_to_cv2(img_bytes)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray    = clahe.apply(gray)
    faces   = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(40,40), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0: return None
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    if w < 40 or h < 40: return None
    img_h, img_w = img.shape[:2]
    px, py = int(w*0.20), int(h*0.20)
    x1, y1 = max(0, x-px), max(0, y-py)
    x2, y2 = min(img_w, x+w+px), min(img_h, y+h+py)
    crop = img[y1:y2, x1:x2]
    if cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 20: return None
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def face_to_hash(face_bytes: bytes) -> str:
    return hashlib.sha256(face_bytes).hexdigest()


def ocr_plate(img_bytes: bytes) -> Optional[str]:
    reader    = get_ocr_reader()
    img       = _bytes_to_cv2(img_bytes)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)
    results   = reader.readtext(processed, detail=1, paragraph=False)
    tokens    = [re.sub(r"[^A-Z0-9]", "", t.upper()) for _, t, _ in results if t.strip()]
    for tok in tokens:
        m = _RE_PLATE.fullmatch(tok)
        if m: return re.sub(r"[\s\-]", "", m.group(0)).upper()
    joined = "".join(tokens)
    m = _RE_PLATE.search(joined)
    if m: return re.sub(r"[\s\-]", "", m.group(0)).upper()
    for tok in tokens:
        if 6 <= len(tok) <= 10 and re.search(r"\d{4}$", tok): return tok
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 5.  FASTAPI APP
# ──────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    init_db()
    db = SessionLocal()
    seed_npcs(db)
    db.close()
    # Start simulation in background thread
    sim_thread = threading.Thread(target=_start_simulation, daemon=True, name="OmniCity-Sim")
    sim_thread.start()
    log.info("Trust Gateway v4.2 online → http://0.0.0.0:8000")
    yield
    log.info("Trust Gateway shutting down.")


app = FastAPI(
    title="OmniCity Trust Gateway + Simulation Bridge",
    description="Autonomous Urban Operating System — read-only citizen terminal API",
    version="4.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/database", StaticFiles(directory=str(REGISTRY_ROOT)), name="database")
app.mount("/faces",    StaticFiles(directory=str(FACE_DIR)),      name="faces")
app.mount("/uploads",  StaticFiles(directory=str(UPLOAD_DIR)),    name="uploads")

# ── Serve HTML pages ──────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/index.html", response_class=FileResponse)
def serve_index():
    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/dashboard.html", response_class=FileResponse)
def serve_dashboard():
    return FileResponse(str(BASE_DIR / "dashboard.html"))

@app.get("/cctv_monitor.html", response_class=FileResponse)
def serve_cctv():
    return FileResponse(str(BASE_DIR / "cctv_monitor.html"))

@app.get("/citizen_walker.html", response_class=FileResponse)
def serve_walker():
    return FileResponse(str(BASE_DIR / "citizen_walker.html"))

# ──────────────────────────────────────────────────────────────────────────────
# 6.  NEW: SIMULATION BRIDGE ENDPOINTS  (/api/v1/*)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/live-map")
def api_live_map():
    """
    Primary polling endpoint for the Citizen Terminal's road canvas.
    Returns edge density scores (for colour coding road segments) and
    signal states (for live junction badges) from the running sim.

    Response shape:
        tick, sim_time,
        edge_densities: { "r1,c1-r2,c2": 0.0–1.0, … },
        signal_states:  { "r,c": "RED"|"YELLOW"|"GREEN", … },
        active_incidents: [ IncidentPacket.to_dict(), … ],
        active_green_waves: [ GreenWaveEvent.to_dict(), … ],
        grid_health, water_health, avg_density, active_npcs
    """
    return _sim_live_map()


@app.get("/api/v1/sentinel")
def api_sentinel(limit: int = Query(30, ge=1, le=100)):
    """
    AI Sentinel feed: CV-4/5/6 detections + verified incident packets.
    Read-only — the simulation runs the CV pipelines autonomously.
    """
    return _sim_sentinel(limit)


@app.get("/api/v1/traffic")
def api_traffic():
    """
    ML-1 Route Optimizer output:
      - active_green_waves: emergency corridors currently holding signals GREEN
      - wave_history: last 10 completed green-wave events
      - jam_relief_events: last 10 anti-jam signal extensions by ML-1
      - top_congested: 10 highest-density road segments right now
      - avg_density: city-wide normalised density score
    """
    return _sim_traffic()


@app.get("/api/v1/city-snapshot")
def api_city_snapshot():
    """Full city health snapshot for the Overview tab header stats."""
    return _sim_snapshot()


@app.post("/api/v1/analyze-feed")
async def api_analyze_feed(
    node_row: int = Form(...),
    node_col: int = Form(...),
    image:    UploadFile = File(None),   # optional real camera frame
    night:    bool       = Form(False),  # set True for night-time frames
):
    """
    Trigger a CV pipeline scan on a specific camera node.

    • If an image file is uploaded AND cv_models.py is available, the full
      real CV-1→CV-6→ML-2 pipeline runs and returns a rich incident report.
    • Otherwise, falls back to the simulation's synthetic camera feed.
    """
    node = (node_row, node_col)

    # ── Real image path ───────────────────────────────────────────────────────
    if image is not None and _cv_pipeline is not None:
        raw = await image.read()
        try:
            result = _cv_pipeline.analyze(raw, node=node, night=night)
            # If ML-2 fires a high-priority alert, inject it into the live sim
            if result.get("sentinel_alert") and _sim is not None:
                pkt = result.get("incident_packet")
                if pkt and pkt.get("green_wave_required"):
                    # Trigger an emergency Green Wave from node → nearest edge
                    dest = _sim.skeleton.random_node()
                    _sim.ml1.trigger_emergency_wave(
                        origin=node, destination=dest,
                        vehicle_type=pkt.get("vehicle_type", "AMBULANCE"),
                        tick=_sim.tick, trigger="EMERGENCY",
                    )
                    log.info(
                        "CV-triggered Green Wave: node=%s vtype=%s",
                        node, pkt.get("vehicle_type"),
                    )
            return result
        except Exception as exc:
            log.exception("CVModelPipeline.analyze failed: %s", exc)
            # Fall through to sim fallback below

    # ── Simulation fallback ───────────────────────────────────────────────────
    if _sim is None:
        return {"error": "Simulation offline", "node": [node_row, node_col]}
    return _sim.api.get_camera_feed(node)


@app.get("/api/v1/green-waves")
def api_green_waves():
    """Active and historical Green Wave events from ML-1."""
    if _sim is None:
        return {"active": [], "history": []}
    return {
        "active":  _sim.api.ml1.active_waves(),
        "history": _sim.api.ml1.wave_history(20),
    }


@app.get("/api/v1/utility-credits")
def api_utility_credits():
    """Smart-billing: citizens currently in failure zones + auto-credit totals."""
    if _sim is None:
        return {"failure_zone_size": 0, "citizens_affected": 0, "total_credits_issued": 0, "records": []}
    return _sim.api.compute_utility_credits()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  EXISTING ENDPOINTS  (unchanged from v3 — all kept intact)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    db  = SessionLocal()
    cnt = db.execute(text("SELECT COUNT(*) FROM citizens")).scalar()
    db.close()
    sim_tick = _sim.tick if _sim else None
    return {"status": "online", "citizens_in_db": cnt, "mock_mode": False,
            "simulation_tick": sim_tick, "simulation_online": _sim is not None}


@app.get("/citizens/count")
def citizen_count():
    db    = SessionLocal()
    total = db.execute(text("SELECT COUNT(*) FROM citizens")).scalar()
    human = db.execute(text("SELECT COUNT(*) FROM citizens WHERE is_npc=0")).scalar()
    db.close()
    return {"total": total, "humans": human, "npcs": total - human}


@app.post("/register/verify-aadhaar")
async def verify_aadhaar(file: UploadFile = File(...)):
    raw     = await file.read()
    quality = check_image_quality(raw)
    if not quality["ok"]:
        raise HTTPException(status_code=422, detail=quality["reason"])
    try:
        parsed = ocr_aadhaar(raw)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR pipeline error: {exc}")
    face_bytes = extract_face_from_aadhaar(raw)
    if face_bytes is None:
        raise HTTPException(status_code=422,
            detail="Could not detect a clear face. Ensure the photo side is fully visible.")
    session_token  = str(uuid.uuid4())
    face_path      = FACE_DIR / f"temp_{session_token}.jpg"
    face_path.write_bytes(face_bytes)
    orig_ext       = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    if orig_ext not in (".jpg",".jpeg",".png",".webp"): orig_ext = ".jpg"
    (FACE_DIR / f"temp_aadhaar_{session_token}{orig_ext}").write_bytes(raw)
    return {"session_token": session_token, "extracted": parsed, "face_detected": True,
            "face_preview": f"/faces/temp_{session_token}.jpg",
            "ocr_confidence": parsed["confidence"], "mock_mode": False}


@app.post("/register/verify-vehicle-plate")
async def verify_vehicle_plate(front: UploadFile = File(...), rear: UploadFile = File(None)):
    front_raw = await front.read()
    q = check_image_quality(front_raw)
    if not q["ok"]: raise HTTPException(422, f"Front image: {q['reason']}")
    plate_from_front = ocr_plate(front_raw)
    fid = str(uuid.uuid4())
    (UPLOAD_DIR / f"vehicle_front_{fid}.jpg").write_bytes(front_raw)
    plate_from_rear = None; rear_path_str = None
    if rear:
        rear_raw = await rear.read()
        q2 = check_image_quality(rear_raw)
        if not q2["ok"]: raise HTTPException(422, f"Rear image: {q2['reason']}")
        plate_from_rear = ocr_plate(rear_raw)
        rp = UPLOAD_DIR / f"vehicle_rear_{fid}.jpg"; rp.write_bytes(rear_raw)
        rear_path_str = f"/uploads/vehicle_rear_{fid}.jpg"
    detected = plate_from_front or plate_from_rear
    return {"vehicle_image_id": fid, "front_img": f"/uploads/vehicle_front_{fid}.jpg",
            "rear_img": rear_path_str, "detected_plate": detected,
            "plate_confidence": "high" if detected else "failed", "mock_mode": False}


@app.post("/register/submit")
async def submit_registration(
    session_token: str = Form(...), full_name: str = Form(...),
    age: int = Form(...), gender: str = Form(...),
    dob: str = Form(None), address_text: str = Form(None),
    aadhaar_last4: str = Form(None),
    has_vehicle: bool = Form(False), vehicle_image_id: str = Form(None),
    plate_no_typed: str = Form(None), plate_no_ocr: str = Form(None),
    v_type: str = Form(None), v_model: str = Form(None), v_color: str = Form(None),
):
    db = SessionLocal()
    try:
        face_tmp = FACE_DIR / f"temp_{session_token}.jpg"
        if not face_tmp.exists():
            raise HTTPException(400, "Session expired. Re-upload your Aadhaar.")
        face_bytes   = face_tmp.read_bytes()
        face_hash    = face_to_hash(face_bytes)
        aadhaar_hash = _sha256(f"{session_token}:{aadhaar_last4 or 'XXXX'}")
        aadhaar_tmp: Optional[Path] = None
        for _ext in (".jpg",".jpeg",".png",".webp"):
            c = FACE_DIR / f"temp_aadhaar_{session_token}{_ext}"
            if c.exists(): aadhaar_tmp = c; break
        for dup_filter, field, msg in [
            (CitizenDB.aadhaar_hash == aadhaar_hash, "Aadhaar", "Aadhaar ID"),
            (CitizenDB.face_hash    == face_hash,    "Face",    "face"),
        ]:
            dup = db.query(CitizenDB).filter(dup_filter, CitizenDB.is_npc == False).first()
            if dup:
                raise HTTPException(409, f"🚨 FRAUD DETECTED: This {msg} is already registered.")
        final_plate = None
        if has_vehicle:
            final_plate = (plate_no_typed or plate_no_ocr or "").strip().upper()
            if not final_plate:
                raise HTTPException(422, "Plate number undetermined. Enter manually.")
            if db.query(VehicleDB).filter(VehicleDB.plate_no == final_plate).first():
                raise HTTPException(409, f"🚨 Plate {final_plate} already registered.")
        citizen_uid  = str(uuid.uuid4())
        grid_node    = [random.randint(0,19), random.randint(0,19)]
        safe_name    = re.sub(r'[\\/:*?"<>|]', "", full_name.strip())
        safe_name    = re.sub(r"\s+", "_", safe_name)
        folder_name  = f"{safe_name}_{citizen_uid}"
        citizen_folder = os.path.join(CITIZENS_DIR, folder_name)
        info_folder    = os.path.join(citizen_folder, "Info")
        os.makedirs(info_folder, exist_ok=True)
        info_txt_path = os.path.join(info_folder, "info.txt")
        info_lines = [
            "="*60, "  OMNICITYAI TRUST GATEWAY — CITIZEN DOSSIER", "="*60, "",
            f"  UID              : {citizen_uid}",
            f"  VERIFIED_NAME    : {full_name.strip().upper()}",
            f"  DOB              : {dob or 'NOT PROVIDED'}",
            f"  AGE              : {age}", f"  GENDER           : {gender}",
            f"  ADDRESS          : {address_text or 'NOT PROVIDED'}",
            f"  AADHAAR_LAST4    : {aadhaar_last4 or 'XXXX'}",
            f"  AADHAAR_HASH     : {aadhaar_hash}",
            f"  FACE_HASH        : {face_hash}",
            f"  KARMA_SCORE      : 1.0",
            f"  GRID_NODE        : {json.dumps(grid_node)}",
            f"  IS_NPC           : False",
            f"  REGISTERED_AT    : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "", "="*60,
        ]
        with open(info_txt_path, "w", encoding="utf-8") as f: f.write("\n".join(info_lines))
        face_dest = os.path.join(info_folder, "face_crop.jpg")
        with open(face_dest, "wb") as f: f.write(face_bytes)
        face_tmp.unlink(missing_ok=True)
        aadhaar_dest: Optional[str] = None
        if aadhaar_tmp and aadhaar_tmp.exists():
            aadhaar_dest = os.path.join(info_folder, f"aadhaar_front{aadhaar_tmp.suffix}")
            import shutil as _shutil; _shutil.move(str(aadhaar_tmp), aadhaar_dest)
        vehicle_front_dest = vehicle_rear_dest = None
        if has_vehicle and final_plate:
            vehicle_folder = os.path.join(citizen_folder, "Vehicle")
            os.makedirs(vehicle_folder, exist_ok=True)
            if vehicle_image_id:
                for _ext in (".jpg",".jpeg",".png",".webp"):
                    sf = UPLOAD_DIR / f"vehicle_front_{vehicle_image_id}{_ext}"
                    if sf.exists():
                        vehicle_front_dest = os.path.join(vehicle_folder, f"vehicle_front{_ext}")
                        import shutil as _shutil; _shutil.move(str(sf), vehicle_front_dest); break
                for _ext in (".jpg",".jpeg",".png",".webp"):
                    sr = UPLOAD_DIR / f"vehicle_rear_{vehicle_image_id}{_ext}"
                    if sr.exists():
                        vehicle_rear_dest = os.path.join(vehicle_folder, f"vehicle_rear{_ext}")
                        import shutil as _shutil; _shutil.move(str(sr), vehicle_rear_dest); break
            with open(os.path.join(vehicle_folder, "vehicle_info.txt"), "w") as f:
                f.write("\n".join(["="*60,"  VEHICLE DOSSIER","="*60,"",
                                   f"  PLATE_NO : {final_plate}", f"  OWNER_UID: {citizen_uid}",
                                   f"  V_TYPE   : {v_type or 'N/A'}", f"  V_MODEL  : {v_model or 'N/A'}",
                                   f"  V_COLOR  : {v_color or 'N/A'}", "","="*60]))
        citizen = CitizenDB(
            uuid=citizen_uid, verified_name=full_name.strip().upper(),
            aadhaar_hash=aadhaar_hash, face_hash=face_hash, face_image_path=face_dest,
            karma_score=KARMA_START, address_node=json.dumps(grid_node), age=age, gender=gender,
            dob=dob, address_text=address_text,
            aadhaar_number=f"XXXX-XXXX-{aadhaar_last4}" if aadhaar_last4 else None,
            is_npc=False, is_verified=True, profile_img=face_dest,
        )
        db.add(citizen); db.flush()
        if has_vehicle and final_plate:
            db.add(VehicleDB(vin=str(uuid.uuid4()), plate_no=final_plate, owner_id=citizen.id,
                             v_color=v_color, v_type=v_type, v_model=v_model,
                             front_img=vehicle_front_dest, rear_img=vehicle_rear_dest))
        db.commit()
        total = db.execute(text("SELECT COUNT(*) FROM citizens")).scalar()
        # Refresh CV-8 plate registry
        _refresh_vehicle_plates()
        return JSONResponse(content={
            "status": "REGISTERED", "citizen_uid": citizen_uid, "citizen_number": total,
            "verified_name": full_name.strip().upper(),
            "face_preview": f"/database/citizens/{folder_name}/Info/face_crop.jpg",
            "karma_score": KARMA_START, "grid_node": grid_node, "vehicle_plate": final_plate,
            "registry_folder": citizen_folder,
            "message": (f"Welcome to OmniCity, {full_name.strip().split()[0].title()}. "
                        f"You are Citizen #{total}. Your identity is wired into the city's nervous system."),
        })
    except HTTPException: db.rollback(); raise
    except Exception as exc: db.rollback(); raise HTTPException(500, f"Registration error: {exc}")
    finally: db.close()


@app.post("/auth/login")
async def login(aadhaar_last4: str = Form(...), name: str = Form(...)):
    target_name = name.strip().upper()
    found_info: dict = {}
    for entry in os.scandir(CITIZENS_DIR):
        if not entry.is_dir(): continue
        info_txt = os.path.join(entry.path, "Info", "info.txt")
        if not os.path.exists(info_txt): continue
        parsed: dict = {}
        with open(info_txt, encoding="utf-8") as f:
            for line in f:
                if ":" in line and not line.strip().startswith("="):
                    key, _, val = line.partition(":")
                    parsed[key.strip()] = val.strip()
        if parsed.get("VERIFIED_NAME") == target_name:
            found_info = parsed; break
    if not found_info:
        db = SessionLocal()
        citizen = db.query(CitizenDB).filter(CitizenDB.verified_name == target_name).first()
        db.close()
        if not citizen: raise HTTPException(401, "Citizen not found. Please register first.")
        return {"status": "authenticated", "citizen_uid": citizen.uuid, "name": citizen.verified_name,
                "karma_score": citizen.karma_score, "grid_node": json.loads(citizen.address_node)}
    stored_last4 = found_info.get("AADHAAR_LAST4", "")
    if stored_last4 and stored_last4 != "XXXX":
        if aadhaar_last4.strip() != stored_last4:
            raise HTTPException(401, "Aadhaar last-4 digits do not match.")
    try:    grid_node = json.loads(found_info.get("GRID_NODE", "[0,0]"))
    except: grid_node = [0, 0]
    try:    karma = float(found_info.get("KARMA_SCORE", "1.0"))
    except: karma = 1.0
    return {"status": "authenticated", "citizen_uid": found_info.get("UID",""),
            "name": found_info.get("VERIFIED_NAME", target_name),
            "karma_score": karma, "grid_node": grid_node}


@app.post("/chhaya/toggle")
async def chhaya_toggle(citizen_uid: str = Form(...), activate: bool = Form(...)):
    """Toggle Chhaya escort. On activation, runs AI-1 trust check first."""
    grid_node = [0, 0]; found_name = citizen_uid
    karma = 1.0; false_reports = 0; prev_penalties = 0; verif_level = 1

    # ── Look up citizen ───────────────────────────────────────────────────────
    for entry in os.scandir(CITIZENS_DIR):
        if not entry.is_dir(): continue
        info_txt = os.path.join(entry.path, "Info", "info.txt")
        if not os.path.exists(info_txt): continue
        parsed: dict = {}
        with open(info_txt, encoding="utf-8") as fh:
            for line in fh:
                if ":" in line and not line.strip().startswith("="):
                    key, _, val = line.partition(":")
                    parsed[key.strip()] = val.strip()
        if parsed.get("UID") == citizen_uid:
            found_name = parsed.get("VERIFIED_NAME", citizen_uid)
            try:   grid_node = json.loads(parsed.get("GRID_NODE","[0,0]"))
            except: grid_node = [0, 0]
            try:   karma = float(parsed.get("KARMA_SCORE","1.0"))
            except: karma = 1.0
            break
    else:
        db  = SessionLocal()
        cit = db.query(CitizenDB).filter(CitizenDB.uuid == citizen_uid).first()
        db.close()
        if cit:
            found_name = cit.verified_name
            karma      = float(cit.karma_score or 1.0)
            try:   grid_node = json.loads(cit.address_node)
            except: grid_node = [0, 0]

    # ── AI-1 Trust Check before allowing escort ───────────────────────────────
    # Use karma score directly (already normalised to 0-100 via _migrate_score)
    ks = max(0.0, min(100.0, _migrate_score(karma)))
    penalty = karma_tier(ks)
    trust_result = {"trust_score": round(ks), "penalty": penalty, "fraud_probability": 0.0}
    log.info("AI-1 trust check for %s: score=%.1f penalty=%s", citizen_uid[:8], ks, penalty)

    # Block escort if trust score too low
    if activate and trust_result["penalty"] == "ban":
        return {
            "status": "BLOCKED", "citizen_uid": citizen_uid, "active": False,
            "trust_score": trust_result["trust_score"],
            "message": f"Chhaya escort denied. Trust score {trust_result['trust_score']}/100 is too low (BAN tier). Improve your karma to use this service.",
        }

    # ── Arm simulation tracking ───────────────────────────────────────────────
    if activate and _sim is not None:
        try:
            node = tuple(grid_node)
            log.info("Chhaya ACTIVE: ML-1 tracking node %s for citizen %s", node, citizen_uid[:8])
        except Exception:
            pass

    # ── Karma reward for using Chhaya ─────────────────────────────────────────
    if activate:
        try:
            db = SessionLocal()
            _apply_karma_delta(db, citizen_uid, +2.0, "CHHAYA_USAGE",
                source="SYSTEM", description="Chhaya escort activated", node=str(grid_node))
            db.commit(); db.close()
        except Exception: pass

    status = "ESCORT_ACTIVE" if activate else "STANDBY"
    return {
        "status":        status,
        "citizen_uid":   citizen_uid,
        "name":          found_name,
        "grid_node":     grid_node,
        "active":        activate,
        "trust_score":   trust_result["trust_score"],
        "trust_penalty": trust_result["penalty"],
        "fraud_prob":    trust_result["fraud_probability"],
        "message": (f"Chhaya escort activated. Node {grid_node}. Trust: {trust_result['trust_score']}/100."
                    if activate else "Chhaya escort deactivated."),
    }


@app.post("/chhaya/telemetry")
async def chhaya_telemetry(
    citizen_uid:    str   = Form(...),
    node_row:       int   = Form(...),
    node_col:       int   = Form(...),
    risk_zone:      float = Form(0.2),
    sim_time:       str   = Form("22:00"),
    path_progress:  float = Form(0.5),
    nearby_ids:     str   = Form("[]"),   # JSON list of {"id","distance","angle"}
):
    """
    AI-2 live telemetry endpoint.
    Called every few seconds while a Chhaya escort is active.
    Returns danger assessment and recommended action.
    """
    try:
        tracks = json.loads(nearby_ids)
    except Exception:
        tracks = []

    if not _AI_MODELS_AVAILABLE:
        return {"danger_level": 0, "is_followed": False,
                "recommended_action": "continue", "stalker_id": None,
                "alert_message": "AI-2 unavailable (ai_models.py missing)"}

    payload = build_chhaya_payload(
        user_node=[node_row, node_col],
        cctv_tracks=tracks,
        risk_zone=risk_zone,
        sim_time=sim_time,
        path_progress=path_progress,
    )
    result = process_chhaya_telemetry(payload)

    # If danger is HIGH → inject patrol alert into simulation
    if result["recommended_action"] == "alert_police" and _sim is not None:
        try:
            _sim.ml1.trigger_emergency_wave(
                origin=tuple([node_row, node_col]),
                destination=_sim.skeleton.random_node(),
                vehicle_type="POLICE",
                tick=_sim.tick,
                trigger=f"CHHAYA_ALERT:{citizen_uid[:8]}",
            )
            log.info("Chhaya patrol dispatched for citizen %s at node [%d,%d]",
                     citizen_uid[:8], node_row, node_col)
        except Exception as de:
            log.debug("Chhaya dispatch error: %s", de)

        # Apply karma bonus for being in a dangerous situation (brave citizen)
        try:
            db = SessionLocal()
            _apply_karma_delta(db, citizen_uid, +3.0, "CHHAYA_THREAT_SURVIVED",
                source="SYSTEM",
                description=f"Chhaya: patrol alerted at node [{node_row},{node_col}]",
                node=str([node_row, node_col]))
            db.commit(); db.close()
        except Exception: pass

    return {
        "citizen_uid":        citizen_uid,
        "node":               [node_row, node_col],
        "danger_level":       result["danger_level"],
        "is_followed":        result["is_followed"],
        "recommended_action": result["recommended_action"],
        "stalker_id":         result["stalker_id"],
        "alert_message":      result["alert_message"],
        "sim_time":           sim_time,
    }


@app.get("/citizen/trust-score/{citizen_uid}")
async def get_trust_score(citizen_uid: str):
    """
    AI-1: Compute real-time trust score for a citizen using their karma event history.
    """
    if not _AI_MODELS_AVAILABLE:
        return {"trust_score": 50, "penalty": "warn", "fraud_probability": 0.0,
                "error": "AI-1 model unavailable"}
    try:
        db = SessionLocal()
        cit = db.query(CitizenDB).filter(CitizenDB.uuid == citizen_uid).first()
        events = db.query(KarmaEventDB).filter(
            KarmaEventDB.citizen_uid == citizen_uid
        ).order_by(KarmaEventDB.id.desc()).limit(200).all()
        db.close()

        karma = float(cit.karma_score if cit else 1.0)
        ks    = karma if karma > 3 else karma * 33.3
        tr    = sum(1 for e in events if e.delta and e.delta > 0)
        fr    = sum(1 for e in events if "FAKE" in (e.event_type or ""))
        pp    = sum(1 for e in events if e.delta and e.delta < -5)
        community = max(0, min(10, int(ks / 10)))

        payload = citizen_to_trust_payload(
            true_reports=tr, false_reports=fr,
            verification_level=2 if cit else 0,
            loitering_flags=0, restricted_flags=0,
            community_score=community,
            previous_penalties=pp,
        )
        result = compute_trust_score(payload)
        result["karma_score"]    = round(ks, 1)
        result["true_reports"]   = tr
        result["false_reports"]  = fr
        result["prev_penalties"] = pp
        return result
    except Exception as e:
        log.debug("Trust score error: %s", e)
        return {"trust_score": 50, "penalty": "warn", "fraud_probability": 0.0}



# ─────────────────────────────────────────────────────────────────────────────
# CHHAYA SOS — Emergency Help Dispatch
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chhaya/sos")
async def chhaya_sos(
    citizen_uid: str   = Form(...),
    node_row:    int   = Form(...),
    node_col:    int   = Form(...),
    situation:   str   = Form("EMERGENCY"),   # short description from user
    danger_level:int   = Form(50),
):
    """
    Citizen pressed the SOS/HELP button.
    1. Dispatches POLICE + AMBULANCE via ML-1 Green Wave to the citizen's node.
    2. Calls Claude AI to generate situation-specific survival guidance.
    3. Returns guidance + estimated arrival info.
    """
    import uuid as _uuid
    from datetime import datetime, timezone

    ticket_id = f"SOS-{_uuid.uuid4().hex[:6].upper()}"
    node      = [node_row, node_col]
    timestamp = datetime.now(timezone.utc).isoformat()
    lat       = round(25.317 + node_row * 0.003, 6)
    lon       = round(82.973 + node_col * 0.003, 6)

    log.info("SOS received: citizen=%s node=%s situation=%s danger=%d",
             citizen_uid[:8], node, situation, danger_level)

    # ── 1. Dispatch Police + Ambulance via Green Wave ─────────────────────────
    vehicles_dispatched = []
    eta_ticks           = []

    if _sim is not None:
        for vtype in ["POLICE", "AMBULANCE"]:
            try:
                origin = _sim.skeleton.random_node()   # vehicle spawns from random node
                wave = _sim.ml1.trigger_emergency_wave(
                    origin=tuple(origin),
                    destination=tuple(node),
                    vehicle_type=vtype,
                    tick=_sim.tick,
                    trigger=f"SOS:{ticket_id}",
                )
                # Rough ETA: Manhattan distance * ~2 ticks per node
                dist = abs(origin[0]-node_row) + abs(origin[1]-node_col)
                eta  = max(1, dist * 2)

                # ── Build full route for the canvas ──────────────────────────
                route_path = []
                if wave and wave.path:
                    route_path = [list(n) for n in wave.path]   # full, not truncated
                else:
                    # Fallback: compute BFS path if wave failed
                    try:
                        import networkx as _nx
                        def _em(u, v, d):
                            return d.get("latency", 1) * (1 + d.get("density_score", 0))
                        rp = _nx.dijkstra_path(
                            _sim.api.skeleton.G, tuple(origin), tuple(node), weight=_em
                        )
                        route_path = [list(n) for n in rp]
                    except Exception:
                        pass

                vehicles_dispatched.append({
                    "type":        vtype,
                    "origin_node": list(origin),
                    "eta_ticks":   eta,
                    "eta_seconds": eta * 2,
                    "route_path":  route_path,
                })
                eta_ticks.append(eta)
                log.info("SOS: %s dispatched from %s → %s (ETA ~%d ticks)",
                         vtype, origin, node, eta)
            except Exception as de:
                log.debug("SOS dispatch error %s: %s", vtype, de)
    else:
        # Simulation offline — simulate dispatch with straight-line fallback routes
        p_origin = [node_row+3, node_col+2]
        a_origin = [node_row-2, node_col+4]
        def _straight(o, d):
            path = []
            r, c = o[0], o[1]
            while r != d[0] or c != d[1]:
                if r < d[0]: r += 1
                elif r > d[0]: r -= 1
                elif c < d[1]: c += 1
                elif c > d[1]: c -= 1
                path.append([r, c])
            return path
        vehicles_dispatched = [
            {"type": "POLICE",    "origin_node": p_origin, "eta_ticks": 10, "eta_seconds": 20,
             "route_path": [[p_origin[0], p_origin[1]]] + _straight(p_origin, node)},
            {"type": "AMBULANCE", "origin_node": a_origin, "eta_ticks": 14, "eta_seconds": 28,
             "route_path": [[a_origin[0], a_origin[1]]] + _straight(a_origin, node)},
        ]

    avg_eta_s = int(sum(v["eta_seconds"] for v in vehicles_dispatched) / max(1, len(vehicles_dispatched)))

    # ── 2. AI guidance via Claude ─────────────────────────────────────────────
    guidance_steps = []
    guidance_raw   = ""

    try:
        import json as _json

        prompt = f"""You are the OmniCity AI Emergency Guidance System for a smart Indian city.
A citizen just pressed the SOS button. Give them IMMEDIATE, PRACTICAL survival guidance.

SITUATION: {situation}
DANGER LEVEL: {danger_level}/100
LOCATION: GPS {lat}°N, {lon}°E — Node [{node_row},{node_col}]
DISPATCHED: Police + Ambulance are en route. ETA ~{avg_eta_s} seconds.
TIME: {timestamp[:16]}

Give exactly 5 SHORT numbered steps the citizen should do RIGHT NOW while waiting.
Be specific to their situation. Keep each step under 15 words.
End with a calm reassurance sentence.

Respond ONLY with JSON (no markdown):
{{
  "steps": ["step1","step2","step3","step4","step5"],
  "reassurance": "short calm sentence",
  "severity_assessment": "one sentence about how serious this is",
  "do_not": "one most important thing NOT to do"
}}"""

        _groq_key = "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"
        _groq_headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {_groq_key}",
        }
        _groq_payload = {
            "model":       "llama-3.3-70b-versatile",
            "max_tokens":  600,
            "temperature": 0.2,
            "messages":    [{"role": "user", "content": prompt}],
        }
        if _HTTPX_AVAILABLE:
            import httpx as _httpx_sos
            async with _httpx_sos.AsyncClient(timeout=15.0) as _cl:
                _resp = await _cl.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=_groq_headers, json=_groq_payload,
                )
            data = _resp.json()
        else:
            import aiohttp as _aiohttp
            async with _aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=_groq_headers, json=_groq_payload,
                    timeout=_aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()

        raw   = data["choices"][0]["message"]["content"]
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = _json.loads(clean)
        guidance_steps = parsed.get("steps", [])
        guidance_raw   = parsed

    except Exception as ge:
        log.debug("SOS AI guidance error: %s", ge)
        # Hardcoded fallback guidance by situation type
        sit = situation.lower()
        if "weapon" in sit or "threat" in sit or "armed" in sit:
            guidance_steps = [
                "Stay low and move away from the threat immediately.",
                "Do NOT confront the armed person — evacuate silently.",
                "Lock yourself in a room or building if possible.",
                "Stay on the phone with emergency — keep mic on.",
                "When police arrive, keep hands visible and stay calm.",
            ]
        elif "fire" in sit or "smoke" in sit:
            guidance_steps = [
                "Cover mouth with cloth and stay low below the smoke.",
                "Move toward the nearest exit — do NOT use elevators.",
                "Feel doors before opening — if hot, find another exit.",
                "Once outside, move upwind and stay 50m from the fire.",
                "Do NOT re-enter the building for any reason.",
            ]
        elif "accident" in sit or "medical" in sit or "injur" in sit:
            guidance_steps = [
                "Do NOT move the injured person unless in immediate danger.",
                "Apply pressure to any bleeding wound with clean cloth.",
                "Keep the person warm and conscious — talk to them.",
                "Do not give food or water to unconscious persons.",
                "Clear a path for the ambulance — flag them down.",
            ]
        elif "follow" in sit or "stalk" in sit:
            guidance_steps = [
                "Move into a crowded, well-lit public area immediately.",
                "Enter a shop, restaurant, or any building with people.",
                "Do NOT go home — you will reveal your address.",
                "Call out loudly if the person approaches you.",
                "Stay visible until police arrive — stand near CCTV.",
            ]
        else:
            guidance_steps = [
                "Stay calm — police and ambulance are already en route.",
                "Move to a safe, visible, well-lit location.",
                "Keep your phone battery available — stay reachable.",
                "Do not engage with any threat — preserve yourself.",
                "When help arrives, identify yourself clearly.",
            ]
        guidance_raw = {
            "steps":               guidance_steps,
            "reassurance":         "Help is on the way. You are not alone.",
            "severity_assessment": "Your SOS has been received and units are dispatched.",
            "do_not":              "Do not panic. Focus on each step.",
        }

    # ── 3. Karma: small penalty for false SOS (trust engine will handle it)
    # For now give neutral — real verification happens later
    try:
        db = SessionLocal()
        _apply_karma_delta(db, citizen_uid, +1.0, "SOS_ACTIVATED",
            source="SYSTEM",
            description=f"SOS {ticket_id} at node {node}",
            node=str(node))
        db.commit(); db.close()
    except Exception: pass

    return {
        "ticket_id":           ticket_id,
        "status":              "DISPATCHED",
        "node":                node,
        "gps":                 {"lat": lat, "lon": lon},
        "vehicles_dispatched": vehicles_dispatched,
        "avg_eta_seconds":     avg_eta_s,
        "guidance":            guidance_steps,
        "guidance_full":       guidance_raw,
        "timestamp":           timestamp,
        "message":             f"🚨 SOS RECEIVED — Police + Ambulance dispatched. ETA ~{avg_eta_s}s. Follow the guidance below.",
    }


REPORT_DIR = REGISTRY_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

_CATEGORY_MAP = {
    "GRID_BLACKOUT": "Infra_GridBlackout", "PIPE_BURST": "Infra_PipeBurst",
    "ACCIDENT": "Accident", "ROAD_DEBRIS": "Accident_Debris",
    "THEFT": "Crime_Theft", "WEAPON": "Crime_Weapon",
    "FIRE": "Fire_Smoke", "OTHER": "Other",
}

# ── Authority → vehicle type mapping ─────────────────────────────────────────
_AUTHORITY_VEHICLE = {
    "Police":       "POLICE",
    "EMS":          "AMBULANCE",
    "Fire Brigade": "FIRE",
    "Jal Sansthan": "WATER",
    "UPPCL":        "UTILITY",
    "PWD":          "MAINTENANCE",
}

# ── SLA hours per category ────────────────────────────────────────────────────
_SLA_MAP = {
    "WEAPON": 1, "FIRE": 1, "ACCIDENT": 2,
    "THEFT": 4, "GRID_BLACKOUT": 4, "PIPE_BURST": 6,
    "ROAD_DEBRIS": 12, "OTHER": 24,
}

# ── Karma deltas for report outcomes ─────────────────────────────────────────
_KARMA_VERIFIED_BY_LEVEL  = {"LOW": +5.0,  "MEDIUM": +10.0, "HIGH": +15.0, "CRITICAL": +20.0}
_KARMA_FAKE_BY_SEVERITY   = {"LOW": -10.0, "MEDIUM": -20.0, "HIGH": -30.0}


async def _ai_verify_report(
    category: str,
    description: str,
    photo1_bytes: bytes,
    photo2_bytes: bytes,
    node_arr: list,
) -> dict:
    """
    Send both photos + report metadata to Groq (llama-4-scout vision) for verification.
    Returns a structured verdict dict.
    """
    import base64 as _b64
    import json as _json

    GROQ_API_KEY = "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"
    GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq vision model

    def _encode(raw: bytes) -> str:
        return _b64.b64encode(raw).decode()

    system_prompt = (
        "You are OmniCity AI Verification Engine — an advanced CV/AI system that verifies "
        "citizen incident reports for a smart Indian city platform. "
        "You receive 2 photos uploaded by the citizen and the report metadata. "
        "Your job:\n"
        "1. Determine if the photos ACTUALLY show the claimed incident type (real vs fake).\n"
        "2. Assess the severity level: LOW | MEDIUM | HIGH | CRITICAL.\n"
        "3. Determine which authorities to dispatch (can be multiple): "
        "   Police, EMS, Fire Brigade, Jal Sansthan, UPPCL, PWD.\n"
        "4. Decide if a Green Wave (emergency signal corridor) is needed.\n"
        "5. Generate a professional incident summary for the Public Ledger.\n"
        "6. Assess how realistic/credible the photos are (not AI-generated, not recycled).\n\n"
        "Be STRICT — if photos don't match the claim or look fake/unrelated, mark as FAKE.\n"
        "Respond ONLY with a valid JSON object, no markdown, no preamble."
    )

    json_schema = (
        "{\n"
        '  "verified": true or false,\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "severity": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",\n'
        '  "incident_type": "mapped incident type string",\n'
        '  "fake_reason": null or "reason if fake",\n'
        '  "authorities": ["Police","EMS","Fire Brigade","Jal Sansthan","UPPCL","PWD"],\n'
        '  "green_wave_needed": true or false,\n'
        '  "green_wave_vehicles": ["AMBULANCE","POLICE","FIRE"],\n'
        '  "ledger_summary": "2-3 sentence professional incident summary for public ledger",\n'
        '  "work_status": "STARTED",\n'
        '  "sla_hours": number between 1 and 24,\n'
        '  "karma_verdict": "REWARD" or "PENALTY" or "NEUTRAL",\n'
        '  "karma_reason": "short explanation"\n'
        "}"
    )

    # Groq uses OpenAI-compatible chat format with image_url for vision
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"REPORT METADATA:\n"
                        f"Category: {category}\n"
                        f"Description: {description}\n"
                        f"Grid Node: {node_arr}\n\n"
                        f"The citizen uploaded 2 photos as evidence. Analyze both carefully.\n\n"
                        f"Return ONLY this JSON (no markdown fences):\n{json_schema}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode(photo1_bytes)}",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode(photo2_bytes)}",
                    },
                },
            ],
        },
    ]

    payload = {
        "model":       GROQ_MODEL,
        "max_tokens":  1000,
        "messages":    messages,
        "temperature": 0.1,
    }
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }

    if _HTTPX_AVAILABLE:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload,
            )
        data = resp.json()
    else:
        import aiohttp as _aiohttp
        async with _aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload,
                timeout=_aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()

    if "error" in data:
        raise RuntimeError(f"Groq API error: {data['error'].get('message', str(data['error']))}")

    raw   = data["choices"][0]["message"]["content"]
    # Strip markdown fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.lstrip("```json").lstrip("```").rstrip("```").strip()
    verdict = _json.loads(clean)
    return verdict


async def _dispatch_to_sim(verdict: dict, node_arr: list, ticket_id: str) -> None:
    """
    Inject the verified incident into the live simulation:
    – trigger Green Wave(s) for each emergency vehicle type
    – log dispatch event
    """
    if _sim is None:
        return
    try:
        node = tuple(node_arr)
        if verdict.get("green_wave_needed"):
            for vtype in verdict.get("green_wave_vehicles", []):
                try:
                    dest = _sim.skeleton.random_node()
                    _sim.ml1.trigger_emergency_wave(
                        origin=node,
                        destination=dest,
                        vehicle_type=vtype,
                        tick=_sim.tick,
                        trigger=f"CITIZEN_REPORT:{ticket_id}",
                    )
                    log.info("Green Wave dispatched: vtype=%s node=%s ticket=%s", vtype, node, ticket_id)
                except Exception as gwe:
                    log.debug("Green Wave dispatch error: %s", gwe)
        # Inject incident into sentinel feed
        try:
            inc_type = verdict.get("incident_type", "Other")
            severity = verdict.get("severity", "MEDIUM")
            _sim.incident_engine.inject_citizen_report(
                incident_type=inc_type,
                node=node,
                severity=severity,
                ticket_id=ticket_id,
                authorities=verdict.get("authorities", []),
            )
        except Exception as ie:
            log.debug("Incident injection error: %s", ie)
    except Exception as e:
        log.debug("Dispatch error: %s", e)


@app.post("/report/submit")
async def submit_report(
    citizen_uid: str        = Form(...),
    category:    str        = Form(...),
    description: str        = Form(...),
    grid_node:   str        = Form("null"),
    photo1:      UploadFile = File(...),
    photo2:      UploadFile = File(...),
):
    """
    AI-Verified Incident Report.
    Requires 2 photos. Claude vision verifies authenticity, assigns severity,
    dispatches authorities, triggers Green Waves, and updates citizen karma.
    """
    # ── Basic validation ──────────────────────────────────────────────────────
    if not description.strip():
        raise HTTPException(422, "Description cannot be empty.")
    if not category:
        raise HTTPException(422, "Category is required.")

    # ── Image quality check ───────────────────────────────────────────────────
    p1_bytes = await photo1.read()
    p2_bytes = await photo2.read()
    for label, raw in [("Photo 1", p1_bytes), ("Photo 2", p2_bytes)]:
        q = check_image_quality(raw)
        if not q["ok"]:
            raise HTTPException(422, f"{label}: {q['reason']}")

    mapped_type = _CATEGORY_MAP.get(category.upper(), "Other")
    try:    node_arr = json.loads(grid_node)
    except: node_arr = [0, 0]

    ticket_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    timestamp  = datetime.now(timezone.utc).isoformat()

    # ── Save photos to disk ───────────────────────────────────────────────────
    p1_path = UPLOAD_DIR / f"{ticket_id}_photo1.jpg"
    p2_path = UPLOAD_DIR / f"{ticket_id}_photo2.jpg"
    p1_path.write_bytes(p1_bytes)
    p2_path.write_bytes(p2_bytes)

    # ── AI Verification ───────────────────────────────────────────────────────
    try:
        verdict = await _ai_verify_report(
            category=category, description=description.strip(),
            photo1_bytes=p1_bytes, photo2_bytes=p2_bytes, node_arr=node_arr,
        )
    except Exception as ve:
        log.exception("AI verification failed: %s", ve)
        # Fail gracefully — mark as PENDING_MANUAL_REVIEW
        err_msg = str(ve)
        log.error("AI verification failed: %s", err_msg)
        raise HTTPException(503, detail=f"AI verification failed: {err_msg}")

    verified  = verdict.get("verified", False)
    severity  = verdict.get("severity", "LOW")
    sla_hours = verdict.get("sla_hours", _SLA_MAP.get(category.upper(), 24))

    # ── Write ticket to ledger ────────────────────────────────────────────────
    ticket = {
        "ticket_id":       ticket_id,
        "citizen_uid":     citizen_uid,
        "category":        category,
        "incident_type":   verdict.get("incident_type", mapped_type),
        "description":     description.strip(),
        "ledger_summary":  verdict.get("ledger_summary", description.strip()),
        "grid_node":       node_arr,
        "status":          "OPEN" if verified else "REJECTED",
        "work_status":     verdict.get("work_status", "STARTED") if verified else "REJECTED",
        "priority":        severity,
        "severity":        severity,
        "verified":        verified,
        "ai_confidence":   verdict.get("confidence", 0.0),
        "fake_reason":     verdict.get("fake_reason"),
        "authorities":     verdict.get("authorities", []),
        "green_wave":      verdict.get("green_wave_needed", False),
        "sla_hours":       sla_hours,
        "sla_deadline":    timestamp,
        "timestamp_utc":   timestamp,
        "photo1_url":      f"/uploads/{ticket_id}_photo1.jpg",
        "photo2_url":      f"/uploads/{ticket_id}_photo2.jpg",
        "karma_verdict":   verdict.get("karma_verdict", "NEUTRAL"),
        "karma_reason":    verdict.get("karma_reason", ""),
    }
    (REPORT_DIR / f"{ticket_id}.json").write_text(json.dumps(ticket, indent=2), encoding="utf-8")

    # ── Apply Karma ───────────────────────────────────────────────────────────
    karma_delta = 0.0
    karma_event_type = "CIVIC_CONTRIBUTION"
    if verdict.get("karma_verdict") == "REWARD" and verified:
        karma_delta      = _KARMA_VERIFIED_BY_LEVEL.get(severity, +10.0)
        karma_event_type = "REPORT_VERIFIED"
    elif verdict.get("karma_verdict") == "PENALTY" or not verified:
        karma_delta      = _KARMA_FAKE_BY_SEVERITY.get(severity, -15.0)
        karma_event_type = "VERIFIED_FAKE_REPORT" if not verified else "FAKE_REPORT"

    if karma_delta != 0.0:
        try:
            db = SessionLocal()
            _apply_karma_delta(
                db, citizen_uid, karma_delta, karma_event_type,
                source="SYSTEM",
                description=f"Report {ticket_id}: {verdict.get('karma_reason','AI verdict')}",
                node=str(node_arr),
            )
            db.commit()
            db.close()
        except Exception as ke:
            log.debug("Karma update failed: %s", ke)

    # ── Dispatch to simulation (non-blocking) ─────────────────────────────────
    if verified:
        import asyncio
        asyncio.create_task(_dispatch_to_sim(verdict, node_arr, ticket_id))

    log.info("Report %s: verified=%s severity=%s authorities=%s karma_delta=%+.1f",
             ticket_id, verified, severity, verdict.get("authorities",[]), karma_delta)

    return {
        "status":          "VERIFIED" if verified else "REJECTED",
        "ticket_id":       ticket_id,
        "verified":        verified,
        "incident_type":   verdict.get("incident_type", mapped_type),
        "severity":        severity,
        "ai_confidence":   verdict.get("confidence", 0.0),
        "authorities":     verdict.get("authorities", []),
        "green_wave":      verdict.get("green_wave_needed", False),
        "ledger_summary":  verdict.get("ledger_summary", ""),
        "work_status":     ticket["work_status"],
        "karma_delta":     karma_delta,
        "karma_verdict":   verdict.get("karma_verdict", "NEUTRAL"),
        "karma_reason":    verdict.get("karma_reason", ""),
        "sla_hours":       sla_hours,
        "timestamp":       timestamp,
        "fake_reason":     verdict.get("fake_reason"),
        "message": (
            f"✅ Report {ticket_id} VERIFIED — {', '.join(verdict.get('authorities',[]))} dispatched. "
            f"Karma: {karma_delta:+.1f}"
            if verified else
            f"❌ Report {ticket_id} REJECTED — {verdict.get('fake_reason','Photos do not match claim.')} "
            f"Karma: {karma_delta:+.1f}"
        ),
    }


@app.patch("/report/update-status/{ticket_id}")
async def update_report_status(ticket_id: str, work_status: str = Form(...)):
    """
    Update the work_status of a report: STARTED → IN_PROGRESS → DONE.
    Called by the simulation or admin; DONE tickets are auto-removed from ledger after 60s.
    """
    path = REPORT_DIR / f"{ticket_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Ticket {ticket_id} not found.")
    ticket = json.loads(path.read_text(encoding="utf-8"))
    ticket["work_status"] = work_status
    if work_status == "DONE":
        ticket["status"]    = "CLOSED"
        ticket["closed_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")
    return {"status": "OK", "ticket_id": ticket_id, "work_status": work_status}


@app.get("/report/list")
def list_reports(limit: int = 10):
    files = sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    now = datetime.now(timezone.utc)
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            # Compute SLA remaining / breached
            try:
                filed_at  = datetime.fromisoformat(data.get("timestamp_utc",""))
                sla_h     = data.get("sla_hours", 24)
                elapsed_h = (now - filed_at).total_seconds() / 3600
                remaining = sla_h - elapsed_h
                data["sla_breached"]  = remaining < 0
                data["sla_remaining"] = f"{max(0,int(remaining))}h {int((remaining%1)*60):02d}m" if remaining > 0 else "BREACHED"
            except Exception:
                data["sla_breached"]  = False
                data["sla_remaining"] = "—"
            results.append(data)
        except Exception:
            pass
    return {"reports": results, "total": len(list(REPORT_DIR.glob("*.json")))}



@app.get("/report/my/{citizen_uid}")
def my_reports(citizen_uid: str, limit: int = 20):
    """Return only this citizen's reports, newest first."""
    files = sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    now = datetime.now(timezone.utc)
    for f in files:
        if len(results) >= limit:
            break
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("citizen_uid") != citizen_uid:
                continue
            try:
                filed_at  = datetime.fromisoformat(data.get("timestamp_utc", ""))
                sla_h     = data.get("sla_hours", 24)
                remaining = sla_h - (now - filed_at).total_seconds() / 3600
                data["sla_breached"]  = remaining < 0
                data["sla_remaining"] = f"{max(0,int(remaining))}h {int((remaining%1)*60):02d}m" if remaining > 0 else "BREACHED"
            except Exception:
                data["sla_breached"]  = False
                data["sla_remaining"] = "—"
            results.append(data)
        except Exception:
            pass
    return {"reports": results, "total": len(results)}

# ──────────────────────────────────────────────────────────────────────────────
# CITIZEN PROFILE  —  /citizen/profile/{uid}
# Returns full citizen record + vehicle + karma history for the profile card
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/citizen/profile/{citizen_uid}")
def citizen_profile(citizen_uid: str):
    """
    Full citizen profile: demographics, vehicle, karma breakdown, report stats.
    Reads from the physical registry folder first (source of truth),
    falls back to SQLite for any NPCs or edge cases.
    """
    profile: dict = {}

    # ── 1. Scan physical registry ─────────────────────────────────────────────
    for entry in os.scandir(CITIZENS_DIR):
        if not entry.is_dir(): continue
        info_txt = os.path.join(entry.path, "Info", "info.txt")
        if not os.path.exists(info_txt): continue
        parsed: dict = {}
        with open(info_txt, encoding="utf-8") as fh:
            for line in fh:
                if ":" in line and not line.strip().startswith("="):
                    k, _, v = line.partition(":")
                    parsed[k.strip()] = v.strip()
        if parsed.get("UID") == citizen_uid:
            profile["citizen_uid"]   = citizen_uid
            profile["name"]          = parsed.get("VERIFIED_NAME", "—")
            profile["age"]           = parsed.get("AGE", "—")
            profile["gender"]        = parsed.get("GENDER", "—")
            profile["dob"]           = parsed.get("DOB", "—")
            profile["address"]       = parsed.get("ADDRESS", "—")
            profile["aadhaar_last4"] = parsed.get("AADHAAR_NUMBER","").replace("X","●").strip()
            profile["grid_node"]     = parsed.get("GRID_NODE", "[0,0]")
            try:
                profile["karma_score"] = float(parsed.get("KARMA_SCORE", "1.0"))
            except ValueError:
                profile["karma_score"] = 1.0
            # Face image — try registry path
            face_path = os.path.join(entry.path, "Info", "face_crop.jpg")
            if os.path.exists(face_path):
                safe = entry.name
                profile["face_url"] = f"/database/citizens/{safe}/Info/face_crop.jpg"
            else:
                profile["face_url"] = None
            break

    # ── 2. SQLite fallback ────────────────────────────────────────────────────
    if not profile:
        db  = SessionLocal()
        cit = db.query(CitizenDB).filter(CitizenDB.uuid == citizen_uid).first()
        if cit:
            profile["citizen_uid"]   = citizen_uid
            profile["name"]          = cit.verified_name
            profile["age"]           = str(cit.age or "—")
            profile["gender"]        = cit.gender or "—"
            profile["dob"]           = cit.dob or "—"
            profile["address"]       = cit.address_text or "—"
            profile["aadhaar_last4"] = cit.aadhaar_number or "—"
            profile["grid_node"]     = cit.address_node or "[0,0]"
            profile["karma_score"]   = float(cit.karma_score or 1.0)
            profile["face_url"]      = cit.face_image_path
        db.close()

    if not profile:
        raise HTTPException(status_code=404, detail="Citizen not found.")

    # ── 3. Vehicle info ───────────────────────────────────────────────────────
    db       = SessionLocal()
    vehicle  = db.query(VehicleDB).filter(VehicleDB.owner_id ==
               db.query(CitizenDB.id).filter(CitizenDB.uuid == citizen_uid).scalar_subquery()
               ).first()
    db.close()

    profile["vehicle"] = {
        "plate":  vehicle.plate_no  if vehicle else None,
        "model":  vehicle.v_model   if vehicle else None,
        "type":   vehicle.v_type    if vehicle else None,
        "color":  vehicle.v_color   if vehicle else None,
    } if vehicle else None

    # ── 4. Report statistics (karma breakdown) ────────────────────────────────
    all_reports   = list(REPORT_DIR.glob("*.json"))
    citizen_rpts  = []
    for f in all_reports:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("citizen_uid") == citizen_uid:
                citizen_rpts.append(data)
        except Exception:
            pass

    open_count   = sum(1 for r in citizen_rpts if r.get("status") == "OPEN")
    closed_count = sum(1 for r in citizen_rpts if r.get("status") == "CLOSED")
    karma        = profile["karma_score"]

    # ── Migrate legacy 0–3 scores to 0–100 ─────────────────────────────────────
    if karma <= 3.0:
        karma = round(karma * (100.0 / 3.0), 2)
        profile["karma_score"] = karma

    # ── Karma event history for this citizen ─────────────────────────────────
    db2 = SessionLocal()
    events = (db2.query(KarmaEventDB)
                 .filter(KarmaEventDB.citizen_uid == citizen_uid)
                 .order_by(KarmaEventDB.created_at.desc())
                 .limit(20).all())
    db2.close()

    total_gains    = sum(e.delta for e in events if e.delta > 0)
    total_losses   = abs(sum(e.delta for e in events if e.delta < 0))
    fake_reports   = sum(1 for e in events if "FAKE" in e.event_type)
    verified_rpts  = sum(1 for e in events if e.event_type == "REPORT_VERIFIED")
    chhaya_uses    = sum(1 for e in events if e.event_type == "CHHAYA_USAGE")

    unlocked_count = sum(1 for min_s,_,_ in KARMA_FACILITIES if karma >= min_s)

    profile["stats"] = {
        "reports_total":    len(citizen_rpts),
        "reports_open":     open_count,
        "reports_closed":   closed_count,
        "karma_score":      karma,
        "karma_tier":       karma_tier(karma),
        "karma_tier_color": karma_tier_color(karma),
        "rewards_unlocked": unlocked_count,
        "facilities":       facilities_for(karma),
        # Breakdown for progress bars (0–100 scale each capped at sensible max)
        "karma_gains":      round(min(total_gains,  100), 2),
        "karma_losses":     round(min(total_losses, 100), 2),
        "karma_verified":   round(min(verified_rpts * 10, 50), 2),
        "karma_reports":    round(min(len(citizen_rpts) * 5, 40), 2),
        "karma_chhaya":     round(min(chhaya_uses * 3, 20), 2),
        "karma_penalty":    round(min(total_losses, 50), 2),
        "fake_reports":     fake_reports,
        # Recent event log (last 10)
        "recent_events": [
            {"event_type": e.event_type, "delta": e.delta,
             "score_after": e.score_after, "source": e.source,
             "description": e.description,
             "created_at": e.created_at.isoformat()}
            for e in events[:10]
        ],
    }

    return profile


# ──────────────────────────────────────────────────────────────────────────────
# KARMA ENGINE ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/karma/event")
async def karma_event(
    citizen_uid: str  = Form(...),
    event_type:  str  = Form(...),   # must be a key in KARMA_EVENTS
    source:      str  = Form("SYSTEM"),  # CCTV | CITIZEN_REPORT | SYSTEM | ADMIN
    source_uid:  str  = Form(None),   # UID of reporting citizen (if applicable)
    description: str  = Form(None),
    node:        str  = Form(None),
    custom_delta:float= Form(None),   # optional override (ADMIN only)
):
    """
    Apply a karma delta to a citizen.
    Sources:
      CCTV            — AI Sentinel detected a violation/positive act
      CITIZEN_REPORT  — another verified citizen filed a corroborated report
      SYSTEM          — automated (e.g. fake-report detection, streak bonus)
      ADMIN           — manual override

    If source=CITIZEN_REPORT, the reporting citizen (source_uid) must exist
    and must have karma >= 50 (only trusted citizens can affect others).
    """
    # Validate event type
    if event_type not in KARMA_EVENTS and custom_delta is None:
        raise HTTPException(422, f"Unknown event_type '{event_type}'. "
                                 f"Valid types: {list(KARMA_EVENTS.keys())}")

    delta = custom_delta if custom_delta is not None else KARMA_EVENTS[event_type]

    # Verify the reporting citizen has enough karma to file an accusation
    if source == "CITIZEN_REPORT" and source_uid:
        db_check = SessionLocal()
        reporter = db_check.query(CitizenDB).filter(CitizenDB.uuid == source_uid).first()
        db_check.close()
        if reporter:
            reporter_karma = float(reporter.karma_score or KARMA_START)
            if reporter_karma <= 3.0:   # migrate
                reporter_karma *= (100.0 / 3.0)
            if reporter_karma < 50.0:
                raise HTTPException(403,
                    "Reporting citizen's karma is too low to submit verified reports. "
                    "Minimum karma 50 required.")

    db = SessionLocal()
    try:
        result = _apply_karma_delta(
            db, citizen_uid, delta, event_type,
            source=source, source_uid=source_uid,
            description=description, node=node,
        )
        db.commit()
        new_score = result["new_score"]
        log.info("Karma event: %s | citizen=%s | delta=%.1f | score=%.1f | tier=%s",
                 event_type, citizen_uid[:8], delta, new_score, result["tier"])
        return {
            "status":       "OK",
            "citizen_uid":  citizen_uid,
            "event_type":   event_type,
            "delta":        delta,
            "old_score":    result["old_score"],
            "new_score":    new_score,
            "tier":         result["tier"],
            "tier_color":   karma_tier_color(new_score),
            "facilities":   facilities_for(new_score),
            "message":      f"Karma updated: {result['old_score']:.1f} → {new_score:.1f} ({delta:+.1f})",
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception:
        db.rollback()
        log.exception("Karma event failed")
        raise HTTPException(500, "Karma update failed.")
    finally:
        db.close()


@app.get("/karma/history/{citizen_uid}")
def karma_history(citizen_uid: str, limit: int = 30):
    """Full karma event log for a citizen — most recent first."""
    db = SessionLocal()
    events = (db.query(KarmaEventDB)
                .filter(KarmaEventDB.citizen_uid == citizen_uid)
                .order_by(KarmaEventDB.created_at.desc())
                .limit(min(limit, 100)).all())
    db.close()
    return {
        "citizen_uid": citizen_uid,
        "events": [
            {"event_type":  e.event_type,
             "delta":       e.delta,
             "score_after": e.score_after,
             "source":      e.source,
             "source_uid":  e.source_uid,
             "description": e.description,
             "node":        e.node,
             "created_at":  e.created_at.isoformat()}
            for e in events
        ],
        "total": len(events),
    }


@app.get("/karma/leaderboard")
def karma_leaderboard(limit: int = 20):
    """Top/bottom citizens by karma score (excludes NPCs)."""
    db = SessionLocal()
    top = (db.query(CitizenDB)
             .filter(CitizenDB.is_npc == False)
             .order_by(CitizenDB.karma_score.desc())
             .limit(limit).all())
    db.close()
    return {
        "leaderboard": [
            {"rank":         i + 1,
             "citizen_uid":  c.uuid,
             "name":         c.verified_name,
             "karma_score":  round(float(c.karma_score or KARMA_START) *
                                   (100.0/3.0 if float(c.karma_score or 1) <= 3.0 else 1.0), 1),
             "tier":         karma_tier(float(c.karma_score or KARMA_START) *
                                        (100.0/3.0 if float(c.karma_score or 1) <= 3.0 else 1.0))}
            for i, c in enumerate(top)
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
