"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OmniCity AI  —  ai_models.py                                              ║
║   AI-1 Citizen Trust Engine  +  AI-2 Chhaya Escort Tracker  v1.0.0         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Exact port of the two notebooks into production-ready Python modules.      ║
║                                                                              ║
║  AI-1  RandomForestRegressor  Citizen Trust Score (0–100)                  ║
║        → fraud_probability, penalty tier, karma recommendations             ║
║                                                                              ║
║  AI-2  RandomForestRegressor  Chhaya Stalking / Danger Detection            ║
║        → is_followed, danger_level (0–100), recommended_action             ║
║                                                                              ║
║  Both models train once on 10 000 synthetic samples at import time          ║
║  and are then cached as module-level singletons for zero-latency inference. ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger("ai_models")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# AI-1 — CITIZEN TRUST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_AI1_MODEL: Optional[RandomForestRegressor] = None
_AI1_COLS:  List[str] = [
    "true_reports", "false_reports", "verification_level",
    "loitering_flags", "restricted_flags", "community_score", "previous_penalties",
]


def _train_ai1() -> RandomForestRegressor:
    """
    Train AI-1 on 10 000 synthetic citizen profiles exactly as in the notebook.
    Returns fitted RandomForestRegressor.
    """
    log.info("AI-1  Generating 10 000 synthetic citizen logs …")
    np.random.seed(42)
    n = 10_000

    data = {
        "true_reports":       np.random.randint(0, 50, n),
        "false_reports":      np.random.randint(0,  10, n),
        "verification_level": np.random.randint(0,   3, n),
        "loitering_flags":    np.random.randint(0,   5, n),
        "restricted_flags":   np.random.randint(0,   3, n),
        "community_score":    np.random.randint(0,  10, n),
        "previous_penalties": np.random.randint(0,   5, n),
    }
    df = pd.DataFrame(data)

    base  = 50 + (df["true_reports"] * 0.5) + \
                 (df["verification_level"] * 5) + \
                 (df["community_score"] * 2.5)
    pens  = (df["false_reports"]      * 4)  + \
            (df["loitering_flags"]    * 2)  + \
            (df["restricted_flags"]   * 8)  + \
            (df["previous_penalties"] * 15)

    df["target"] = np.clip(base - pens, 0, 100)

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(df[_AI1_COLS], df["target"])
    log.info("AI-1  Trust Model trained ✓")
    return model


def _load_ai1() -> RandomForestRegressor:
    global _AI1_MODEL
    if _AI1_MODEL is None:
        _AI1_MODEL = _train_ai1()
    return _AI1_MODEL


# ── Public API ────────────────────────────────────────────────────────────────

def compute_trust_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI-1 inference.  Accepts the nested JSON payload from the backend:

    {
      "incident_history":   {"true_reports": int, "false_reports": int},
      "verification_level": int,          # 0=None 1=Face 2=Full
      "behaviour_flags":    {"loitering": int, "restricted_area_attempt": int},
      "community_score":    int,
      "previous_penalties": int
    }

    Returns:
    {
      "trust_score":       int   0–100,
      "penalty":           str   "none"|"warn"|"restrict"|"ban",
      "fraud_probability": float 0.0–1.0
    }
    """
    model = _load_ai1()

    features = pd.DataFrame([{
        "true_reports":       payload["incident_history"]["true_reports"],
        "false_reports":      payload["incident_history"]["false_reports"],
        "verification_level": payload["verification_level"],
        "loitering_flags":    payload["behaviour_flags"]["loitering"],
        "restricted_flags":   payload["behaviour_flags"]["restricted_area_attempt"],
        "community_score":    payload["community_score"],
        "previous_penalties": payload["previous_penalties"],
    }])

    raw_score   = model.predict(features)[0]
    trust_score = int(round(float(np.clip(raw_score, 0, 100))))

    # Penalty tier (identical to notebook)
    if trust_score >= 75:
        penalty = "none"
    elif trust_score >= 50:
        penalty = "warn"
    elif trust_score >= 30:
        penalty = "restrict"
    else:
        penalty = "ban"

    # Fraud probability (identical to notebook formula)
    false_r   = payload["incident_history"]["false_reports"]
    verif_lvl = payload["verification_level"]
    prev_pen  = payload["previous_penalties"]

    base_fraud       = false_r * 0.15
    unverif_pen      = 0.25 if verif_lvl == 0 else 0.0
    history_pen      = prev_pen * 0.10
    fraud_prob       = min(1.0, base_fraud + unverif_pen + history_pen)

    return {
        "trust_score":       trust_score,
        "penalty":           penalty,
        "fraud_probability": round(float(fraud_prob), 2),
    }


def citizen_to_trust_payload(
    true_reports:        int   = 0,
    false_reports:       int   = 0,
    verification_level:  int   = 0,
    loitering_flags:     int   = 0,
    restricted_flags:    int   = 0,
    community_score:     int   = 5,
    previous_penalties:  int   = 0,
) -> Dict[str, Any]:
    """Helper: build the nested JSON payload from flat DB fields."""
    return {
        "incident_history":   {"true_reports": true_reports, "false_reports": false_reports},
        "verification_level": verification_level,
        "behaviour_flags":    {"loitering": loitering_flags, "restricted_area_attempt": restricted_flags},
        "community_score":    community_score,
        "previous_penalties": previous_penalties,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI-2 — CHHAYA ESCORT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

_AI2_MODEL: Optional[RandomForestRegressor] = None
_AI2_COLS:  List[str] = [
    "nearest_person_distance", "nearest_person_angle",
    "risk_zone", "is_night",
]


def _train_ai2() -> RandomForestRegressor:
    """
    Train AI-2 on 10 000 synthetic night-route samples exactly as in the notebook.
    Returns fitted RandomForestRegressor.
    """
    log.info("AI-2  Generating 10 000 simulated night routes …")
    np.random.seed(42)
    n = 10_000

    data = {
        "nearest_person_distance": np.random.uniform(0.5, 50.0, n),
        "nearest_person_angle":    np.random.uniform(0,  180.0, n),
        "risk_zone":               np.random.uniform(0.0,  1.0, n),
        "is_night":                np.random.choice([1, 1, 0],   n),
    }
    df = pd.DataFrame(data)

    def _true_danger(row: pd.Series) -> float:
        danger = 0.0
        if   row["nearest_person_distance"] < 5.0:  danger += 50
        elif row["nearest_person_distance"] < 15.0: danger += 25
        if row["nearest_person_angle"] < 15.0:      danger += 30
        danger += row["risk_zone"] * 20
        if row["is_night"] == 1:                    danger += 10
        return min(100.0, danger)

    df["target"] = df.apply(_true_danger, axis=1)

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(df[_AI2_COLS], df["target"])
    log.info("AI-2  Chhaya Danger Model trained ✓")
    return model


def _load_ai2() -> RandomForestRegressor:
    global _AI2_MODEL
    if _AI2_MODEL is None:
        _AI2_MODEL = _train_ai2()
    return _AI2_MODEL


# ── Public API ────────────────────────────────────────────────────────────────

def process_chhaya_telemetry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI-2 inference.  Accepts live telemetry JSON from the backend:

    {
      "user_position":          [lat, lon],
      "expected_path_progress": 0.0–1.0,
      "cctv_tracks": [
        {"id": "P123", "distance": 4.2, "angle": 12},
        ...
      ],
      "risk_zone": 0.0–1.0,
      "time":      "HH:MM"
    }

    Returns:
    {
      "is_followed":         bool,
      "danger_level":        int   0–100,
      "recommended_action":  str   "continue"|"reroute"|"alert_police",
      "stalker_id":          str | None,
      "alert_message":       str
    }
    """
    model = _load_ai2()

    # Parse time → is_night
    try:
        hour     = int(str(payload.get("time", "12:00")).split(":")[0])
        is_night = 1 if (hour >= 19 or hour <= 5) else 0
    except Exception:
        is_night = 0

    # Find nearest / most threatening person from CCTV tracks
    nearest_dist  = 100.0
    nearest_angle = 180.0
    stalker_id    = None

    for track in payload.get("cctv_tracks", []):
        d = float(track.get("distance", 100))
        if d < nearest_dist:
            nearest_dist  = d
            nearest_angle = float(track.get("angle", 180))
            stalker_id    = track.get("id")

    features = pd.DataFrame([{
        "nearest_person_distance": nearest_dist,
        "nearest_person_angle":    nearest_angle,
        "risk_zone":               float(payload.get("risk_zone", 0.0)),
        "is_night":                is_night,
    }])

    raw_danger   = model.predict(features)[0]
    danger_level = int(round(float(np.clip(raw_danger, 0, 100))))

    # Action thresholds (identical to notebook)
    if danger_level > 60:
        action      = "alert_police"
        is_followed = True
        alert_msg   = f"⚠ THREAT DETECTED — Patrol alerted. Stay in lit areas."
    elif danger_level >= 30:
        action      = "reroute"
        is_followed = True
        alert_msg   = f"⚡ Suspicious proximity. AI recommends rerouting."
    else:
        action      = "continue"
        is_followed = False
        stalker_id  = None
        alert_msg   = "✓ Route clear. No threats detected."

    return {
        "is_followed":        is_followed,
        "danger_level":       danger_level,
        "recommended_action": action,
        "stalker_id":         stalker_id if is_followed else None,
        "alert_message":      alert_msg,
    }


def build_chhaya_payload(
    user_node:     list,
    cctv_tracks:   Optional[List[Dict]] = None,
    risk_zone:     float = 0.2,
    sim_time:      str   = "22:00",
    path_progress: float = 0.5,
) -> Dict[str, Any]:
    """Helper: build a telemetry payload from sim data."""
    lat = 25.317 + user_node[0] * 0.003
    lon = 82.973 + user_node[1] * 0.003
    return {
        "user_position":          [round(lat, 6), round(lon, 6)],
        "expected_path_progress": path_progress,
        "cctv_tracks":            cctv_tracks or [],
        "risk_zone":              risk_zone,
        "time":                   sim_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WARM-UP (called once at backend startup)
# ─────────────────────────────────────────────────────────────────────────────

def warmup() -> None:
    """Pre-train both models so the first API call is instant."""
    _load_ai1()
    _load_ai2()
    log.info("ai_models  Both AI-1 and AI-2 warmed up ✓")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  OmniCity AI — ai_models.py  SELF-TEST")
    print("="*65)

    # AI-1 test (from notebook)
    print("\n[AI-1] Trust Engine …")
    payload1 = {
        "incident_history":   {"true_reports": 12, "false_reports": 1},
        "verification_level": 2,
        "behaviour_flags":    {"loitering": 0, "restricted_area_attempt": 0},
        "community_score":    3,
        "previous_penalties": 0,
    }
    out1 = compute_trust_score(payload1)
    print(f"  trust_score      : {out1['trust_score']}")
    print(f"  penalty          : {out1['penalty']}")
    print(f"  fraud_probability: {out1['fraud_probability']}")
    assert out1["penalty"] in ("none","warn","restrict","ban"), "Bad penalty"
    print("  ✓ AI-1 OK")

    # AI-2 test (from notebook — should return alert_police)
    print("\n[AI-2] Chhaya Tracker …")
    payload2 = {
        "user_position":          [25.317, 82.973],
        "expected_path_progress": 0.76,
        "cctv_tracks": [
            {"id": "P123", "distance": 4.2,  "angle": 12},
            {"id": "P991", "distance": 8.0,  "angle": 15},
        ],
        "risk_zone": 0.64,
        "time": "22:14",
    }
    out2 = process_chhaya_telemetry(payload2)
    print(f"  is_followed      : {out2['is_followed']}")
    print(f"  danger_level     : {out2['danger_level']}")
    print(f"  recommended_action: {out2['recommended_action']}")
    print(f"  stalker_id       : {out2['stalker_id']}")
    assert out2["is_followed"] == True, "Should detect stalking"
    assert out2["recommended_action"] == "alert_police", "Should alert police"
    print("  ✓ AI-2 OK")

    print("\n" + "="*65)
    print("  All self-tests passed!")
    print("="*65 + "\n")
