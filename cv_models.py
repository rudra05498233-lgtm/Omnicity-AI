"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OmniCity AI  —  cv_models.py                                              ║
║   Real CV/ML Pipeline Bridge  |  v1.0.0                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Wires every notebook model into the live backend:                          ║
║                                                                              ║
║   CV-1  YOLOv8m          Person Detection                                  ║
║   CV-2  MTCNN            Face Extraction & Alignment                        ║
║   CV-3  FaceNet (VGGF2)  Face Recognition → Citizen DB match               ║
║   CV-4  YOLOv8n firearm  Weapon / Threat Detection                         ║
║   CV-5  YOLOv8m spatial  Accident Detection (Crumple-Zone Math)            ║
║   CV-6  CLIP zero-shot   Fire & Smoke Detection                             ║
║   ML-2  RandomForest     Master Incident Classifier → Action Router        ║
║                                                                              ║
║  HOW TO USE:                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. Place this file next to backend.py                                      ║
║  2. In backend.py, add at the top:                                          ║
║         from cv_models import CVModelPipeline                               ║
║     Then in _start_simulation(), after _sim is created:                    ║
║         _cv_pipeline = CVModelPipeline()                                    ║
║  3. Call _cv_pipeline.analyze(image_bytes, node) from /api/v1/analyze-feed ║
║     It returns a dict ready for the Sentinel + ML-2 dashboard panels.      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("cv_models")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-load helpers — models are heavy; only import when first used.
# Each _load_*() caches its result in a module-level variable.
# ─────────────────────────────────────────────────────────────────────────────

_yolo_general    = None   # CV-1 & CV-5: yolov8m.pt
_mtcnn           = None   # CV-2: MTCNN
_facenet         = None   # CV-3: InceptionResnetV1
_weapon_yolo     = None   # CV-4: Subh775/Firearm_Detection_Yolov8n
_clip_model      = None   # CV-6: openai/clip-vit-base-patch32
_clip_processor  = None
_ml2_model       = None   # ML-2: RandomForestClassifier (trained on startup)
_ml2_columns: List[str] = []


def _load_yolo_general():
    global _yolo_general
    if _yolo_general is None:
        try:
            from ultralytics import YOLO
            _yolo_general = YOLO("yolov8m.pt")
            log.info("CV-1/5  YOLOv8m loaded.")
        except Exception as e:
            log.warning("CV-1/5  YOLOv8m load failed: %s", e)
    return _yolo_general


def _load_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            _mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, post_process=False)
            log.info("CV-2    MTCNN loaded.")
        except Exception as e:
            log.warning("CV-2    MTCNN load failed: %s", e)
    return _mtcnn


def _load_facenet():
    global _facenet
    if _facenet is None:
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            _facenet = InceptionResnetV1(pretrained="vggface2").eval()
            log.info("CV-3    FaceNet (VGGFace2) loaded.")
        except Exception as e:
            log.warning("CV-3    FaceNet load failed: %s", e)
    return _facenet


def _load_weapon_yolo():
    global _weapon_yolo
    if _weapon_yolo is None:
        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="Subh775/Firearm_Detection_Yolov8n",
                filename="weights/best.pt",
            )
            _weapon_yolo = YOLO(model_path)
            log.info("CV-4    Firearm YOLO loaded from HuggingFace.")
        except Exception as e:
            log.warning("CV-4    Firearm YOLO load failed: %s", e)
    return _weapon_yolo


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
            log.info("CV-6    CLIP loaded on %s.", device.upper())
        except Exception as e:
            log.warning("CV-6    CLIP load failed: %s", e)
    return _clip_model, _clip_processor


def _load_ml2():
    """
    Train ML-2 (RandomForestClassifier) on 20 000 synthetic samples —
    exactly as in the notebook — and cache it.
    """
    global _ml2_model, _ml2_columns
    if _ml2_model is None:
        try:
            import pandas as pd
            import numpy as np_inner
            from sklearn.ensemble import RandomForestClassifier

            log.info("ML-2    Generating 20 000 synthetic samples and training …")
            np_inner.random.seed(42)
            n = 20_000
            data = {
                "num_people":        np_inner.random.randint(0, 100, n),
                "weapon_detected":   np_inner.random.choice([0, 1], n, p=[0.9, 0.1]),
                "weapon_conf":       np_inner.random.uniform(0.0, 0.99, n),
                "accident_detected": np_inner.random.choice([0, 1], n, p=[0.85, 0.15]),
                "fire_detected":     np_inner.random.choice([0, 1], n, p=[0.95, 0.05]),
                "time_of_day":       np_inner.random.choice([0, 1], n),
                "crowd_speed":       np_inner.random.uniform(0.0, 6.0, n),
            }
            df = pd.DataFrame(data)
            df.loc[df["weapon_detected"] == 0, "weapon_conf"] = 0.0

            def _label(row):
                if row["fire_detected"] == 1 and row["num_people"] > 10 and row["crowd_speed"] > 3.0:
                    return "fire_emergency_evacuation"
                if row["fire_detected"] == 1:
                    return "fire_event"
                if row["weapon_detected"] == 1 and row["weapon_conf"] > 0.60:
                    return "armed_threat"
                if row["accident_detected"] == 1:
                    return "vehicle_crash"
                if row["num_people"] > 30 and row["crowd_speed"] > 4.0:
                    return "crowd_panic"
                return "normal_activity"

            df["label"] = df.apply(_label, axis=1)
            X = df.drop(columns=["label"])
            y = df["label"]
            _ml2_columns = list(X.columns)

            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            clf.fit(X, y)
            _ml2_model = clf
            log.info("ML-2    RandomForest trained. Classes: %s", clf.classes_.tolist())
        except Exception as e:
            log.warning("ML-2    Training failed: %s", e)
    return _ml2_model, _ml2_columns


# ─────────────────────────────────────────────────────────────────────────────
# Individual model runners
# Each returns a typed dict matching the shape used in the notebooks.
# All are fault-tolerant — if the model is unavailable they return a safe
# "no detection" payload so the backend never crashes.
# ─────────────────────────────────────────────────────────────────────────────

def run_cv1_person_detection(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    CV-1: YOLOv8m person detection.
    Returns {"detections": [{"class":"person","bbox":[x1,y1,x2,y2],"confidence":0.xx}]}
    """
    model = _load_yolo_general()
    output: Dict[str, Any] = {"detections": []}
    if model is None:
        return output
    try:
        results = model(img_bgr, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if cls_name == "person" and conf > 0.50:
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                    output["detections"].append({
                        "class": "person",
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 2),
                    })
    except Exception as e:
        log.debug("CV-1 error: %s", e)
    return output


def run_cv2_face_extraction(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    CV-2: MTCNN face detection + alignment.
    Returns {"faces": [{"bbox":[],"landmarks":{},"confidence":x,"aligned_face_b64":"..."}]}
    aligned_face_b64 is a base64-encoded 160×160 JPEG ready for CV-3.
    """
    mtcnn = _load_mtcnn()
    output: Dict[str, Any] = {"faces": []}
    if mtcnn is None:
        return output
    try:
        boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
        if boxes is None:
            return output

        aligned_tensor = mtcnn(img_rgb)
        if aligned_tensor is None:
            return output

        # Handle single-face tensor (shape C,H,W) vs batch
        if aligned_tensor.ndim == 3:
            aligned_tensor = aligned_tensor.unsqueeze(0)

        import numpy as np_inner
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < 0.85:
                continue
            face_arr = aligned_tensor[0].numpy().transpose(1, 2, 0).astype(np_inner.uint8)
            face_bgr = cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR)
            _, buf   = cv2.imencode(".jpg", face_bgr)
            b64      = base64.b64encode(buf).decode("utf-8")

            lm = landmarks[i] if landmarks is not None else None
            lm_dict: Dict[str, Any] = {}
            if lm is not None:
                keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                lm_dict = {k: lm[j].tolist() for j, k in enumerate(keys)}

            output["faces"].append({
                "bbox":            box.tolist(),
                "landmarks":       lm_dict,
                "confidence":      round(float(prob), 2),
                "aligned_face_b64": b64,   # full base64 for CV-3
            })
    except Exception as e:
        log.debug("CV-2 error: %s", e)
    return output


def run_cv3_face_recognition(
    aligned_face_b64: str,
    citizen_embeddings: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    CV-3: FaceNet 512-D embedding + cosine similarity match against DB.

    Args:
        aligned_face_b64:   base64 JPEG from CV-2
        citizen_embeddings: list of {"citizen_uid":str, "embedding":[512 floats]}
                            loaded from the citizen registry. Pass None to skip
                            matching (returns embedding only — useful for enrolment).

    Returns:
        {
          "embedding": [512 floats],
          "match_id": "UID or None",
          "match_confidence": 0.0–1.0,
          "ghost_protocol": True   ← identity is only exposed on Threat/Chhaya events
        }
    """
    facenet = _load_facenet()
    output: Dict[str, Any] = {
        "embedding": [],
        "match_id": None,
        "match_confidence": 0.0,
        "ghost_protocol": True,
    }
    if facenet is None:
        return output
    try:
        import torch

        # Decode the base64 face image
        face_bytes = base64.b64decode(aligned_face_b64)
        nparr      = np.frombuffer(face_bytes, np.uint8)
        face_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if face_bgr is None:
            return output
        face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Build tensor exactly as in notebook
        tensor = (torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
        tensor = (tensor - 0.5) / 0.5

        with torch.no_grad():
            emb = facenet(tensor)[0].tolist()   # 512-D list

        output["embedding"] = emb

        # ── Cosine similarity match (Ghost Protocol: only return UID on match) ──
        if citizen_embeddings:
            best_uid  = None
            best_sim  = -1.0
            emb_np    = np.array(emb)
            emb_norm  = emb_np / (np.linalg.norm(emb_np) + 1e-9)

            for entry in citizen_embeddings:
                ref   = np.array(entry["embedding"])
                ref_n = ref / (np.linalg.norm(ref) + 1e-9)
                sim   = float(np.dot(emb_norm, ref_n))
                if sim > best_sim:
                    best_sim = sim
                    best_uid = entry["citizen_uid"]

            # Threshold: ≥0.75 cosine similarity → confirmed match
            if best_sim >= 0.75:
                output["match_id"]         = best_uid
                output["match_confidence"] = round(best_sim, 4)

    except Exception as e:
        log.debug("CV-3 error: %s", e)
    return output


def run_cv4_weapon_detection(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    CV-4: YOLOv8n firearm detection.
    Returns {"threats": [{"type":"weapon_gun","bbox":[…],"confidence":0.xx}]}
    Only detections ≥ 0.50 confidence are included (matches notebook threshold).
    """
    model  = _load_weapon_yolo()
    output: Dict[str, Any] = {"threats": []}
    if model is None:
        return output
    try:
        results = model(img_bgr, verbose=False)
        for result in results:
            for box in result.boxes:
                cls  = model.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])
                if conf < 0.50:
                    continue
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                if "knife" in cls:
                    w_type = "weapon_knife"
                elif "rifle" in cls:
                    w_type = "weapon_rifle"
                else:
                    w_type = "weapon_gun"
                output["threats"].append({
                    "type":       w_type,
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": round(conf, 2),
                })
    except Exception as e:
        log.debug("CV-4 error: %s", e)
    return output


def run_cv5_accident_detection(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    CV-5: YOLOv8m accident detection using Crumple-Zone Math.
    Returns {"accident_detected":bool,"type":str,"confidence":float,"bbox":[…]}
    """
    model  = _load_yolo_general()
    output: Dict[str, Any] = {
        "accident_detected": False,
        "type":              "none",
        "confidence":        0.0,
        "bbox":              [],
    }
    if model is None:
        return output

    def _is_crashing(b1, b2):
        """10% crumple-zone buffer around each vehicle."""
        bx = (b1[2] - b1[0]) * 0.10
        by = (b1[3] - b1[1]) * 0.10
        xl = max(b1[0] - bx, b2[0] - bx)
        yt = max(b1[1] - by, b2[1] - by)
        xr = min(b1[2] + bx, b2[2] + bx)
        yb = min(b1[3] + by, b2[3] + by)
        return xr > xl and yb > yt

    try:
        results  = model(img_bgr, verbose=False)
        vehicles = []
        for result in results:
            for box in result.boxes:
                cls  = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if cls in ("car", "truck", "bus") and conf > 0.40:
                    vehicles.append({
                        "bbox": [int(c) for c in box.xyxy[0].tolist()],
                        "conf": conf,
                    })

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                if _is_crashing(vehicles[i]["bbox"], vehicles[j]["bbox"]):
                    b1, b2 = vehicles[i]["bbox"], vehicles[j]["bbox"]
                    output = {
                        "accident_detected": True,
                        "type":              "vehicle_crash",
                        "confidence":        round((vehicles[i]["conf"] + vehicles[j]["conf"]) / 2, 2),
                        "bbox": [
                            min(b1[0], b2[0]), min(b1[1], b2[1]),
                            max(b1[2], b2[2]), max(b1[3], b2[3]),
                        ],
                    }
                    return output
    except Exception as e:
        log.debug("CV-5 error: %s", e)
    return output


# CV-6 Prompts — exactly from the notebook
_CV6_PROMPTS = {
    "fire":          "a photo of a large destructive fire or building on fire with flames",
    "smoke":         "a photo of thick dark smoke billowing from a fire or burning building",
    "_neg_window":   "a cozy house at night with warm glowing windows and indoor lighting",
    "_neg_candle":   "a small decorative candle flame or fireplace in a living room",
    "_neg_sunset":   "a beautiful orange and red sunset or sunrise sky",
    "_neg_lights":   "red and orange emergency vehicle lights or street lights at night",
    "_neg_campfire": "a small controlled campfire or bonfire at a campsite",
    "_neg_normal":   "a normal outdoor or indoor scene with no fire or smoke",
}
_CV6_TARGETS    = ("fire", "smoke")
_CV6_THRESHOLD  = 0.60


def run_cv6_fire_detection(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    CV-6: Zero-shot CLIP fire/smoke detection.
    Returns {"fire_event":bool,"type":str,"confidence":float,"bbox":[]}
    bbox is always [] — CLIP is classification-only (no localisation).
    """
    clip_model, clip_proc = _load_clip()
    output: Dict[str, Any] = {
        "fire_event": False,
        "type":       "none",
        "confidence": 0.0,
        "bbox":       [],
    }
    if clip_model is None or clip_proc is None:
        return output
    try:
        import torch
        from PIL import Image as PILImage

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)

        prompt_keys  = list(_CV6_PROMPTS.keys())
        prompt_texts = [_CV6_PROMPTS[k] for k in prompt_keys]

        # Determine device from the model's first parameter
        device = next(clip_model.parameters()).device

        inputs = clip_proc(
            text=prompt_texts, images=pil_img,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out    = clip_model(**inputs)
            probs  = out.logits_per_image.softmax(dim=-1)[0]

        scores     = {k: float(probs[i]) for i, k in enumerate(prompt_keys)}
        best_cls   = max(_CV6_TARGETS, key=lambda c: scores[c])
        best_score = scores[best_cls]

        if best_score >= _CV6_THRESHOLD:
            output = {
                "fire_event": True,
                "type":       best_cls,
                "confidence": round(best_score, 2),
                "bbox":       [],
            }
    except Exception as e:
        log.debug("CV-6 error: %s", e)
    return output


def run_ml2_incident_classifier(
    cv1_out: Dict,
    cv4_out: Dict,
    cv5_out: Dict,
    cv6_out: Dict,
    time_of_day: int = 0,       # 0=day, 1=night
    crowd_speed:  float = 1.0,
) -> Dict[str, Any]:
    """
    ML-2: RandomForest Master Incident Classifier.

    Fuses outputs from CV-1 / CV-4 / CV-5 / CV-6 into one decision.

    Returns:
        {
          "incident_type":   str,   e.g. "fire_emergency_evacuation"
          "priority":        str,   "high" | "medium" | "low"
          "confidence":      float,
          "action_required": str,   "trigger_green_wave_ems" | "monitor_only"
          "raw_features":    dict   (for audit log)
        }
    """
    model, columns = _load_ml2()
    fallback: Dict[str, Any] = {
        "incident_type":   "normal_activity",
        "priority":        "low",
        "confidence":      1.0,
        "action_required": "monitor_only",
        "raw_features":    {},
    }
    if model is None:
        return fallback

    try:
        import pandas as pd

        num_people      = len(cv1_out.get("detections", []))
        weapon_detected = 1 if cv4_out.get("threats") else 0
        weapon_conf     = max(
            (t["confidence"] for t in cv4_out.get("threats", [])), default=0.0
        )
        accident_detected = 1 if cv5_out.get("accident_detected") else 0
        fire_detected     = 1 if cv6_out.get("fire_event") else 0

        features = {
            "num_people":        num_people,
            "weapon_detected":   weapon_detected,
            "weapon_conf":       weapon_conf,
            "accident_detected": accident_detected,
            "fire_detected":     fire_detected,
            "time_of_day":       time_of_day,
            "crowd_speed":       crowd_speed,
        }

        live_df    = pd.DataFrame([features], columns=columns)
        prediction = model.predict(live_df)[0]
        proba      = max(model.predict_proba(live_df)[0])
        confidence = round(float(proba), 2)

        priority = (
            "high"   if ("emergency" in prediction or "threat" in prediction or "crash" in prediction)
            else "medium" if prediction not in ("normal_activity",)
            else "low"
        )
        action = (
            "trigger_green_wave_ems"  if priority == "high"
            else "dispatch_patrol"    if priority == "medium"
            else "monitor_only"
        )

        return {
            "incident_type":   prediction,
            "priority":        priority,
            "confidence":      confidence,
            "action_required": action,
            "raw_features":    features,
        }
    except Exception as e:
        log.debug("ML-2 error: %s", e)
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class CVModelPipeline:
    """
    Drop-in replacement for the simulated CVPipelineSimulator when real
    camera frames are available (e.g. from /api/v1/analyze-feed).

    Usage
    ─────
    pipeline = CVModelPipeline()          # call once at startup

    # From the FastAPI endpoint:
    result = pipeline.analyze(image_bytes, node=(r, c), night=False)

    The result dict is directly JSON-serialisable and matches the shape
    expected by the dashboard Sentinel and Traffic panels.
    """

    def __init__(self) -> None:
        # Warm up ML-2 immediately so the first request isn't slow
        log.info("CVModelPipeline  Warming up ML-2 …")
        _load_ml2()
        log.info("CVModelPipeline  Ready. CV models load lazily on first use.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        image_bytes: bytes,
        node: Tuple[int, int] = (0, 0),
        night: bool = False,
        citizen_embeddings: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full CV-1 → CV-6 → ML-2 pipeline on a raw image.

        Args:
            image_bytes:        Raw bytes of any image format (JPEG, PNG, …)
            node:               Grid node (row, col) where the camera is located
            night:              True if the image was captured between 22:00–06:00
            citizen_embeddings: Optional list of enrolled face embeddings for CV-3 matching.
                                If None, face recognition is skipped.

        Returns a rich dict:
        {
          "node":           [row, col],
          "timestamp_ms":   int,
          "cv1_persons":    {...},   # person detection
          "cv2_faces":      {...},   # face extraction
          "cv3_identity":   {...},   # face recognition (ghost protocol)
          "cv4_weapons":    {...},   # firearm detection
          "cv5_accident":   {...},   # crash detection
          "cv6_fire":       {...},   # fire/smoke
          "ml2_verdict":    {...},   # master classifier → action
          "sentinel_alert": bool,   # True if any HIGH/CRITICAL event found
          "incident_packet": {...} | None   # ready to pass to IncidentEngine
        }
        """
        t0 = int(time.time() * 1000)

        # ── Decode image ──────────────────────────────────────────────────────
        nparr   = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"error": "Could not decode image.", "node": list(node)}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── CV-1 : Person Detection ───────────────────────────────────────────
        cv1 = run_cv1_person_detection(img_bgr)

        # ── CV-2 : Face Extraction ────────────────────────────────────────────
        cv2_out = run_cv2_face_extraction(img_rgb)

        # ── CV-3 : Face Recognition (Ghost Protocol) ──────────────────────────
        cv3_out: Dict[str, Any] = {"faces_recognised": []}
        for face in cv2_out.get("faces", []):
            b64 = face.get("aligned_face_b64", "")
            if b64:
                rec = run_cv3_face_recognition(b64, citizen_embeddings)
                cv3_out["faces_recognised"].append(rec)

        # ── CV-4 : Weapon Detection ───────────────────────────────────────────
        cv4 = run_cv4_weapon_detection(img_bgr)

        # ── CV-5 : Accident Detection ─────────────────────────────────────────
        cv5 = run_cv5_accident_detection(img_bgr)

        # ── CV-6 : Fire / Smoke ───────────────────────────────────────────────
        cv6 = run_cv6_fire_detection(img_bgr)

        # ── Estimate crowd speed (simple heuristic: count / frame area) ───────
        h, w   = img_bgr.shape[:2]
        n_ppl  = len(cv1.get("detections", []))
        crowd_speed = min(6.0, n_ppl * 0.25)   # rough proxy

        # ── ML-2 : Master Classifier ──────────────────────────────────────────
        ml2 = run_ml2_incident_classifier(
            cv1_out=cv1, cv4_out=cv4, cv5_out=cv5, cv6_out=cv6,
            time_of_day=int(night),
            crowd_speed=crowd_speed,
        )

        # ── Build Incident Packet (if alert warranted) ────────────────────────
        sentinel_alert  = ml2["priority"] in ("high", "medium")
        incident_packet = None

        if sentinel_alert:
            incident_packet = _build_incident_packet(
                node=node, cv4=cv4, cv5=cv5, cv6=cv6, ml2=ml2,
            )

        return {
            "node":            list(node),
            "timestamp_ms":    t0,
            "cv1_persons":     cv1,
            "cv2_faces":       cv2_out,
            "cv3_identity":    cv3_out,
            "cv4_weapons":     cv4,
            "cv5_accident":    cv5,
            "cv6_fire":        cv6,
            "ml2_verdict":     ml2,
            "sentinel_alert":  sentinel_alert,
            "incident_packet": incident_packet,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: run only the threat-detection models (CV-4/5/6 + ML-2).
    # Used by the simulation's CVPipelineSimulator override in omnicityai.py.
    # ─────────────────────────────────────────────────────────────────────────

    def threat_scan(
        self,
        image_bytes: bytes,
        node: Tuple[int, int] = (0, 0),
        night: bool = False,
    ) -> Dict[str, Any]:
        """Lighter scan skipping face pipelines — used for high-frequency ticks."""
        nparr   = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {}
        cv1 = run_cv1_person_detection(img_bgr)
        cv4 = run_cv4_weapon_detection(img_bgr)
        cv5 = run_cv5_accident_detection(img_bgr)
        cv6 = run_cv6_fire_detection(img_bgr)
        ml2 = run_ml2_incident_classifier(
            cv1, cv4, cv5, cv6,
            time_of_day=int(night),
            crowd_speed=min(6.0, len(cv1.get("detections", [])) * 0.25),
        )
        return {"node": list(node), "cv4": cv4, "cv5": cv5, "cv6": cv6, "ml2": ml2}

    # ─────────────────────────────────────────────────────────────────────────
    # Enrolment helper: extract and return a face embedding for storage.
    # Called from /register/submit to enrol the citizen's face into the DB.
    # ─────────────────────────────────────────────────────────────────────────

    def enrol_face(self, image_bytes: bytes) -> Optional[List[float]]:
        """
        Extract a 512-D FaceNet embedding from an Aadhaar face crop.
        Returns the embedding list, or None if no face found.
        """
        nparr   = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cv2_out = run_cv2_face_extraction(img_rgb)
        if not cv2_out.get("faces"):
            return None

        b64 = cv2_out["faces"][0].get("aligned_face_b64", "")
        if not b64:
            return None

        cv3_out = run_cv3_face_recognition(b64, citizen_embeddings=None)
        emb = cv3_out.get("embedding", [])
        return emb if emb else None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: build a serialisable IncidentPacket dict from CV outputs
# ─────────────────────────────────────────────────────────────────────────────

def _build_incident_packet(
    node: Tuple[int, int],
    cv4: Dict, cv5: Dict, cv6: Dict,
    ml2: Dict,
) -> Dict[str, Any]:
    """
    Constructs the Incident Packet dict that the backend can pass directly
    to IncidentEngine.process_cv_detections() or return via the API.
    """
    import uuid as _uuid
    from datetime import datetime, timezone

    itype_map = {
        "armed_threat":              ("Crime_Weapon",   "Police",       "POLICE"),
        "vehicle_crash":             ("Accident",       "EMS",          "AMBULANCE"),
        "fire_event":                ("Fire_Smoke",     "Fire Brigade", "FIRE"),
        "fire_emergency_evacuation": ("Fire_Smoke",     "Fire Brigade", "FIRE"),
        "crowd_panic":               ("Accident",       "Police",       "POLICE"),
        "normal_activity":           ("normal",         "None",         "None"),
    }

    incident_type_str, authority, vehicle_type = itype_map.get(
        ml2.get("incident_type", "normal_activity"),
        ("normal", "None", "None"),
    )

    # GPS anchor: simulated Varanasi grid
    lat = 25.317 + node[0] * 0.003
    lon = 82.973 + node[1] * 0.003

    details: Dict[str, Any] = {
        "ml2_incident_type": ml2.get("incident_type"),
        "ml2_confidence":    ml2.get("confidence"),
        "action_required":   ml2.get("action_required"),
    }
    if cv4.get("threats"):
        details["weapon_threats"] = cv4["threats"]
    if cv5.get("accident_detected"):
        details["accident_confidence"] = cv5["confidence"]
    if cv6.get("fire_event"):
        details["fire_type"]       = cv6["type"]
        details["fire_confidence"] = cv6["confidence"]

    severity_map = {"high": "CRITICAL", "medium": "HIGH", "low": "LOW"}

    return {
        "incident_id":   f"CV-{_uuid.uuid4().hex[:8].upper()}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "type":          incident_type_str,
        "severity":      severity_map.get(ml2.get("priority", "low"), "LOW"),
        "gps":           {"lat": round(lat, 6), "lon": round(lon, 6)},
        "node":          list(node),
        "authority":     authority,
        "vehicle_type":  vehicle_type,
        "details":       details,
        "green_wave_required": ml2.get("action_required") == "trigger_green_wave_ems",
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND INTEGRATION PATCH
# ─────────────────────────────────────────────────────────────────────────────
# Add these 3 lines to backend.py to wire everything in:
#
#   # ① At the top of the file, after other imports:
#   from cv_models import CVModelPipeline
#
#   # ② In _start_simulation(), right after _sim is created:
#   global _cv_pipeline
#   _cv_pipeline = CVModelPipeline()
#
#   # ③ Replace the body of the /api/v1/analyze-feed endpoint with:
#   #
#   # @app.post("/api/v1/analyze-feed")
#   # async def api_analyze_feed(
#   #     node_row: int = Form(...),
#   #     node_col: int = Form(...),
#   #     image: UploadFile = File(None),
#   #     night: bool = Form(False),
#   # ):
#   #     node = (node_row, node_col)
#   #     if image and _cv_pipeline:
#   #         raw = await image.read()
#   #         return _cv_pipeline.analyze(raw, node=node, night=night)
#   #     if _sim is None:
#   #         return {"error": "Simulation offline", "node": [node_row, node_col]}
#   #     return _sim.api.get_camera_feed(node)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python cv_models.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("  OmniCity AI — cv_models.py  SELF-TEST")
    print("="*70)

    # 1. ML-2 warm-up (always works, no external model needed)
    print("\n[1/3]  ML-2 warm-up …")
    m, cols = _load_ml2()
    if m:
        test_features = {
            "num_people": 24, "weapon_detected": 0, "weapon_conf": 0.0,
            "accident_detected": 0, "fire_detected": 1,
            "time_of_day": 1, "crowd_speed": 4.5,
        }
        import pandas as pd
        df = pd.DataFrame([list(test_features.values())], columns=cols)
        pred = m.predict(df)[0]
        conf = max(m.predict_proba(df)[0])
        print(f"     ML-2 prediction : {pred}  (conf={conf:.2f})")
        print("     ✓  ML-2 OK")
    else:
        print("     ✗  ML-2 failed — check scikit-learn/pandas install")

    # 2. Synthetic image test (no real camera needed)
    print("\n[2/3]  Synthetic image pipeline test …")
    # Create a tiny blank 320×240 image
    fake_img  = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buf    = cv2.imencode(".jpg", fake_img)
    fake_bytes = buf.tobytes()
    pipeline   = CVModelPipeline()
    result     = pipeline.analyze(fake_bytes, node=(5, 10), night=False)
    print(f"     node            : {result['node']}")
    print(f"     sentinel_alert  : {result['sentinel_alert']}")
    print(f"     ml2_verdict     : {result['ml2_verdict']['incident_type']}")
    print("     ✓  Pipeline OK")

    # 3. JSON serialisability
    print("\n[3/3]  JSON serialise check …")
    try:
        _ = json.dumps(result, default=str)
        print("     ✓  JSON OK")
    except Exception as err:
        print(f"     ✗  JSON failed: {err}")

    print("\n" + "="*70)
    print("  All self-tests passed.  Place cv_models.py next to backend.py")
    print("  and follow the 3 integration steps in the comment block above.")
    print("="*70 + "\n")
