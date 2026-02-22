#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
from picamera2 import Picamera2


TRUST_LEVELS = ("UNKNOWN", "Guest", "Friend", "OWNER")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

DEFAULT_DETECT_MODEL = os.path.join(_THIS_DIR, "models", "face_detection_yunet.onnx")
DEFAULT_RECOG_MODEL = os.path.join(_THIS_DIR, "models", "face_recognition_sface.onnx")
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "people", "face_db.json")
DEFAULT_TRUST_MAP_PATH = os.path.join(_PROJECT_ROOT, "data", "people", "trust_map.json")
DEFAULT_CAPTURE_DIR = os.path.join(_PROJECT_ROOT, "data", "captures")


def now_ts() -> float:
    return time.time()


def parse_size(size_str: str) -> Tuple[int, int]:
    try:
        w_str, h_str = size_str.lower().split("x")
        w = int(w_str)
        h = int(h_str)
    except ValueError as e:
        raise ValueError(f"Invalid --size format '{size_str}'. Expected WxH (e.g. 640x480).") from e
    if w <= 0 or h <= 0:
        raise ValueError("Camera size must be positive.")
    return w, h


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _load_json_dict(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in '{path}', got {type(data).__name__}")
    return data


def _save_json_dict(path: str, data: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def _resolve_capture_output(capture_out: str, name: str) -> str:
    # Treat as a directory if it already exists as dir OR if no file extension was provided.
    looks_like_directory = os.path.isdir(capture_out) or os.path.splitext(capture_out)[1] == ""
    if looks_like_directory:
        return os.path.join(capture_out, f"enroll_{name}.jpg")
    return capture_out


def _detect_largest_face(detector: cv2.FaceDetectorYN, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    _, faces = detector.detect(frame_bgr)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: float(f[2] * f[3]))


def _extract_embedding(
    recognizer: cv2.FaceRecognizerSF,
    frame_bgr: np.ndarray,
    face: np.ndarray,
) -> np.ndarray:
    aligned = recognizer.alignCrop(frame_bgr, face)
    feat = recognizer.feature(aligned)
    return feat.flatten().astype(np.float32)


def enroll(
    name: str,
    trust: Optional[str],
    samples: int,
    max_attempts: int,
    interval_s: float,
    camera_size: Tuple[int, int],
    detect_model_path: str,
    recog_model_path: str,
    detector_score_threshold: float,
    detector_nms_threshold: float,
    db_path: str,
    trust_map_path: str,
    capture_out: Optional[str],
) -> Dict[str, Any]:
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"format": "RGB888", "size": camera_size})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.4)

    embeddings: List[np.ndarray] = []
    last_frame_bgr = None

    detector = cv2.FaceDetectorYN.create(
        detect_model_path,
        "",
        camera_size,
        score_threshold=detector_score_threshold,
        nms_threshold=detector_nms_threshold,
        top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(recog_model_path, "")

    try:
        for attempt in range(1, max_attempts + 1):
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            last_frame_bgr = frame_bgr

            face = _detect_largest_face(detector, frame_bgr)
            if face is None:
                print(f"[{attempt}/{max_attempts}] No face detected")
                time.sleep(interval_s)
                continue

            emb = _extract_embedding(recognizer, frame_bgr, face)
            embeddings.append(l2_normalize(emb))
            print(f"[{attempt}/{max_attempts}] Captured sample {len(embeddings)}/{samples}")

            if len(embeddings) >= samples:
                break

            time.sleep(interval_s)
    finally:
        picam2.stop()

    if len(embeddings) == 0:
        raise RuntimeError("Enrollment failed: no usable face sample was captured")

    if len(embeddings) < samples:
        raise RuntimeError(
            f"Enrollment failed: captured only {len(embeddings)}/{samples} samples within {max_attempts} attempts"
        )

    avg_emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    avg_emb = l2_normalize(avg_emb)

    db = _load_json_dict(db_path)
    trust_map = _load_json_dict(trust_map_path)

    db[name] = avg_emb.tolist()

    existing_trust = str(trust_map.get(name, "Guest"))
    if existing_trust not in TRUST_LEVELS:
        existing_trust = "Guest"
    final_trust = trust if trust is not None else existing_trust
    if final_trust not in TRUST_LEVELS:
        final_trust = "Guest"
    trust_map[name] = final_trust

    _save_json_dict(db_path, db)
    _save_json_dict(trust_map_path, trust_map)

    saved_capture_path = None
    if capture_out is not None and last_frame_bgr is not None:
        out_path = _resolve_capture_output(capture_out, name)
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        cv2.imwrite(out_path, last_frame_bgr)
        saved_capture_path = out_path

    return {
        "type": "FACE_ENROLL_RESULT",
        "ts": now_ts(),
        "ok": True,
        "name": name,
        "trust_level": final_trust,
        "samples": len(embeddings),
        "db_path": db_path,
        "trust_map_path": trust_map_path,
        "capture_path": saved_capture_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Enroll a person into face DB/trust map")
    sub = ap.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add/update one person from camera")
    p_add.add_argument("name", type=str, help="Identity name to store")
    p_add.add_argument("--trust", type=str, choices=TRUST_LEVELS, default=None)
    p_add.add_argument("--samples", type=int, default=3)
    p_add.add_argument("--max-attempts", type=int, default=12)
    p_add.add_argument("--interval-s", type=float, default=0.25)
    p_add.add_argument("--size", type=str, default="640x480")
    p_add.add_argument("--detector-score-threshold", type=float, default=0.90)
    p_add.add_argument("--detector-nms-threshold", type=float, default=0.30)
    p_add.add_argument("--detect-model", type=str, default=DEFAULT_DETECT_MODEL)
    p_add.add_argument("--recog-model", type=str, default=DEFAULT_RECOG_MODEL)
    p_add.add_argument("--db", type=str, default=DEFAULT_DB_PATH)
    p_add.add_argument("--trust-map", type=str, default=DEFAULT_TRUST_MAP_PATH)
    p_add.add_argument("--capture-out", type=str, default=DEFAULT_CAPTURE_DIR)

    args = ap.parse_args()

    if args.command == "add":
        name = args.name.strip()
        if not name:
            raise SystemExit("Name cannot be empty")
        if args.samples <= 0:
            raise SystemExit("--samples must be > 0")
        if args.max_attempts < args.samples:
            raise SystemExit("--max-attempts must be >= --samples")

        cam_size = parse_size(args.size)

        result = enroll(
            name=name,
            trust=args.trust,
            samples=args.samples,
            max_attempts=args.max_attempts,
            interval_s=max(0.0, args.interval_s),
            camera_size=cam_size,
            detect_model_path=args.detect_model,
            recog_model_path=args.recog_model,
            detector_score_threshold=args.detector_score_threshold,
            detector_nms_threshold=args.detector_nms_threshold,
            db_path=args.db,
            trust_map_path=args.trust_map,
            capture_out=args.capture_out,
        )
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
