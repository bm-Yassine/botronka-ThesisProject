import os
import time
import json
import cv2
import numpy as np
from picamera2 import Picamera2

DETECT_MODEL = "src/vision/models/face_detection_yunet.onnx"
RECOG_MODEL  = "src/vision/models/face_recognition_sface.onnx"
DB_PATH      = "data/people/face_db.json"

CAPTURE_SIZE = (640, 480)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

def capture_frame():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": CAPTURE_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)
    frame_rgb = picam2.capture_array()
    picam2.stop()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

def init_models(w, h):
    detector = cv2.FaceDetectorYN.create(
        DETECT_MODEL, "", (w, h),
        score_threshold=0.9, nms_threshold=0.3, top_k=5000
    )
    recognizer = cv2.FaceRecognizerSF.create(RECOG_MODEL, "")
    return detector, recognizer

def detect_largest_face(detector, frame_bgr):
    _, faces = detector.detect(frame_bgr)
    if faces is None:
        return None
    # pick largest by area
    best = max(faces, key=lambda f: f[2] * f[3])
    return best  # [x,y,w,h,score, ...landmarks...]

def get_embedding(recognizer, frame_bgr, face):
    # face is [x,y,w,h,score,...]
    # SFace wants face box in a specific format, OpenCV uses the detector output directly
    aligned = recognizer.alignCrop(frame_bgr, face)
    feat = recognizer.feature(aligned)
    return feat.flatten().astype(np.float32)

def enroll(name):
    frame = capture_frame()
    h, w = frame.shape[:2]
    detector, recognizer = init_models(w, h)

    face = detect_largest_face(detector, frame)
    if face is None:
        print("No face detected. Try again with better lighting / closer distance.")
        return

    emb = get_embedding(recognizer, frame, face)
    db = load_db()
    db[name] = emb.tolist()
    save_db(db)

    print(f"Enrolled '{name}' with 1 embedding.")
    os.makedirs("data/captures", exist_ok=True)
    cv2.imwrite(f"data/captures/enroll_{name}.jpg", frame)
    print(f"Saved capture: data/captures/enroll_{name}.jpg")

def recognize(threshold=0.60):
    frame = capture_frame()
    h, w = frame.shape[:2]
    detector, recognizer = init_models(w, h)

    face = detect_largest_face(detector, frame)
    if face is None:
        print("No face detected.")
        return

    emb = get_embedding(recognizer, frame, face)
    db = load_db()
    if not db:
        print("DB empty. Enroll someone first.")
        return

    best_name = "UNKNOWN"
    best_sim = -1.0

    for name, ref_list in db.items():
        ref = np.array(ref_list, dtype=np.float32)
        sim = cosine_sim(emb, ref)
        if sim > best_sim:
            best_sim = sim
            best_name = name

    print(f"Best match: {best_name}  similarity={best_sim:.3f}  threshold={threshold}")

    # Save debug frame with bbox
    x, y, bw, bh = map(int, face[:4])
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    cv2.putText(frame, f"{best_name} {best_sim:.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    os.makedirs("data/captures", exist_ok=True)
    out = "data/captures/recognize_result.jpg"
    cv2.imwrite(out, frame)
    print("Saved:", out)

def usage():
    print("Usage:")
    print("  python3 face_recog_cli.py enroll <name>")
    print("  python3 face_recog_cli.py recognize [threshold]")
    print("")
    print("Examples:")
    print("  python3 face_recog_cli.py enroll yassine")
    print("  python3 face_recog_cli.py recognize 0.62")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        usage()
        raise SystemExit(1)

    cmd = sys.argv[1].lower()

    if cmd == "enroll" and len(sys.argv) >= 3:
        enroll(sys.argv[2])
    elif cmd == "recognize":
        thr = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.60
        recognize(threshold=thr)
    else:
        usage()
        raise SystemExit(1)
