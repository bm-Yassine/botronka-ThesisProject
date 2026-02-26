import time
import cv2
import numpy as np
from picamera2 import Picamera2

MODEL_PATH = "src/vision/models/face_detection_yunet.onnx"
OUT_PATH = "data/captures/detect_result.jpg"

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    time.sleep(0.5)  # camera warmup
    frame_rgb = picam2.capture_array()  # RGB
    picam2.stop()

    h, w = frame_rgb.shape[:2]

    # OpenCV uses BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # YuNet face detector
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (w, h),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )

    _, faces = detector.detect(frame_bgr)

    if faces is None:
        print("No faces detected.")
        cv2.imwrite(OUT_PATH, frame_bgr)
        print("Saved:", OUT_PATH)
        return

    print(f"Detected faces: {len(faces)}")

    for f in faces:
        x, y, bw, bh = map(int, f[:4])
        score = float(f[4])
        cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"{score:.2f}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(OUT_PATH, frame_bgr)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
