import time
import cv2
from picamera2 import Picamera2

SIZE = (640, 480)  # try 640x480 first (usually largest FoV for many modules)

def main():
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"format": "RGB888", "size": SIZE})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.5)

    # Inspect scaler crop ranges
    sc = picam2.camera_controls.get("ScalerCrop", None)
    print("ScalerCrop control:", sc)
    # sc is typically (min, max, default)
    if sc:
        _, max_crop, default_crop = sc
        print("Default crop:", default_crop)
        print("Max (full FoV) crop:", max_crop)

    # Capture default
    frame1 = picam2.capture_array()
    cv2.imwrite("tests/out/fov_default.jpg", cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))
    print("Saved tests/out/fov_default.jpg")

    # Force full FoV (max crop)
    if sc:
        picam2.set_controls({"ScalerCrop": max_crop})
        time.sleep(0.3)  # allow control to apply

    frame2 = picam2.capture_array()
    cv2.imwrite("tests/out/fov_full.jpg", cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR))
    print("Saved tests/out/fov_full.jpg")

    picam2.stop()

if __name__ == "__main__":
    main()
