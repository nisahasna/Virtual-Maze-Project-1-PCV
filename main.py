import sys
import time
from collections import deque

import cv2
import numpy as np

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR = True

WINDOW_MAIN = "Virtual Maze"
WINDOW_MASK = "Mask (Hasil Segmentasi)"
WINDOW_RESULT = "Result (Hanya Area Biru)"
WINDOW_CTRL = "Controls (HSV & Morfologi)"

DEFAULT_HSV = {
    "H Min": 100, "H Max": 130,
    "S Min": 69, "S Max": 255,
    "V Min": 70,  "V Max": 255,
}
DEFAULT_KERNEL = 2   
DEFAULT_ITER = 2    

MIN_CONTOUR_AREA = 1500
SMOOTH_MODE = "ema"
EMA_ALPHA = 0.35
MA_WINDOW = 5

def nothing(_):
    pass


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka kamera dengan index {index}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Kamera terbuka. Resolusi: {actual_w}x{actual_h}")
    return cap


def create_controls() -> None:
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("H Min", WINDOW_CTRL, DEFAULT_HSV["H Min"], 179, nothing)
    cv2.createTrackbar("H Max", WINDOW_CTRL, DEFAULT_HSV["H Max"], 179, nothing)
    cv2.createTrackbar("S Min", WINDOW_CTRL, DEFAULT_HSV["S Min"], 255, nothing)
    cv2.createTrackbar("S Max", WINDOW_CTRL, DEFAULT_HSV["S Max"], 255, nothing)
    cv2.createTrackbar("V Min", WINDOW_CTRL, DEFAULT_HSV["V Min"], 255, nothing)
    cv2.createTrackbar("V Max", WINDOW_CTRL, DEFAULT_HSV["V Max"], 255, nothing)
    cv2.createTrackbar("Kernel", WINDOW_CTRL, DEFAULT_KERNEL, 10, nothing)
    cv2.createTrackbar("Iterasi", WINDOW_CTRL, DEFAULT_ITER, 5, nothing)


def read_controls():
    h_min = cv2.getTrackbarPos("H Min", WINDOW_CTRL)
    h_max = cv2.getTrackbarPos("H Max", WINDOW_CTRL)
    s_min = cv2.getTrackbarPos("S Min", WINDOW_CTRL)
    s_max = cv2.getTrackbarPos("S Max", WINDOW_CTRL)
    v_min = cv2.getTrackbarPos("V Min", WINDOW_CTRL)
    v_max = cv2.getTrackbarPos("V Max", WINDOW_CTRL)
    k = cv2.getTrackbarPos("Kernel", WINDOW_CTRL)
    iters = cv2.getTrackbarPos("Iterasi", WINDOW_CTRL)

    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lower, upper, 2 * k + 1, max(1, iters)


def reset_controls() -> None:
    for name, value in DEFAULT_HSV.items():
        cv2.setTrackbarPos(name, WINDOW_CTRL, value)
    cv2.setTrackbarPos("Kernel", WINDOW_CTRL, DEFAULT_KERNEL)
    cv2.setTrackbarPos("Iterasi", WINDOW_CTRL, DEFAULT_ITER)
    print("[INFO] Trackbar di-reset ke nilai awal.")


def segment_color(frame, lower, upper):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)


def clean_mask(mask, ksize, iters):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return closed

def find_largest_contour(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    return largest

def compute_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

class CursorSmoother:
    def __init__(self, mode="ema", alpha=0.35, window=5):
        self.mode = mode
        self.alpha = alpha
        self.history = deque(maxlen=window)
        self.ema = None

    def update(self,x,y):
        if self.mode == "ma":
            self.history.append((x,y))
            xs, ys = zip(*self.history)
            return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

        if self.ema is None:
            self.ema = (float(x), float(y))
        else:
            ex, ey = self.ema
            self.ema = (self.alpha * x + (1 - self.alpha) * ex,
                        self.alpha * y + (1 - self.alpha) * ey)
        return int(self.ema[0]), int(self.ema[1])

    def reset(self):
        self.history.clear()
        self.ema = None


def draw_hud(frame, fps: float, mode: str, detected: bool) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    status = "TERDETEKSI" if detected else "TIDAK TERDETEKSI"
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, f"Tangan: {status}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Smoothing: {mode.upper()}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "q/ESC: keluar | p: print | r: reset | m: ganti smoothing",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_tracking(frame, contour, raw_pt, smooth_pt) -> None:
    cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)       
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 1)
    cv2.circle(frame, raw_pt, 5, (0, 0, 255), -1)              
    cv2.circle(frame, smooth_pt, 10, (0, 255, 0), -1)
    cv2.circle(frame, smooth_pt, 12, (255, 255, 255), 2)


def main() -> None:
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_AUTOSIZE)
    create_controls()

    smoother = CursorSmoother(SMOOTH_MODE, EMA_ALPHA, MA_WINDOW)
    smooth_mode = SMOOTH_MODE

    fps = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari kamera.")
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        lower, upper, ksize, iters = read_controls()
        mask = clean_mask(segment_color(frame, lower, upper), ksize, iters)

        
        detected = False
        contour = find_largest_contour(mask, MIN_CONTOUR_AREA)
        if contour is not None:
            raw_pt = compute_centroid(contour)
            if raw_pt is not None:
                smooth_pt = smoother.update(*raw_pt)
                draw_tracking(frame, contour, raw_pt, smooth_pt)
                detected = True
        if not detected:
            smoother.reset()

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        draw_hud(frame, fps, smooth_mode, detected)
        cv2.imshow(WINDOW_MAIN, frame)
        cv2.imshow(WINDOW_MASK, mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            print(f"[HSV] lower={lower.tolist()} upper={upper.tolist()} "
                  f"kernel={ksize} iterasi={iters}")
        elif key == ord("r"):
            reset_controls()
        elif key == ord("m"):
            smooth_mode = "ma" if smooth_mode == "ema" else "ema"
            smoother = CursorSmoother(smooth_mode, EMA_ALPHA, MA_WINDOW)
            print(f"[INFO] Mode smoothing: {smooth_mode.upper()}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program selesai.")


if __name__ == "__main__":
    main()