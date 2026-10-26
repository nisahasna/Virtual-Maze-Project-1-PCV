import sys
import time

import cv2
import numpy as np

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR = True

WINDOW_MAIN = "Virtual Maze - Camera Feed"
WINDOW_MASK = "Mask (Hasil Segmentasi)"
WINDOW_RESULT = "Result (Hanya Area Biru)"
WINDOW_CTRL = "Controls (HSV & Morfologi)"

# Nilai awal rentang HSV untuk warna BIRU.
# Ingat: di OpenCV, Hue berskala 0-179 (bukan 0-360), S & V berskala 0-255.
DEFAULT_HSV = {
    "H Min": 100, "H Max": 130,
    "S Min": 120, "S Max": 255,
    "V Min": 70,  "V Max": 255,
}
DEFAULT_KERNEL = 2   # ukuran kernel sebenarnya = 2*nilai + 1 (selalu ganjil)
DEFAULT_ITER = 2     # berapa kali opening/closing diulang


def nothing(_):
    """Callback kosong yang dibutuhkan createTrackbar."""
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
    """Membuat jendela kontrol berisi trackbar HSV dan morfologi."""
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
    """Membaca nilai trackbar saat ini. Mengembalikan (lower, upper, ksize, iters)."""
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
    ksize = 2 * k + 1          # dipaksa ganjil (1, 3, 5, ..., 21)
    iters = max(1, iters)      # minimal 1 kali
    return lower, upper, ksize, iters


def reset_controls() -> None:
    """Mengembalikan semua trackbar ke nilai awal."""
    for name, value in DEFAULT_HSV.items():
        cv2.setTrackbarPos(name, WINDOW_CTRL, value)
    cv2.setTrackbarPos("Kernel", WINDOW_CTRL, DEFAULT_KERNEL)
    cv2.setTrackbarPos("Iterasi", WINDOW_CTRL, DEFAULT_ITER)
    print("[INFO] Trackbar di-reset ke nilai awal.")


def segment_color(frame, lower, upper):
    """BGR -> (blur) -> HSV -> inRange. Hasilnya mask biner (255 = biru)."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)   # redam noise sensor kamera
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)


def clean_mask(mask, ksize, iters):
    """
    Opening  (erosi -> dilasi): menghapus bintik putih kecil di luar objek.
    Closing  (dilasi -> erosi): menutup lubang kecil di dalam objek.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return closed


def draw_hud(frame, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "q/ESC: keluar | p: print nilai | r: reset",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW_RESULT, cv2.WINDOW_AUTOSIZE)
    create_controls()

    fps = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari kamera.")
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        # 1. Baca parameter dari trackbar
        lower, upper, ksize, iters = read_controls()

        # 2. Segmentasi warna biru, lalu bersihkan noise
        raw_mask = segment_color(frame, lower, upper)
        mask = clean_mask(raw_mask, ksize, iters)

        # 3. Tampilkan hanya piksel asli yang lolos mask
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        draw_hud(frame, fps)
        cv2.imshow(WINDOW_MAIN, frame)
        cv2.imshow(WINDOW_MASK, mask)
        cv2.imshow(WINDOW_RESULT, result)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            print(f"[HSV] lower={lower.tolist()} upper={upper.tolist()} "
                  f"kernel={ksize} iterasi={iters}")
        elif key == ord("r"):
            reset_controls()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program selesai.")


if __name__ == "__main__":
    main()