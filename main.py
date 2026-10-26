import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

try:                      
    import winsound
except ImportError:
    winsound = None

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR = True

WINDOW_CAMERA = "Camera Tracking"
WINDOW_GAME = "Virtual Maze"
WINDOW_MASK = "Mask (Hasil Segmentasi)"
WINDOW_CTRL = "Controls (HSV & Morfologi)"

DEFAULT_HSV = {
    "H Min": 100, "H Max": 130,
    "S Min": 120, "S Max": 255,
    "V Min": 70,  "V Max": 255,
}
DEFAULT_KERNEL = 2
DEFAULT_ITER = 2

MIN_CONTOUR_AREA = 1500
SMOOTH_MODE = "ema"
EMA_ALPHA = 0.35
MA_WINDOW = 5

WIRE_CONTROL_POINTS = [
    (0.14, 0.80), (0.14, 0.52), (0.17, 0.40), (0.22, 0.50), (0.25, 0.32),
    (0.30, 0.16), (0.36, 0.18), (0.39, 0.34), (0.41, 0.55), (0.46, 0.64),
    (0.52, 0.58), (0.54, 0.42), (0.58, 0.32), (0.63, 0.40), (0.66, 0.55),
    (0.71, 0.48), (0.74, 0.30), (0.79, 0.16), (0.83, 0.22), (0.86, 0.42),
    (0.86, 0.80),
]
LOOP_RADIUS_RATIO = 0.05   
WIRE_THICKNESS = 4

COLLISION_RADIUS = 0

GAME_BG_LEVELS = [0.6, 1.0, 0.0]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURSOR_ASSET_PATH = os.path.join(BASE_DIR, "assets", "cursor.png")
CURSOR_HIT_ASSET_PATH = os.path.join(BASE_DIR, "assets", "cursor_hit.png")  
ASSET_SCALE = 1.8          
ASSET_ANCHOR = (0.5, 0.5)  

MAX_LIVES = 3              
TIME_LIMIT = 60.0          
HOLD_TO_START = 1.2        
HOLD_TO_RESUME = 0.5       
HIT_FLASH = 0.5            
SOUND_ENABLED = True      
HIGHSCORE_PATH = os.path.join(BASE_DIR, "highscore.json")


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
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iters)


def find_largest_contour(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest if cv2.contourArea(largest) >= min_area else None


def compute_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


class CursorSmoother:
    def __init__(self, mode="ema", alpha=0.35, window=5):
        self.mode = mode
        self.alpha = alpha
        self.history = deque(maxlen=window)
        self.ema = None

    def update(self, x, y):
        if self.mode == "ma":
            self.history.append((x, y))
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


def catmull_rom(points, samples=16):
    pts = np.array(points, dtype=np.float64)
    pts = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for t in np.linspace(0.0, 1.0, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-2])
    return np.array(out)


def build_maze(width, height):
    loop_r = max(12, int(LOOP_RADIUS_RATIO * height))
    corridor_half = loop_r - WIRE_THICKNESS // 2 - 1

    ctrl = [(x * width, y * height) for x, y in WIRE_CONTROL_POINTS]
    wire_pts = np.round(catmull_rom(ctrl)).astype(np.int32)

    maze = np.full((height, width), 255, dtype=np.uint8)
    cv2.polylines(maze, [wire_pts], False, 0, thickness=2 * corridor_half)
    for p in wire_pts:
        cv2.circle(maze, (int(p[0]), int(p[1])), corridor_half, 0, -1)

    start = (int(wire_pts[0][0]), int(wire_pts[0][1]))
    finish = (int(wire_pts[-1][0]), int(wire_pts[-1][1]))
    return maze, wire_pts, start, finish, loop_r


def draw_label(img, text, center, bg, fg):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = center
    p1 = (x - tw // 2 - 8, y - th // 2 - 7)
    p2 = (x + tw // 2 + 8, y + th // 2 + 7)
    cv2.rectangle(img, p1, p2, bg, -1)
    cv2.rectangle(img, p1, p2, (60, 60, 60, 255), 1)
    cv2.putText(img, text, (x - tw // 2, y + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)


def render_wire_layer(maze, wire_pts, start, finish, loop_r, show_guide=True):
    h, w = maze.shape
    layer = np.zeros((h, w, 4), dtype=np.uint8)

    if show_guide:
        layer[maze == 0] = (36, 36, 36, 36)

    cv2.polylines(layer, [wire_pts], False, (30, 30, 30, 255), WIRE_THICKNESS + 2, cv2.LINE_AA)
    cv2.polylines(layer, [wire_pts], False, (205, 205, 210, 255), WIRE_THICKNESS, cv2.LINE_AA)
    cv2.polylines(layer, [wire_pts - 1], False, (250, 250, 250, 255), 1, cv2.LINE_AA)

    for cx in (start[0], finish[0]):
        cy = int(0.62 * h)
        cv2.rectangle(layer, (cx - 5, cy - 10), (cx + 5, cy + 10), (30, 30, 200, 255), -1)

    cv2.circle(layer, start, loop_r, (0, 200, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(layer, finish, loop_r, (0, 200, 255, 255), 2, cv2.LINE_AA)

    label_y = int(0.87 * h)
    draw_label(layer, "START", (start[0], label_y), (245, 245, 245, 255), (20, 20, 20, 255))
    draw_label(layer, "FINISH", (finish[0], label_y), (90, 180, 60, 255), (255, 255, 255, 255))
    return layer


def prepare_layer(layer):
    premult = layer[..., :3].astype(np.float32)
    inv_alpha = 1.0 - layer[..., 3:4].astype(np.float32) / 255.0
    return premult, inv_alpha


def compose_game_view(frame, prepared, brightness):
    premult, inv_alpha = prepared
    bg = frame.astype(np.float32) * brightness
    return np.clip(bg * inv_alpha + premult, 0, 255).astype(np.uint8)


def check_collision(maze, pt, radius):
    h, w = maze.shape
    x, y = pt
    if not (0 <= x < w and 0 <= y < h):
        return True
    if radius <= 0:
        return maze[y, x] == 255

    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    roi = maze[y0:y1, x0:x1]
    circle = np.zeros_like(roi)
    cv2.circle(circle, (x - x0, y - y0), radius, 255, -1)
    return cv2.countNonZero(cv2.bitwise_and(roi, circle)) > 0


def in_zone(pt, center, radius):
    return math.hypot(pt[0] - center[0], pt[1] - center[1]) <= radius


def _load_rgba(path, target_w):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)      
    h, w = img.shape[:2]
    target_h = max(1, int(round(h * target_w / w)))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def load_cursor_assets(loop_r):
    target_w = max(8, int(2 * loop_r * ASSET_SCALE))
    normal = _load_rgba(CURSOR_ASSET_PATH, target_w)
    if normal is None:
        print(f"[WARN] Aset kursor tidak ditemukan/terbaca: {CURSOR_ASSET_PATH}")
        print("       Memakai cincin sebagai cadangan.")
        return None, None

    hit = _load_rgba(CURSOR_HIT_ASSET_PATH, target_w)
    if hit is None:                                        
        hit = normal.copy()
        red = np.array([0, 0, 255], dtype=np.float32)
        hit[..., :3] = (0.5 * hit[..., :3] + 0.5 * red).astype(np.uint8)
    print(f"[INFO] Aset kursor dimuat: {CURSOR_ASSET_PATH} ({normal.shape[1]}x{normal.shape[0]} px)")
    return normal, hit


def overlay_rgba(dst, asset, center, anchor=(0.5, 0.5)):
    ah, aw = asset.shape[:2]
    x0 = int(center[0] - anchor[0] * aw)
    y0 = int(center[1] - anchor[1] * ah)
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + aw, dst.shape[1]), min(y0 + ah, dst.shape[0])
    if cx0 >= cx1 or cy0 >= cy1:
        return
    patch = asset[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    roi = dst[cy0:cy1, cx0:cx1].astype(np.float32)
    dst[cy0:cy1, cx0:cx1] = (patch[..., :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)


SOUND_ALIASES = {
    "start": "SystemDefault",
    "hit": "SystemHand",
    "win": "SystemAsterisk",
    "gameover": "SystemExclamation",
}


def play_sound(kind):
    if not SOUND_ENABLED or winsound is None:
        return
    try:
        winsound.PlaySound(SOUND_ALIASES[kind], winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception:
        pass


def load_best_time():
    try:
        with open(HIGHSCORE_PATH, "r", encoding="utf-8") as f:
            return float(json.load(f)["best_time"])
    except Exception:
        return None


def save_best_time(value):
    try:
        with open(HIGHSCORE_PATH, "w", encoding="utf-8") as f:
            json.dump({"best_time": round(value, 2)}, f)
    except Exception as e:
        print(f"[WARN] Gagal menyimpan rekor: {e}")


READY, PLAYING, RETURN, WIN, GAME_OVER = "READY", "PLAYING", "RETURN", "WIN", "GAME_OVER"


class Game:
    def __init__(self, start, finish, zone_r):
        self.start = start
        self.finish = finish
        self.zone_r = zone_r
        self.best_time = load_best_time()
        self.reset()

    def reset(self):
        self.state = READY
        self.lives = MAX_LIVES
        self.hits = 0
        self.time_start = None
        self.final_elapsed = 0.0
        self.hold_since = None
        self.hold_progress = 0.0
        self.prev_collided = False
        self.hit_flash_until = 0.0
        self.reason = ""
        self.score = 0
        self.new_record = False

    def elapsed(self, now):
        if self.time_start is None:
            return 0.0
        if self.state in (WIN, GAME_OVER):
            return self.final_elapsed
        return now - self.time_start

    def remaining(self, now):
        return max(0.0, TIME_LIMIT - self.elapsed(now))

    def _finish_round(self, new_state, now):
        self.state = new_state
        self.final_elapsed = now - self.time_start

    def update(self, detected, cursor, collided, now):
        events = []

        if self.state in (READY, RETURN):
            need = HOLD_TO_START if self.state == READY else HOLD_TO_RESUME
            if detected and in_zone(cursor, self.start, self.zone_r):
                if self.hold_since is None:
                    self.hold_since = now
                self.hold_progress = min(1.0, (now - self.hold_since) / need)
                if self.hold_progress >= 1.0:
                    if self.state == READY:
                        self.time_start = now
                        events.append("start")
                    self.state = PLAYING
                    self.hold_since, self.hold_progress = None, 0.0
                    self.prev_collided = False
            else:
                self.hold_since, self.hold_progress = None, 0.0

        if self.state in (PLAYING, RETURN) and self.remaining(now) <= 0:
            self.reason = "WAKTU HABIS"
            self._finish_round(GAME_OVER, now)
            events.append("gameover")
            return events

        if self.state == PLAYING and detected:
            if collided and not self.prev_collided:
                self.lives -= 1
                self.hits += 1
                self.hit_flash_until = now + HIT_FLASH
                if self.lives <= 0:
                    self.reason = "NYAWA HABIS"
                    self._finish_round(GAME_OVER, now)
                    events.append("gameover")
                else:
                    self.state = RETURN
                    self.hold_since, self.hold_progress = None, 0.0
                    events.append("hit")
            elif not collided and in_zone(cursor, self.finish, self.zone_r):
                self._finish_round(WIN, now)
                t = self.final_elapsed
                self.score = max(0, int(1000 + 300 * self.lives - 10 * t))
                if self.best_time is None or t < self.best_time:
                    self.best_time = t
                    self.new_record = True
                    save_best_time(t)
                events.append("win")
            self.prev_collided = collided
        elif not detected:
            self.prev_collided = False

        return events


def put_text_outline(img, text, org, scale, color, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def put_text_center(img, text, y, scale, color, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    put_text_outline(img, text, ((img.shape[1] - tw) // 2, y), scale, color, thickness)


def draw_heart(img, center, size, filled):
    t = np.linspace(0, 2 * np.pi, 40)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([center[0] + x * size / 17.0, center[1] + y * size / 17.0], axis=1)
    pts = pts.astype(np.int32)
    if filled:
        cv2.fillPoly(img, [pts], (50, 50, 230))
        cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts], True, (160, 160, 160), 2, cv2.LINE_AA)


def draw_camera_hud(view, fps, mode, detected, game_state):
    cv2.putText(view, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    status = "TERDETEKSI" if detected else "TIDAK TERDETEKSI"
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(view, f"Tangan: {status}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(view, f"Smoothing: {mode.upper()} | Game: {game_state}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    h = view.shape[0]
    cv2.putText(view, "q: keluar | SPACE: ulang game | p: print HSV | r: reset HSV",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view, "m: smoothing | g: pita panduan | b: latar | h: debug",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_tracking(view, contour, raw_pt, smooth_pt):
    cv2.drawContours(view, [contour], -1, (0, 255, 255), 2)
    cv2.circle(view, raw_pt, 5, (0, 0, 255), -1)
    cv2.circle(view, smooth_pt, 10, (0, 255, 0), -1)
    cv2.circle(view, smooth_pt, 12, (255, 255, 255), 2)


def draw_cursor(view, cursor, loop_r, show_hit, assets, debug_ring):
    normal, hit = assets
    if normal is not None:
        overlay_rgba(view, hit if show_hit else normal, cursor, ASSET_ANCHOR)
    else:
        ring = (0, 0, 255) if show_hit else (255, 255, 255)
        cv2.circle(view, cursor, loop_r, ring, 3, cv2.LINE_AA)
        cv2.circle(view, cursor, 2, ring, -1)
    if debug_ring:
        cv2.circle(view, cursor, loop_r, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(view, cursor, 2, (0, 255, 255), -1)


def draw_led(view, on):
    led = (view.shape[1] - 32, 30)
    if on:
        cv2.circle(view, led, 18, (0, 0, 120), -1)
        cv2.circle(view, led, 10, (0, 0, 255), -1)
    else:
        cv2.circle(view, led, 10, (0, 0, 70), -1)
    cv2.circle(view, led, 12, (200, 200, 200), 1, cv2.LINE_AA)


def draw_game_hud(view, game, detected, now):
    h, w = view.shape[:2]

    for i in range(MAX_LIVES):
        draw_heart(view, (26 + i * 34, 30), 12, filled=(i < game.lives))

    remaining = game.remaining(now)
    tcolor = (255, 255, 255) if remaining > 10 else (60, 60, 255)
    put_text_center(view, f"WAKTU {remaining:4.1f}", 34, 0.8, tcolor, 2)
    if game.best_time is not None:
        put_text_center(view, f"Terbaik: {game.best_time:.1f} dtk", 56, 0.5, (210, 210, 210), 1)

    if game.state in (READY, PLAYING, RETURN) and not detected:
        msg, col = "TANGAN TIDAK TERDETEKSI", (0, 165, 255)
    elif game.state == READY:
        msg, col = "Taruh tangan di lingkaran START", (255, 255, 255)
    elif game.state == RETURN:
        msg, col = "KENA! Kembali ke lingkaran START", (80, 80, 255)
    elif game.state == PLAYING:
        msg, col = "Ikuti kawat sampai FINISH", (0, 230, 0)
    else:
        msg = ""
    if msg:
        put_text_center(view, msg, h - 20, 0.6, col, 2)


def draw_start_progress(view, game):
    if game.hold_progress > 0:
        cv2.ellipse(view, game.start, (game.zone_r + 7, game.zone_r + 7), 0, -90,
                    -90 + 360 * game.hold_progress, (0, 255, 255), 4, cv2.LINE_AA)


def draw_end_screen(view, game):
    h, w = view.shape[:2]
    view[:] = (view.astype(np.float32) * 0.35).astype(np.uint8)

    if game.state == WIN:
        put_text_center(view, "YOU WIN!", h // 2 - 50, 1.8, (0, 230, 0), 4)
        put_text_center(view, f"Waktu: {game.final_elapsed:.1f} dtk   Skor: {game.score}",
                        h // 2 - 5, 0.8, (255, 255, 255), 2)
        if game.new_record:
            put_text_center(view, "REKOR BARU!", h // 2 + 30, 0.8, (0, 215, 255), 2)
    else:
        put_text_center(view, "GAME OVER", h // 2 - 50, 1.8, (60, 60, 255), 4)
        put_text_center(view, game.reason, h // 2 - 5, 0.8, (255, 255, 255), 2)
        put_text_center(view, f"Tabrakan: {game.hits}", h // 2 + 30, 0.7, (200, 200, 200), 2)

    put_text_center(view, "Tekan SPACE untuk main lagi", h // 2 + 80, 0.7, (255, 255, 0), 2)


def draw_game_view(view, game, cursor, loop_r, detected, collided, assets, debug_ring, now):
    show_hit = now < game.hit_flash_until or (game.state == PLAYING and collided)

    if detected:
        draw_cursor(view, cursor, loop_r, show_hit, assets, debug_ring)
    draw_start_progress(view, game)
    draw_led(view, show_hit)
    draw_game_hud(view, game, detected, now)

    if game.state in (WIN, GAME_OVER):
        draw_end_screen(view, game)


def main() -> None:
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW_GAME, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_AUTOSIZE)
    create_controls()

    ret, first = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame pertama.")
        sys.exit(1)
    fh, fw = first.shape[:2]

    cv2.moveWindow(WINDOW_CAMERA, 0, 0)
    cv2.moveWindow(WINDOW_GAME, fw + 20, 0)

    maze, wire_pts, start, finish, loop_r = build_maze(fw, fh)
    layers = {
        True: prepare_layer(render_wire_layer(maze, wire_pts, start, finish, loop_r, True)),
        False: prepare_layer(render_wire_layer(maze, wire_pts, start, finish, loop_r, False)),
    }
    assets = load_cursor_assets(loop_r)
    game = Game(start, finish, loop_r)

    show_guide = True
    debug_ring = False
    bg_index = 0

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

        camera_view = frame.copy()  

        lower, upper, ksize, iters = read_controls()
        mask = clean_mask(segment_color(frame, lower, upper), ksize, iters)

        detected = False
        smooth_pt = (0, 0)
        contour = find_largest_contour(mask, MIN_CONTOUR_AREA)
        if contour is not None:
            raw_pt = compute_centroid(contour)
            if raw_pt is not None:
                smooth_pt = smoother.update(*raw_pt)
                draw_tracking(camera_view, contour, raw_pt, smooth_pt)
                detected = True
        if not detected:
            smoother.reset()

        now = time.time()
        collided = detected and check_collision(maze, smooth_pt, COLLISION_RADIUS)
        for event in game.update(detected, smooth_pt, collided, now):
            play_sound(event)

        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        draw_camera_hud(camera_view, fps, smooth_mode, detected, game.state)
        cv2.imshow(WINDOW_CAMERA, camera_view)

        game_view = compose_game_view(frame, layers[show_guide], GAME_BG_LEVELS[bg_index])
        draw_game_view(game_view, game, smooth_pt, loop_r, detected, collided,
                       assets, debug_ring, now)
        cv2.imshow(WINDOW_GAME, game_view)

        cv2.imshow(WINDOW_MASK, mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == 32:                       
            game.reset()
            print("[INFO] Game diulang.")
        elif key == ord("p"):
            print(f"[HSV] lower={lower.tolist()} upper={upper.tolist()} "
                  f"kernel={ksize} iterasi={iters}")
        elif key == ord("r"):
            reset_controls()
        elif key == ord("g"):
            show_guide = not show_guide
            print(f"[INFO] Pita panduan: {'TAMPIL' if show_guide else 'SEMBUNYI (mode sulit)'}")
        elif key == ord("b"):
            bg_index = (bg_index + 1) % len(GAME_BG_LEVELS)
            print(f"[INFO] Kecerahan latar game: {GAME_BG_LEVELS[bg_index]}")
        elif key == ord("h"):
            debug_ring = not debug_ring
        elif key == ord("m"):
            smooth_mode = "ma" if smooth_mode == "ema" else "ema"
            smoother = CursorSmoother(smooth_mode, EMA_ALPHA, MA_WINDOW)
            print(f"[INFO] Mode smoothing: {smooth_mode.upper()}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program selesai.")


if __name__ == "__main__":
    main()