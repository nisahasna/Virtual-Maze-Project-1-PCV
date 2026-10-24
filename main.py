# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:37:25 2026

@author: ASUS
"""

import cv2
import numpy as np
import time


HSV_LOWER_1 = np.array([0,  30,  50])
HSV_UPPER_1 = np.array([25, 255, 255])

HSV_LOWER_2 = np.array([165, 20,  50])
HSV_UPPER_2 = np.array([179, 180, 255])

MIN_CONTOUR_AREA = 3000

MORPH_KERNEL_SIZE = (5, 5)

NUM_ZONES = 4

COLOR_ZONE_LINE  = (200, 200, 200)
COLOR_CONTOUR    = (0, 255, 0) 
COLOR_CENTROID   = (0, 0, 255)   
COLOR_BBOX       = (255, 165, 0)
COLOR_TEXT_OK    = (0, 255, 100)
COLOR_TEXT_MISS  = (0, 60, 255)
COLOR_FPS        = (180, 180, 180)

def create_mask(frame: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, HSV_LOWER_1, HSV_UPPER_1)
    mask2 = cv2.inRange(hsv, HSV_LOWER_2, HSV_UPPER_2)
    mask  = cv2.bitwise_or(mask1, mask2)
    
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def detect_object(mask: np.ndarray):
    contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None
    
    best_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(best_contour) < MIN_CONTOUR_AREA:
        return None

    return best_contour

def get_centroid(contour: np.ndarray):
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def get_zone(cx: int, frame_width: int) -> int:
    zone_index = int(cx * NUM_ZONES / frame_width)
    return min(zone_index, NUM_ZONES - 1)

def draw_zones(frame: np.ndarray, active_zone: int | None = None) -> None:
    h, w = frame.shape[:2]
    zone_width = w // NUM_ZONES

    for i in range(1, NUM_ZONES):
        x = i * zone_width
        cv2.line(frame, (x, 0), (x, h), COLOR_ZONE_LINE, 1)
    
    for i in range(NUM_ZONES):
        label_x = i * zone_width + zone_width // 2 - 20
        cv2.putText(
            frame, f"Z{i + 1}",
            (label_x, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            COLOR_ZONE_LINE, 1, cv2.LINE_AA
        )
        
    if active_zone is not None:
        overlay = frame.copy()
        x1 = active_zone * zone_width
        x2 = x1 + zone_width
        cv2.rectangle(overlay, (x1, 0), (x2, h), (0, 255, 180), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
def draw_detection(
    frame: np.ndarray,
    contour,
    centroid,
    active_zone: int | None
) -> None:
    
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, COLOR_CONTOUR, 2)

        x, y, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_BBOX, 2)

    if centroid is not None:
        cx, cy = centroid
        
        cv2.circle(frame, (cx, cy), 8, COLOR_CENTROID, -1)
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 1)
    
        zone_label = f"Zone {active_zone + 1}" if active_zone is not None else ""
        cv2.putText(
            frame, zone_label,
            (cx + 15, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            COLOR_TEXT_OK, 2, cv2.LINE_AA
        )

def draw_status(
    frame: np.ndarray,
    detected: bool,
    active_zone: int | None,
    fps: float
) -> None:
    h, w = frame.shape[:2]

    if detected and active_zone is not None:
        status_text = f"DETECTED  |  Zone {active_zone + 1} of {NUM_ZONES}"
        color = COLOR_TEXT_OK
    else:
        status_text = "Object Not Detected"
        color = COLOR_TEXT_MISS

    cv2.putText(
        frame, status_text,
        (10, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        color, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        COLOR_FPS, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, "Q: Quit  |  R: Reset",
        (w - 220, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        COLOR_FPS, 1, cv2.LINE_AA
    )

# MAIN LOOP
def main() -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam tidak dapat dibuka.")
        print("        Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
        return

    print("[INFO] Webcam berhasil dibuka.")
    print("[INFO] Tekan Q untuk keluar, R untuk reset.")
    print(f"[INFO] Mendeteksi HSV range-1: {HSV_LOWER_1.tolist()} - {HSV_UPPER_1.tolist()}")
    print(f"[INFO] Mendeteksi HSV range-2: {HSV_LOWER_2.tolist()} - {HSV_UPPER_2.tolist()}")

    detection_reset = False

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame tidak berhasil dibaca. Mencoba lagi...")
            continue

        frame = cv2.flip(frame, 1)
        frame_w = frame.shape[1]

        mask = create_mask(frame)

        contour     = None
        centroid    = None
        active_zone = None

        if not detection_reset:
            contour = detect_object(mask)

            if contour is not None:
                centroid = get_centroid(contour)

            if centroid is not None:
                cx, _ = centroid
                active_zone = get_zone(cx, frame_w)

        detected = (centroid is not None)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        display_frame = frame.copy()
        draw_zones(display_frame, active_zone)
        draw_detection(display_frame, contour, centroid, active_zone)
        draw_status(display_frame, detected, active_zone, fps)
        
        cv2.imshow("ColorBeat | Hand Detection", display_frame)

        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            mask_display, "Binary Mask (HSV Segmentation)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (200, 200, 200), 1, cv2.LINE_AA
        )
        cv2.imshow("ColorBeat | Binary Mask", mask_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[INFO] Program dihentikan oleh pengguna.")
            break
        elif key == ord('r') or key == ord('R'):
            detection_reset = not detection_reset
            status = "AKTIF (deteksi dijeda)" if detection_reset else "NONAKTIF (deteksi berjalan)"
            print(f"[INFO] Reset toggle: {status}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam dilepaskan. Semua jendela ditutup.")


if __name__ == "__main__":
    main()