# Virtual Maze – Wire Loop Game Berbasis Deteksi Tangan

**Project 1: Mini Game Based on Hand Detection**
Mata kuliah **Pengolahan Citra dan Video (PCV)**

Game *wire loop* yang dikendalikan dengan gerakan tangan (memakai **sarung tangan biru**) melalui webcam. Dibuat **tanpa** library machine learning / hand tracking (tanpa MediaPipe, YOLO, maupun OpenPose); seluruh deteksi memakai pengolahan citra klasik dengan OpenCV.

## Daftar Isi

1. [Penjelasan Project Game](#1-penjelasan-project-game-virtual-maze)
2. [Tema Project](#2-tema-project)
3. [Fitur Utama](#3-fitur-utama)
4. [Implementasi Requirement](#4-implementasi-requirement)
5. [Struktur dan Arsitektur Project](#5-struktur-dan-arsitektur-project)
6. [Cara Menjalankan](#6-cara-menjalankan)
7. [Screenshot dan Demo](#7-screenshot-dan-demo)
8. [Catatan Pengembangan](#8-catatan-pengembangan)

---

## 1. Penjelasan Project Game (Virtual Maze)

**Virtual Maze** adalah versi digital dari permainan ketangkasan *wire loop*: pemain harus menggerakkan sebuah cincin sepanjang kawat berkelok tanpa menyentuhnya. Pada versi ini tidak ada controller fisik. Webcam menangkap tangan pemain, sistem mencari posisinya, dan posisi itulah yang menjadi kursor di dalam game.

**Cara bermain:**

1. Pakai sarung tangan biru dan arahkan ke kamera.
2. Gerakkan tangan sampai kursor berada di lingkaran **START** (hijau), lalu **tahan 1,2 detik** untuk memulai.
3. Ikuti kawat melewati koridor berkelok menuju lingkaran **FINISH** (kuning).
4. Menyentuh dinding mengurangi **1 nyawa** dan pemain harus kembali ke START (tahan 0,5 detik) untuk melanjutkan. Timer tetap berjalan.
5. **Menang** jika sampai FINISH tanpa menabrak. **Kalah** jika nyawa (3) atau waktu (60 detik) habis.

Skor saat menang: `max(0, 1000 + 300 × sisa_nyawa − 10 × waktu_detik)`.

**Alur status game:**

```
READY ──(tahan kursor di START 1,2 dtk)──► PLAYING ──(masuk FINISH)──► WIN
                                              │
                                  menabrak dinding: nyawa -1
                                              │
                          nyawa > 0 ──► RETURN ──(tahan di START 0,5 dtk)──► PLAYING
                          nyawa = 0 ──► GAME OVER

Waktu habis (60 dtk) saat PLAYING / RETURN ──► GAME OVER
```

---

## 2. Tema Project

**Interaksi tanpa sentuhan (touchless) berbasis pengolahan citra klasik.**

Proyek ini menunjukkan bahwa sebuah antarmuka berbasis gerakan tangan bisa dibangun hanya dengan teknik dasar pengolahan citra, tanpa model ML atau library hand tracking siap pakai. Tantangan utamanya ada pada pipeline visi komputer: segmentasi warna, pembersihan noise, pencarian objek, dan penghitungan posisi, yang kemudian dipakai sebagai input permainan ketangkasan.

---

## 3. Fitur Utama

- **Deteksi tangan murni pengolahan citra klasik**: HSV, morfologi, kontur, image moments.
- **Trackbar interaktif** untuk rentang HSV, ukuran kernel, dan jumlah iterasi morfologi, sehingga mudah dikalibrasi untuk kondisi cahaya dan warna sarung tangan yang berbeda.
- **Kursor stabil** dengan *smoothing* (Exponential Moving Average atau Moving Average, dapat diganti saat program berjalan).
- **Labirin berupa mask biner** yang dibuat otomatis dari kurva kawat (spline Catmull-Rom), mudah diubah bentuknya lewat daftar titik kontrol.
- **Deteksi tabrakan berbasis pixel** pada mask labirin.
- **Game lengkap**: nyawa, timer, skor, *hold-to-start*, layar YOU WIN dan GAME OVER.
- **Aset kursor PNG** (opsional), dengan cincin sebagai cadangan.
- **Tiga tampilan sekaligus**: kamera dengan tracking, tampilan game, dan hasil segmentasi (mask).
- **Mode debug** untuk melihat titik persis yang dipakai dalam deteksi tabrakan.

---

## 4. Implementasi Requirement

| Requirement | Implementasi | Lokasi di `main.py` |
|---|---|---|
| Tanpa ML / hand tracking pihak ketiga | Hanya memakai `cv2` (OpenCV) dan `numpy` | seluruh file |
| Konversi BGR → HSV | `cv2.cvtColor(..., cv2.COLOR_BGR2HSV)` setelah Gaussian Blur 5×5 untuk meredam noise | `segment_color()` |
| Segmentasi warna dengan `cv2.inRange()` dan rentang yang dapat diatur | 6 trackbar (H/S/V min-max) pada jendela *Controls*; `p` mencetak nilai, `r` me-reset | `create_controls()`, `read_controls()`, `segment_color()` |
| Operasi morfologi (Opening / Closing) | Opening lalu Closing dengan kernel elips; ukuran kernel dan iterasi dari trackbar | `clean_mask()` |
| Kontur terbesar (`cv2.findContours`) | `RETR_EXTERNAL`, ambil kontur dengan area terbesar, abaikan jika < 1500 px | `find_largest_contour()` |
| Image moments (M00, M10, M01) → centroid | `cx = M10 / M00`, `cy = M01 / M00` | `compute_centroid()` |
| Smoothing kursor (Moving Average / Exponential Smoothing) | Kelas `CursorSmoother` dengan mode `ema` (α = 0,35) dan `ma` (window 5); di-reset saat tangan hilang; `m` untuk ganti mode | `CursorSmoother` |
| Peta labirin biner (hitam = jalur aman, putih = dinding) | Dibangun dengan fungsi gambar OpenCV (`polylines`, `circle`) dari kurva Catmull-Rom | `catmull_rom()`, `build_maze()` |
| Collision detection berbasis nilai pixel | Baca pixel `mask[y, x]` pada posisi kursor: putih (255) = tabrakan; di luar frame juga dihitung tabrakan | `check_collision()` |
| Konsekuensi tabrakan | Nyawa berkurang dan pemain harus kembali ke garis START | `Game.update()` |
| Feed kamera sebagai latar, berdampingan dengan game board | Jendela *Camera Tracking* dan *Virtual Maze* dibuka berdampingan; latar kamera di jendela game diredupkan (0,6) | `compose_game_view()`, `main()` |
| Timer, nyawa, dan status game (START, PLAYING, GAME OVER, YOU WIN) | Timer dan ikon hati pada HUD; status READY/PLAYING/RETURN/WIN/GAME_OVER dikelola kelas `Game` | `Game`, `draw_game_hud()`, `draw_end_screen()` |

---

## 5. Struktur dan Arsitektur Project

### Struktur direktori

```
.
├── main.py              # seluruh program: deteksi tangan + labirin + game
├── assets/
│   ├── cursor.png       # kursor normal (opsional)
│   └── cursor_hit.png   # kursor saat menabrak (opsional)
└── README.md
```

### Arsitektur pipeline

Setiap frame melewati alur berikut di dalam loop utama:

```
Webcam
  │  flip horizontal (mirror)
  ▼
┌──────────────── DETEKSI TANGAN ────────────────┐
│ Gaussian Blur → BGR→HSV → inRange (trackbar)   │
│ → Opening → Closing → kontur terbesar          │
│ → image moments → centroid → smoothing         │
└───────────────────────┬────────────────────────┘
                        │ posisi kursor (cx, cy)
                        ▼
┌──────────────── LOGIKA GAME ───────────────────┐
│ check_collision(maze_mask, cursor)             │
│ Game.update(): state, nyawa, timer, skor       │
└───────────────────────┬────────────────────────┘
                        ▼
┌──────────────── TAMPILAN ──────────────────────┐
│ Camera Tracking │ Virtual Maze │ Mask           │
└────────────────────────────────────────────────┘
```

### Pembagian fungsi dalam `main.py`

| Bagian | Fungsi / kelas utama |
|---|---|
| Kamera | `open_camera()` |
| Segmentasi dan morfologi | `create_controls()`, `read_controls()`, `reset_controls()`, `segment_color()`, `clean_mask()` |
| Pelacakan tangan | `find_largest_contour()`, `compute_centroid()`, `CursorSmoother` |
| Labirin dan rendering | `catmull_rom()`, `build_maze()`, `render_wire_layer()`, `prepare_layer()`, `compose_game_view()` |
| Collision | `check_collision()`, `in_zone()` |
| Aset kursor | `load_cursor_assets()`, `overlay_rgba()` |
| State game | `Game` (READY, PLAYING, RETURN, WIN, GAME_OVER) |
| HUD dan layar | `draw_game_hud()`, `draw_start_progress()`, `draw_end_screen()`, `draw_camera_hud()`, `draw_cursor()`, `draw_tracking()` |
| Program utama | `main()` |

---

## 6. Cara Menjalankan

### Persyaratan

- Python 3.8 atau lebih baru
- Webcam
- Library: `opencv-python` dan `numpy`

### Instalasi

```bash
# (opsional) buat virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install opencv-python numpy
```

### Menjalankan

```bash
python main.py
```

Akan muncul empat jendela:

| Jendela | Isi |
|---|---|
| **Camera Tracking** | Feed kamera, kontur tangan, titik centroid mentah (merah) dan terhaluskan (hijau), FPS |
| **Virtual Maze** | Tampilan game: kawat, START/FINISH, kursor, nyawa, timer |
| **Mask (Hasil Segmentasi)** | Mask biner hasil segmentasi warna dan morfologi |
| **Controls (HSV & Morfologi)** | Trackbar H/S/V min-max, ukuran kernel, jumlah iterasi |

### Kalibrasi sebelum bermain

1. Pakai sarung tangan biru dan arahkan ke kamera.
2. Perhatikan jendela **Mask**: tangan harus tampak sebagai blok putih bersih di atas latar hitam.
3. Atur trackbar **H/S/V** sampai hasilnya bersih. Naikkan **Kernel** atau **Iterasi** untuk membuang bintik-bintik putih kecil.
4. Tekan `p` untuk mencetak nilai HSV ke terminal, lalu salin ke `DEFAULT_HSV` di `main.py` agar menjadi nilai awal.

Nilai awal HSV (skala OpenCV, H 0–179): **H 100–130, S 120–255, V 70–255** (biru).

> Jika tidak memakai sarung tangan biru, sesuaikan rentang HSV lewat trackbar. Untuk warna kulit, mulailah dari kira-kira H 0–20, S 30–150, V 60–255.

### Kontrol keyboard

| Tombol | Fungsi |
|---|---|
| `SPACE` | Ulang game |
| `q` / `ESC` | Keluar |
| `p` | Cetak nilai HSV, kernel, dan iterasi saat ini ke terminal |
| `r` | Reset trackbar ke nilai awal |
| `m` | Ganti mode smoothing (EMA ⇄ Moving Average) |
| `h` | Tampilkan/sembunyikan debug (cincin kuning dan titik collision) |

### Konfigurasi

Parameter ada di bagian atas `main.py`:

| Konstanta | Fungsi | Default |
|---|---|---|
| `CAMERA_INDEX` | Indeks webcam | `0` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Resolusi kamera | `640` / `480` |
| `MIRROR` | Balik frame secara horizontal | `True` |
| `DEFAULT_HSV` | Rentang HSV awal | biru (H 100–130) |
| `MIN_CONTOUR_AREA` | Area minimum kontur agar dianggap tangan | `1500` |
| `SMOOTH_MODE` | `"ema"` atau `"ma"` | `"ema"` |
| `EMA_ALPHA` | Faktor EMA (kecil = lebih halus tapi lebih lambat) | `0.35` |
| `MA_WINDOW` | Jumlah frame untuk Moving Average | `5` |
| `WIRE_CONTROL_POINTS` | Titik kontrol bentuk kawat (rasio 0–1) | lihat kode |
| `LOOP_RADIUS_RATIO` | Radius cincin kursor dan zona START/FINISH (rasio tinggi frame) | `0.05` |
| `COLLISION_RADIUS` | `0` = tepat 1 pixel; `>0` = patch lingkaran | `0` |
| `GAME_BG_BRIGHTNESS` | Kecerahan feed kamera di jendela game | `0.6` |
| `MAX_LIVES` | Jumlah nyawa | `3` |
| `TIME_LIMIT` | Batas waktu (detik) | `60` |
| `HOLD_TO_START` | Lama menahan di START untuk mulai (detik) | `1.2` |
| `HOLD_TO_RESUME` | Lama menahan di START setelah menabrak (detik) | `0.5` |
| `HIT_FLASH` | Lama kursor berwarna merah setelah menabrak (detik) | `0.5` |

### Aset kursor (opsional)

Kursor dapat diganti dengan PNG berlatar transparan di folder `assets/`:

- `assets/cursor.png`: kursor normal.
- `assets/cursor_hit.png`: kursor saat menabrak. Jika tidak ada, kursor normal diberi semburat merah otomatis.

Jika `assets/cursor.png` tidak ditemukan, game memakai kursor berbentuk **cincin** sebagai cadangan. Ukuran aset diskalakan otomatis, sehingga ukuran file aslinya bebas.

---

## 7. Screenshot dan Demo

### Screenshot

| Camera Tracking | Virtual Maze | Mask Segmentasi |
|---|---|---|
| ![Camera Tracking](assets/Camera-Tracking.png) | ![Virtual Maze](access/Virtual-Maze.png) | ![Mask](access/Mask-Hasil-Segmentasi.png) |

| Trackbar HSV | Layar YOU WIN | Layar GAME OVER |
|---|---|---|
| ![Controls](assets/Trackbar%20HSV.png) | ![Game Over](assets/Layar%20Game%20Over.png) |

```### Demo

![Demo gameplay](docs/demo.gif)

Video lengkap: *(tautan video demo)*```

---

## 8. Catatan Pengembangan

### Tahapan pengembangan

Proyek dibangun bertahap, satu milestone per commit (gaya *Conventional Commits*):

| Tahap | Isi | Contoh pesan commit |
|---|---|---|
| 1 | Inisialisasi webcam dan jendela dasar | `feat: add webcam capture and base windows` |
| 2 | Segmentasi HSV, trackbar, dan operasi morfologi | `feat: add HSV segmentation with morphological filtering` |
| 3 | Kontur terbesar, centroid (image moments), dan smoothing | `feat: add centroid tracking with cursor smoothing` |
| 4 | Peta labirin (mask) dan deteksi tabrakan pixel | `feat: add maze mask and pixel-based collision detection` |
| 5 | Game loop: nyawa, timer, skor, state, dan layar akhir | `feat: add game state machine with lives, timer, and score` |

### Keputusan desain

- **Collision berbasis mask, bukan geometri.** Labirin disimpan sebagai citra biner sehingga pengecekan tabrakan cukup membaca satu pixel (O(1)), dan bentuk jalur bisa diubah bebas tanpa mengubah logika collision.
- **Tabrakan dihitung per kejadian.** Nyawa hanya berkurang saat kursor *mulai* menyentuh dinding, bukan tiap frame selama menempel.
- **Smoother di-reset saat tangan hilang**, agar kursor tidak "meluncur" dari posisi lama ketika tangan muncul lagi.
- **Gaussian Blur sebelum konversi HSV** untuk meredam noise sensor sebelum `inRange`.
- **Hold-to-start** mencegah game mulai tidak sengaja dan memberi pemain waktu bersiap.
- **Frame di-mirror** supaya gerakan di layar searah dengan gerakan tangan.

### Keterbatasan

- Sangat bergantung pada **pencahayaan** dan **warna sarung tangan**; benda lain berwarna serupa di latar dapat ikut terdeteksi (yang dipakai adalah kontur terbesar).
- Kursor adalah **centroid seluruh tangan**, bukan ujung jari, sehingga kurang presisi untuk gerakan halus.
- Tabrakan dicek pada **satu titik** (centroid), sementara cincin kursor yang tampil lebih besar. Gunakan `COLLISION_RADIUS` untuk membuatnya lebih ketat.
- Hanya ada **satu level** labirin.

### Rencana pengembangan

- Menyimpan rekor waktu (*highscore*) dan menambah efek suara.
- Beberapa level atau generator labirin acak, serta tingkat kesulitan (lebar koridor, jumlah nyawa).
- Kalibrasi otomatis: klik pada tangan untuk mengambil sampel warna HSV.
- Deteksi ujung jari (misalnya dengan *convex hull* dan *convexity defects*) agar kontrol lebih presisi, tetap dengan teknik klasik.
- Memindahkan konfigurasi ke file terpisah (misalnya JSON) agar tidak perlu mengedit kode.

---

## Teknologi

- [Python](https://www.python.org/)
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)