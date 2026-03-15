"""Kontrol model GLTF/GLB 3D dengan gesture tangan. Lihat README.md untuk dokumentasi."""

import os
import sys
import math
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

try:
    import trimesh
except ImportError:
    print("Jalankan dulu: pip install 'trimesh[easy]'")
    sys.exit(1)


# ============================================================
# Pengaturan MediaPipe
# ============================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

JALUR_MODEL_TANGAN = "hand_landmarker.task"
URL_MODEL_TANGAN = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(JALUR_MODEL_TANGAN):
    print("Mengunduh model MediaPipe...")
    urllib.request.urlretrieve(URL_MODEL_TANGAN, JALUR_MODEL_TANGAN)

KONEKSI_TANGAN = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def hitung_jari(landmarks, sisi_tangan):
    """Hitung jari yang terangkat. Mengembalikan list [jempol, telunjuk, tengah, manis, kelingking]."""
    jari = []
    ujung = [4, 8, 12, 16, 20]

    # Jempol: bandingkan posisi X ujung dengan ruas sebelumnya
    if sisi_tangan == "Right":
        jari.append(1 if landmarks[ujung[0]].x > landmarks[ujung[0] - 1].x else 0)
    else:
        jari.append(1 if landmarks[ujung[0]].x < landmarks[ujung[0] - 1].x else 0)

    # 4 jari lainnya: bandingkan posisi Y ujung dengan 2 ruas di bawahnya
    for i in range(1, 5):
        jari.append(1 if landmarks[ujung[i]].y < landmarks[ujung[i] - 2].y else 0)

    return jari


# ============================================================
# Muat Model GLTF
# ============================================================
FOLDER_MODEL = "blue_archive_shirokos_rifle"

daftar_file = sorted([
    f for f in os.listdir(FOLDER_MODEL)
    if f.lower().endswith((".gltf", ".glb"))
])

if not daftar_file:
    print(f"Tidak ada file .gltf / .glb di folder '{FOLDER_MODEL}/'")
    sys.exit(1)


def muat_gltf(jalur):
    """Muat file GLTF/GLB, bake tekstur ke warna vertex, normalisasi ukuran."""
    print(f"  Memuat: {os.path.basename(jalur)} ...", end=" ", flush=True)

    data = trimesh.load(jalur)

    if isinstance(data, trimesh.Scene):
        # Gabung semua sub-mesh secara manual agar warna tekstur terjaga
        daftar_vertex, daftar_face, daftar_warna = [], [], []
        offset = 0

        for nama, geom in data.geometry.items():
            if not hasattr(geom, "vertices"):
                continue

            # Bake tekstur menjadi warna per vertex
            try:
                warna = geom.visual.to_color().vertex_colors[:, :3]
            except Exception:
                warna = np.full((len(geom.vertices), 3), 180, dtype=np.uint8)

            # Terapkan transformasi dari scene graph
            try:
                transform = data.graph.get(nama)[0]
                titik = trimesh.transformations.transform_points(geom.vertices, transform)
            except Exception:
                titik = np.array(geom.vertices)

            daftar_vertex.append(titik)
            daftar_face.append(geom.faces + offset)
            daftar_warna.append(warna)
            offset += len(geom.vertices)

        vertices = np.vstack(daftar_vertex).astype(np.float64)
        faces = np.vstack(daftar_face).astype(np.int32)
        warna_vertex = np.vstack(daftar_warna).astype(np.float64) / 255.0
    else:
        vertices = np.array(data.vertices, dtype=np.float64)
        faces = np.array(data.faces, dtype=np.int32)
        try:
            wv = data.visual.to_color().vertex_colors[:, :3]
            warna_vertex = wv.astype(np.float64) / 255.0
        except Exception:
            warna_vertex = np.full((len(vertices), 3), 0.7)

    # Pusatkan ke origin
    pusat = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    vertices -= pusat

    # Normalisasi skala ke bola satuan (radius = 1)
    radius_maks = np.max(np.linalg.norm(vertices, axis=1))
    if radius_maks > 0:
        vertices /= radius_maks

    print(f"OK ({len(faces)} face)")
    return vertices, faces, warna_vertex


print("Memuat model GLTF...")
daftar_model = []
for f in daftar_file:
    try:
        v, fc, warna = muat_gltf(os.path.join(FOLDER_MODEL, f))
        daftar_model.append((f, v, fc, warna))
    except Exception as e:
        print(f"GAGAL: {e}")

if not daftar_model:
    print("Tidak ada model yang berhasil dimuat!")
    sys.exit(1)

print(f"Total {len(daftar_model)} model siap.\n")


# ============================================================
# Renderer Software
# ============================================================
UKURAN_RENDER = 320


def buat_rotasi(rx_derajat, ry_derajat):
    """Buat matriks rotasi 3x3 (sumbu Y dulu, lalu sumbu X)."""
    rx, ry = math.radians(rx_derajat), math.radians(ry_derajat)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    return np.array([
        [ cy,   sy * sx,  sy * cx],
        [  0,   cx,      -sx     ],
        [-sy,   cy * sx,  cy * cx],
    ], dtype=np.float64)


def render_mesh(vertices, faces, warna, rx, ry,
                ukuran=UKURAN_RENDER, jarak_z=2.5, fov=50):
    """Render mesh ke gambar BGRA (H x W x 4) menggunakan proyeksi perspektif."""
    R = buat_rotasi(rx, ry)

    # Transformasi vertex: rotasi lalu geser ke depan kamera
    titik = vertices @ R.T
    titik[:, 2] += jarak_z

    # Proyeksi perspektif ke koordinat layar
    focal = ukuran / (2.0 * math.tan(math.radians(fov / 2)))
    z_aman = titik[:, 2].clip(min=0.01)
    x2d = (titik[:, 0] / z_aman * focal + ukuran / 2).astype(np.int32)
    y2d = (-titik[:, 1] / z_aman * focal + ukuran / 2).astype(np.int32)

    # Ambil posisi 3D tiap sudut segitiga
    v0 = titik[faces[:, 0]]
    v1 = titik[faces[:, 1]]
    v2 = titik[faces[:, 2]]

    # Hitung normal tiap face
    normal = np.cross(v1 - v0, v2 - v0)
    panjang = np.linalg.norm(normal, axis=1, keepdims=True)
    normal /= (panjang + 1e-10)

    # Backface culling: sembunyikan face yang membelakangi kamera
    terlihat = normal[:, 2] < 0

    # Urutkan berdasarkan kedalaman (face terjauh digambar duluan)
    kedalaman = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3
    idx_terlihat = np.where(terlihat)[0]
    urutan = idx_terlihat[np.argsort(-kedalaman[idx_terlihat])]

    # Pencahayaan: arah cahaya dari kamera ke scene (sumbu +Z)
    arah_cahaya = np.array([-0.2, 0.3, 0.8], dtype=np.float64)
    arah_cahaya /= np.linalg.norm(arah_cahaya)
    kecerahan = np.clip(normal @ (-arah_cahaya), 0.25, 1.0)

    # Hitung warna akhir per face (rata-rata 3 vertex x kecerahan)
    warna_face = (warna[faces[:, 0]] + warna[faces[:, 1]] + warna[faces[:, 2]]) / 3
    warna_lit = warna_face * kecerahan[:, None] * 255
    warna_bgr = warna_lit[:, ::-1].clip(0, 255).astype(np.int32)

    # Siapkan koordinat segitiga
    titik_x = x2d[faces]
    titik_y = y2d[faces]

    # Gambar semua segitiga ke kanvas BGRA
    gambar = np.zeros((ukuran, ukuran, 4), dtype=np.uint8)

    for i in urutan:
        segitiga = np.array([
            [titik_x[i, 0], titik_y[i, 0]],
            [titik_x[i, 1], titik_y[i, 1]],
            [titik_x[i, 2], titik_y[i, 2]],
        ], dtype=np.int32)
        c = warna_bgr[i]
        cv2.fillConvexPoly(gambar, segitiga, (int(c[0]), int(c[1]), int(c[2]), 255))

    return gambar


def tempelkan_rgba(latar, rgba, cx, cy, skala=1.0):
    """Tempelkan gambar BGRA ke latar belakang dengan alpha blending."""
    lh, lw = latar.shape[:2]
    gh, gw = rgba.shape[:2]

    if skala != 1.0 and skala > 0:
        gw_baru = max(10, int(gw * skala))
        gh_baru = max(10, int(gh * skala))
        rgba = cv2.resize(rgba, (gw_baru, gh_baru), interpolation=cv2.INTER_LINEAR)
        gh, gw = rgba.shape[:2]

    x1, y1 = int(cx - gw / 2), int(cy - gh / 2)
    x2, y2 = x1 + gw, y1 + gh

    # Hitung area yang valid (tidak keluar layar)
    gx1 = max(0, -x1);  gy1 = max(0, -y1)
    gx2 = gw - max(0, x2 - lw)
    gy2 = gh - max(0, y2 - lh)
    lx1, ly1 = max(0, x1), max(0, y1)
    lx2 = lx1 + (gx2 - gx1)
    ly2 = ly1 + (gy2 - gy1)

    if gx2 <= gx1 or gy2 <= gy1:
        return latar

    area = latar[ly1:ly2, lx1:lx2].astype(np.float32)
    potongan = rgba[gy1:gy2, gx1:gx2]
    alpha = potongan[:, :, 3:4].astype(np.float32) / 255.0
    bgr = potongan[:, :, :3].astype(np.float32)

    latar[ly1:ly2, lx1:lx2] = (bgr * alpha + area * (1.0 - alpha)).astype(np.uint8)
    return latar


# ============================================================
# State Aplikasi
# ============================================================
indeks_model = 0
posisi_x, posisi_y = 320, 240
rotasi_x, rotasi_y = 20.0, 0.0
zoom = 1.0

prev_ix, prev_iy = None, None
prev_mid_x = None
prev_mid_y = None


# ============================================================
# Loop Utama
# ============================================================
opsi_mp = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=JALUR_MODEL_TANGAN),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Kamera aktif!")
print("Gesture: 1jari=Pindah  2jari=PutarY  3jari=PutarX  5jari=ZoomIn  Kepalan=ZoomOut  Jempol=Kunci")
print("Keyboard: R=Reset  N/P=Ganti model  Q=Keluar\n")

hitungan_frame = 0
waktu_fps = time.time()

with HandLandmarker.create_from_options(opsi_mp) as detektor:
    while True:
        berhasil, frame = kamera.read()
        if not berhasil:
            break

        frame = cv2.flip(frame, 1)
        t, l = frame.shape[:2]
        hitungan_frame += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gambar_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hasil = detektor.detect(gambar_mp)

        mode_saat_ini = "tidak_ada"

        if hasil.hand_landmarks:
            tangan = hasil.hand_landmarks[0]
            sisi = hasil.handedness[0][0].category_name
            jari = hitung_jari(tangan, sisi)
            jumlah_jari = sum(jari)

            # Koordinat ujung jari penting
            ix = int(tangan[8].x * l);   iy = int(tangan[8].y * t)
            mx = int(tangan[12].x * l);  my = int(tangan[12].y * t)
            rx = int(tangan[16].x * l);  ry = int(tangan[16].y * t)

            # Gambar kerangka tangan
            for lm in tangan:
                cv2.circle(frame, (int(lm.x * l), int(lm.y * t)), 3, (0, 200, 255), -1)
            for a, b in KONEKSI_TANGAN:
                cv2.line(frame,
                         (int(tangan[a].x * l), int(tangan[a].y * t)),
                         (int(tangan[b].x * l), int(tangan[b].y * t)),
                         (180, 180, 180), 1)

            # -- 1 JARI (telunjuk) -> Pindah model --
            if jari == [0, 1, 0, 0, 0]:
                mode_saat_ini = "pindah"
                if prev_ix is not None:
                    posisi_x += ix - prev_ix
                    posisi_y += iy - prev_iy
                prev_ix, prev_iy = ix, iy
                prev_mid_x = prev_mid_y = None
                cv2.circle(frame, (ix, iy), 12, (0, 255, 100), -1)
                cv2.circle(frame, (ix, iy), 15, (255, 255, 255), 2)

            # -- 2 JARI (V) -> Putar sumbu Y --
            elif jari[1] == 1 and jari[2] == 1 and jari[3] == 0 and jari[4] == 0:
                mode_saat_ini = "putar_y"
                tengah_x = (ix + mx) // 2
                if prev_mid_x is not None:
                    rotasi_y += (tengah_x - prev_mid_x) * 0.9
                prev_mid_x = tengah_x
                prev_ix = prev_iy = prev_mid_y = None
                cv2.arrowedLine(frame, (ix, iy), (mx, my), (255, 140, 0), 2)

            # -- 3 JARI -> Putar sumbu X --
            elif jari[1] == 1 and jari[2] == 1 and jari[3] == 1 and jari[4] == 0:
                mode_saat_ini = "putar_x"
                tengah_y = (iy + my + ry) // 3
                if prev_mid_y is not None:
                    rotasi_x += (tengah_y - prev_mid_y) * 0.9
                prev_mid_y = tengah_y
                prev_ix = prev_iy = prev_mid_x = None
                cv2.arrowedLine(frame, (ix, iy), (rx, ry), (0, 180, 255), 2)

            # -- 5 JARI (buka) -> Zoom in --
            elif jumlah_jari == 5:
                mode_saat_ini = "zoom_masuk"
                zoom = min(zoom + 0.04, 4.0)
                prev_ix = prev_iy = prev_mid_x = prev_mid_y = None

            # -- KEPALAN (tutup) -> Zoom out --
            elif jumlah_jari == 0:
                mode_saat_ini = "zoom_keluar"
                zoom = max(zoom - 0.04, 0.2)
                prev_ix = prev_iy = prev_mid_x = prev_mid_y = None

            # -- JEMPOL -> Kunci posisi --
            elif jari == [1, 0, 0, 0, 0]:
                mode_saat_ini = "kunci"
                prev_ix = prev_iy = prev_mid_x = prev_mid_y = None

            else:
                prev_ix = prev_iy = prev_mid_x = prev_mid_y = None
        else:
            prev_ix = prev_iy = prev_mid_x = prev_mid_y = None

        # -- Render model 3D dan tempelkan ke frame --
        nama_file, vert, face, warna = daftar_model[indeks_model]
        try:
            hasil_render = render_mesh(vert, face, warna, rotasi_x, rotasi_y)
            frame = tempelkan_rgba(frame, hasil_render, posisi_x, posisi_y, zoom)
        except Exception as e:
            cv2.putText(frame, f"Gagal render: {e}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # -- Hitung FPS --
        sekarang = time.time()
        fps = 1.0 / max(sekarang - waktu_fps, 0.001)
        waktu_fps = sekarang

        # -- Tampilan antarmuka --
        info_mode = {
            "pindah":      ("PINDAH",     (0, 255, 100)),
            "putar_y":     ("PUTAR Y",    (255, 140, 0)),
            "putar_x":     ("PUTAR X",    (0, 200, 255)),
            "zoom_masuk":  ("ZOOM MASUK", (0, 255, 255)),
            "zoom_keluar": ("ZOOM KELUAR",(255, 100, 100)),
            "kunci":       ("KUNCI",      (120, 120, 255)),
            "tidak_ada":   ("---",        (150, 150, 150)),
        }
        label, warna_label = info_mode.get(mode_saat_ini, ("---", (150, 150, 150)))

        # Baris atas: nama model & FPS
        cv2.rectangle(frame, (0, 0), (l, 36), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"[{indeks_model + 1}/{len(daftar_model)}] {nama_file}  FPS:{fps:.0f}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Baris bawah: mode & info rotasi/zoom
        cv2.rectangle(frame, (0, t - 36), (l, t), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"Mode: {label}   RotX:{rotasi_x:.1f}  RotY:{rotasi_y:.1f}  Zoom:{zoom:.1f}x",
                    (8, t - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, warna_label, 1)

        # Petunjuk di kanan atas
        petunjuk = [
            "1 jari: Pindah",
            "2 jari: Putar Y",
            "3 jari: Putar X",
            "5 jari: Zoom masuk",
            "Kepalan: Zoom keluar",
            "Jempol: Kunci",
            "N/P: Ganti model",
            "R: Reset",
        ]
        for i, teks in enumerate(petunjuk):
            cv2.putText(frame, teks, (l - 190, 52 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1)

        cv2.imshow("Kontrol Model GLTF", frame)

        # -- Kontrol keyboard --
        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        elif tombol == ord("r"):
            posisi_x, posisi_y = l // 2, t // 2
            rotasi_x, rotasi_y = 20.0, 0.0
            zoom = 1.0
        elif tombol in (ord("+"), ord("=")):
            zoom = min(zoom + 0.1, 4.0)
        elif tombol == ord("-"):
            zoom = max(zoom - 0.1, 0.2)
        elif tombol == ord("n"):
            indeks_model = (indeks_model + 1) % len(daftar_model)
            rotasi_x, rotasi_y = 20.0, 0.0
        elif tombol == ord("p"):
            indeks_model = (indeks_model - 1) % len(daftar_model)
            rotasi_x, rotasi_y = 20.0, 0.0

kamera.release()
cv2.destroyAllWindows()
print("Selesai.")
