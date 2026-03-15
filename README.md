# Kontrol Model GLTF 3D dengan Gesture Tangan

Aplikasi real-time untuk mengontrol model 3D (GLTF/GLB) menggunakan gesture tangan melalui webcam. Menggunakan software renderer murni tanpa memerlukan OpenGL.

## Cara Menjalankan

```bash
cd "python/face detection"
source venv/bin/activate
python gltf_control.py
```

## Gesture Tangan

| Gesture | Fungsi |
|---------|--------|
| 1 jari (telunjuk) | Pindah model |
| 2 jari (V) | Putar sumbu Y (kiri-kanan) |
| 3 jari | Putar sumbu X (atas-bawah) |
| 5 jari (buka) | Zoom masuk |
| Kepalan (tutup) | Zoom keluar |
| Jempol | Kunci posisi |

## Keyboard

| Tombol | Fungsi |
|--------|--------|
| `R` | Reset posisi & rotasi |
| `N` / `P` | Model berikutnya / sebelumnya |
| `+` / `-` | Zoom masuk / keluar |
| `Q` | Keluar |

## Struktur Folder

```
face detection/
  gltf_control.py                 # Script utama
  hand_landmarker.task            # Model MediaPipe (otomatis terunduh)
  blue_archive_shirokos_rifle/    # Folder aset model 3D
    scene.gltf
    scene.bin
    textures/
  venv/                           # Virtual environment Python
```

## Menambah Model

Taruh file `.gltf` atau `.glb` di dalam folder yang ditentukan pada variabel `FOLDER_MODEL` di `gltf_control.py`. Model akan otomatis dimuat saat aplikasi dijalankan.

Sumber model gratis:
- [Sketchfab](https://sketchfab.com) (filter: Free + Downloadable, format GLB)
- [Quaternius](https://quaternius.com) (model low-poly gratis)

## Dependensi

```bash
pip install opencv-python mediapipe numpy "trimesh[easy]"
```
