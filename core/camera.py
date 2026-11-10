"""
Modul Penangkapan Video OpenCV & Pengelola Kamera Manual (core/camera.py)
Proyek: Sistem Inspeksi QC Sendok Otomatis (PBLIF3-MD02) - YOLO26
Pengembang: Muhammad Farhan (Backend & AI) & M. Khairil Candra (Frontend & Testing)

Fitur:
- Penangkapan frame video thread-safe dan non-blocking untuk performa MJPEG optimal
- Pemilihan kamera manual: Kamera Laptop (0) dan IP Webcam HP (HTTP / MJPEG Stream)
- Cek ketersediaan koneksi cepat (socket pre-check) untuk mencegah hang pada stream jaringan
- Normalisasi otomatis URL IP Webcam (misal: 192.168.1.5:8080 -> http://192.168.1.5:8080/video)
- Penyimpanan persisten URL IP Webcam di camera_config.json
- Fallback frame grafis kamera standar profesional (ikon kamera modern + instruksi aktivasi)
- Auto-reconnect periodik saat IP Webcam HP terputus atau baru diaktifkan
"""

import os

# Konfigurasi low-latency OpenCV FFmpeg untuk menghilangkan buffer delay pada IP Webcam stream
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"

import cv2
import json
import time
import socket
import threading
import numpy as np
from datetime import datetime
from urllib.parse import urlparse

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "camera_config.json")
DEFAULT_IP_URL = "http://192.168.1.100:8080/video"


def load_saved_ip_url() -> str:
    """Memuat URL IP Webcam yang tersimpan di camera_config.json jika ada."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                url = data.get("ip_webcam_url", DEFAULT_IP_URL)
                return normalize_ip_url(url)
    except Exception as e:
        print(f"[Camera] Gagal membaca camera_config.json: {e}")
    return DEFAULT_IP_URL


def save_ip_url(url: str):
    """Menyimpan URL IP Webcam ke camera_config.json secara persisten."""
    try:
        normalized = normalize_ip_url(url)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"ip_webcam_url": normalized}, f, indent=2)
    except Exception as e:
        print(f"[Camera] Gagal menyimpan camera_config.json: {e}")


def normalize_ip_url(url: str) -> str:
    """
    Menormalkan input URL atau IP dari pengguna agar valid untuk stream OpenCV.
    Contoh:
    - '192.168.1.5:8080' -> 'http://192.168.1.5:8080/video'
    - 'http://192.168.1.5:8080' -> 'http://192.168.1.5:8080/video'
    - 'http://192.168.1.5:8080/video' -> 'http://192.168.1.5:8080/video'
    """
    if not url:
        return DEFAULT_IP_URL
    url = str(url).strip()
    if not (url.startswith("http://") or url.startswith("https://") or url.startswith("rtsp://")):
        url = "http://" + url

    parsed = urlparse(url)
    if not parsed.path or parsed.path == "/":
        url = f"{url.rstrip('/')}/video"
    return url


def is_ip_reachable(url: str, timeout: float = 1.2) -> bool:
    """
    Cek cepat apakah host dan port URL dapat dijangkau sebelum inisialisasi VideoCapture
    agar terhindar dari hang thread hingga 30 detik jika server belum jalan.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class CameraManager:
    """
    Kelas pengelola kamera untuk menangkap video real-time
    berdasarkan pilihan manual pengguna (Kamera Laptop atau IP Webcam HP).
    """

    def __init__(self, default_source=0):
        self.lock = threading.Lock()
        # Default: 0 = Kamera Laptop, 'ipwebcam' = IP Webcam HP
        self.current_source = default_source
        self.ip_webcam_url = load_saved_ip_url()
        self.cap = None
        self.is_connected = False
        self.status_message = "Memulai kamera..."

        # Frame buffer
        self.current_raw_frame = None
        self.running = False
        self.capture_thread = None

        # Resolusi standar (640x480 untuk performa inferensi & streaming stabil)
        self.width = 640
        self.height = 480

        # Inisialisasi kamera awal (default: Kamera Laptop di indeks 0)
        self._init_camera(self.current_source)

        # Jalankan worker pembacaan frame non-blocking
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

    def set_ip_webcam_url(self, ip_url: str):
        """Memperbarui URL IP Webcam dan menyimpannya secara persisten."""
        if not ip_url:
            return
        normalized = normalize_ip_url(ip_url)
        self.ip_webcam_url = normalized
        save_ip_url(normalized)
        print(f"[Camera] URL IP Webcam disetel ke: {normalized}")

    def _is_ip_source(self, source_id):
        """Mengecek apakah source_id merupakan sumber IP Webcam."""
        if source_id == "ipwebcam" or source_id == 1 or source_id == "1":
            return True
        if isinstance(source_id, str) and (source_id.startswith("http://") or source_id.startswith("https://") or source_id.startswith("rtsp://")):
            return True
        return False

    def _open_capture(self, source_id):
        """Membuka VideoCapture dengan toleransi backend yang tepat untuk masing-masing tipe kamera."""
        # 1. Sumber IP Webcam (HTTP/MJPEG/RTSP stream)
        if self._is_ip_source(source_id):
            stream_url = self.ip_webcam_url
            if isinstance(source_id, str) and (source_id.startswith("http://") or source_id.startswith("https://") or source_id.startswith("rtsp://")):
                stream_url = source_id

            # Pre-check ketersediaan port jaringan secara instan
            if not is_ip_reachable(stream_url, timeout=1.2):
                return None

            try:
                cap = cv2.VideoCapture(stream_url)
                if cap.isOpened():
                    # Set buffer size ke 1 frame untuk mencegah akumulasi delay/lag
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception:
                pass
            return None

        # 2. Kamera Hardware Lokal (misal Kamera Laptop indeks 0)
        try:
            device_index = int(source_id)
        except Exception:
            device_index = 0

        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(device_index, backend)
                if cap.isOpened():
                    return cap
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception:
                pass

        return None

    def _init_camera(self, source_id, ip_url=None):
        """Membuka perangkat kamera yang dipilih secara manual."""
        if ip_url:
            self.set_ip_webcam_url(ip_url)

        # Normalisasi source_id jika diteruskan string "1" atau int 1 -> 'ipwebcam'
        if source_id == 1 or source_id == "1":
            source_id = "ipwebcam"
        elif source_id == 0 or source_id == "0":
            source_id = 0

        is_ip_cam = self._is_ip_source(source_id)

        with self.lock:
            old_cap = self.cap
            try:
                new_cap = self._open_capture(source_id)
                if new_cap is not None and new_cap.isOpened():
                    if not is_ip_cam:
                        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        new_cap.set(cv2.CAP_PROP_FPS, 30)

                    # Lepaskan kamera lama secara bersih
                    if old_cap is not None and old_cap is not new_cap:
                        try:
                            old_cap.release()
                        except Exception:
                            pass

                    self.cap = new_cap
                    self.current_source = source_id

                    # Cek pembacaan beberapa frame awal
                    frame_ok = False
                    for _ in range(6):
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None and not self._is_invalid_or_green(test_frame):
                            if test_frame.shape[1] != self.width or test_frame.shape[0] != self.height:
                                test_frame = cv2.resize(test_frame, (self.width, self.height))
                            self.current_raw_frame = test_frame
                            frame_ok = True
                            break
                        time.sleep(0.05)

                    cam_name = "IP Webcam (HP)" if is_ip_cam else "Kamera Laptop"
                    if frame_ok:
                        self.is_connected = True
                        self.status_message = f"{cam_name} aktif dan terhubung"
                    else:
                        self.is_connected = False
                        if is_ip_cam:
                            self.status_message = f"Terhubung ke {self.ip_webcam_url}, menunggu sinyal video..."
                        else:
                            self.status_message = f"{cam_name} aktif (Menunggu sinyal video)"

                    return True

                else:
                    if old_cap is not None:
                        try:
                            old_cap.release()
                        except Exception:
                            pass
                    self.cap = None
                    self.is_connected = False
                    self.current_source = source_id
                    cam_name = "IP Webcam (HP)" if is_ip_cam else "Kamera Laptop"
                    if is_ip_cam:
                        self.status_message = f"Tidak dapat terhubung ke IP Webcam ({self.ip_webcam_url}). Pastikan 'Start server' aktif di aplikasi IP Webcam HP."
                    else:
                        self.status_message = f"Perangkat {cam_name} belum terdeteksi di Windows."
                    return False

            except Exception as e:
                self.is_connected = False
                self.status_message = f"Kesalahan koneksi kamera: {str(e)}"
                return False

    def _is_invalid_or_green(self, frame):
        """
        Mendeteksi apakah frame kosong, rusak, atau memiliki sinyal uji kosong.
        """
        if frame is None or frame.size == 0:
            return True
        try:
            b_mean = float(np.mean(frame[:, :, 0]))
            g_mean = float(np.mean(frame[:, :, 1]))
            r_mean = float(np.mean(frame[:, :, 2]))
            mean_all = float(np.mean(frame))

            # 1. Layar hijau khas test signal driver virtual / uninitialized frame
            if g_mean > 65 and b_mean < 45 and r_mean < 45:
                return True
            if g_mean > 80 and g_mean > (b_mean + r_mean) * 1.8:
                return True

            # 2. Layar hitam kosong total
            if mean_all < 1.0:
                return True

            return False
        except Exception:
            return False

    def _generate_fallback_frame(self, message=None, sub_message=None):
        """
        Membuat antarmuka visual standar kamera standby dengan grafis kamera modern
        dan petunjuk aktivasi yang jelas.
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background dark slate #0b0f19 (BGR: 25, 15, 11)
        img[:] = (25, 15, 11)

        cx, cy = self.width // 2, self.height // 2 - 35

        # 1. Grafis Kamera Modern
        # Bodi Kamera
        bw, bh = 140, 90
        bx1, by1 = cx - bw // 2, cy - bh // 2 + 10
        bx2, by2 = cx + bw // 2, cy + bh // 2 + 10
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (71, 55, 41), -1)      # Slate-700
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (100, 80, 60), 2)     # Border

        # Viewfinder / Top notch
        tw, th = 44, 16
        tx1, ty1 = cx - tw // 2 - 20, by1 - th
        tx2, ty2 = cx + tw // 2 - 20, by1
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (71, 55, 41), -1)
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (100, 80, 60), 2)

        # Flash indicator dot
        cv2.circle(img, (cx + 42, by1 + 22), 6, (148, 163, 184), -1)

        # Lensa Kamera (Concentric Circles)
        cv2.circle(img, (cx, cy + 10), 34, (51, 39, 29), -1)
        cv2.circle(img, (cx, cy + 10), 34, (148, 163, 184), 2)
        cv2.circle(img, (cx, cy + 10), 22, (30, 22, 16), -1)
        cv2.circle(img, (cx, cy + 10), 12, (51, 65, 85), -1)
        # Efek pantulan cahaya pada kaca lensa
        cv2.circle(img, (cx - 7, cy + 3), 4, (200, 210, 220), -1)

        # 2. Teks Standby Standar
        text1 = "KAMERA STANDBY"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw1, th1), _ = cv2.getTextSize(text1, font, 0.65, 2)
        cv2.putText(img, text1, (cx - tw1 // 2, cy + 90),
                    font, 0.65, (226, 232, 240), 2, cv2.LINE_AA)

        is_ip_cam = self._is_ip_source(self.current_source)

        if is_ip_cam:
            line1 = message or "IP Webcam HP: Belum Terhubung"
            line2 = sub_message or f"URL: {self.ip_webcam_url}"
            line3 = "Di HP: Buka app IP Webcam -> Tap 'Start server'"
        else:
            line1 = message or "Menghubungkan ke Kamera Laptop..."
            line2 = sub_message or "Pastikan kamera laptop tidak digunakan aplikasi lain"
            line3 = ""

        (tw2, th2), _ = cv2.getTextSize(line1, font, 0.42, 1)
        cv2.putText(img, line1, (cx - tw2 // 2, cy + 118),
                    font, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

        (tw3, th3), _ = cv2.getTextSize(line2, font, 0.38, 1)
        cv2.putText(img, line2, (cx - tw3 // 2, cy + 140),
                    font, 0.38, (56, 189, 248), 1, cv2.LINE_AA)  # cyan text

        if line3:
            (tw4, th4), _ = cv2.getTextSize(line3, font, 0.35, 1)
            cv2.putText(img, line3, (cx - tw4 // 2, cy + 160),
                        font, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

        return img

    def _capture_worker(self):
        """
        Worker thread untuk membaca frame terbaru dari kamera secara kontinu.
        Dilengkapi auto-reconnect periodik saat IP Webcam terputus atau baru diaktifkan.
        """
        consecutive_failures = 0
        last_reconnect_time = 0

        while self.running:
            is_ip_cam = self._is_ip_source(self.current_source)

            if self.cap is not None and self.cap.isOpened():
                try:
                    if is_ip_cam:
                        # Grab langsung frame terbaru, membuang buffer lama
                        ret = self.cap.grab()
                        if ret:
                            ret, frame = self.cap.retrieve()
                        else:
                            frame = None
                    else:
                        ret, frame = self.cap.read()

                    if ret and frame is not None and not self._is_invalid_or_green(frame):
                        # Resize ke resolusi standar (640x480) jika resolusi stream berbeda
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))

                        with self.lock:
                            self.current_raw_frame = frame
                            self.is_connected = True
                            cam_name = "IP Webcam (HP)" if is_ip_cam else "Kamera Laptop"
                            self.status_message = f"{cam_name} aktif dan terhubung"
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > 15:  # ~0.5 detik tanpa frame valid
                            with self.lock:
                                self.is_connected = False
                                if is_ip_cam:
                                    self.status_message = f"IP Webcam terputus. Mencoba reconnect ke {self.ip_webcam_url}..."
                                else:
                                    self.status_message = "Menunggu sinyal video..."

                        # Jika kegagalan berulang, lakukan reconnect berkala (setiap 2 detik)
                        now = time.time()
                        if consecutive_failures > 25 and (now - last_reconnect_time > 2.0):
                            last_reconnect_time = now
                            with self.lock:
                                try:
                                    self.cap.release()
                                except Exception:
                                    pass
                                self.cap = self._open_capture(self.current_source)
                                if self.cap is not None and self.cap.isOpened() and not is_ip_cam:
                                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                        time.sleep(0.02)
                except Exception:
                    consecutive_failures += 1
                    time.sleep(0.03)
            else:
                with self.lock:
                    self.is_connected = False
                now = time.time()
                if now - last_reconnect_time > 2.0:
                    last_reconnect_time = now
                    with self.lock:
                        self.cap = self._open_capture(self.current_source)
                        if self.cap is not None and self.cap.isOpened() and not is_ip_cam:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                time.sleep(0.08)

    def switch_camera(self, source_id, ip_url=None):
        """
        Beralih ke sumber kamera yang dipilih secara manual oleh pengguna.
        Mengembalikan tuple (status_sukses: bool, pesan: str).
        """
        if ip_url:
            self.set_ip_webcam_url(ip_url)

        if source_id == 1 or source_id == "1":
            source_id = "ipwebcam"
        elif source_id == 0 or source_id == "0":
            source_id = 0
        elif isinstance(source_id, str) and (source_id.startswith("http://") or source_id.startswith("https://")):
            self.set_ip_webcam_url(source_id)
            source_id = "ipwebcam"

        is_ip_cam = self._is_ip_source(source_id)
        cam_name = "IP Webcam (HP)" if is_ip_cam else "Kamera Laptop"

        # Jika sudah di sumber yang sama dan terhubung (tanpa update ip_url baru)
        if source_id == self.current_source and self.is_connected and not ip_url:
            return True, f"{cam_name} sudah aktif."

        success = self._init_camera(source_id)
        if success:
            return True, f"Beralih ke {cam_name}."
        else:
            if is_ip_cam:
                return False, f"Tidak dapat terhubung ke IP Webcam ({self.ip_webcam_url}). Pastikan HP & Laptop dalam 1 Wi-Fi dan 'Start server' sudah aktif di aplikasi IP Webcam HP."
            else:
                return False, f"Tidak dapat membuka {cam_name}. Pastikan kamera laptop tidak digunakan aplikasi lain."

    def get_raw_frame(self):
        """
        Mengambil frame raw BGR terbaru.
        Jika kamera offline atau memancarkan frame kosong, kembalikan fallback frame standar.
        """
        with self.lock:
            frame = self.current_raw_frame.copy() if (self.is_connected and self.current_raw_frame is not None) else None

        if self._is_invalid_or_green(frame):
            return self._generate_fallback_frame()

        return frame

    def get_available_cameras(self):
        """Mendeteksi dan mengembalikan daftar sumber kamera yang tersedia di sistem."""
        return [
            {"id": 0, "name": "Kamera Laptop (Default)"},
            {"id": "ipwebcam", "name": "IP Webcam (Kamera HP)"}
        ]

    def get_cameras_info(self):
        """Mengembalikan dictionary status kamera untuk REST API /api/cameras."""
        return {
            "status": "success",
            "current_source": self.current_source,
            "cameras": self.get_available_cameras(),
            "ip_webcam_url": self.ip_webcam_url,
            "is_connected": self.is_connected,
            "status_message": self.status_message
        }

    def get_camera_status(self):
        """Mengembalikan status kamera periodik untuk polling frontend /api/camera_status."""
        return {
            "is_connected": self.is_connected,
            "status_message": self.status_message,
            "current_source": self.current_source,
            "ip_webcam_url": self.ip_webcam_url
        }

    def release(self):
        """Menghentikan thread dan melepaskan resource kamera."""
        self.running = False
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
