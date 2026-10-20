import cv2
import threading
import time
import uuid
import numpy as np

class CameraManager:
    """
    Mengelola perangkat input kamera, pemindaian perangkat (DirectShow/pygrabber),
    dan switching kamera (Laptop webcam vs Iriun Webcam / Smartphone).
    Mendukung DirectShow (CAP_DSHOW) untuk inisialisasi instan di Windows dan
    fitur Auto-Switching ketika Iriun Webcam terhubung atau terputus.
    """
    def __init__(self, default_source=0, auto_switch=True):
        self.lock = threading.Lock()
        self.default_source = default_source
        self.current_source = default_source
        self.cap = None
        self.running = False
        self.thread = None
        self.monitor_thread = None
        self.latest_frame = None
        self.is_connected = False
        self.status_message = "Memulai kamera..."
        
        # Fitur Auto-Switch
        self.auto_switch_enabled = auto_switch
        self.user_manually_selected = False
        self.last_event = None
        self.iriun_connected = False

        self._start(default_source)
        self._start_monitor()

    @staticmethod
    def get_available_cameras():
        """
        Mendeteksi daftar kamera yang tersedia di sistem Windows.
        Menggunakan pygrabber (DirectShow) jika tersedia, dengan fallback OpenCV.
        """
        cameras = []
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for idx, name in enumerate(devices):
                display_name = name
                if "iriun" in name.lower():
                    display_name = f"Iriun Webcam (Kamera HP) - Index {idx}"
                elif idx == 0:
                    display_name = f"{name} (Kamera Laptop) - Index {idx}"
                else:
                    display_name = f"{name} - Index {idx}"
                cameras.append({"id": idx, "name": display_name, "raw_name": name})
        except Exception:
            pass

        # Fallback jika pygrabber tidak menemukan perangkat
        if not cameras:
            for idx in range(3):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        name = "Kamera Laptop" if idx == 0 else f"Kamera Eksternal {idx}"
                        cameras.append({"id": idx, "name": f"{name} (Index {idx})", "raw_name": name})
                    cap.release()

        # Jika tetap belum terdeteksi, sediakan pilihan default 0 dan 1
        if not cameras:
            cameras = [
                {"id": 0, "name": "Kamera Laptop (Index 0)", "raw_name": "USB Camera"},
                {"id": 1, "name": "Iriun Webcam (Kamera HP - Index 1)", "raw_name": "Iriun Webcam"}
            ]

        return cameras

    def _find_iriun_index(self):
        """Mencari ID indeks Iriun Webcam jika terdeteksi di perangkat."""
        cameras = self.get_available_cameras()
        for cam in cameras:
            if "iriun" in cam["name"].lower() or "iriun" in cam.get("raw_name", "").lower():
                return cam["id"]
        return None

    def _find_laptop_index(self):
        """Mencari indeks kamera laptop/bawaan (bukan Iriun Webcam)."""
        cameras = self.get_available_cameras()
        for cam in cameras:
            name = cam["name"].lower()
            raw = cam.get("raw_name", "").lower()
            if "iriun" not in name and "iriun" not in raw:
                return cam["id"]
        return 0

    def _check_iriun_connection(self):
        """
        Mendeteksi apakah smartphone terhubung ke aplikasi Iriun Webcam di PC.
        Iriun Webcam membuat koneksi socket TCP ESTABLISHED saat HP terhubung via WiFi atau USB.
        """
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                try:
                    name = (proc.info['name'] or '').lower()
                    if 'iriun' in name:
                        for conn in proc.net_connections(kind='inet'):
                            if conn.status == 'ESTABLISHED':
                                return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return False

    def _open_capture(self, source):
        """
        Membuka VideoCapture secara stabil dan cepat di Windows.
        Mengutamakan cv2.CAP_DSHOW agar kamera laptop/USB langsung terbuka dalam <1 detik.
        """
        try:
            idx = int(source)
            # 1. Coba cv2.CAP_DSHOW (DirectShow Windows - instan & andal untuk webcam)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                for _ in range(5):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        print(f"[CameraManager] Kamera {idx} berhasil dibuka via DirectShow (DSHOW).")
                        return cap
                    time.sleep(0.1)
                cap.release()

            # 2. Fallback ke backend default (MSMF) jika DSHOW tidak merespons
            print(f"[CameraManager] DirectShow tidak merespons pada indeks {idx}, mencoba default backend...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                for _ in range(5):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        print(f"[CameraManager] Kamera {idx} berhasil dibuka via default backend.")
                        return cap
                    time.sleep(0.1)
                cap.release()

            print(f"[CameraManager] Gagal membuka kamera pada indeks {idx}.")
            return None
        except (ValueError, TypeError):
            # Jika user memberikan URL streaming (RTSP/HTTP MJPEG stream)
            return cv2.VideoCapture(str(source))

    def _start(self, source=0):
        with self.lock:
            self.current_source = source
            print(f"[CameraManager] Memulai inisialisasi Kamera Laptop (Indeks {source})...")
            self.cap = self._open_capture(source)

            # Jika indeks yang diminta gagal, coba cari indeks kamera non-Iriun lainnya
            if (self.cap is None or not self.cap.isOpened()) and isinstance(source, int):
                alt_idx = self._find_laptop_index()
                if alt_idx != source:
                    print(f"[CameraManager] Indeks {source} gagal, mencoba alternatif indeks {alt_idx}...")
                    self.current_source = alt_idx
                    self.cap = self._open_capture(alt_idx)

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.latest_frame = frame
                    self.is_connected = True
                    self.status_message = "Kamera aktif"
                    print(f"[CameraManager] Kamera aktif siap digunakan. Resolusi: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    self.is_connected = False
                    self.status_message = "Menunggu frame kamera..."
            else:
                self.is_connected = False
                self.status_message = "Kamera tidak dapat dibuka"
                print(f"[CameraManager] PERINGATAN: Kamera {source} tidak dapat dibuka. Pastikan tidak sedang digunakan aplikasi lain.")

            self.running = True
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()

    def _start_monitor(self):
        """Memulai background thread untuk memantau koneksi Iriun Webcam."""
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()

    def _monitor_worker(self):
        """
        Thread pemantau otomatis:
        - Jika Iriun Webcam terhubung, otomatis beralih ke kamera Iriun.
        - Jika Iriun Webcam terputus, otomatis kembali ke kamera laptop (default).
        """
        # Beri jeda 3 detik agar kamera laptop stabil terlebih dahulu
        time.sleep(3.0)
        last_state = None

        while self.running:
            try:
                iriun_is_connected = self._check_iriun_connection()
                self.iriun_connected = iriun_is_connected
                iriun_id = self._find_iriun_index()

                if self.auto_switch_enabled and iriun_id is not None:
                    # KASUS 1: Iriun baru saja terhubung -> Auto-switch ke Iriun
                    if iriun_is_connected and self.current_source != iriun_id and not self.user_manually_selected:
                        print(f"[Auto-Switch] Iriun Webcam terhubung! Beralih dari Kamera {self.current_source} ke Iriun (ID: {iriun_id}).")
                        success, _ = self.switch_camera(iriun_id, manual=False)
                        if success:
                            self.last_event = {
                                "id": str(uuid.uuid4()),
                                "type": "auto_switch",
                                "source": iriun_id,
                                "name": "Iriun Webcam (Kamera HP)",
                                "message": "Iriun Webcam terhubung! Otomatis beralih ke Kamera HP."
                            }

                    # KASUS 2: Sedang pakai Iriun, lalu Iriun terputus -> Auto-switch kembali ke kamera laptop
                    elif not iriun_is_connected and self.current_source == iriun_id and last_state is True:
                        laptop_id = self._find_laptop_index()
                        print(f"[Auto-Switch] Iriun Webcam terputus! Beralih kembali ke kamera laptop (ID: {laptop_id}).")
                        success, _ = self.switch_camera(laptop_id, manual=False)
                        if success:
                            self.last_event = {
                                "id": str(uuid.uuid4()),
                                "type": "auto_switch",
                                "source": laptop_id,
                                "name": "Kamera Laptop",
                                "message": "Iriun Webcam terputus. Otomatis kembali ke Kamera Laptop."
                            }

                # Reset flag override user jika sambungan Iriun terputus
                if last_state != iriun_is_connected and not iriun_is_connected:
                    self.user_manually_selected = False

                last_state = iriun_is_connected
            except Exception as e:
                print(f"[Monitor Worker Error] {e}")

            time.sleep(1.2)

    def switch_camera(self, source, manual=False):
        """
        Beralih ke sumber kamera baru secara dinamis.
        """
        with self.lock:
            if manual:
                iriun_id = self._find_iriun_index()
                # Jika user memilih manual kamera laptop saat Iriun terhubung, tandai manual override
                if iriun_id is not None and str(source) != str(iriun_id):
                    self.user_manually_selected = True
                else:
                    self.user_manually_selected = False

            self.status_message = "Mengalihkan sumber kamera..."
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.current_source = source
            new_cap = self._open_capture(source)
            if new_cap and new_cap.isOpened():
                ret, frame = new_cap.read()
                if ret and frame is not None:
                    self.cap = new_cap
                    self.latest_frame = frame
                    self.is_connected = True
                    self.status_message = "Kamera aktif"
                    return True, "Berhasil beralih ke sumber kamera."

            # Gagal membuka sumber baru
            if new_cap is not None:
                new_cap.release()
            self.is_connected = False
            self.status_message = f"Gagal membuka kamera (ID: {source})"
            return False, f"Gagal membuka kamera {source}. Pastikan kamera aktif dan tidak digunakan aplikasi lain."

    def set_auto_switch(self, enabled: bool):
        """Mengaktifkan atau menonaktifkan fitur auto-switch."""
        self.auto_switch_enabled = bool(enabled)
        self.user_manually_selected = False

        if self.auto_switch_enabled and self.iriun_connected:
            iriun_id = self._find_iriun_index()
            if iriun_id is not None and self.current_source != iriun_id:
                self.switch_camera(iriun_id, manual=False)
                self.last_event = {
                    "id": str(uuid.uuid4()),
                    "type": "auto_switch",
                    "source": iriun_id,
                    "name": "Iriun Webcam (Kamera HP)",
                    "message": "Auto-switch aktif! Otomatis beralih ke Kamera HP."
                }
        return self.auto_switch_enabled

    def _capture_worker(self):
        """Background thread worker untuk membaca frame terus menerus."""
        consecutive_failures = 0
        while self.running:
            with self.lock:
                cap = self.cap

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0
                    with self.lock:
                        self.latest_frame = frame
                        self.is_connected = True
                        self.status_message = "Kamera aktif"
                    time.sleep(0.01)
                    continue
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 20:
                        with self.lock:
                            self.is_connected = False
                            self.status_message = "Kamera terputus atau tidak merespons"
            else:
                with self.lock:
                    self.is_connected = False

            time.sleep(0.08)

    def _create_placeholder(self, text="KAMERA TIDAK TERHUBUNG"):
        """Membuat frame placeholder informatif jika kamera terputus."""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (25, 20, 15)

        # Border
        cv2.rectangle(placeholder, (10, 10), (630, 470), (45, 35, 25), 2)

        # Teks Panduan & Status
        cv2.putText(placeholder, text.upper(), (80, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 240), 2, cv2.LINE_AA)
        cv2.putText(placeholder, "1. Pastikan shutter fisik kamera laptop terbuka.", (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "2. Pastikan tidak ada aplikasi lain (Zoom/Teams) memakai kamera.", (70, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "3. Untuk menghubungkan HP, buka aplikasi Iriun Webcam di HP.", (70, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 200, 130), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "4. Gunakan dropdown di atas untuk memilih kamera secara manual.", (70, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 170, 240), 1, cv2.LINE_AA)
        return placeholder

    def get_frame(self):
        """Mengembalikan tuple: (frame, is_live)"""
        with self.lock:
            if self.is_connected and self.latest_frame is not None:
                return self.latest_frame.copy(), True
            msg = self.status_message or "KAMERA TIDAK TERHUBUNG"
            return self._create_placeholder(msg), False

    def get_status(self):
        """Mengembalikan dictionary status kamera terkini."""
        return {
            "current_source": self.current_source,
            "is_connected": self.is_connected,
            "status_message": self.status_message,
            "auto_switch_enabled": self.auto_switch_enabled,
            "iriun_connected": self.iriun_connected,
            "last_event": self.last_event
        }

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
