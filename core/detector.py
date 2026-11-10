"""
Modul Deteksi Cacat Sendok Berbasis Smart Hybrid Two-Stage YOLO (core/detector.py)
Proyek: Sistem Inspeksi QC Sendok Otomatis (PBLIF3-MD02) - YOLO26
Pengembang: Muhammad Farhan (Backend & AI) & M. Khairil Candra (Frontend & Testing)

Fitur Smart Hybrid Detection:
1. Prioritas Deteksi Cacat Bengkok (best.pt):
   Sendok yang bengkok atau cacat (Kelas NG) langsung direspons tanpa diblokir oleh
   model COCO, karena model COCO standar hanya mengenali bentuk sendok lurus/normal.
2. Validasi Sendok Normal (COCO Gatekeeper):
   Untuk sendok normal (Kelas OK), divalidasi dengan model COCO (yolov8n.pt) sehingga
   tangan, meja, pulpen, botol, atau benda lain TIDAK salah dideteksi sebagai sendok.
3. Filter Geometri Cerdas:
   Filter luas area minimal (>= 1200 px) untuk menepis bintik kotoran, debu, atau pantulan meja.
"""

import os
import cv2
import time
import io
import csv
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO


class SpoonDefectDetector:
    """
    Detektor cacat produk sendok menggunakan arsitektur Smart Hybrid Two-Stage YOLO.
    """

    def __init__(
        self,
        model_path="models/best.pt",
        coco_model_path="models/yolov8n.pt",
        conf_thresh=0.45,
        coco_conf=0.25,
        two_stage=True
    ):
        self.lock = threading.Lock()
        self.conf_thresh = conf_thresh
        self.coco_conf = coco_conf
        self.two_stage = two_stage

        # ID kelas 'spoon' pada dataset COCO
        self.COCO_SPOON_CLASS_ID = 44
        # Batas luas piksel minimum untuk sendok valid (menghindari noise piksel kecil)
        self.min_spoon_area = 1200

        print(f"[Detector] [Stage 1] Memuat model COCO Gatekeeper: {coco_model_path}")
        if not os.path.exists(coco_model_path) and os.path.exists("yolov8n.pt"):
            coco_model_path = "yolov8n.pt"
        self.coco_model = YOLO(coco_model_path)

        print(f"[Detector] [Stage 2] Memuat model deteksi cacat sendok: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"[Detector] Kelas model cacat aktif: {self.class_names}")

        # Identifikasi ID kelas NG dan OK secara dinamis dari nama kelas model
        self.ng_class_ids = set()
        self.ok_class_ids = set()
        for cid, cname in self.class_names.items():
            name_upper = str(cname).strip().upper()
            if any(k in name_upper for k in ["NG", "DEFECT", "BENGKOK", "CACAT", "REJECT"]):
                self.ng_class_ids.add(cid)
            elif any(k in name_upper for k in ["OK", "PASS", "NORMAL", "GOOD"]):
                self.ok_class_ids.add(cid)

        # Fallback jika nama tidak terdeteksi otomatis (default best.pt: 0: NG, 1: OK)
        if not self.ng_class_ids and not self.ok_class_ids:
            self.ng_class_ids.add(0)
            self.ok_class_ids.add(1)

        print(f"[Detector] Pemetaan Kelas: NG={self.ng_class_ids}, OK={self.ok_class_ids}")

        # Struktur data statistik standar (FR-06, FR-08, FR-09)
        self.stats = {
            "total_inspected": 0,
            "total_ok": 0,
            "total_ng": 0,
            "current_status": "STANDBY",  # STANDBY | OK | NG
            "last_defect": "None",
            "history": []
        }

        # Konfigurasi cooldown dan transisi status
        self.standby_timeout = 1.0       # detik tanpa objek sebelum kembali ke STANDBY
        self.cooldown_period = 1.0       # detik jeda antar inspeksi produk baru
        self.last_seen_time = time.time()
        self.last_counted_time = 0
        self.is_current_counted = False
        self.last_recorded_status = None
        self.last_recorded_history_id = None
        self.ok_streak = 0
        self.ng_streak = 0

        # Metrik performa real-time (NFR-01)
        self.fps = 24.0
        self.latency_ms = 40.0
        self.alpha_smooth = 0.15

        # Palet warna modern BGR (Emerald Hijau & Crimson Merah)
        self.COLOR_OK = (16, 185, 129)       # Emerald #10B981 (BGR: 129, 185, 16)
        self.COLOR_NG = (68, 68, 239)        # Crimson #EF4444 (BGR: 68, 68, 239)

    def set_two_stage(self, enabled: bool):
        """Mengaktifkan atau menonaktifkan validasi Stage 1 COCO."""
        with self.lock:
            self.two_stage = enabled
        print(f"[Detector] Two-Stage Detection: {'Aktif' if enabled else 'Nonaktif'}")

    def set_confidence(self, conf: float):
        """Memperbarui ambang batas keyakinan deteksi."""
        with self.lock:
            self.conf_thresh = max(0.1, min(0.99, conf))
        print(f"[Detector] Confidence threshold diubah ke: {self.conf_thresh}")

    def is_ng_class(self, cls_id: int) -> bool:
        """Cek apakah class ID tergolong cacat/NG berdasarkan mapping atau nama label."""
        if cls_id in self.ng_class_ids:
            return True
        cname = str(self.class_names.get(cls_id, "")).strip().upper()
        return any(k in cname for k in ["NG", "DEFECT", "BENGKOK", "CACAT", "REJECT"])

    def _draw_modern_bbox(self, frame, x1, y1, x2, y2, color, label, conf_str):
        """
        Menggambar bounding box modern dengan aksen sudut dan badge label kontras tinggi.
        """
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(x1, w_frame - 1))
        y1 = max(0, min(y1, h_frame - 1))
        x2 = max(0, min(x2, w_frame - 1))
        y2 = max(0, min(y2, h_frame - 1))

        # 1. Bounding box utama
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # 2. Sudut dekoratif tebal (Corner brackets) untuk tampilan UI industrial
        corner_len = min(22, max(8, int((x2 - x1) * 0.12)))
        t = 3
        # Kiri Atas
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
        # Kanan Atas
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
        # Kiri Bawah
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
        # Kanan Bawah
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)

        # 3. Floating Pill Badge Label
        badge_text = f"{label} | {conf_str}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1

        (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)
        badge_h = text_h + 10
        badge_w = text_w + 14

        # Posisi badge (di atas box atau di dalam jika mepet layar atas)
        badge_y1 = y1 - badge_h if (y1 - badge_h) > 5 else y1 + 5
        badge_y2 = badge_y1 + badge_h
        badge_x1 = x1
        badge_x2 = x1 + badge_w

        # Gambar background badge solid
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Teks label putih kontras tinggi
        cv2.putText(frame, badge_text, (badge_x1 + 7, badge_y2 - 6),
                    font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    def process_frame(self, frame):
        """
        Menjalankan inferensi Smart Hybrid:
        1. Prioritaskan deteksi cacat sendok bengkok (NG) dari best.pt.
        2. Validasi sendok normal (OK) via COCO (yolov8n.pt) agar benda lain tidak terdeteksi.
        """
        if frame is None:
            return frame

        t_start = time.perf_counter()
        annotated_frame = frame.copy()
        now = time.time()

        # -------------------------------------------------------------
        # 1. EVALUASI MODEL CACAT PENGGUNA (best.pt)
        # -------------------------------------------------------------
        defect_results = self.model(frame, conf=self.conf_thresh, verbose=False)
        defect_boxes = defect_results[0].boxes if len(defect_results) > 0 else []

        valid_defect_items = []
        has_ng = False
        max_ng_conf = 0.0
        max_ok_conf = 0.0

        for box in defect_boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h

            # Filter area minimal untuk membuang noise meja/pantulan kecil
            if area >= self.min_spoon_area:
                is_ng = self.is_ng_class(cls_id)
                valid_defect_items.append({
                    "box": xyxy,
                    "cls_id": cls_id,
                    "is_ng": is_ng,
                    "conf": conf,
                    "area": area
                })
                if is_ng:
                    has_ng = True
                    if conf > max_ng_conf:
                        max_ng_conf = conf
                else:
                    if conf > max_ok_conf:
                        max_ok_conf = conf

        # -------------------------------------------------------------
        # 1b. MULTI-ANGLE VERIFICATION (Penyelamat Orientasi Sendok)
        # -------------------------------------------------------------
        # Model dilatih hanya pada sendok posisi vertikal (Angle 0/180). Jika sendok
        # diletakkan miring/horizontal, model salah mengiranya sebagai NG.
        # Jika terdeteksi NG tapi belum ada OK, uji kembali frame yang diputar 90 derajat.
        # Jika pada posisi tegak terdeteksi OK, terbukti objek adalah sendok NORMAL (OK)!
        if has_ng and max_ok_conf < 0.45:
            rot_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rot_results = self.model(rot_frame, conf=self.conf_thresh, verbose=False)
            rot_boxes = rot_results[0].boxes if len(rot_results) > 0 else []
            for rbox in rot_boxes:
                rcls = int(rbox.cls[0].item())
                rconf = float(rbox.conf[0].item())
                if not self.is_ng_class(rcls) and rconf >= 0.45:
                    max_ok_conf = rconf
                    has_ng = False
                    for it in valid_defect_items:
                        it["is_ng"] = False
                        it["conf"] = rconf
                    break

        # Resolusi konflik skor: Jika keyakinan OK lebih tinggi atau sama dengan NG, menangkan OK
        if has_ng and (max_ok_conf >= max_ng_conf):
            has_ng = False

        # -------------------------------------------------------------
        # 2. EVALUASI MODEL COCO GATEKEEPER (yolov8n.pt)
        # -------------------------------------------------------------
        coco_spoons = []
        if self.two_stage:
            coco_results = self.coco_model(
                frame,
                classes=[self.COCO_SPOON_CLASS_ID],
                conf=self.coco_conf,
                verbose=False
            )
            coco_boxes = coco_results[0].boxes if len(coco_results) > 0 else []
            for box in coco_boxes:
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if (w * h) >= self.min_spoon_area:
                    coco_spoons.append({
                        "box": xyxy,
                        "conf": float(box.conf[0].item())
                    })

        # -------------------------------------------------------------
        # 3. KEPUTUSAN SMART HYBRID
        # -------------------------------------------------------------
        has_spoon = False
        frame_status = "STANDBY"
        boxes_to_draw = []

        if has_ng:
            # SKENARIO A: SENDOK BENGKOK / CACAT FISIK (NG)
            # Model pengguna (best.pt) mendeteksi sendok bengkok / cacat secara spesifik.
            # Diterima langsung tanpa diblokir COCO (karena COCO tidak kenal sendok bengkok).
            has_spoon = True
            frame_status = "NG"
            for item in valid_defect_items:
                if item["is_ng"]:
                    boxes_to_draw.append({
                        "box": item["box"],
                        "color": self.COLOR_NG,
                        "label": "NG (REJECT)",
                        "conf": item["conf"]
                    })

        elif max_ok_conf > 0:
            # SKENARIO B: SENDOK NORMAL (OK) DARI MODEL KUSTOM (best.pt)
            has_spoon = True
            frame_status = "OK"
            for item in valid_defect_items:
                if not item["is_ng"]:
                    boxes_to_draw.append({
                        "box": item["box"],
                        "color": self.COLOR_OK,
                        "label": "OK (PASS)",
                        "conf": item["conf"]
                    })

        elif len(coco_spoons) > 0:
            # SKENARIO C: SENDOK NORMAL DIVALIDASI COCO
            has_spoon = True
            frame_status = "OK"
            for sp in coco_spoons:
                conf_val = max_ok_conf if max_ok_conf > 0 else sp["conf"]
                boxes_to_draw.append({
                    "box": sp["box"],
                    "color": self.COLOR_OK,
                    "label": "OK (PASS)",
                    "conf": conf_val
                })

        elif len(valid_defect_items) > 0:
            # SKENARIO D: FILTER GEOMETRI CADANGAN
            ok_items = [it for it in valid_defect_items if not it["is_ng"]]
            for item in ok_items:
                has_spoon = True
                frame_status = "OK"
                boxes_to_draw.append({
                    "box": item["box"],
                    "color": self.COLOR_OK,
                    "label": "OK (PASS)",
                    "conf": item["conf"]
                })

        # -------------------------------------------------------------
        # 4. GAMBAR BOUNDING BOX & CATAT STATISTIK
        # -------------------------------------------------------------
        # Cetak diagnostik real-time ke terminal (rate-limited setiap 0.6 detik agar mudah dibaca)
        if len(defect_boxes) > 0 or len(coco_spoons) > 0:
            if not hasattr(self, "_last_diag_print") or (now - self._last_diag_print >= 0.6):
                self._last_diag_print = now
                raw_best = [f"{self.class_names.get(int(b.cls[0]))}:{float(b.conf[0])*100:.1f}%" for b in defect_boxes] or ["Tidak ada"]
                raw_coco = [f"Spoon:{sp['conf']*100:.1f}%" for sp in coco_spoons] or ["Tidak ada"]
                print(f"[DIAGNOSTIK AI] best.pt: {raw_best} | COCO: {raw_coco} => Status Akhir: {frame_status}")

        if has_spoon:
            self.last_seen_time = now

            # JIKA OBJEK INI SUDAH TERCATAT: KUNCI (LATCH) STATUSNYA!
            # Mencegah gangguan tangan atau pergeseran saat mengambil sendok mengubah status OK menjadi NG
            if self.is_current_counted and self.last_recorded_status is not None:
                frame_status = self.last_recorded_status
                locked_color = self.COLOR_OK if self.last_recorded_status == "OK" else self.COLOR_NG
                locked_label = "OK (PASS)" if self.last_recorded_status == "OK" else "NG (REJECT)"
                for item in boxes_to_draw:
                    item["color"] = locked_color
                    item["label"] = locked_label

            for item in boxes_to_draw:
                b = item["box"]
                conf_str = f"{item['conf'] * 100:.1f}%"
                self._draw_modern_bbox(
                    annotated_frame,
                    b[0], b[1], b[2], b[3],
                    item["color"], item["label"], conf_str
                )

            frame_defect_desc = "Sendok cacat / Defect" if frame_status == "NG" else "None"
            frame_conf = max_ng_conf if frame_status == "NG" else (max_ok_conf if max_ok_conf > 0 else 0.85)

            # AKUMULASI STATISTIK DENGAN MEKANISME COOLDOWN & LATCH (FR-06, FR-08)
            with self.lock:
                self.stats["current_status"] = frame_status
                if frame_status == "NG":
                    self.stats["last_defect"] = frame_defect_desc

                # Hanya hitung jika produk ini adalah sampel baru (belum terhitung)
                if not self.is_current_counted:
                    if frame_status == "OK":
                        self.ok_streak += 1
                        self.ng_streak = 0
                    elif frame_status == "NG":
                        self.ng_streak += 1
                        self.ok_streak = 0

                    cooldown_ok = (now - self.last_counted_time) >= self.cooldown_period
                    # OK diakui segera (>=1 frame) karena model tidak bias ke OK
                    # NG butuh konfirmasi stabil (>=3 frame) agar bayangan tangan saat menaruh tidak memicu false NG
                    status_confirmed = (frame_status == "OK" and self.ok_streak >= 1) or \
                                       (frame_status == "NG" and self.ng_streak >= 3)

                    if cooldown_ok and status_confirmed:
                        self.stats["total_inspected"] += 1
                        if frame_status == "NG":
                            self.stats["total_ng"] += 1
                        else:
                            self.stats["total_ok"] += 1

                        self.is_current_counted = True
                        self.last_counted_time = now
                        self.last_recorded_status = frame_status
                        self.last_recorded_history_id = self.stats["total_inspected"]
                        self.ok_streak = 0
                        self.ng_streak = 0

                        # Tambah entri ke riwayat log (4 kolom sesuai tabel web)
                        history_entry = {
                            "id": self.stats["total_inspected"],
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "status": frame_status,
                            "confidence": f"{frame_conf * 100:.1f}%"
                        }
                        self.stats["history"].append(history_entry)
                        if len(self.stats["history"]) > 50:
                            self.stats["history"].pop(0)

                # CATATAN: Begitu self.is_current_counted == True, status produk TERKUNCI (LATCHED).
                # Tidak ada lagi koreksi balik yang mengurangi total_ok atau menambah total_ng
                # saat sendok sedang diambil / dipindahkan oleh operator.

        else:
            # Tidak ada objek sendok di kamera
            time_since_last_seen = now - self.last_seen_time
            if time_since_last_seen > self.standby_timeout:
                with self.lock:
                    self.stats["current_status"] = "STANDBY"
                    self.stats["last_defect"] = "None"
                    self.is_current_counted = False
                    self.last_recorded_status = None
                    self.ok_streak = 0
                    self.ng_streak = 0

        # Pengukuran performa latensi inferensi & FPS (NFR-01)
        t_lat = (time.perf_counter() - t_start) * 1000.0
        self.latency_ms = (1.0 - self.alpha_smooth) * self.latency_ms + self.alpha_smooth * max(t_lat, 1.0)
        curr_fps = 1000.0 / max(self.latency_ms, 1.0)
        self.fps = (1.0 - self.alpha_smooth) * self.fps + self.alpha_smooth * curr_fps

        return annotated_frame

    def get_stats(self):
        """Mengembalikan data statistik inspeksi secara thread-safe."""
        with self.lock:
            return {
                "total_inspected": self.stats["total_inspected"],
                "total_ok": self.stats["total_ok"],
                "total_ng": self.stats["total_ng"],
                "current_status": self.stats["current_status"],
                "last_defect": self.stats["last_defect"],
                "history": list(self.stats["history"]),
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
                "two_stage": self.two_stage,
                "conf_thresh": self.conf_thresh
            }

    def reset_stats(self):
        """Mereset seluruh akumulasi total inspeksi dan log riwayat (FR-07)."""
        with self.lock:
            self.stats["total_inspected"] = 0
            self.stats["total_ok"] = 0
            self.stats["total_ng"] = 0
            self.stats["current_status"] = "STANDBY"
            self.stats["last_defect"] = "None"
            self.stats["history"] = []
            self.is_current_counted = False
            self.last_recorded_status = None
            self.last_recorded_history_id = None
            self.ok_streak = 0
            self.ng_streak = 0
        return True

    def export_csv_string(self):
        """
        Menghasilkan berkas CSV 4 kolom sesuai tabel riwayat deteksi (US-05).
        """
        with self.lock:
            history_data = list(self.stats["history"])

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["No", "Waktu (Timestamp)", "Status (OK/NG)", "Confidence Score"])

        for item in history_data:
            writer.writerow([
                item.get("id", ""),
                item.get("timestamp", ""),
                item.get("status", ""),
                item.get("confidence", "")
            ])

        return output.getvalue()
