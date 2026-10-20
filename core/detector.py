import os
import cv2
import time
from ultralytics import YOLO

class DefectDetector:
    """
    Class untuk mengelola inferensi YOLOv8n dan pemrosesan frame OpenCV.
    
    Arsitektur Two-Stage AI (Dual-Model Pipeline):
    - Stage 1 (Verifikasi Objek): Menggunakan base model YOLOv8 (COCO class 44: 'spoon')
      untuk memastikan objek yang diarahkan ke kamera BENAR-BENAR SENDOK.
      Objek lain seperti tangan, buku, botol, mug, dll. otomatis ditolak (STANDBY).
    - Stage 2 (Inspeksi Cacat): Menggunakan custom model fine-tuned 'models/best.pt'
      untuk mendeteksi jenis cacat sendok (Bent, Scratch).
      
    Logika Keputusan Kualifikasi QC:
    1. NG (Reject) : Terdeteksi cacat 'Bent' atau 'Scratch' pada sendok.
    2. OK (Pass)   : Objek sendok terverifikasi AI dan BEBAS dari cacat.
    3. Standby     : Tidak ada sendok (area kosong atau benda selain sendok).
    """

    def __init__(self, model_path="models/best.pt", spoon_model_path="yolov8n.pt", conf_threshold=0.25, imgsz=320):
        self.model_path = model_path
        self.spoon_model_path = spoon_model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        
        # 1. Muat Stage 1 Model: Pretrained Spoon Verifier (COCO class 44 = spoon)
        try:
            self.spoon_model = YOLO(self.spoon_model_path)
            print(f"[DefectDetector] Stage 1 (Spoon Verifier) '{self.spoon_model_path}' berhasil dimuat!")
        except Exception as e:
            print(f"[DefectDetector Peringatan] Gagal memuat '{self.spoon_model_path}': {e}")
            self.spoon_model = None

        # 2. Muat Stage 2 Model: Custom Defect Model (Bent, Scratch)
        self.defect_model = None
        self.defect_model_loaded = False
        self.class_names = {}
        self._load_defect_model()
        
        # Tracking statistik inspeksi
        self.stats = {
            "total_inspected": 0,
            "total_ok": 0,
            "total_ng": 0,
            "current_status": "STANDBY",
            "last_defect": "",
            "history": []
        }
        
        # State machine debounce (1 siklus pencatatan per sendok)
        self.object_inspected = False
        self.last_action_time = 0
        self.cooldown_period = 2.0
        self.last_model_check_time = 0

    def _load_defect_model(self):
        """Memuat model defect jika file best.pt tersedia di folder models."""
        if os.path.exists(self.model_path):
            try:
                self.defect_model = YOLO(self.model_path)
                self.class_names = self.defect_model.names if hasattr(self.defect_model, 'names') else {0: 'Bent', 1: 'Scratch'}
                self.defect_model_loaded = True
                print(f"[DefectDetector] Stage 2 (Defect Model) '{self.model_path}' berhasil dimuat!")
                print(f"[DefectDetector] Daftar kelas cacat: {self.class_names}")
                return True
            except Exception as e:
                print(f"[DefectDetector] Gagal memuat defect model '{self.model_path}': {e}")
                self.defect_model_loaded = False
                return False
        else:
            self.defect_model_loaded = False
            return False

    def _draw_brackets(self, img, pt1, pt2, color, thickness=2, length=25):
        """Menggambar corner bracket modern di sudut kotak inspeksi."""
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)

    def process_frame(self, frame):
        """
        Mengeksekusi inferensi Two-Stage AI pada frame kamera:
        1. Verifikasi apakah objek adalah Sendok.
        2. Jika ya, periksa apakah terdapat Cacat (Bent/Scratch).
        """
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        current_time = time.time()
        annotated_frame = frame.copy()

        # Cek secara periodik apakah file best.pt baru saja dimasukkan (hot-reload)
        if not self.defect_model_loaded and (current_time - self.last_model_check_time > 3.0):
            self.last_model_check_time = current_time
            if os.path.exists(self.model_path):
                self._load_defect_model()

        # Koordinat Area Panduan Inspeksi di tengah layar
        roi_x1, roi_y1 = int(w * 0.18), int(h * 0.12)
        roi_x2, roi_y2 = int(w * 0.82), int(h * 0.88)

        # -------------------------------------------------------------
        # TAHAP 1: DETEKSI CACAT (CUSTOM DEFECT MODEL)
        # -------------------------------------------------------------
        has_defects = False
        highest_defect_info = None

        if self.defect_model_loaded:
            try:
                defect_results = self.defect_model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    imgsz=self.imgsz,
                    verbose=False
                )
                defect_boxes = defect_results[0].boxes
                if defect_boxes is not None and len(defect_boxes) > 0:
                    has_defects = True
                    # Render bounding box cacat dari Ultralytics
                    annotated_frame = defect_results[0].plot()

                    defects = []
                    for b in defect_boxes:
                        cid = int(b.cls[0].item())
                        c_conf = float(b.conf[0].item())
                        c_name = self.class_names.get(cid, f"Defect_{cid}")
                        defects.append((c_name, c_conf))
                    highest_defect_info = max(defects, key=lambda x: x[1])
            except Exception as e:
                print(f"[DefectDetector Error] Prediksi cacat gagal: {e}")

        # -------------------------------------------------------------
        # TAHAP 2: VERIFIKASI KEBERADAAN SENDOK (STAGE 1 AI VERIFIER)
        # Benda selain sendok (buku, botol, tangan, cangkir) tidak akan lolos di sini!
        # -------------------------------------------------------------
        spoon_detected = False
        spoon_boxes_list = []

        if self.spoon_model is not None:
            try:
                # Class 44 di dataset COCO adalah 'spoon'
                spoon_results = self.spoon_model.predict(
                    source=frame,
                    classes=[44],
                    conf=0.25,
                    imgsz=self.imgsz,
                    verbose=False
                )
                sp_boxes = spoon_results[0].boxes
                if sp_boxes is not None and len(sp_boxes) > 0:
                    spoon_detected = True
                    for sb in sp_boxes:
                        xyxy = sb.xyxy[0].cpu().numpy().astype(int)
                        conf_val = float(sb.conf[0].item())
                        spoon_boxes_list.append((xyxy, conf_val))
            except Exception as e:
                print(f"[DefectDetector Error] Verifikasi sendok gagal: {e}")

        # -------------------------------------------------------------
        # EVALUASI 3-STATE QUALITY CONTROL
        # -------------------------------------------------------------

        # KASUS 1: NG (REJECT) - Ditemukan cacat Scratch atau Bent
        if has_defects and highest_defect_info:
            d_name, d_conf = highest_defect_info
            conf_percent = int(d_conf * 100)

            # Visual Bracket Merah
            self._draw_brackets(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 240), thickness=3)

            # Banner NG di atas
            cv2.rectangle(annotated_frame, (10, 10), (w - 10, 48), (20, 20, 200), -1)
            cv2.rectangle(annotated_frame, (10, 10), (w - 10, 48), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame,
                f"DEFECT DETECTED: {d_name.upper()} ({conf_percent}%) | STATUS: REJECT (NG)",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Catat statistik 1 kali per siklus produk
            if not self.object_inspected or (current_time - self.last_action_time > self.cooldown_period):
                self.stats["total_inspected"] += 1
                self.stats["total_ng"] += 1
                
                log_entry = {
                    "id": self.stats["total_inspected"],
                    "timestamp": time.strftime("%H:%M:%S"),
                    "status": "NG",
                    "defect_type": d_name,
                    "confidence": f"{conf_percent}%"
                }
                self.stats["history"].append(log_entry)
                if len(self.stats["history"]) > 50:
                    self.stats["history"].pop(0)

                self.object_inspected = True
                self.last_action_time = current_time

            self.stats["current_status"] = "NG"
            self.stats["last_defect"] = f"{d_name} ({conf_percent}%)"

        # KASUS 2: OK (PASS) - Objek TERVERIFIKASI SENDOK dan BEBAS CACAT
        elif spoon_detected and not has_defects:
            # Visual Bracket Hijau Emerald
            self._draw_brackets(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 220, 100), thickness=3)

            # Gambar Bounding Box Hijau di sekeliling sendok
            for xyxy, sp_conf in spoon_boxes_list:
                sx1, sy1, sx2, sy2 = xyxy
                cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), (0, 220, 100), 2)
                
                # Label box sendok
                lbl_text = f"SPOON: OK (PASS) {int(sp_conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (sx1, max(0, sy1 - 22)), (sx1 + tw + 6, sy1), (0, 200, 90), -1)
                cv2.putText(
                    annotated_frame,
                    lbl_text,
                    (sx1 + 3, max(15, sy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # Banner Hijau di bagian atas
            cv2.rectangle(annotated_frame, (10, 10), (w - 10, 48), (20, 140, 50), -1)
            cv2.rectangle(annotated_frame, (10, 10), (w - 10, 48), (0, 220, 100), 2)
            cv2.putText(
                annotated_frame,
                "PRODUCT QUALIFIED: SENDOK NORMAL | STATUS: PASS (OK)",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Catat statistik 1 kali per siklus produk
            if not self.object_inspected or (current_time - self.last_action_time > self.cooldown_period):
                self.stats["total_inspected"] += 1
                self.stats["total_ok"] += 1
                
                log_entry = {
                    "id": self.stats["total_inspected"],
                    "timestamp": time.strftime("%H:%M:%S"),
                    "status": "OK",
                    "defect_type": "Normal (Bebas Cacat)",
                    "confidence": "100%"
                }
                self.stats["history"].append(log_entry)
                if len(self.stats["history"]) > 50:
                    self.stats["history"].pop(0)

                self.object_inspected = True
                self.last_action_time = current_time

            self.stats["current_status"] = "OK"
            self.stats["last_defect"] = "Normal (Bebas Cacat)"

        # KASUS 3: STANDBY - Bukan sendok (benda lain diabaikan) atau area kosong
        else:
            self.object_inspected = False

            if current_time - self.last_action_time > 1.5:
                self.stats["current_status"] = "STANDBY"

            # Visual Bracket Biru / Cyan
            self._draw_brackets(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 190, 0), thickness=2)

            # Teks status di pojok kiri atas
            mode_text = "QC STANDBY | MENUNGGU OBJEK SENDOK"
            if not self.defect_model_loaded:
                mode_text = "QC STANDBY | MENUNGGU 'best.pt' DI FOLDER models/"

            cv2.putText(
                annotated_frame,
                mode_text,
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        return annotated_frame

    def reset(self):
        """Mereset seluruh data akumulasi inspeksi."""
        self.stats["total_inspected"] = 0
        self.stats["total_ok"] = 0
        self.stats["total_ng"] = 0
        self.stats["current_status"] = "STANDBY"
        self.stats["last_defect"] = ""
        self.stats["history"] = []
        self.object_inspected = False
        self.last_action_time = 0