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
        
        # Inisialisasi font untuk rendering bounding box modern bergaya UI mockup (image.png)
        self.font_main = None
        self.font_status = None
        self._init_fonts()

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

    def _init_fonts(self):
        """Memuat font antialiased untuk bounding box modern sesuai template image.png."""
        try:
            from PIL import ImageFont
            font_candidates = [
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arial.ttf"
            ]
            chosen = None
            for p in font_candidates:
                if os.path.exists(p):
                    chosen = p
                    break
            if chosen:
                self.font_main = ImageFont.truetype(chosen, 13)
                self.font_status = ImageFont.truetype(chosen, 11)
            else:
                self.font_main = ImageFont.load_default()
                self.font_status = ImageFont.load_default()
        except Exception as e:
            print(f"[DefectDetector] Peringatan memuat font: {e}")
            self.font_main = None
            self.font_status = None

    def _draw_modern_box(self, img, box, label, conf, status="NG", is_ng=True):
        """
        Menggambar bounding box modern bersudut bulat (rounded) dengan badge label
        dan badge status (NG/OK) persis seperti desain image.png.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 - x1 < 15 or y2 - y1 < 15:
            return img

        # Skema Warna sesuai image.png
        if is_ng:
            box_color = (239, 68, 68)      # Modern Red #EF4444 (RGB)
        else:
            box_color = (16, 185, 129)     # Modern Emerald Green #10B981 (RGB)
        dark_bg = (15, 23, 42)             # Slate 900 #0F172A (RGB)

        try:
            from PIL import Image, ImageDraw
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            box_w = x2 - x1
            box_h = y2 - y1
            radius = min(10, box_w // 4, box_h // 4)

            # 1. Garis Luar Bounding Box Bulat (Rounded Box)
            draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, outline=box_color, width=2)

            # 2. Badge Kiri Atas: 'Label: Conf%' (misal: 'Bent: 89.4%')
            conf_str = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"
            label_text = f"{label}: {conf_str}" if label else conf_str

            font_m = self.font_main
            font_s = self.font_status

            if font_m is not None:
                bbox = font_m.getbbox(label_text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            else:
                tw, th = len(label_text) * 8, 14

            margin = 5
            bx1 = x1 + margin
            by1 = y1 + margin
            bx2 = min(x2 - margin, bx1 + tw + 14)
            by2 = min(y2 - margin, by1 + th + 8)

            if bx2 > bx1 and by2 > by1:
                badge_radius = min(5, (bx2 - bx1) // 2, (by2 - by1) // 2)
                draw.rounded_rectangle([bx1, by1, bx2, by2], radius=badge_radius, fill=box_color)
                if font_m is not None:
                    draw.text((bx1 + 7, by1 + 3), label_text, font=font_m, fill=(255, 255, 255))

            # 3. Badge Kanan Bawah: 'NG' atau 'OK'
            st_text = status.upper()
            if font_s is not None:
                s_bbox = font_s.getbbox(st_text)
                stw = s_bbox[2] - s_bbox[0]
                sth = s_bbox[3] - s_bbox[1]
            else:
                stw, sth = len(st_text) * 8, 12

            sx2 = x2 - margin
            sy2 = y2 - margin
            sx1 = max(x1 + margin, sx2 - stw - 10)
            sy1 = max(y1 + margin, sy2 - sth - 6)

            if (sx1 > bx1 or sy1 > by2) and sx2 > sx1 and sy2 > sy1:
                st_radius = min(4, (sx2 - sx1) // 2, (sy2 - sy1) // 2)
                draw.rounded_rectangle([sx1, sy1, sx2, sy2], radius=st_radius, fill=dark_bg)
                if font_s is not None:
                    draw.text((sx1 + 5, sy1 + 2), st_text, font=font_s, fill=(255, 255, 255))

            return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

        except Exception:
            # Fallback ke OpenCV standar jika PIL terkendala
            bgr_color = (box_color[2], box_color[1], box_color[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 2)
            cv2.putText(img, f"{label}: {int(conf*100)}%", (x1 + 6, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)
            cv2.putText(img, status, (x2 - 25, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            return img

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
        defect_detections = []

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
                    defects = []
                    for b in defect_boxes:
                        cid = int(b.cls[0].item())
                        c_conf = float(b.conf[0].item())
                        c_name = self.class_names.get(cid, f"Defect_{cid}")
                        xyxy = b.xyxy[0].cpu().numpy().astype(int)
                        defects.append((c_name, c_conf))
                        defect_detections.append((xyxy, c_name, c_conf))
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
        # EVALUASI 3-STATE QUALITY CONTROL & BOUNDING BOX MODERN (IMAGE.PNG)
        # -------------------------------------------------------------

        # KASUS 1: NG (REJECT) - Ditemukan cacat Scratch atau Bent
        if has_defects and highest_defect_info:
            d_name, d_conf = highest_defect_info
            conf_percent = int(d_conf * 100)

            # Render Bounding Box Cacat Modern Bergaya image.png (Merah, Pill Bent: 89.4%, Badge NG)
            for xyxy, name, conf_val in defect_detections:
                annotated_frame = self._draw_modern_box(
                    annotated_frame,
                    xyxy,
                    label=name,
                    conf=conf_val,
                    status="NG",
                    is_ng=True
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
            self.stats["last_defect"] = f"{d_name}: {d_conf * 100:.1f}%"

        # KASUS 2: OK (PASS) - Objek TERVERIFIKASI SENDOK dan BEBAS CACAT
        elif spoon_detected and not has_defects:
            # Render Bounding Box Sendok Hijau Modern (OK)
            for xyxy, sp_conf in spoon_boxes_list:
                annotated_frame = self._draw_modern_box(
                    annotated_frame,
                    xyxy,
                    label="Spoon",
                    conf=sp_conf,
                    status="OK",
                    is_ng=False
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