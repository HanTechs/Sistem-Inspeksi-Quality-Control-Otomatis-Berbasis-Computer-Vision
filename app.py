"""
Entry Point Server Flask & API Dashboard QC Sendok (app.py)
Proyek: Sistem Inspeksi QC Sendok Otomatis (PBLIF3-MD02)
Pengembang: Muhammad Farhan (Backend & AI) & M. Khairil Candra (Frontend & Testing)

Fitur:
- Routing Dashboard Web Kiosk Mode (HTML5 + Tailwind CSS)
- Endpoint Streaming Video MJPEG Real-Time (/video_feed)
- REST API Statistik Inspeksi Real-Time (/api/stats)
- REST API Manajemen Sumber Kamera & Auto-Switch (/api/cameras, /api/camera_status, /api/set_camera)
- REST API Reset Akumulasi Statistik Batch (/api/reset)
- REST API Ekspor Riwayat Log Inspeksi ke Berkas CSV (/api/export_csv)
"""

import os
import cv2
import time
import signal
import sys
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request

from core.camera import CameraManager
from core.detector import SpoonDefectDetector

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Konfigurasi Path Model Deteksi Cacat Sendok & Validasi Objek
MODEL_DEFECT_PATH = os.path.join("models", "best.pt")
MODEL_COCO_PATH = os.path.join("models", "yolov8n.pt")

# Inisialisasi Modul Kamera dan Detektor AI (Default: Kamera Laptop di Indeks 0)
print("[System] Menginisialisasi CameraManager...")
camera_manager = CameraManager(default_source=0)

print("[System] Menginisialisasi SpoonDefectDetector (Two-Stage AI: COCO + best.pt)...")
defect_detector = SpoonDefectDetector(
    model_path=MODEL_DEFECT_PATH,
    coco_model_path=MODEL_COCO_PATH,
    conf_thresh=0.45,
    coco_conf=0.25,
    two_stage=True
)


def generate_mjpeg_frames():
    """
    Generator streaming video real-time berformat MJPEG.
    Mengambil frame dari kamera, melakukan inferensi Two-Stage YOLO,
    dan melakukan encoding ke format JPEG.
    """
    while True:
        try:
            # Ambil frame mentah terbaru dari kamera
            raw_frame = camera_manager.get_raw_frame()

            # Proses inferensi AI & anotasi bounding box
            annotated_frame = defect_detector.process_frame(raw_frame)

            # Encode frame ke format JPEG
            success, buffer = cv2.imencode(
                ".jpg",
                annotated_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )

            if not success:
                time.sleep(0.01)
                continue

            frame_bytes = buffer.tobytes()

            # Format MJPEG multipart stream
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        except Exception as e:
            time.sleep(0.05)


# -------------------------------------------------------------
# ROUTE UTAMA DASHBOARD WEB (FR-01)
# -------------------------------------------------------------
@app.route("/")
def index():
    """Menampilkan antarmuka utama Web Dashboard QC (Kiosk Mode)."""
    return render_template("index.html")


# -------------------------------------------------------------
# ENDPOINT VIDEO FEED MJPEG (FR-02)
# -------------------------------------------------------------
@app.route("/video_feed")
def video_feed():
    """Endpoint aliran video streaming MJPEG dengan anotasi bounding box real-time."""
    return Response(
        generate_mjpeg_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------------------------------------------------
# REST API STATISTIK INSPEKSI (FR-06, FR-08, FR-09)
# -------------------------------------------------------------
@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Mengembalikan metrik statistik inspeksi, status kualifikasi, dan riwayat log terbaru."""
    stats_data = defect_detector.get_stats()
    return jsonify(stats_data)


# -------------------------------------------------------------
# REST API MANAJEMEN SUMBER KAMERA & SWITCHER (FR-02)
# -------------------------------------------------------------
@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    """Mengembalikan daftar perangkat kamera yang tersedia beserta status koneksi."""
    cameras_info = camera_manager.get_cameras_info()
    return jsonify(cameras_info)


@app.route("/api/camera_status", methods=["GET"])
def get_camera_status():
    """Mengembalikan status kamera dan event auto-switch terkini untuk polling frontend."""
    status = camera_manager.get_camera_status()
    return jsonify(status)


@app.route("/api/set_camera", methods=["POST"])
def set_camera():
    """Mengubah sumber kamera aktif (Kamera Laptop vs IP Webcam HP)."""
    data = request.get_json(silent=True) or {}
    camera_id = data.get("camera_id")
    ip_url = data.get("ip_url")

    if camera_id is None:
        return jsonify({"status": "error", "message": "Parameter camera_id wajib diisi."}), 400

    success, message = camera_manager.switch_camera(camera_id, ip_url=ip_url)
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "current_source": camera_manager.current_source,
        "ip_webcam_url": camera_manager.ip_webcam_url,
        "is_connected": camera_manager.is_connected
    })


@app.route("/api/set_ip_webcam", methods=["POST"])
def set_ip_webcam():
    """Mengatur dan menghubungkan alamat URL stream IP Webcam (HP)."""
    data = request.get_json(silent=True) or {}
    ip_url = data.get("ip_url")
    if not ip_url:
        return jsonify({"status": "error", "message": "Parameter ip_url wajib diisi."}), 400

    camera_manager.set_ip_webcam_url(ip_url)
    success, message = camera_manager.switch_camera("ipwebcam")
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "current_source": camera_manager.current_source,
        "ip_webcam_url": camera_manager.ip_webcam_url,
        "is_connected": camera_manager.is_connected
    })


@app.route("/api/set_detector_config", methods=["POST"])
def set_detector_config():
    """Mengatur konfigurasi sensitivitas detektor AI dan mode Two-Stage."""
    data = request.get_json(silent=True) or {}
    if "conf_thresh" in data:
        defect_detector.set_confidence(float(data["conf_thresh"]))
    if "two_stage" in data:
        defect_detector.set_two_stage(bool(data["two_stage"]))
    return jsonify({
        "status": "success",
        "conf_thresh": defect_detector.conf_thresh,
        "two_stage": defect_detector.two_stage
    })


@app.route("/api/set_auto_switch", methods=["POST"])
def set_auto_switch():
    """Endpoint kompatibilitas (Auto-switch dinonaktifkan, menggunakan pemilihan manual)."""
    return jsonify({
        "status": "success",
        "message": "Mode pemilihan kamera manual aktif."
    })


# -------------------------------------------------------------
# REST API RESET COUNTER STATISTIK (FR-07)
# -------------------------------------------------------------
@app.route("/api/reset", methods=["POST"])
def reset_statistics():
    """Mereset akumulasi total inspeksi, OK, NG, dan riwayat batch produksi."""
    defect_detector.reset_stats()
    return jsonify({
        "status": "success",
        "message": "Akumulasi statistik berhasil direset."
    })


# -------------------------------------------------------------
# REST API EKSPOR LOG RIWAYAT INSPEKSI KE CSV (US-05)
# -------------------------------------------------------------
@app.route("/api/export_csv", methods=["GET"])
def export_csv():
    """Mengunduh berkas CSV berisi rekaman seluruh riwayat deteksi inspeksi."""
    csv_content = defect_detector.export_csv_string()
    filename = f"log_inspeksi_sendok_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Cache-Control": "no-cache"
        }
    )


def graceful_exit(sig, frame):
    """Menutup resource kamera dan server secara bersih saat menerima sinyal shutdown."""
    print("\n[System] Menerima sinyal penghentian. Melepaskan resource kamera...")
    camera_manager.release()
    sys.exit(0)


signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)


if __name__ == "__main__":
    print("[System] Menjalankan Server Flask QC Sendok pada http://localhost:5000")
    # Menjalankan Flask dengan threaded=True untuk konkurensi streaming & API
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
