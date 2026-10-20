import cv2
import time
from flask import Flask, render_template, Response, jsonify, request
from core.detector import DefectDetector
from core.camera import CameraManager

# Inisialisasi Flask, AI Engine, dan Camera Manager
# default_source=0 (Kamera Laptop), auto_switch=True (Otomatis pindah ke Iriun saat terhubung)
app = Flask(__name__)
detector = DefectDetector()
camera_manager = CameraManager(default_source=0, auto_switch=True)

def generate_frames():
    """
    Menangkap video dari camera_manager, memprosesnya dengan YOLO secara real-time,
    dan menyalurkannya sebagai byte stream MJPEG ke browser.
    """
    while True:
        frame, is_live = camera_manager.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        # Jika kamera aktif, jalankan inferensi YOLO
        if is_live:
            annotated_frame = detector.process_frame(frame)
        else:
            annotated_frame = frame

        # Konversi array gambar numpy ke format JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            time.sleep(0.03)
            continue

        frame_bytes = buffer.tobytes()

        # Menghasilkan byte stream format MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Batasi ke ~30 FPS untuk efisiensi streaming
        time.sleep(0.01)

@app.route('/')
def dashboard():
    """Merender halaman utama UI Dashboard."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """FR-02: Endpoint penyalur stream video MJPEG."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Endpoint untuk mendapatkan daftar perangkat kamera dan status aktif."""
    available_cameras = CameraManager.get_available_cameras()
    status = camera_manager.get_status()
    return jsonify({
        "status": "success",
        "cameras": available_cameras,
        "current_source": status["current_source"],
        "is_connected": status["is_connected"],
        "status_message": status["status_message"],
        "auto_switch_enabled": status["auto_switch_enabled"],
        "iriun_connected": status["iriun_connected"],
        "last_event": status["last_event"]
    })

@app.route('/api/set_camera', methods=['POST'])
def set_camera():
    """Endpoint untuk mengganti sumber kamera secara manual."""
    data = request.get_json(silent=True) or request.form
    camera_id = data.get('camera_id')

    if camera_id is None:
        return jsonify({"status": "error", "message": "Parameter camera_id diperlukan."}), 400

    try:
        # Jika berupa string angka, konversi ke integer
        if isinstance(camera_id, str) and camera_id.isdigit():
            camera_id = int(camera_id)
        
        success, message = camera_manager.switch_camera(camera_id, manual=True)
        if success:
            return jsonify({
                "status": "success",
                "message": message,
                "current_source": camera_manager.current_source
            })
        else:
            return jsonify({
                "status": "error",
                "message": message,
                "current_source": camera_manager.current_source
            }), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/set_auto_switch', methods=['POST'])
def set_auto_switch():
    """Endpoint untuk mengaktifkan/menonaktifkan auto-switch ke Iriun Webcam."""
    data = request.get_json(silent=True) or request.form
    enabled = data.get('enabled', True)
    if isinstance(enabled, str):
        enabled = enabled.lower() in ['true', '1', 'yes']

    new_val = camera_manager.set_auto_switch(enabled)
    return jsonify({
        "status": "success",
        "auto_switch_enabled": new_val,
        "message": f"Auto-switch {'diaktifkan' if new_val else 'dinonaktifkan'}."
    })

@app.route('/api/camera_status', methods=['GET'])
def camera_status():
    """Endpoint untuk memantau status koneksi kamera dan event auto-switch secara real-time."""
    return jsonify(camera_manager.get_status())

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """FR-09: Endpoint REST API untuk menarik data statistik secara asynchronous."""
    return jsonify(detector.stats)

@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """FR-07: Endpoint REST API untuk mereset counter statistik ke angka nol."""
    detector.reset()
    return jsonify({"status": "success", "message": "Counter berhasil direset."})

if __name__ == '__main__':
    try:
        # PENTING: use_reloader=False wajib digunakan di Windows saat mengelola
        # perangkat keras (webcam) agar Flask tidak menduplikasi proses
        # yang saling berebut mengunci (device lock) kamera laptop.
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
    finally:
        camera_manager.release()