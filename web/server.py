"""
Веб-сервер визуализации для проекта "Анти-проспалка" (Eye Monitor).

Запускает ту же логику детекции, что и main/main, но вместо окна OpenCV
раздаёт:
  GET  /              -> красивый дашборд (web/ui/index.html)
  GET  /video_feed    -> MJPEG-поток (аннотированное видео с камеры)
  GET  /api/state     -> текущее состояние в JSON (карточки + график)
  POST /api/arduino   -> приём телеметрии с Arduino (опционально)
  GET  /api/arduino   -> последние данные с Arduino

Зависимости:  pip install flask flask-cors opencv-python
Запуск:      python web/server.py   (затем открыть http://localhost:5000)
"""

import os
import time
import threading

try:
    import cv2
except ImportError:  # если OpenCV не установлен — сервер всё равно поднимет UI в демо-режиме
    cv2 = None
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_DIR = os.path.join(os.path.dirname(BASE_DIR), "main")

FACE_XML = os.path.join(DET_DIR, "haarcascade_frontalface_default.txt")
EYE_XML = os.path.join(DET_DIR, "haarcascade_eye_tree_eyeglasses.txt")

EYE_TIMEOUT_SEC = 10.0          # через сколько секунд без глаз = "уснул"
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
POLL_INTERVAL = 0.1             # период обновления состояния / графика

app = Flask(__name__, static_folder="ui", static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# Общее состояние: пишет поток обработки, читают API и фронt
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
start_time = time.time()

state = {
    "camera_ok": False,
    "fps": 0.0,
    "faces": 0,
    "eyes": 0,
    "alert": False,             # True => глаза не видно дольше EYE_TIMEOUT_SEC
    "seconds_since_eyes": 0.0,
    "uptime_sec": 0.0,
    "frames_processed": 0,
    "history": [],              # точки для графика на фронте
}


def set_state(**kwargs):
    with state_lock:
        state.update(kwargs)


def get_state():
    with state_lock:
        return dict(state, history=list(state["history"]))


# ---------------------------------------------------------------------------
# Кадровый буфер для MJPEG-стрима
# ---------------------------------------------------------------------------
frame_lock = threading.Lock()
latest_jpeg = None  # bytes


def publish_frame(jpeg_bytes):
    global latest_jpeg
    with frame_lock:
        latest_jpeg = jpeg_bytes


def snapshot_jpeg():
    with frame_lock:
        return latest_jpeg


# ---------------------------------------------------------------------------
# История состояний (график "бодрствование/сон")
# ---------------------------------------------------------------------------
history = []
history_lock = threading.Lock()


def push_history_point(alert, eyes_count):
    point = {
        "t": round(time.time() - start_time, 1),
        "eyes": eyes_count,
        "awake": 0 if alert else 1,
        "alert": 1 if alert else 0,
    }
    with history_lock:
        history.append(point)
        if len(history) > 600:
            del history[:-600]
    with state_lock:
        state["history"] = list(history)


# ---------------------------------------------------------------------------
# Основной цикл обработки видео (алгоритм как в main/main)
# ---------------------------------------------------------------------------
def demo_loop():
    """Нет камеры / нет OpenCV — гоним состояние и историю, чтобы UI жил."""
    set_state(camera_ok=False)
    last_seen = time.time()
    while True:
        time.sleep(POLL_INTERVAL)
        now = time.time()
        idle = now - last_seen
        alert = idle >= EYE_TIMEOUT_SEC
        set_state(
            alert=alert,
            seconds_since_eyes=round(idle, 1),
            uptime_sec=round(now - start_time, 1),
        )
        push_history_point(alert, 0)


def process_loop():
    if cv2 is None or not hasattr(cv2, "CascadeClassifier"):
        print("[!] OpenCV без модуля objdetect (или не установлен) — демо-режим.")
        demo_loop()
        return

    if not (os.path.exists(FACE_XML) and os.path.exists(EYE_XML)):
        print("[!] Не найдены файлы каскадов:\n{}\n{}".format(FACE_XML, EYE_XML))
        demo_loop()
        return

    face_cascade = cv2.CascadeClassifier(FACE_XML)
    eye_cascade = cv2.CascadeClassifier(EYE_XML)
    if face_cascade.empty() or eye_cascade.empty():
        print("[!] Не удалось загрузить каскады Haar. Проверьте пути/формат.")
        demo_loop()
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[!] Камера не открыта — демо-режим (без видео, API работает).")
        demo_loop()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    set_state(camera_ok=True)
    last_eyes_seen_at = time.time()
    frames_count = 0
    fps_t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        eyes_total = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 120), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25)
            )
            filtered = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes
                        if ey + eh * 0.5 < h * 0.65]
            eyes_total += len(filtered)

            for (ex, ey, ew, eh) in filtered:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 80, 0), 2)
                cx, cy = ex + ew // 2, ey + eh // 2
                cv2.circle(roi_color, (cx, cy), 2, (0, 0, 255), -1)

        now = time.time()
        if eyes_total > 0:
            last_eyes_seen_at = now

        alert = (now - last_eyes_seen_at) >= EYE_TIMEOUT_SEC
        if alert:
            hh, ww = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (ww, int(0.12 * hh)), (0, 0, 0), -1)
            cv2.putText(frame, "PROSIPAYASYA!!!",
                        (int(0.05 * ww), int(0.09 * hh)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)

        frames_count += 1
        if now - fps_t0 >= 1.0:
            fps = frames_count / (now - fps_t0)
            frames_count = 0
            fps_t0 = now
        else:
            fps = state.get("fps", 0.0)

        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        publish_frame(jpeg.tobytes())

        set_state(
            fps=round(fps, 1),
            faces=len(faces),
            eyes=eyes_total,
            alert=alert,
            seconds_since_eyes=round(now - last_eyes_seen_at, 1),
            uptime_sec=round(now - start_time, 1),
            frames_processed=state.get("frames_processed", 0) + 1,
        )
        push_history_point(alert, eyes_total)


# ---------------------------------------------------------------------------
# Маршруты
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            jpg = snapshot_jpeg()
            if jpg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpg + b"\r\n")
            time.sleep(0.033)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(get_state())


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True})


# Телеметрия с Arduino: POST {"servo_angle": 45, "buzzer": 1, ...}
arduino_store = {"last_seen": 0.0, "data": {}}


@app.route("/api/arduino", methods=["POST"])
def api_arduino_post():
    payload = request.get_json(force=True, silent=True) or {}
    arduino_store["last_seen"] = time.time()
    arduino_store["data"] = payload
    return jsonify({"received": True})


@app.route("/api/arduino")
def api_arduino_get():
    age = time.time() - arduino_store["last_seen"] if arduino_store["last_seen"] else None
    return jsonify({
        "connected": age is not None and age < 15,
        "seconds_ago": round(age, 1) if age is not None else None,
        "data": arduino_store["data"],
    })


def main():
    worker = threading.Thread(target=process_loop, daemon=True)
    worker.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
