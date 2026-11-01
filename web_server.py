#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анти-проспалка: веб-дашборд поверх твоего main.py.

Запуск (рядом должны лежать haarcascade_*.txt, как у основного скрипта):
    pip install flask flask-cors opencv-python pyserial
    python web_server.py

Затем открой в браузере:  http://localhost:5000

Что делает:
  - Читает камеру и считает глаза/лица (та же логика, что в main.py)
  - Если глаз нет >= EYE_TIMEOUT_SEC сек -> тревога (и команда на Arduino "1\n")
  - Раздаёт живое видео в браузер (MJPEG) + JSON-телеметрию для красивого UI
  - Окно OpenCV не открывается — всё в браузере. Закрытие: Ctrl+C в консоли.
"""

import os
import sys
import time
import json
import threading
import traceback

import cv2

try:
    from flask import Flask, Response, jsonify, request, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("Нужен Flask:  pip install flask flask-cors")
    sys.exit(1)

# ========= НАСТРОЙКИ (те же, что в основном скрипте) =========
EYE_TIMEOUT_SEC = 10.0

ARDUINO_PORT = "COM7"      # ← свой порт Arduino (Windows: COM7, Linux/Mac: /dev/ttyUSB0)
ARDUINO_BAUD = 9600
USE_ARDUINO = True         # False = работать без Arduino

CAM_INDEX = 0
WANTED_W, WANTED_H = 1280, 720
MIRROR = False             # True = зеркалить картинку
PORT = 5000                # порт веб-интерфейса

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_XML = os.path.join(BASE_DIR, "haarcascade_frontalface_default.txt")
EYE_XML = os.path.join(BASE_DIR, "haarcascade_eye_tree_eyeglasses.txt")

# ========================================================================


class ArduinoAlarm:
    """Копия твоего класса — отправка '1\\n'/'0\\n' только при смене состояния."""

    def __init__(self, port, baud=9600):
        self.ser = None
        self.state = False
        self.connected = False
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)
            self.connected = True
            print(f"✅ Arduino подключен: {port}")
        except Exception as e:
            print(f"⚠️ Arduino не подключен: {e}")
            self.ser = None

    def set(self, state: bool):
        if self.ser is None or self.state == state:
            return
        try:
            self.ser.write(b"1\n" if state else b"0\n")
            self.state = state
            print("🚨 >>> Arduino: ALARM ON" if state else "✅ >>> Arduino: ALARM OFF")
        except Exception as e:
            print(f"Ошибка отправки команды на Arduino: {e}")

    def close(self):
        if self.ser is None:
            return
        try:
            self.ser.write(b"0\n")
            time.sleep(0.05)
            self.ser.close()
        except Exception:
            pass


class VisionEngine:
    """Детекция в отдельном потоке; кадры и состояние отдаём в веб."""

    def __init__(self, alarm):
        self.alarm = alarm
        self.lock = threading.Lock()
        self.jpeg = None          # последний закодированный кадр
        self.faces = []           # [[x,y,w,h], ...]
        self.eyes = []            # [[x,y,w,h], ...] (в координатах всего кадра)
        self.eyes_detected = False
        self.last_eyes_seen_at = time.time()
        self.time_without_eyes = 0.0
        self.alarm_needed = False
        self.fps = 0.0
        self.frame_count = 0
        self.running = True
        self.camera_ok = False
        self.error_msg = ""

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _open_camera(self):
        # CAP_DSHOW — как в твоём скрипте (быстрый старт на Windows)
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        cap = cv2.VideoCapture(CAM_INDEX, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAM_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WANTED_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WANTED_H)
        return cap

    def _loop(self):
        # Каскады
        face_cascade = cv2.CascadeClassifier(FACE_XML)
        eye_cascade = cv2.CascadeClassifier(EYE_XML)
        if face_cascade.empty() or eye_cascade.empty():
            self.error_msg = f"Не удалось загрузить каскады рядом со скриптом:\n{FACE_XML}\n{EYE_XML}"
            print("❌", self.error_msg)
            return

        cap = self._open_camera()
        if not cap.isOpened():
            self.error_msg = "Не удалось открыть веб-камеру (CAM_INDEX=%d)" % CAM_INDEX
            print("❌", self.error_msg)
            return
        self.camera_ok = True

        try:
            while self.running:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                if MIRROR:
                    frame = cv2.flip(frame, 1)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
                if isinstance(faces, tuple) and len(faces) > 0:
                    faces = faces[0]  # старые версии OpenCV

                eyes_all = []
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(
                        roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25))
                    if isinstance(eyes, tuple) and len(eyes) > 0:
                        eyes = eyes[0]
                    filtered = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes
                                if ey + eh * 0.5 < h * 0.65]
                    if filtered:
                        eyes_all.extend((x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in filtered)
                        for (ex, ey, ew, eh) in filtered:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                            cx, cy = ex + ew // 2, ey + eh // 2
                            cv2.circle(roi_color, (cx, cy), 2, (0, 0, 255), -1)

                eyes_detected = len(eyes_all) > 0
                now = time.time()
                if eyes_detected:
                    self.last_eyes_seen_at = now
                time_without_eyes = now - self.last_eyes_seen_at
                alarm_needed = time_without_eyes >= EYE_TIMEOUT_SEC

                # Рисуем поверх кадра то же, что было в окне OpenCV
                fh, fw = frame.shape[:2]
                if alarm_needed:
                    cv2.rectangle(frame, (0, 0), (fw, int(0.12 * fh)), (0, 0, 0), -1)
                    cv2.putText(frame, "PROSIPAYASYA!!!",
                                (int(0.05 * fw), int(0.09 * fh)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
                remaining = max(0, EYE_TIMEOUT_SEC - time_without_eyes)
                cv2.putText(frame, f"Timer: {remaining:.1f}s", (10, fh - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255) if not alarm_needed else (0, 0, 255),
                            2, cv2.LINE_AA)

                # Тревога на Arduino
                if self.alarm:
                    self.alarm.set(alarm_needed)

                # Сжимаем JPEG для браузера (масштабируем, чтобы не тормозило)
                small = cv2.resize(frame, (640, int(640 * fh / fw)))
                okj, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if okj:
                    with self.lock:
                        self.jpeg = buf.tobytes()
                        self.faces = [[int(a) for a in f] for f in faces]
                        self.eyes = [[int(a) for a in e] for e in eyes_all]
                        self.eyes_detected = eyes_detected
                        self.time_without_eyes = time_without_eyes
                        self.alarm_needed = alarm_needed
                        self.frame_count += 1
                        dt = time.time() - t0
                        inst_fps = 1.0 / dt if dt > 0 else 0
                        self.fps = round(self.fps * 0.9 + inst_fps * 0.1, 1)
                time.sleep(0.001)
        except Exception:
            self.error_msg = traceback.format_exc()
            print(self.error_msg)
        finally:
            cap.release()

    def get_state(self):
        with self.lock:
            return {
                "camera_ok": self.camera_ok,
                "error": self.error_msg,
                "faces": self.faces,
                "eyes": self.eyes,
                "eyes_detected": self.eyes_detected,
                "time_without_eyes": round(self.time_without_eyes, 2),
                "alarm": self.alarm_needed,
                "timeout_sec": EYE_TIMEOUT_SEC,
                "fps": self.fps,
                "frames": self.frame_count,
                "arduino_connected": bool(self.alarm and self.alarm.connected),
                "arduino_alarm": bool(self.alarm and self.alarm.state),
                "ts": time.time(),
            }

    def get_jpeg(self):
        with self.lock:
            return self.jpeg


app = Flask(__name__, static_folder=None)
CORS(app)
engine = None


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "web_ui.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            jpg = engine.get_jpeg()
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.033)  # ~30 fps максимум
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(engine.get_state())


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True})


def main():
    global engine
    print("=" * 50)
    print("👁️  Анти-проспалка — веб-дашборд")
    print("=" * 50)
    if not os.path.exists(FACE_XML) or not os.path.exists(EYE_XML):
        print(f"⚠️  Каскады не найдены рядом с web_server.py ({BASE_DIR}).")
        print("   Скопируй сюда haarcascade_frontalface_default.txt и haarcascade_eye_tree_eyeglasses.txt")

    alarm = ArduinoAlarm(ARDUINO_PORT, ARDUINO_BAUD) if USE_ARDUINO else None
    engine = VisionEngine(alarm)
    engine.start()

    print(f"🌐 Открой в браузере:  http://localhost:{PORT}")
    print("   Остановить: Ctrl+C")
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        engine.running = False
        if alarm:
            alarm.close()


if __name__ == "__main__":
    main()
