import os
import cv2
import time  # таймер для контроля 10 секунд

CASCADE_DIR = r"C:\Users\Public\opencv_haarcascades"
FACE_XML = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
EYE_XML  = os.path.join(CASCADE_DIR, "haarcascade_eye_tree_eyeglasses.xml")

EYE_TIMEOUT_SEC = 10.0  # порог в секундах

def main():
    # Проверим, что файлы на месте
    if not (os.path.exists(FACE_XML) and os.path.exists(EYE_XML)):
        raise FileNotFoundError(f"Не найдены файлы каскадов:\n{FACE_XML}\n{EYE_XML}")

    face_cascade = cv2.CascadeClassifier(FACE_XML)
    eye_cascade  = cv2.CascadeClassifier(EYE_XML)
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Не удалось загрузить каскады Haar. Проверьте пути/доступ.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть веб-камеру.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    MIRROR = False

    last_eyes_seen_at = time.time()  # когда в последний раз видели глаза

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if MIRROR:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            eyes_detected = False  # флаг: нашли ли глаза на этом кадре

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi_gray  = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25)
                )
                filtered_eyes = [
                    (ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes
                    if ey + eh * 0.5 < h * 0.65
                ]

                if len(filtered_eyes) > 0:
                    eyes_detected = True

                for (ex, ey, ew, eh) in filtered_eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                    cx, cy = ex + ew // 2, ey + eh // 2
                    cv2.circle(roi_color, (cx, cy), 2, (0, 0, 255), -1)

            # Обновляем таймер, если глаза найдены
            now = time.time()
            if eyes_detected:
                last_eyes_seen_at = now

            # Если глаза не видны дольше порога — показываем предупреждение
            if now - last_eyes_seen_at >= EYE_TIMEOUT_SEC:
                h, w = frame.shape[:2]
                msg = "PROSIPAYASYA!!!"
                # Полоска под текст для читаемости
                cv2.rectangle(frame, (0, 0), (w, int(0.12 * h)), (0, 0, 0), -1)
                # Крупный красный текст
                cv2.putText(frame, msg, (int(0.05 * w), int(0.09 * h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)

            cv2.imshow('Eye detection (q/ESC to quit)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()