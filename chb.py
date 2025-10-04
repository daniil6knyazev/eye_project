import os
import cv2

# Путь к каскадам (без кириллицы)
CASCADE_DIR = r"C:\Users\Public\opencv_haarcascades"
FACE_XML = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
EYE_XML  = os.path.join(CASCADE_DIR, "haarcascade_eye_tree_eyeglasses.xml")  # можно заменить на 'haarcascade_eye.xml'

# Настройки
MIRROR_PREVIEW = True   # отзеркаливать картинку для вывода (по желанию)
NIGHT_MODE = True       # включить усиление для тёмной сцены

def main():
    # Проверим каскады
    if not (os.path.exists(FACE_XML) and os.path.exists(EYE_XML)):
        raise FileNotFoundError(f"Не найдены каскады:\n{FACE_XML}\n{EYE_XML}")

    face_cascade = cv2.CascadeClassifier(FACE_XML)
    eye_cascade  = cv2.CascadeClassifier(EYE_XML)
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Не удалось загрузить каскады Haar.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть веб-камеру.")

    # Разрешение: в темноте лучше 640x480 (быстрее и стабильнее), но можно оставить 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 if NIGHT_MODE else 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 if NIGHT_MODE else 720)

    # Попытка включить автоэкспозицию (может игнорироваться драйвером)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 — авто для некоторых backend'ов
    # cap.set(cv2.CAP_PROP_EXPOSURE, -5)         # пример: ручная экспозиция (если выключили авто)

    # Создаём CLAHE один раз (локальное выравнивание контраста)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            # В серый канал
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if NIGHT_MODE:
                # Небольшое шумоподавление (в темноте шумит матрица)
                gray = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)
                # Локальное выравнивание контраста
                gray = clahe.apply(gray)
                # Чуть увеличим яркость/контраст (alpha) и добавим небольшой сдвиг (beta)
                gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=8)
            else:
                # В дневном режиме хватит глобального выравнивания
                gray = cv2.equalizeHist(gray)

            # Параметры детекции (в ночном режиме — чуть «мягче»)
            face_scale = 1.05 if NIGHT_MODE else 1.1
            face_neighbors = 4 if NIGHT_MODE else 5
            face_min_size = (80, 80) if NIGHT_MODE else (100, 100)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=face_scale, minNeighbors=face_neighbors, minSize=face_min_size
            )

            # Рисуем белые рамки на ч/б изображении
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x + w, y + h), 255, 2)

                roi_gray = gray[y:y + h, x:x + w]

                eye_scale = 1.05 if NIGHT_MODE else 1.1
                eye_neighbors = 8 if NIGHT_MODE else 10
                eye_min_size = (18, 18) if NIGHT_MODE else (25, 25)

                eyes = eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=eye_scale, minNeighbors=eye_neighbors, minSize=eye_min_size
                )

                # Фильтр: глаза чаще в верхней части лица
                filtered_eyes = [
                    (ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes
                    if ey + eh * 0.5 < h * 0.65
                ]

                for (ex, ey, ew, eh) in filtered_eyes:
                    cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), 255, 2)
                    cx, cy = ex + ew // 2, ey + eh // 2
                    cv2.circle(roi_gray, (cx, cy), 2, 255, -1)

            cv2.imshow('Eye detection q/ESC for exit', gray)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()