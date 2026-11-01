# Eye Monitor — веб-дашборд визуализации

Красивый сайт для проекта «анти-проспалка»: детекция глаз (Haar-каскады OpenCV,
та же логика, что в `main/main`) + интеграция с Arduino.

## Что умеет
- 🎥 **Живая трансляция** с камеры прямо в браузере (MJPEG-поток `/video_feed`)
  с рамками лиц/глаз и надписью *PROSIPAYASYA!!!* при тревоге
- 📊 **Дашборд**: статус глаз, FPS, число лиц/глаз, аптайм, прогресс-бар
  «тревога через N секунд»
- 📈 **График бодрствования во времени** (canvas, без внешних библиотек) —
  красные зоны = сон
- 🔌 **Arduino**: эндпоинт `POST /api/arduino` принимает JSON-телеметрию
  (ESP32 по Wi-Fi или скрипт-мост через USB-serial), карточка на дашборде
  показывает связь и последние данные
- 🔄 Демо-режим: если камеры нет, UI всё равно работает и обновляется

## Запуск

```bash
pip install flask flask-cors opencv-python
python web/server.py
# открыть http://localhost:5000
```

Индекс камеры можно задать переменной окружения: `CAMERA_INDEX=1 python web/server.py`

## API
| Эндпоинт | Метод | Описание |
|---|---|---|
| `/` | GET | сам дашборд |
| `/video_feed` | GET | MJPEG-видеопоток |
| `/api/state` | GET | JSON: alert, eyes, faces, fps, history… (обновление на фронте каждые 0.5 с) |
| `/api/arduino` | POST | принять JSON от Arduino, напр. `{"servo_angle":45,"buzzer":1}` |
| `/api/arduino` | GET | статус связи и последние данные |

## Пример скетча для Arduino (ESP32, отправляет телеметрию)

```cpp
#include <WiFi.h>
#include <HTTPClient.h>

const char* SERVER = "http://192.168.1.50:5000/api/arduino"; // IP компа

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER);
    http.addHeader("Content-Type", "application/json");
    String body = "{\"servo_angle\":" + String(millis() % 180) + ",\"buzzer\":0}";
    http.POST(body);
    http.end();
  }
  delay(2000);
}
```

Для обычной Arduino UNO по USB — маленький python-мост:

```python
import serial, requests, json
ser = serial.Serial("COM3", 9600)
while True:
    line = ser.readline().decode(errors="ignore").strip()
    try:
        requests.post("http://localhost:5000/api/arduino", json=json.loads(line))
    except Exception:
        pass
```

## Структура
```
web/
├── server.py      # Flask-сервер + поток детекции (логика как в main/main)
└── ui/
    └── index.html # дашборд (HTML/CSS/JS, один файл, без сборки)
```
