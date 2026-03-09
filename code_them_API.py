import os
import glob as _glob
import ctypes

# Preload libcusparseLt.so.0 for Jetson Orin (JetPack 6 + pip torch)
# Must be done BEFORE importing torch using ctypes.CDLL (LD_LIBRARY_PATH is too late)
_matches = _glob.glob(os.path.expanduser(
    "~/.local/lib/python*/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"
))
if _matches:
    ctypes.CDLL(_matches[0])

os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"
import torch
from ultralytics import YOLO
import cv2
import time
import threading
from queue import Queue
import signal
import sys
import requests
import numpy as np
import asyncio
from amqtt.client import MQTTClient
import json
# ================= CHECK GPU =================
DEVICE = 0   # TensorRT device id
MODEL_PATH = "yolov8n.engine"
model = YOLO(MODEL_PATH, task="detect")
torch.backends.cudnn.benchmark = True

# ================= CONFIG =================
TOTAL_VIDEO = 25
DETECT_FPS = 100
RESIZE = (416, 416)   # tốt hơn cho Nano
BATCH_SIZE = 1        # RTX 2050 safe
RTSP_URL = "rtsp://100.82.253.83:8554/cam"

QUEUE_PER_CAM = 2     # 🔥 quan trọng: nhỏ để realtime

# ================= LOG CONFIG =================
LOG_INTERVAL = 0.5
LOG_FILE = "camera_stats.csv"

# ================= MQTT CONFIG =================
MQTT_BROKER = "100.82.253.83"
MQTT_PORT = 1883
MQTT_TOPIC = "camera/stats"
CLIENT_ID = "machine_a_camera_ai"
API_SEND_INTERVAL = 0.5
MQTT_URI = f"mqtt://{MQTT_BROKER}:{MQTT_PORT}/"


# ================= INIT LOG FILE =================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,cam_id,fps,people,is_night\n")

# ================= CAMERA STATE =================
init_time = int(time.time())
camera_state = {
    vid: {
        "timestamp": init_time,
        "fps": 0.0,
        "people": 0,
        "is_night": 0,
    }
    for vid in range(TOTAL_VIDEO)
}
state_lock = threading.Lock()
last_log_time = {}
last_time = {}

# ================= QUEUE PER CAMERA =================
frame_queues = {
    vid: Queue(maxsize=QUEUE_PER_CAM)
    for vid in range(TOTAL_VIDEO)
}

# ================= DAY / NIGHT =================
def get_brightness(frame):
    """Trả về nhãn độ sáng: 0=tối/hồng ngoại | 1=mờ | 2=trung bình | 3=sáng"""

    # --- Phát hiện chế độ hồng ngoại (IR) ---
    # Khi bật IR, camera chuyển sang ảnh xám → saturation gần bằng 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_saturation = hsv[:, :, 1].mean()
    if mean_saturation < 20:   # ngưỡng: < 20/255 ≈ ảnh xám = đang bật IR
        return "0"             # coi như ban đêm / tối

    # --- Tính độ sáng bình thường (ban ngày) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    brightness = p50 * 0.5 + (p10 + p90) * 0.25

    if brightness < 45:
        return "0"   # tối (ban đêm, không đèn)
    elif brightness < 85:
        return "1"   # mờ (hoàng hôn, ánh đèn đường)
    elif brightness < 145:
        return "2"   # trung bình (trong nhà, trời흐림)
    else:
        return "3"   # sáng (ban ngày, đủ ánh sáng)

# ================= SINGLE RTSP WORKER =================
def single_rtsp_worker():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)


    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Không mở được RTSP")
        return

    last_sent = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if now - last_sent < 1.0 / DETECT_FPS:
            continue
        last_sent = now

        frame = cv2.resize(frame, RESIZE)

        # 🔥 fan-out cho 20 cam logic (DROP FRAME CŨ)
        for vid in range(TOTAL_VIDEO):
            q = frame_queues[vid]
            if q.full():
                q.get_nowait()
            q.put(frame)

# ================= YOLO WORKER (ROUND ROBIN) =================
def yolo_worker():
    cam_cursor = 0

    while True:
        frames = []
        ids = []

        while len(frames) < BATCH_SIZE:
            q = frame_queues[cam_cursor]
            if not q.empty():
                frame = q.get()
                frames.append(frame)
                ids.append(cam_cursor)

            cam_cursor = (cam_cursor + 1) % TOTAL_VIDEO
            time.sleep(0.001)

        start = time.time()
        with torch.no_grad():
            batch_results = model.predict(
                source=frames,
                device=0,
                imgsz=416,
                classes=[0],
                verbose=False,
                stream=False
            )

        end = time.time()

        for vid, res, frame in zip(ids, batch_results, frames):
            prev = last_time.get(vid)
            fps = 1.0 / max(end - prev, 1e-6) if prev else 0.0
            last_time[vid] = end
            brightness = get_brightness(frame)
            num_person = len(res.boxes) if res.boxes else 0

            now = time.time()
            if now - last_log_time.get(vid, 0) >= LOG_INTERVAL:
                last_log_time[vid] = now
                with state_lock:
                    camera_state[vid] = {
                        "timestamp": int(now),
                        "fps": round(fps, 2),
                        "people": num_person,
                        "is_night": brightness,
                    }

# ================= LOG WRITER =================
def log_writer_worker():
    header = "timestamp,cam_id,fps,people,is_night\n"
    while True:
        time.sleep(LOG_INTERVAL)
        with state_lock:
            lines = [header]
            for vid in range(TOTAL_VIDEO):
                s = camera_state[vid]
                lines.append(
                    f"{s['timestamp']},{vid},{s['fps']},"
                    f"{s['people']},{s['is_night']}\n"
                )
        with open(LOG_FILE, "w", buffering=1) as f:
            f.writelines(lines)

# ================= MQTT SENDER (AMQTT / asyncio) =================
async def _async_mqtt_sender():
    """Coroutine gửi dữ liệu camera qua AMQTT, tự reconnect nếu mất kết nối"""
    # reconnect_retries=0 → tắt auto-reconnect nội bộ của amqtt,
    # để vòng while bên ngoài tự xử lý reconnect sạch hơn
    _cfg = {"reconnect_retries": 0, "reconnect_max_interval": 5}
    while True:
        client = MQTTClient(client_id=CLIENT_ID, config=_cfg)
        try:
            await client.connect(MQTT_URI)
            print(f"✅ Đã kết nối MQTT broker (AMQTT): {MQTT_BROKER}")
        except Exception as e:
            print(f"⚠️  AMQTT chưa kết nối được: {e} — thử lại sau 5s")
            await asyncio.sleep(5)
            continue

        try:
            while True:
                await asyncio.sleep(API_SEND_INTERVAL)
                with state_lock:
                    payload = {
                        "cameras": [
                            {
                                "cam_id": str(vid),
                                "timestamp": camera_state[vid]["timestamp"],
                                "fps": camera_state[vid]["fps"],
                                "people": camera_state[vid]["people"],
                                "light_level": int(camera_state[vid]["is_night"])
                            }
                            for vid in range(TOTAL_VIDEO)
                        ]
                    }
                await client.publish(
                    MQTT_TOPIC,
                    json.dumps(payload).encode(),
                    qos=0x01
                )
        except Exception as e:
            print(f"❌ AMQTT send error: {e} — reconnect sau 3s")
            await asyncio.sleep(3)
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass


def mqtt_sender_worker():
    """Chạy async MQTT sender trong thread riêng với event loop của nó"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_async_mqtt_sender())
# ================= START =================
threading.Thread(target=single_rtsp_worker, daemon=True).start()
threading.Thread(target=yolo_worker, daemon=True).start()
threading.Thread(target=log_writer_worker, daemon=True).start()
threading.Thread(target=mqtt_sender_worker, daemon=True).start()

print("✅ Camera AI pipeline started (ROUND ROBIN MODE)")
print(f"📹 Tổng số camera ảo: {TOTAL_VIDEO}")
# ================= SIGNAL HANDLER =================
running = True
def signal_handler(sig, frame):
    global running
    print("\n🛑 Đang tắt chương trình...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

while running:
    time.sleep(1)

print("👋 Đã tắt chương trình an toàn.")
time.sleep(0.5)  # Cho threads TRT/CUDA có thời gian dừng
os._exit(0)      # Dùng os._exit để tránh lỗi FATAL khi TRT cleanup

