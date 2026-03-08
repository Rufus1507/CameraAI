import os
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

# ================= API CONFIG =================
API_URL = "http://192.168.58.3:8000/api/camera_stats"
API_SEND_INTERVAL = 0.5  # Gửi API mỗi 0.5 giây



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

# ================= DAY / NIGHT (0 = tối, 1 = sáng) =================
def get_brightness(frame):
    """Trả về độ sáng 0 → 1 (0 = tối nhất, 1 = sáng nhất)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean() / 255.0  # Chuẩn hóa về 0-1
    return round(brightness, 2)

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

# ================= API SENDER =================
def api_sender_worker():
    """Gửi dữ liệu camera đến FastAPI server"""
    while True:
        time.sleep(API_SEND_INTERVAL)
        try:
            with state_lock:
                # Tạo payload từ camera_state
                payload = {
                    "cameras": [
                        {
                            "cam_id": vid,
                            "timestamp": camera_state[vid]["timestamp"],
                            "fps": camera_state[vid]["fps"],
                            "people": camera_state[vid]["people"],
                            "brightness": camera_state[vid]["is_night"],
                        }
                        for vid in range(TOTAL_VIDEO)
                    ]
                }
            
            # Gửi POST request đến API
            response = requests.post(
                API_URL,
                json=payload,
                timeout=2  # Timeout 2 giây
            )
            
            if response.status_code == 200:
                pass  # Thành công, không log để tránh spam
            else:
                print(f"⚠️ API response: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Không kết nối được API server")
        except requests.exceptions.Timeout:
            print("⏱️ API timeout")
        except Exception as e:
            print(f"❌ API error: {e}")

# ================= START =================
threading.Thread(target=single_rtsp_worker, daemon=True).start()
threading.Thread(target=yolo_worker, daemon=True).start()
threading.Thread(target=log_writer_worker, daemon=True).start()
threading.Thread(target=api_sender_worker, daemon=True).start()

print("✅ Camera AI pipeline started (ROUND ROBIN MODE)")
print(f"📹 Tổng số camera ảo: {TOTAL_VIDEO}")
print(f"📡 API endpoint: {API_URL}")

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
sys.exit(0)
