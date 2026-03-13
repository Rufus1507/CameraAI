import os
import glob as _glob
import ctypes
import logging

# ── Preload libcusparseLt.so.0 for Jetson Orin (BEFORE importing torch) ──────
_matches = _glob.glob(os.path.expanduser(
    "~/.local/lib/python*/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"
))
if _matches:
    ctypes.CDLL(_matches[0])

os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

# ── Tắt spam H264 decode error từ FFmpeg / OpenCV ────────────────────────────
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
logging.getLogger("libav").setLevel(logging.CRITICAL)

import torch
from ultralytics import YOLO
import cv2
import time
import threading
from queue import Queue, Empty
import signal
import requests
import numpy as np
import asyncio
from amqtt.client import MQTTClient
import json
import sqlite3
from collections import deque

# ── Suppress verbose FFmpeg output at C level ─────────────────────────────────
try:
    import ctypes as _ct
    _libav = _ct.cdll.LoadLibrary("libavcodec.so")
    _libav.av_log_set_level(16)   # AV_LOG_ERROR=16
except Exception:
    pass

# =============================================================================
# DATABASE
# =============================================================================
DB_PATH = "cameras.db"

def load_cameras():
    """Đọc danh sách camera đang bật từ DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            SELECT cam_id, rtsp_url
            FROM cameras
            WHERE enabled = 1
            ORDER BY cam_id
        """)
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"⚠️  load_cameras error: {e}")
        return []
    return [{"id": cam_id, "rtsp": rtsp_url} for cam_id, rtsp_url in rows]

# =============================================================================
# GPU / MODEL
# =============================================================================
DEVICE     = 0
MODEL_PATH = "yolov8n.engine"
model      = YOLO(MODEL_PATH, task="detect")
torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIG
# =============================================================================
# TRT engine: batch=1 static, imgsz=416 static
# Throughput thực: ~14ms/inference → ~71 fps tổng
# Với 26 cams: 71/26 ≈ 2.7 fps/cam tối đa
# → DETECT_FPS đặt 3 để phù hợp với throughput thực, tránh queue overflow
DETECT_FPS    = 3           # fps gửi frame vào queue mỗi cam (≤ GPU throughput/n_cams)
RESIZE        = (416, 416)  # phải khớp imgsz lúc export TRT engine (KHÔNG được đổi)
BATCH_SIZE    = 1           # TRT engine export với batch=1 static shape
QUEUE_PER_CAM = 3           # buffer nhỏ để tránh dùng frame cũ quá lâu
CONF_THRESH   = 0.25        # ngưỡng confidence tối thiểu
PERSON_CLASS  = 0           # COCO class index cho "person"

# Rolling window để đo FPS chính xác
FPS_WINDOW    = 3.0         # giây: cửa sổ tính FPS (≥ 1/DETECT_FPS * n_cams)

# People count persistence: giữ max của N kết quả gần nhất để tránh spike về 0
PEOPLE_HISTORY = 3          # số frame gần nhất để lấy max

RTSP_RETRY_DELAY = 10       # giây chờ trước khi reconnect

LOG_INTERVAL = 1.0          # ghi CSV mỗi 1 giây
LOG_FILE     = "camera_stats.csv"

DB_POLL_INTERVAL = 10       # giây check cameras.db có thay đổi không

# =============================================================================
# MQTT CONFIG
# =============================================================================
MQTT_BROKER       = "100.82.253.83"
MQTT_PORT         = 1883
MQTT_TOPIC        = "camera/stats"
CLIENT_ID         = "machine_a_camera_ai"
API_SEND_INTERVAL = 1.0
MQTT_URI          = f"mqtt://{MQTT_BROKER}:{MQTT_PORT}/"

# =============================================================================
# SHARED STATE  (tất cả đều protected bởi state_lock)
# =============================================================================
state_lock   = threading.Lock()

CAMERAS      = load_cameras()
CAM_IDS      = [c["id"] for c in CAMERAS]
TOTAL_VIDEO  = len(CAM_IDS)

init_time    = int(time.time())
camera_state = {
    cid: {"timestamp": init_time, "fps": 0.0, "people": 0, "is_night": "0"}
    for cid in CAM_IDS
}
frame_queues = {cid: Queue(maxsize=QUEUE_PER_CAM) for cid in CAM_IDS}

# dict để track thread sống: cam_id → threading.Event (stop-signal)
cam_stop_events: dict[int, threading.Event] = {}

# Rolling window FPS
detect_timestamps: dict[int, list] = {cid: [] for cid in CAM_IDS}

# People count history: deque của PEOPLE_HISTORY kết quả gần nhất
# → dùng max() để tránh spike về 0 do 1 frame bị miss detect
people_history: dict[int, deque] = {
    cid: deque([0] * PEOPLE_HISTORY, maxlen=PEOPLE_HISTORY) for cid in CAM_IDS
}

# Thời điểm detect gần nhất của mỗi cam (để biết cam có đang active không)
last_detect_time: dict[int, float] = {}

STALE_TIMEOUT = 5.0   # giây không detect → coi fps=0

# =============================================================================
# INIT LOG FILE
# =============================================================================
with open(LOG_FILE, "w") as f:
    f.write("timestamp,cam_id,fps,people,is_night\n")

# =============================================================================
# DAY / NIGHT
# =============================================================================
def get_brightness(frame) -> str:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hsv[:, :, 1].mean() < 20:
        return "0"   # IR / tối

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    b = p50 * 0.5 + (p10 + p90) * 0.25

    if   b < 45:  return "0"
    elif b < 85:  return "1"
    elif b < 145: return "2"
    else:         return "3"

# =============================================================================
# RTSP WORKER  (1 thread / camera)
# =============================================================================
def rtsp_worker(cam: dict, stop_event: threading.Event):
    """
    Đọc RTSP stream, gửi frame vào queue đúng nhịp DETECT_FPS.
    Logic đơn giản: đọc liên tục, khi đến thời điểm gửi → resize + put queue.
    """
    cam_id        = cam["id"]
    url           = cam["rtsp"]
    send_interval = 1.0 / DETECT_FPS

    while not stop_event.is_set():
        # ── Mở RTSP ──────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"❌ Cam {cam_id}: không kết nối — thử lại sau {RTSP_RETRY_DELAY}s")
            with state_lock:
                if cam_id in camera_state:
                    camera_state[cam_id]["fps"] = -1.0
            cap.release()
            stop_event.wait(timeout=RTSP_RETRY_DELAY)
            continue

        print(f"✅ Cam {cam_id}: kết nối RTSP thành công")
        fail_count = 0
        last_sent  = 0.0   # perf_counter của lần gửi gần nhất

        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                fail_count += 1
                if fail_count >= 5:
                    print(f"⚠️  Cam {cam_id}: mất kết nối — reconnect sau {RTSP_RETRY_DELAY}s")
                    with state_lock:
                        if cam_id in camera_state:
                            camera_state[cam_id]["fps"] = -1.0
                    cap.release()
                    stop_event.wait(timeout=RTSP_RETRY_DELAY)
                    break
                time.sleep(0.01)
                continue

            fail_count = 0
            now = time.perf_counter()

            # Rate-limit: chỉ gửi frame khi đã qua send_interval
            if now - last_sent < send_interval:
                continue

            last_sent = now

            # Gửi frame vào queue
            frame_resized = cv2.resize(frame, RESIZE)
            q: Queue = frame_queues.get(cam_id)
            if q is not None:
                if q.full():
                    try: q.get_nowait()   # bỏ frame cũ nhất để luôn có frame mới
                    except Empty: pass
                q.put_nowait(frame_resized)

        cap.release()

    print(f"🔴 Cam {cam_id}: thread dừng")

# =============================================================================
# YOLO WORKER  — strict fair round-robin, batch=1
# =============================================================================
def yolo_worker():
    """
    Round-robin công bằng qua tất cả cameras, mỗi lần lấy 1 frame từ 1 cam.
    TRT engine batch=1 static → mỗi inference xử lý đúng 1 frame.
    """
    start_idx = 0

    while True:
        with state_lock:
            cur_ids = list(CAM_IDS)
        n = len(cur_ids)
        if n == 0:
            time.sleep(0.1)
            continue

        # Tìm cam kế tiếp có frame trong queue
        found = False
        for i in range(n):
            cam_id = cur_ids[(start_idx + i) % n]
            q: Queue = frame_queues.get(cam_id)
            if q is None or q.empty():
                continue
            try:
                frame = q.get_nowait()
            except Empty:
                continue

            # Tiến start_idx đến cam tiếp theo
            start_idx = (start_idx + i + 1) % n
            found = True

            # ── GPU Inference ─────────────────────────────────────────────────
            # imgsz=416 BẮT BUỘC khớp với TRT engine (static shape 1×3×416×416)
            # Không truyền classes= → tự filter thủ công để tránh TRT conflict
            with torch.no_grad():
                results = model.predict(
                    source=[frame],
                    device=DEVICE,
                    imgsz=416,
                    conf=CONF_THRESH,
                    verbose=False,
                    stream=False,
                )

            res = results[0]
            now = time.time()

            with state_lock:
                # Rolling window FPS
                ts = detect_timestamps.setdefault(cam_id, [])
                ts.append(now)
                cutoff = now - FPS_WINDOW
                detect_timestamps[cam_id] = [t for t in ts if t > cutoff]
                fps_val = round(len(detect_timestamps[cam_id]) / FPS_WINDOW, 2)

                # Đếm người: filter class=0 thủ công
                raw_person = 0
                if res.boxes is not None and len(res.boxes) > 0:
                    raw_person = int((res.boxes.cls == PERSON_CLASS).sum().item())

                # Persistence: lấy max của N frame gần nhất
                # → tránh báo 0 người chỉ vì 1 frame bị detect sai
                hist = people_history.setdefault(cam_id, deque([0] * PEOPLE_HISTORY, maxlen=PEOPLE_HISTORY))
                hist.append(raw_person)
                stable_person = max(hist)   # lấy giá trị cao nhất trong window

                last_detect_time[cam_id] = now
                brightness = get_brightness(frame)

                if cam_id in camera_state:
                    camera_state[cam_id] = {
                        "timestamp": int(now),
                        "fps":       fps_val,
                        "people":    stable_person,
                        "is_night":  brightness,
                    }
            break   # xử lý xong 1 frame → vòng lặp tiếp theo

        if not found:
            time.sleep(0.002)   # tất cả queue rỗng → short idle

# =============================================================================
# LOG WRITER
# =============================================================================
def log_writer_worker():
    while True:
        time.sleep(LOG_INTERVAL)
        now    = time.time()
        cutoff = now - FPS_WINDOW
        with state_lock:
            cur_ids  = list(CAM_IDS)
            snapshot = {cid: dict(camera_state[cid]) for cid in cur_ids if cid in camera_state}
            fps_snap = {
                cid: round(
                    len([t for t in detect_timestamps.get(cid, []) if t > cutoff]) / FPS_WINDOW, 2
                )
                for cid in cur_ids
            }
            last_det_snap = {cid: last_detect_time.get(cid) for cid in cur_ids}

        lines = ["timestamp,cam_id,fps,people,is_night\n"]
        for cid in sorted(cur_ids):
            s    = snapshot.get(cid)
            last = last_det_snap.get(cid)
            if not s:
                continue
            # Nếu camera không được detect quá STALE_TIMEOUT → fps=0
            stale   = (last is None) or (now - last > STALE_TIMEOUT)
            fps_out = 0.0 if stale else fps_snap.get(cid, 0.0)
            lines.append(
                f"{s['timestamp']},{cid},{fps_out},{s['people']},{s['is_night']}\n"
            )

        try:
            with open(LOG_FILE, "w", buffering=1) as f:
                f.writelines(lines)
        except Exception as e:
            print(f"❌ log_writer error: {e}")

# =============================================================================
# MQTT SENDER
# =============================================================================
async def _async_mqtt_sender():
    _cfg = {"reconnect_retries": 0, "reconnect_max_interval": 5}
    while True:
        client = MQTTClient(client_id=CLIENT_ID, config=_cfg)
        try:
            await client.connect(MQTT_URI)
            print(f"✅ Đã kết nối MQTT broker: {MQTT_BROKER}")
        except Exception as e:
            print(f"⚠️  MQTT chưa kết nối: {e} — thử lại sau 5s")
            await asyncio.sleep(5)
            continue

        try:
            while True:
                await asyncio.sleep(API_SEND_INTERVAL)
                now2    = time.time()
                cutoff2 = now2 - FPS_WINDOW
                with state_lock:
                    cur_ids = list(CAM_IDS)
                    payload = {
                        "cameras": [
                            {
                                "cam_id":      str(cid),
                                "timestamp":   camera_state[cid]["timestamp"],
                                "fps":         round(
                                    len([t for t in detect_timestamps.get(cid, []) if t > cutoff2]) / FPS_WINDOW, 2
                                ),
                                "people":      camera_state[cid]["people"],
                                "light_level": int(camera_state[cid]["is_night"]),
                            }
                            for cid in cur_ids if cid in camera_state
                        ]
                    }
                await client.publish(MQTT_TOPIC, json.dumps(payload).encode(), qos=0x01)
                print(f"📤 Đã gửi MQTT: {len(payload['cameras'])} cameras tới {MQTT_TOPIC}")
        except Exception as e:
            print(f"❌ MQTT send error: {e} — reconnect sau 3s")
            await asyncio.sleep(3)
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

def mqtt_sender_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_async_mqtt_sender())

# =============================================================================
# DB WATCHER  — hot-reload cameras.db mỗi DB_POLL_INTERVAL giây
# =============================================================================
def db_watcher_worker():
    global CAM_IDS, TOTAL_VIDEO

    while True:
        time.sleep(DB_POLL_INTERVAL)

        new_cameras = load_cameras()
        new_ids     = {c["id"] for c in new_cameras}
        new_cam_map = {c["id"]: c for c in new_cameras}

        with state_lock:
            old_ids = set(CAM_IDS)

        added   = new_ids - old_ids
        removed = old_ids - new_ids

        for cid in removed:
            evt = cam_stop_events.pop(cid, None)
            if evt:
                evt.set()
            with state_lock:
                camera_state.pop(cid, None)
                frame_queues.pop(cid, None)
                detect_timestamps.pop(cid, None)
                people_history.pop(cid, None)
                last_detect_time.pop(cid, None)
            print(f"🔴 DB watcher: cam {cid} bị xóa/disable")

        for cid in added:
            cam = new_cam_map[cid]
            with state_lock:
                frame_queues[cid]      = Queue(maxsize=QUEUE_PER_CAM)
                detect_timestamps[cid] = []
                people_history[cid]    = deque([0] * PEOPLE_HISTORY, maxlen=PEOPLE_HISTORY)
                camera_state[cid]      = {
                    "timestamp": int(time.time()),
                    "fps": 0.0, "people": 0, "is_night": "0"
                }
            stop_evt = threading.Event()
            cam_stop_events[cid] = stop_evt
            threading.Thread(
                target=rtsp_worker, args=(cam, stop_evt), daemon=True
            ).start()
            print(f"🟢 DB watcher: cam {cid} mới → khởi thread")

        if added or removed:
            with state_lock:
                CAM_IDS     = sorted(new_ids)
                TOTAL_VIDEO = len(CAM_IDS)
            print(f"📋 DB watcher: tổng {TOTAL_VIDEO} cameras đang chạy")

# =============================================================================
# KHỞI ĐỘNG
# =============================================================================
for cam in CAMERAS:
    evt = threading.Event()
    cam_stop_events[cam["id"]] = evt
    threading.Thread(target=rtsp_worker, args=(cam, evt), daemon=True).start()

threading.Thread(target=yolo_worker,        daemon=True).start()
threading.Thread(target=log_writer_worker,  daemon=True).start()
threading.Thread(target=mqtt_sender_worker, daemon=True).start()
threading.Thread(target=db_watcher_worker,  daemon=True).start()

print("✅ Camera AI pipeline started")
print(f"📹 Tổng số camera: {TOTAL_VIDEO}  |  Batch: {BATCH_SIZE}  |  Detect FPS/cam: {DETECT_FPS}")
print(f"🔄 DB hot-reload mỗi {DB_POLL_INTERVAL}s  |  Log: {LOG_INTERVAL}s  |  People history: {PEOPLE_HISTORY} frames")

# =============================================================================
# SIGNAL HANDLER
# =============================================================================
running = True

def signal_handler(sig, frame):
    global running
    print("\n🛑 Đang tắt chương trình...")
    running = False

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

while running:
    time.sleep(1)

print("👋 Đã tắt chương trình an toàn.")
time.sleep(0.5)
os._exit(0)
