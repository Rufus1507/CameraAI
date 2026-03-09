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
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"   # chặn warn-level từ OpenCV
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

# ── Suppress verbose FFmpeg output at C level ─────────────────────────────────
try:
    import ctypes as _ct
    _libav = _ct.cdll.LoadLibrary("libavcodec.so")
    # AV_LOG_ERROR = 16, AV_LOG_QUIET = -8
    _libav.av_log_set_level(16)
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
DETECT_FPS  = 8           # frame gửi vào queue mỗi cam (giảm xuống để realtime hơn)
RESIZE      = (416, 416)
BATCH_SIZE  = 1           # TRT engine export với batch=1 (static shape)
QUEUE_PER_CAM = 2

RTSP_RETRY_DELAY = 10     # giây chờ trước khi reconnect

LOG_INTERVAL = 1.0        # ghi CSV mỗi 1 giây
LOG_FILE     = "camera_stats.csv"

DB_POLL_INTERVAL = 10     # giây check cameras.db có thay đổi không

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

last_detect_time: dict[int, float] = {}   # dùng để tính FPS

# FPS smoothing: Exponential Moving Average
FPS_EMA_ALPHA  = 0.3    # trọng số mẫu mới (0=bỏ qua hết, 1=tức thời)
STALE_TIMEOUT  = 5.0   # giây: nếu camera không được detect → báo fps=0 trong log

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
# RTSP WORKER  (1 thread / camera, có stop-event để DB watcher có thể kill)
# =============================================================================
def rtsp_worker(cam: dict, stop_event: threading.Event):
    cam_id = cam["id"]
    url    = cam["rtsp"]

    # Interval giữa 2 frame gửi vào queue
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
        last_sent = 0.0
        fail_count = 0

        # ── Đọc frame ────────────────────────────────────────────────────────
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
                time.sleep(0.02)
                continue

            fail_count = 0
            now = time.time()
            if now - last_sent < send_interval:
                continue

            last_sent = now
            frame = cv2.resize(frame, RESIZE)

            q: Queue = frame_queues.get(cam_id)
            if q is not None:
                if q.full():
                    try: q.get_nowait()
                    except Empty: pass
                q.put_nowait(frame)

        cap.release()

    print(f"🔴 Cam {cam_id}: thread dừng")

# =============================================================================
# YOLO WORKER  — strict fair round-robin scheduling
# =============================================================================
def yolo_worker():
    """
    Mỗi lần gọi GPU inference = 1 lần quét hoàn chỉnh qua TẤT CẢ cameras
    bắt đầu từ vị trí xoay vòng (start_idx).

    Đảm bảo:
    - Mỗi camera đóng góp TỐI ĐA 1 frame / batch.
    - start_idx xoay sau mỗi batch → camera nào bị bỏ lần này sẽ được ưu tiên lần sau.
    - Nếu tổng frame < BATCH_SIZE, chạy batch nhỏ hơn thay vì chờ hoặc thiên vị.
    """
    start_idx = 0   # xoay sau mỗi batch để cân bằng camera đứng đầu hàng

    while True:
        # ── Snapshot danh sách camera hiện tại ───────────────────────────────
        with state_lock:
            cur_ids = list(CAM_IDS)
        n = len(cur_ids)
        if n == 0:
            time.sleep(0.1)
            continue

        frames: list[np.ndarray] = []
        ids:    list[int]        = []
        last_i  = 0   # vị trí dừng để start_idx kế tiếp tiếp nối đúng chỗ

        # ── 1 lần quét hoàn chỉnh qua tất cả n cameras ───────────────────────
        for i in range(n):
            cam_id = cur_ids[(start_idx + i) % n]
            q: Queue = frame_queues.get(cam_id)
            if q is not None and not q.empty():
                try:
                    frame = q.get_nowait()
                    frames.append(frame)
                    ids.append(cam_id)
                    last_i = i
                    if len(frames) >= BATCH_SIZE:
                        # batch đầy → lần sau bắt đầu từ camera TIẾP THEO
                        start_idx = (start_idx + last_i + 1) % n
                        break
                except Empty:
                    pass
        else:
            # Đã quét hết vòng → lần sau bắt đầu lại từ đầu (xoay 1 bước)
            start_idx = (start_idx + 1) % n

        # ── Không camera nào có frame → nghỉ ngắn ────────────────────────────
        if not frames:
            time.sleep(0.005)
            continue

        # ── GPU Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            batch_results = model.predict(
                source=frames,
                device=DEVICE,
                imgsz=416,
                classes=[0],
                verbose=False,
                stream=False,
            )

        # ── Cập nhật state cho từng camera trong batch ────────────────────────
        now = time.time()
        with state_lock:
            for vid, res, frame in zip(ids, batch_results, frames):
                prev = last_detect_time.get(vid)
                last_detect_time[vid] = now

                # FPS tức thời
                instant_fps = 1.0 / max(now - prev, 1e-6) if prev else 0.0

                # EMA smoothing: tránh spike do jitter mạng
                old_fps = camera_state[vid]["fps"] if vid in camera_state else 0.0
                if old_fps <= 0:
                    smoothed_fps = round(instant_fps, 2)          # lần đầu: lấy thẳng
                else:
                    smoothed_fps = round(
                        FPS_EMA_ALPHA * instant_fps + (1 - FPS_EMA_ALPHA) * old_fps, 2
                    )

                num_person = len(res.boxes) if res.boxes else 0
                brightness = get_brightness(frame)

                if vid in camera_state:
                    camera_state[vid] = {
                        "timestamp": int(now),
                        "fps":       smoothed_fps,
                        "people":    num_person,
                        "is_night":  brightness,
                    }

# =============================================================================
# LOG WRITER
# =============================================================================
def log_writer_worker():
    while True:
        time.sleep(LOG_INTERVAL)
        now = time.time()
        with state_lock:
            cur_ids = list(CAM_IDS)
            snapshot      = {cid: dict(camera_state[cid])  for cid in cur_ids if cid in camera_state}
            last_det_snap = {cid: last_detect_time.get(cid) for cid in cur_ids}

        lines = ["timestamp,cam_id,fps,people,is_night\n"]
        for cid in sorted(cur_ids):
            s    = snapshot.get(cid)
            last = last_det_snap.get(cid)
            if not s:
                continue

            # Nếu camera chưa được detect hoặc quá STALE_TIMEOUT → fps=0
            stale = (last is None) or (now - last > STALE_TIMEOUT)
            fps_out = 0.0 if stale else s["fps"]

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
                with state_lock:
                    cur_ids = list(CAM_IDS)
                    payload = {
                        "cameras": [
                            {
                                "cam_id":      str(cid),
                                "timestamp":   camera_state[cid]["timestamp"],
                                "fps":         camera_state[cid]["fps"],
                                "people":      camera_state[cid]["people"],
                                "light_level": int(camera_state[cid]["is_night"]),
                            }
                            for cid in cur_ids if cid in camera_state
                        ]
                    }
                await client.publish(MQTT_TOPIC, json.dumps(payload).encode(), qos=0x01)
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
    """
    Mỗi DB_POLL_INTERVAL giây đọc lại cameras.db.
    - Camera mới → tạo queue, state, khởi thread rtsp_worker mới
    - Camera bị xóa/disable → gửi stop-event, dọn queue & state
    """
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

        # ── Xử lý camera bị remove ────────────────────────────────────────────
        for cid in removed:
            evt = cam_stop_events.pop(cid, None)
            if evt:
                evt.set()   # báo thread dừng
            with state_lock:
                camera_state.pop(cid, None)
                frame_queues.pop(cid, None)
            print(f"🔴 DB watcher: cam {cid} bị xóa/disable")

        # ── Xử lý camera mới ─────────────────────────────────────────────────
        for cid in added:
            cam = new_cam_map[cid]
            with state_lock:
                frame_queues[cid]  = Queue(maxsize=QUEUE_PER_CAM)
                camera_state[cid]  = {
                    "timestamp": int(time.time()),
                    "fps": 0.0, "people": 0, "is_night": "0"
                }
            stop_evt = threading.Event()
            cam_stop_events[cid] = stop_evt
            threading.Thread(
                target=rtsp_worker, args=(cam, stop_evt), daemon=True
            ).start()
            print(f"🟢 DB watcher: cam {cid} mới → khởi thread")

        # ── Cập nhật CAM_IDS & TOTAL_VIDEO ───────────────────────────────────
        if added or removed:
            with state_lock:
                CAM_IDS     = sorted(new_ids - removed | new_ids & old_ids)
                TOTAL_VIDEO = len(CAM_IDS)
            print(f"📋 DB watcher: tổng {TOTAL_VIDEO} cameras đang chạy")

# =============================================================================
# KHỞI ĐỘNG
# =============================================================================
# Stop-event cho các camera ban đầu
for cam in CAMERAS:
    evt = threading.Event()
    cam_stop_events[cam["id"]] = evt
    threading.Thread(target=rtsp_worker, args=(cam, evt), daemon=True).start()

threading.Thread(target=yolo_worker,       daemon=True).start()
threading.Thread(target=log_writer_worker, daemon=True).start()
threading.Thread(target=mqtt_sender_worker,daemon=True).start()
threading.Thread(target=db_watcher_worker, daemon=True).start()

print("✅ Camera AI pipeline started")
print(f"📹 Tổng số camera: {TOTAL_VIDEO}  |  Batch size: {BATCH_SIZE}  |  Detect FPS/cam: {DETECT_FPS}")
print(f"🔄 DB hot-reload mỗi {DB_POLL_INTERVAL}s  |  Log interval: {LOG_INTERVAL}s")

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
