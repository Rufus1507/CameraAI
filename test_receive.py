import asyncio
from amqtt.client import MQTTClient
import json
import time
import sqlite3
import os
from datetime import datetime

# ================= CONFIG =================
BROKER_HOST = "localhost"       # broker đang chạy tại broker.py trên cùng máy
BROKER_PORT = 1883
BROKER_URL  = f"mqtt://{BROKER_HOST}:{BROKER_PORT}/"
CLIENT_ID   = "machine_b_receiver"

CAM_DB_PATH           = os.path.join(os.path.dirname(__file__), "cameras.db")
TOPIC_RELOAD_INTERVAL = 10   # giây reload topic mới từ DB

PEOPLE_THRESHOLD = 5
LIGHT_MIN        = 1

LIGHT_LABEL = {0: "dark", 1: "dim", 2: "medium", 3: "bright"}

# ================= ĐỌC TOPICS TỪ DB =================
def load_camera_topics() -> list[str]:
    try:
        conn = sqlite3.connect(CAM_DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT mqtt_topic FROM cameras WHERE status = 'online'")
        rows = cur.fetchall()
        conn.close()
        return [row[0] for row in rows if row[0]]
    except Exception as e:
        print(f"⚠️  Không đọc được cameras.db: {e}")
        return []

# ================= XỬ LÝ MESSAGE =================
async def handle_message(client, topic: str, payload_str: str):
    try:
        data        = json.loads(payload_str)
        now         = datetime.now().strftime("%H:%M:%S")
        people      = int(data.get("people", 0))
        light_level = int(data.get("light_level", 2))

        print(
            f"[{now}] ✅ Nhận | topic={topic} | "
            f"people={people} | light={LIGHT_LABEL.get(light_level, str(light_level))}"
        )

        if people > 0 and light_level <= LIGHT_MIN:
            alert = json.dumps({"topic": topic, "action": "TURN_ON", "people": people, "time": now}).encode()
            await client.publish("alert/lighting", alert, qos=1)
            print(f"  💡 Bật đèn — {topic}")

        if people >= PEOPLE_THRESHOLD:
            alert = json.dumps({"topic": topic, "action": "CROWD_ALERT", "people": people, "time": now}).encode()
            await client.publish("alert/crowd", alert, qos=1)
            print(f"  🚨 Đông người tại {topic} ({people} người)")

    except Exception as e:
        print(f"❌ Lỗi xử lý [{topic}]: {e}")

# ================= RELOAD TOPICS NỀN =================
async def _reload_worker(client, subscribed: set):
    while True:
        await asyncio.sleep(TOPIC_RELOAD_INTERVAL)
        topics = load_camera_topics()
        new_topics = set(topics) - subscribed
        if new_topics:
            await client.subscribe([(t, 1) for t in new_topics])
            subscribed.update(new_topics)
            print(f"📡 Đã subscribe thêm {len(new_topics)} topic mới: {new_topics}")

# ================= MAIN LOOP =================
async def main():
    client = MQTTClient(client_id=CLIENT_ID, config={"reconnect_retries": -1, "reconnect_max_interval": 5})

    while True:
        reload_task = None
        try:
            await client.connect(BROKER_URL)
            print(f"✅ Đã kết nối broker: {BROKER_URL}")

            # Subscribe topics từ DB
            topics = load_camera_topics()
            subscribed: set = set()
            if topics:
                await client.subscribe([(t, 1) for t in topics])
                subscribed = set(topics)
                print(f"📡 Đang lắng nghe {len(topics)} topics:")
                for t in topics:
                    print(f"   • {t}")
            else:
                print("⚠️  Không có topic nào trong DB — subscribe wildcard '#'")
                await client.subscribe([("#", 1)])
                subscribed.add("#")

            # Task reload topic nền
            reload_task = asyncio.create_task(_reload_worker(client, subscribed))

            # Nhận message
            while True:
                message     = await client.deliver_message()
                packet      = message.publish_packet
                topic       = packet.variable_header.topic_name
                payload_str = packet.payload.data.decode("utf-8", errors="replace")
                await handle_message(client, topic, payload_str)

        except asyncio.CancelledError:
            print("🛑 Dừng receiver.")
            return
        except Exception as e:
            print(f"⚠️  Lỗi kết nối: {e} — thử lại sau 3s...")
            await asyncio.sleep(3)
        finally:
            if reload_task and not reload_task.done():
                reload_task.cancel()
                try:
                    await reload_task
                except asyncio.CancelledError:
                    pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng receiver.")
