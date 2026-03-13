import asyncio
from amqtt.broker import Broker

BROKER_CONFIG = {
    "listeners": {
        "default": {
            "type": "tcp",
            "bind": "0.0.0.0:1883",
        }
    },
    "sys_interval": 10,
    "auth": {
        "allow-anonymous": True,
        "plugins": ["auth_anonymous"],
    },
    "topic-check": {
        "enabled": False,
    },
}

async def main():
    broker = Broker(BROKER_CONFIG)
    await broker.start()
    print("🚀 AMQTT Broker đã khởi động — lắng nghe tại 0.0.0.0:1883")
    print("⏳ Đang chờ kết nối... (Ctrl+C để dừng)")
    try:
        await asyncio.get_event_loop().create_future()  # chạy mãi mãi
    except asyncio.CancelledError:
        pass
    finally:
        await broker.shutdown()
        print("🛑 Broker đã dừng.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng broker.")
