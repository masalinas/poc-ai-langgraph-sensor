"""
Simulates machine-07's temperature sensor publishing to the same broker/topic
that agent.py subscribes to. Run this alongside agent.py to see the full
event-driven loop react to real MQTT messages.
"""

import json
import random
import time
from datetime import datetime

import paho.mqtt.client as mqtt

BROKER_HOST = "broker.hivemq.com"
BROKER_PORT = 1883
TOPIC = "veradoc/demo/machine07/temperature"


def main() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    try:
        while True:
            reading = {
                "sensor": TOPIC,
                "value_c": round(random.uniform(60, 95), 1),
                "timestamp": datetime.utcnow().isoformat(),
            }
            client.publish(TOPIC, json.dumps(reading))
            print(f"published -> {reading}")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
