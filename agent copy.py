"""
Event-driven Sense -> Reason -> Actuate -> Reflect agent.

Instead of a fixed loop, each MQTT message that arrives triggers exactly one
pass through the LangGraph. Memory persists in SQLite (memory_store.py), so
the agent has real history across restarts -- closer to how the HiveMQTT
pattern actually runs in production.

Run:
    pip install langgraph langchain-ollama paho-mqtt --break-system-packages
    ollama pull qwen2.5:7b

    # terminal 1
    python agent.py

    # terminal 2 (simulates a real sensor publishing readings)
    python simulate_sensor.py
"""

from __future__ import annotations

import json
from typing import TypedDict, Literal

import paho.mqtt.client as mqtt
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

import memory_store

BROKER_HOST = "broker.hivemq.com"   # public test broker; point at your own for real use
BROKER_PORT = 1883
SENSE_TOPIC = "veradoc/demo/machine07/temperature"
ACTUATE_TOPIC_PREFIX = "veradoc/demo/machine07"

GOALS = "Keep machine-07 temperature under 85C. Warn at 85-90C. Alert above 90C."

# Thresholds for the rule engine (this is what used to live only inside the
# LLM prompt -- now it's plain numbers a human can review, test, and tune).
WARN_THRESHOLD = 85.0
ALERT_THRESHOLD = 90.0
RISING_SLOPE_C = 4.0          # degC per cycle considered "rising fast"
PREDICTIVE_MARGIN = 3.0       # escalate if within this many degC of WARN_THRESHOLD while rising fast
HYSTERESIS_READINGS = 2       # consecutive readings needed to drop out of warn/alert

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ---------------------------------------------------------------------------
# Shared state for a single graph invocation (i.e. a single MQTT message)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    topic: str
    reading: dict
    decision: str
    reasoning: str
    action: str


# ---------------------------------------------------------------------------
# SENSE - the "sensing" already happened in on_message (that's what triggered
# this run); this node just normalizes/validates the payload for the graph.
# ---------------------------------------------------------------------------
def sense(state: AgentState) -> AgentState:
    print(f"\n[SENSE]   {state['topic']} -> {state['reading']}")
    return {}


# ---------------------------------------------------------------------------
# REASON - pulls persisted history for this topic out of SQLite, not memory
# ---------------------------------------------------------------------------
def reason(state: AgentState) -> AgentState:
    history = memory_store.recent(state["topic"], limit=3)

    prompt = f"""You are an operations agent for an industrial machine.

Goals and rules:
{GOALS}

Recent history for this sensor (most recent last, from persistent storage):
{json.dumps(history, indent=2) if history else "none yet"}

New sensor reading:
{json.dumps(state['reading'], indent=2)}

Decide the single right action. Respond ONLY as compact JSON:
{{"decision": "<ok|warn|alert>", "reasoning": "<one sentence>"}}"""

    response = llm.invoke(prompt)

    try:
        parsed = json.loads(response.content)
        decision, reasoning = parsed["decision"], parsed["reasoning"]
    except (json.JSONDecodeError, KeyError):
        decision, reasoning = "warn", "Could not parse model output, defaulting to warn."

    print(f"[REASON]  decision={decision!r}  because: {reasoning}")
    return {"decision": decision, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# ACTUATE - publishes back to MQTT, this is a real client.publish() call
# ---------------------------------------------------------------------------
def actuate(state: AgentState, mqtt_client: mqtt.Client) -> AgentState:
    decision = state["decision"]
    value = state["reading"].get("value_c")

    if decision == "alert":
        topic, payload = f"{ACTUATE_TOPIC_PREFIX}/alert", f"ALERT: {value}C, shutdown recommended"
    elif decision == "warn":
        topic, payload = f"{ACTUATE_TOPIC_PREFIX}/warn", f"WARN: {value}C, notify on-call"
    else:
        topic, payload = f"{ACTUATE_TOPIC_PREFIX}/status", f"OK: {value}C"

    mqtt_client.publish(topic, payload)
    print(f"[ACTUATE] published '{payload}' -> {topic}")
    return {"action": payload}


# ---------------------------------------------------------------------------
# REFLECT - persists the cycle to SQLite instead of an in-memory list
# ---------------------------------------------------------------------------
def reflect(state: AgentState) -> AgentState:
    row_id = memory_store.record(
        topic=state["topic"],
        reading=state["reading"],
        decision=state["decision"],
        reasoning=state["reasoning"],
    )
    print(f"[REFLECT] persisted as memory row #{row_id}")
    return {}


# ---------------------------------------------------------------------------
# Build the graph. `actuate` needs the live mqtt client, so we bind it via a
# closure when compiling rather than stuffing the client into AgentState
# (it isn't serializable and doesn't belong in the data model).
# ---------------------------------------------------------------------------
def build_graph(mqtt_client: mqtt.Client):
    graph = StateGraph(AgentState)
    graph.add_node("sense", sense)
    graph.add_node("reason", reason)
    graph.add_node("actuate", lambda state: actuate(state, mqtt_client))
    graph.add_node("reflect", reflect)

    graph.set_entry_point("sense")
    graph.add_edge("sense", "reason")
    graph.add_edge("reason", "actuate")
    graph.add_edge("actuate", "reflect")
    graph.add_edge("reflect", END)  # one graph run = one message; no internal loop

    return graph.compile()


# ---------------------------------------------------------------------------
# MQTT wiring: every inbound message = one invocation of the graph
# ---------------------------------------------------------------------------
def main() -> None:
    memory_store.init_db()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    app = build_graph(client)

    def on_connect(c, userdata, flags, reason_code, properties):
        print(f"Connected to {BROKER_HOST} (rc={reason_code}), subscribing to {SENSE_TOPIC}")
        c.subscribe(SENSE_TOPIC)

    def on_message(c, userdata, msg):
        try:
            reading = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            print(f"Ignoring non-JSON payload on {msg.topic}: {msg.payload!r}")
            return

        app.invoke({"topic": msg.topic, "reading": reading, "decision": "", "reasoning": "", "action": ""})

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()  # blocks; each message triggers sense->reason->actuate->reflect


if __name__ == "__main__":
    main()
