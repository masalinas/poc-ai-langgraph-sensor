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
def compute_trend(history: list[dict], current_value: float) -> dict:
    """Turns the last 3 stored readings + the current one into a simple,
    auditable trend signal: slope, direction, and consecutive-above counts."""
    values = [h["reading"]["value_c"] for h in history] + [current_value]

    if len(values) >= 2:
        slope = (values[-1] - values[0]) / (len(values) - 1)
    else:
        slope = 0.0

    direction = "rising" if slope > 0.5 else "falling" if slope < -0.5 else "stable"

    consecutive_above_warn = 0
    for v in reversed(values):
        if v >= WARN_THRESHOLD:
            consecutive_above_warn += 1
        else:
            break

    consecutive_below_warn = 0
    for v in reversed(values):
        if v < WARN_THRESHOLD:
            consecutive_below_warn += 1
        else:
            break

    return {
        "slope": round(slope, 2),
        "direction": direction,
        "consecutive_above_warn": consecutive_above_warn,
        "consecutive_below_warn": consecutive_below_warn,
    }


def rule_based_decision(value: float, trend: dict, last_decision: str | None) -> tuple[str, str, bool]:
    """Deterministic decision layer. Returns (decision, reasoning, ambiguous).
    `ambiguous=True` means the rules don't confidently cover this case and
    it should be escalated to the LLM instead of guessed at in code."""

    # Clear-cut cases first -- no LLM needed for any of these.
    if value >= ALERT_THRESHOLD:
        return "alert", f"{value}C >= alert threshold {ALERT_THRESHOLD}C.", False

    if value >= WARN_THRESHOLD:
        # Hysteresis: don't drop straight back to ok after briefly dipping
        # below warn -- require a couple of consecutive readings below it.
        if last_decision == "alert" and trend["consecutive_below_warn"] < HYSTERESIS_READINGS:
            return "alert", "Recently alerted; holding alert until temperature stabilizes below warn threshold.", False
        return "warn", f"{value}C is within warn band ({WARN_THRESHOLD}-{ALERT_THRESHOLD}C).", False

    # Below warn threshold, but rising fast and close to it: genuinely
    # ambiguous -- a fixed rule here would either over-alert on transients
    # or under-alert on real trends. Good candidate for the LLM.
    if trend["direction"] == "rising" and trend["slope"] >= RISING_SLOPE_C and (WARN_THRESHOLD - value) <= PREDICTIVE_MARGIN:
        return "warn", "Provisional: rising fast near warn threshold.", True

    if last_decision in ("warn", "alert") and trend["consecutive_below_warn"] < HYSTERESIS_READINGS:
        return "warn", "Recently in warn/alert; holding until stabilized below threshold.", False

    return "ok", f"{value}C is within normal range, trend {trend['direction']}.", False


def reason(state: AgentState) -> AgentState:
    history = memory_store.recent(state["topic"], limit=3)
    value = state["reading"]["value_c"]
    last_decision = history[-1]["decision"] if history else None

    trend = compute_trend(history, value)
    decision, reasoning, ambiguous = rule_based_decision(value, trend, last_decision)
    engine = "rules"

    if ambiguous:
        prompt = f"""You are an operations agent for an industrial machine.

Goals: {WARN_THRESHOLD}C warn / {ALERT_THRESHOLD}C alert thresholds for machine-07 temperature.

The rule engine flagged this case as ambiguous:
- current value: {value}C
- trend: {json.dumps(trend)}
- last decision: {last_decision}
- rule engine's provisional call: {decision} ({reasoning})

Decide the right action, considering the trend as well as the raw value.
Respond ONLY as compact JSON: {{"decision": "<ok|warn|alert>", "reasoning": "<one sentence>"}}"""

        response = llm.invoke(prompt)
        try:
            parsed = json.loads(response.content)
            decision, reasoning = parsed["decision"], parsed["reasoning"]
            engine = "llm"
        except (json.JSONDecodeError, KeyError):
            # LLM failed to follow format -- fall back to the rule engine's
            # own provisional call rather than guessing.
            engine = "rules_fallback"

    print(f"[REASON]  trend={trend}  decision={decision!r} ({engine})  because: {reasoning}")
    return {"decision": decision, "reasoning": reasoning, "engine": engine}


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
