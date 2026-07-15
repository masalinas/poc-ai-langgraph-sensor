"""
Event-driven Sense -> Reason -> Actuate -> Reflect agent.
 
Instead of a fixed loop, each MQTT message that arrives triggers exactly one
pass through the LangGraph. Memory persists in SQLite (memory_store.py), so
the agent has real history across restarts -- closer to how the HiveMQTT
pattern actually runs in production.
 
Reason is a hybrid: trend.py does deterministic trend analysis + threshold
rules over the last 3 readings (no LLM, no network call). The LLM is only
invoked when the rule engine itself flags a case as ambiguous -- a predicted
threshold crossing or a value hovering right at a boundary. Most cycles never
touch the LLM at all.
 
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
import trend
 
BROKER_HOST = "broker.hivemq.com"   # public test broker; point at your own for real use
BROKER_PORT = 1883
SENSE_TOPIC = "veradoc/demo/machine07/temperature"
ACTUATE_TOPIC_PREFIX = "veradoc/demo/machine07"
 
GOALS = "Keep machine-07 temperature under 85C. Warn at 85-90C. Alert above 90C."
 
# LLM is now only called for ambiguous cases -- constructed lazily so a run
# with no ambiguous cases never even needs Ollama up.
_llm = None
 
 
def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    return _llm
 
 
# ---------------------------------------------------------------------------
# Shared state for a single graph invocation (i.e. a single MQTT message)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    topic: str
    reading: dict
    decision: str
    reasoning: str
    engine: str    # "rules" | "llm" | "rules_fallback" -- who actually decided
    action: str
 
 
# ---------------------------------------------------------------------------
# SENSE - the "sensing" already happened in on_message (that's what triggered
# this run); this node just normalizes/validates the payload for the graph.
# ---------------------------------------------------------------------------
def sense(state: AgentState) -> AgentState:
    print(f"\n[SENSE]   {state['topic']} -> {state['reading']}")
    return {}
 
 
# ---------------------------------------------------------------------------
# REASON - two layers:
#   1. Deterministic: trend.compute_stats + trend.rule_based_decision, using
#      the last 3 persisted readings for this topic. Fast, free, auditable.
#   2. LLM (only if the rules flag the case as ambiguous): gets the same
#      stats plus full history and either confirms or overrides the rule's
#      candidate decision, with a one-sentence justification.
# ---------------------------------------------------------------------------
def reason(state: AgentState) -> AgentState:
    history = memory_store.recent(state["topic"], limit=3)
    current_value = state["reading"]["value_c"]
 
    stats = trend.compute_stats(history, current_value)
    rule_decision, is_ambiguous, note = trend.rule_based_decision(current_value, stats)
 
    print(f"[REASON]  rules -> decision={rule_decision!r} ambiguous={is_ambiguous}  ({note})")
 
    if not is_ambiguous:
        return {"decision": rule_decision, "reasoning": note, "engine": "rules"}
 
    # Only ambiguous cases reach the LLM, and it's asked to weigh in on a
    # specific, narrow question -- not to invent the whole decision from scratch.
    prompt = f"""You are an operations agent for an industrial machine.
 
Goals and rules:
{GOALS}
 
Deterministic analysis of the last {len(stats.values)} readings:
- values (oldest -> newest): {stats.values}
- trend: {stats.trend} ({stats.slope:+.1f}C/cycle)
- projected next reading: {stats.projected_next:.1f}C
- rule engine's candidate decision: "{rule_decision}"
- why this case is ambiguous: {note}
 
Confirm or override the candidate decision. Respond ONLY as compact JSON:
{{"decision": "<ok|warn|alert>", "reasoning": "<one sentence>"}}"""
 
    response = get_llm().invoke(prompt)
 
    try:
        parsed = json.loads(response.content)
        decision, reasoning = parsed["decision"], parsed["reasoning"]
        engine = "llm"
    except (json.JSONDecodeError, KeyError):
        # LLM failed to follow format -> fall back to the rule engine's answer
        decision, reasoning, engine = rule_decision, note, "rules_fallback"
 
    print(f"[REASON]  {engine} -> decision={decision!r}  because: {reasoning}")
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
        engine=state["engine"],
    )
    print(f"[REFLECT] persisted as memory row #{row_id} (engine={state['engine']})")
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
 
        app.invoke({
            "topic": msg.topic,
            "reading": reading,
            "decision": "",
            "reasoning": "",
            "engine": "",
            "action": "",
        })
 
    client.on_connect = on_connect
    client.on_message = on_message
 
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()  # blocks; each message triggers sense->reason->actuate->reflect
 
 
if __name__ == "__main__":
    main()