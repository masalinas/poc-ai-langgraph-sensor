"""
Event-driven Sense -> Reason -> Actuate -> Reflect agent, with a human
approval gate for any decision that came out of the LLM.

Instead of a fixed loop, each MQTT message that arrives triggers exactly one
pass through the LangGraph. Memory persists in SQLite (memory_store.py), so
the agent has real history across restarts -- closer to how the HiveMQTT
pattern actually runs in production.

Reason is a hybrid: trend.py does deterministic trend analysis + threshold
rules over the last 3 readings (no LLM, no network call). The LLM is only
invoked when the rule engine itself flags a case as ambiguous. Most cycles
never touch the LLM at all, and those that don't go straight to actuate.

Human-in-the-loop: any cycle where engine ends up "llm" or "rules_fallback"
is routed to human_review, which calls LangGraph's interrupt() -- the graph
freezes (state persisted in agent_checkpoints.db via a SqliteSaver
checkpointer) and the process publishes an approval request to
veradoc/demo/machine07/approval_request. Nothing resumes until someone
publishes a verdict to veradoc/demo/machine07/approval_response with the
matching thread_id; only then does execution continue into actuate.

Run:
    pip install langgraph langgraph-checkpoint-sqlite langchain-ollama paho-mqtt --break-system-packages
    ollama pull qwen2.5:7b

    # terminal 1
    python agent.py

    # terminal 2 (simulates a real sensor publishing readings)
    python simulate_sensor.py

    # terminal 3, only needed when a case reaches human_review -- watch for
    # "[REVIEW] awaiting approval for thread <id>" in terminal 1, then:
    mosquitto_pub -h broker.hivemq.com -t veradoc/demo/machine07/approval_response \\
        -m '{"thread_id": "<id-from-log>", "approved": true, "operator": "miguel"}'

    # or to reject and override to a different decision:
    mosquitto_pub -h broker.hivemq.com -t veradoc/demo/machine07/approval_response \\
        -m '{"thread_id": "<id-from-log>", "approved": false, "override": "ok", "operator": "miguel", "reason": "known transient spike"}'
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import TypedDict, Literal

import paho.mqtt.client as mqtt
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

import memory_store
import trend

BROKER_HOST = "broker.hivemq.com"   # public test broker; point at your own for real use
BROKER_PORT = 1883
SENSE_TOPIC = "veradoc/demo/machine07/temperature"
ACTUATE_TOPIC_PREFIX = "veradoc/demo/machine07"
APPROVAL_REQUEST_TOPIC = f"{ACTUATE_TOPIC_PREFIX}/approval_request"
APPROVAL_RESPONSE_TOPIC = f"{ACTUATE_TOPIC_PREFIX}/approval_response"

CHECKPOINT_DB = Path(__file__).parent / "agent_checkpoints.db"

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
# HUMAN REVIEW - only reached when `reason` used the LLM (or fell back from
# it). Deterministic rule-only decisions skip this node entirely and go
# straight to actuate. `interrupt()` freezes the graph here -- the process
# doesn't block on input(); LangGraph persists the paused state to
# CHECKPOINT_DB via the checkpointer and this function call simply doesn't
# return until someone calls app.invoke(Command(resume=...), config=...)
# with the same thread_id, which happens in on_approval() below.
# ---------------------------------------------------------------------------
def human_review(state: AgentState) -> AgentState:
    print(f"[REVIEW]  paused for human approval -> {state['decision']!r} ({state['engine']}), waiting on {APPROVAL_RESPONSE_TOPIC}")

    verdict = interrupt({
        "topic": state["topic"],
        "reading": state["reading"],
        "proposed_decision": state["decision"],
        "reasoning": state["reasoning"],
        "engine": state["engine"],
    })

    if verdict.get("approved", False):
        print(f"[REVIEW]  approved by {verdict.get('operator', 'unknown')}")
        return {}  # keep the LLM's decision/reasoning/engine as-is

    override = verdict.get("override", "ok")
    print(f"[REVIEW]  rejected by {verdict.get('operator', 'unknown')}, overriding to {override!r}")
    return {
        "decision": override,
        "reasoning": f"Operator override: {verdict.get('reason', 'no reason given')}",
        "engine": "human_override",
    }


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
# Only route to human_review when the decision came from the LLM (or from a
# failed LLM call falling back to rules) -- deterministic rule-only decisions
# never pause for a human.
# ---------------------------------------------------------------------------
def needs_human_review(state: AgentState) -> Literal["human_review", "actuate"]:
    return "human_review" if state["engine"] in ("llm", "rules_fallback") else "actuate"


# ---------------------------------------------------------------------------
# Build the graph. `actuate` needs the live mqtt client, so we bind it via a
# closure when compiling rather than stuffing the client into AgentState
# (it isn't serializable and doesn't belong in the data model).
# ---------------------------------------------------------------------------
def build_graph(mqtt_client: mqtt.Client):
    graph = StateGraph(AgentState)
    graph.add_node("sense", sense)
    graph.add_node("reason", reason)
    graph.add_node("human_review", human_review)
    graph.add_node("actuate", lambda state: actuate(state, mqtt_client))
    graph.add_node("reflect", reflect)

    graph.set_entry_point("sense")
    graph.add_edge("sense", "reason")
    graph.add_conditional_edges("reason", needs_human_review, {"human_review": "human_review", "actuate": "actuate"})
    graph.add_edge("human_review", "actuate")
    graph.add_edge("actuate", "reflect")
    graph.add_edge("reflect", END)  # one graph run = one message; no internal loop

    # A checkpointer is mandatory for interrupt()/Command(resume=...) to work --
    # it's what lets the graph actually freeze mid-run and pick back up later,
    # potentially in a different process.
    checkpointer = SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# MQTT wiring: every inbound message = one invocation of the graph
# ---------------------------------------------------------------------------
def main() -> None:
    memory_store.init_db()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    app = build_graph(client)

    def on_connect(c, userdata, flags, reason_code, properties):
        print(f"Connected to {BROKER_HOST} (rc={reason_code})")
        c.subscribe(SENSE_TOPIC)
        c.subscribe(APPROVAL_RESPONSE_TOPIC)
        print(f"  subscribed to {SENSE_TOPIC}")
        print(f"  subscribed to {APPROVAL_RESPONSE_TOPIC}")

    def on_sensor_message(c, msg):
        try:
            reading = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            print(f"Ignoring non-JSON payload on {msg.topic}: {msg.payload!r}")
            return

        # A fresh thread_id per message -- this is the key the checkpointer
        # uses to persist/resume this specific run if it gets interrupted.
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        result = app.invoke(
            {"topic": msg.topic, "reading": reading, "decision": "", "reasoning": "", "engine": "", "action": ""},
            config=config,
        )
        handle_possible_interrupt(c, thread_id, result)

    def handle_possible_interrupt(c, thread_id: str, result: dict) -> None:
        """After any invoke/resume, check whether the graph paused at
        human_review. If so, publish the approval request and stop --
        on_approval_message will resume this exact thread_id later."""
        pending = result.get("__interrupt__")
        if not pending:
            return  # graph ran straight through to reflect/END, nothing to do

        payload = pending[0].value
        payload["thread_id"] = thread_id
        c.publish(APPROVAL_REQUEST_TOPIC, json.dumps(payload))
        print(f"[REVIEW]  awaiting approval for thread {thread_id} -> published to {APPROVAL_REQUEST_TOPIC}")

    def on_approval_message(c, msg):
        try:
            verdict = json.loads(msg.payload.decode())
            thread_id = verdict["thread_id"]
        except (json.JSONDecodeError, KeyError):
            print(f"Ignoring malformed approval payload: {msg.payload!r}")
            return

        config = {"configurable": {"thread_id": thread_id}}
        result = app.invoke(Command(resume=verdict), config=config)
        handle_possible_interrupt(c, thread_id, result)

    def on_message(c, userdata, msg):
        if msg.topic == SENSE_TOPIC:
            on_sensor_message(c, msg)
        elif msg.topic == APPROVAL_RESPONSE_TOPIC:
            on_approval_message(c, msg)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()  # blocks; each message triggers sense->reason->[human_review]->actuate->reflect


if __name__ == "__main__":
    main()