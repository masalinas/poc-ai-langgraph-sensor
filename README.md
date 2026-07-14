# Description
PoC Langraph AI Agent integrated with a MQTT Events reasoning about the status of a machine using determenistic and semantic rules.

## Execution
- Start simulate_sensor.py to generate mqtt temperature events,

python simulate_sensor.py

- Start the agent. These are the steps executed by the agent:

    - **Sense**: he "sensing" already happened in on_message (that's what triggered this run); this node just normalizes/validates the payload for the graph
    - **Reason**: pulls persisted history for this topic out of SQLite, not memory
    - **Actuate**: publishes back to MQTT, this is a real client.publish() call
    - **Refelct**: persists the cycle to SQLite instead of an in-memory list

python agent.py