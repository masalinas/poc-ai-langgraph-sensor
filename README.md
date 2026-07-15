# Description
PoC Langraph AI Agent integrated with a MQTT Events reasoning about the status of a machine using determenistic and semantic rules.

## Execution
- **Start Sensor Simulator** to generate mqtt temperature events published in MQTT Hive free MQTT Broker

    ```python
    python simulate_sensor.py
    ```

- **Start Sensor Agent**. to predict the machine status from temperature sensor published in a Hive MQTT Broker. This agent execute these steps using a langgraph workflow:

    - **Sense**: he "sensing" already happened in on_message (that's what triggered this run); this node just normalizes/validates the payload for the graph
    - **Reason**: pulls persisted history for this topic out of SQLite, not memory
    - **Actuate**: publishes back to MQTT, this is a real client.publish() call
    - **Refelct**: persists the cycle to SQLite instead of an in-memory list

    ```python
    python agent.py
    ```

## Explaining Reasoning:

This system is an excellent example of a hybrid architecture (or "Guardrails"). Instead of sending every data point directly to the LLM (which would be slow, costly, and unpredictable), the code uses a deterministic rules engine to filter out easy, unambiguous cases, and only delegates to the LLM when the situation enters a "gray area."

Let's break down how these rules and key concepts work, especially hysteresis and the ambiguous rule that is delegated to the LLM.

1. In control engineering, hysteresis is used to prevent a system from going "crazy" by oscillating between two states (like an air conditioner turning on and off every 2 seconds because the temperature fluctuates by 0.1°C).

    In your code, it is defined as:

        ```python
        HYSTERESIS_READINGS = 2  
        ```

        Consecutive readings needed to exit warning/alert.

    How is it applied?

    - From Alert to Warning (alert $\rightarrow$ warn): If the machine was in a critical state (alert) and the temperature drops slightly below 90°C (entering the warning zone), the system maintains the alert unless it has been below that alert limit for at least 2 consecutive readings (consecutive_below_warn).
    
    - From Warning to Normal (warn $\rightarrow$ ok): If the machine drops below 85°C, it is not immediately marked as ok. The system maintains the warn state as a precaution until stability is proven for 2 consecutive readings below 85°C.


2. Rule Flow (Step-by-Step)

The engine evaluates rules in a strict top-down order (prioritizing safety):

- **Step A**: Immediate Alert (Critical Case)

    ```python
    if value >= ALERT_THRESHOLD:
        return "alert", f"{value}C >= alert threshold {ALERT_THRESHOLD}C.", False
    ``` 

    **Logic**: If the current temperature is greater than or equal to 90°C, an alert is declared immediately. There is nothing to doubt, so the LLM is not needed (ambiguous=False).

- **Step B**: Warning Zone (warn band)
    ```python
    if value >= WARN_THRESHOLD:
        if last_decision == "alert" and trend["consecutive_below_warn"] < HYSTERESIS_READINGS:
            return "alert", "Recently alerted; holding alert...", False
        return "warn", f"{value}C is within warn band...", False
    ```

    **Logic**: If the temperature is between 85°C and 89.9°C, theoretically it is a warn.
    Application of Hysteresis: But if the previous reading was an alert (alert) and we don't have enough consecutive cold readings yet (consecutive_below_warn < 2), it keeps the alert state for safety.

    - **Step C**: The Gray Area (LLM Delegation)
    
    This is the rule you mentioned where the system is not sure and delegates the decision:

    ```python
    if trend["direction"] == "rising" and trend["slope"] >= RISING_SLOPE_C and (WARN_THRESHOLD - value) <= PREDICTIVE_MARGIN:
        return "warn", "Provisional: rising fast near warn threshold.", True
    ```

    When is it triggered? Three conditions must be met at the same time:

    - The temperature is rising (trend["direction"] == "rising").

    - It is rising very fast (trend["slope"] >= 4.0 degrees per cycle).

    - It is dangerously close to the warning limit. Specifically, less than 3°C away (WARN_THRESHOLD - value <= 3.0, meaning the value is between 82°C and 84.9°C).

    **Why is it delegated?**: Although the value (83°C, for example) is technically "normal" (less than 85°C), the fact that it is rising so fast means it will likely cross the limit in the next cycle.

    **The action**: It returns a provisional warn state, but marks ambiguous=True. This tells your code's orchestrator: "Hey, the mathematical rule suspects something is wrong but cannot confirm it absolutely; ask the LLM to analyze the full context of the previous reasoning."

- **Paso D**: Cool-down Hysteresis (Returning to normal)

    ```python
    if last_decision in ("warn", "alert") and trend["consecutive_below_warn"] < HYSTERESIS_READINGS:
            return "warn", "Recently in warn/alert; holding until stabilized...", False
    ```

    **Logic**: If the temperature has already dropped below the danger limit (for example, it is at 80°C), but we recently came from a warn or alert state and it hasn't stabilized (less than 2 safe consecutive readings), the system retains the warn state. This avoids giving a false "all good" message when the system might be experiencing intermittent spikes.

- **Step E**: All OK

    ```python
    return "ok", f"{value}C is within normal range, trend {trend['direction']}.", False
    ```

    **Logic**: If none of the above are met (the temperature is below 85°C, there are no rapid rising trends, and the system has already stabilized after cooling down), the state is safely declared as ok and the LLM does not intervene.

## Execution logs

Some sample agent logs are these ones. In these events we can observe that many are not ambiguous decissions and only one is ambiguous to be confirmed by a human later, but the thread continues, the events to be confirmes are save in SQLLite so they can be query or listed in a UI to be select and confirmed:

```shell
[SENSE]   veradoc/demo/machine07/temperature -> {'sensor': 'veradoc/demo/machine07/temperature', 'value_c': 68.5, 'timestamp': '2026-07-15T13:21:50.360142'}
[REASON]  rules -> decision='ok' ambiguous=False  (68.5C is within normal range, trend is falling.)
[ACTUATE] published 'OK: 68.5C' -> veradoc/demo/machine07/status
[REFLECT] persisted as memory row #198 (engine=rules)

[SENSE]   veradoc/demo/machine07/temperature -> {'sensor': 'veradoc/demo/machine07/temperature', 'value_c': 61.1, 'timestamp': '2026-07-15T13:21:55.360649'}
[REASON]  rules -> decision='ok' ambiguous=False  (61.1C is within normal range, trend is falling.)
[ACTUATE] published 'OK: 61.1C' -> veradoc/demo/machine07/status
[REFLECT] persisted as memory row #199 (engine=rules)

[SENSE]   veradoc/demo/machine07/temperature -> {'sensor': 'veradoc/demo/machine07/temperature', 'value_c': 69.9, 'timestamp': '2026-07-15T13:22:00.361096'}
[REASON]  rules -> decision='ok' ambiguous=False  (69.9C is within normal range, trend is falling.)
[ACTUATE] published 'OK: 69.9C' -> veradoc/demo/machine07/status
[REFLECT] persisted as memory row #200 (engine=rules)

[SENSE]   veradoc/demo/machine07/temperature -> {'sensor': 'veradoc/demo/machine07/temperature', 'value_c': 82.2, 'timestamp': '2026-07-15T13:22:05.361557'}
[REASON]  rules -> decision='warn' ambiguous=True  (82.2C is below 85.0C but rising at +4.6C/cycle, projected to reach 86.8C next cycle.)
[REASON]  llm -> decision='warn'  because: Projected temperature exceeds warning threshold despite current reading being below 85C.
[REVIEW]  paused for human approval -> 'warn' (llm), waiting on veradoc/demo/machine07/approval_response
[REVIEW]  awaiting approval for thread 00482696-0cae-4f43-aedf-1b5ec13ff84a -> published to veradoc/demo/machine07/approval_request

[SENSE]   veradoc/demo/machine07/temperature -> {'sensor': 'veradoc/demo/machine07/temperature', 'value_c': 94.4, 'timestamp': '2026-07-15T13:22:10.361999'}
[REASON]  rules -> decision='alert' ambiguous=False  (94.4C is at/above the 90.0C alert threshold.)
[ACTUATE] published 'ALERT: 94.4C, shutdown recommended' -> veradoc/demo/machine07/alert
[REFLECT] persisted as memory row #201 (engine=rules)
```

## Manage human in the loop events

All ambiguous events are persisted in a SQLite database. You must send for each one a MQTT event like this to be accept or not, then the final machine status for this particular event will be updated in consecuence:

```shell
$ mosquitto_pub -h broker.hivemq.com -t veradoc/demo/machine07/approval_response \
    -m '{"thread_id": "354e3321-8fee-445c-97eb-b4ef300c0b75", "approved": true, "operator": "miguel"}'

[SENSE]   veradoc/demo/machine07/temperature -> {'sensor': 'veradoc/demo/machine07/temperature', 'value_c': 88.1, 'timestamp': '2026-07-15T13:24:45.379469'}
[REASON]  rules -> decision='warn' ambiguous=False  (88.1C is at/above the 85.0C warn threshold.)
[ACTUATE] published 'WARN: 88.1C, notify on-call' -> veradoc/demo/machine07/warn
[REFLECT] persisted as memory row #231 (engine=rules)
[REVIEW]  paused for human approval -> 'warn' (llm), waiting on veradoc/demo/machine07/approval_response
[REVIEW]  approved by miguel
[ACTUATE] published 'WARN: 82.9C, notify on-call' -> veradoc/demo/machine07/warn
[REFLECT] persisted as memory row #232 (engine=llm)

```

## Links

- [Hive MQTT Agent Marketplace](https://app.hivemq.com/act/marketplace)