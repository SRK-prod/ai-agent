# Backend API

Base URL: `http://<host>:<port>` from `MEETING_COPILOT_HOST`/`_PORT` in
`.env` (default `http://127.0.0.1:8765`).

## `GET /health`

```json
{"status": "ok"}
```

## `GET /metrics`

Prometheus exposition format (`text/plain`). Key series:

- `meeting_copilot_stage_latency_seconds{stage="speaker_id"|"stt"|"retrieval"|"llm"}` (histogram)
- `meeting_copilot_total_latency_seconds` (histogram)

## `POST /enroll`

Body:

```json
{"duration_seconds": 45.0}
```

Blocking call (runs in a thread) that records from the default input device
and stores a voice embedding. **Only call this before the meeting
pipeline is actively capturing audio** -- it does not pause the live
pipeline first.

Response:

```json
{"status": "enrolled"}
```

## `WS /ws`

The overlay connects here on boot. The backend pushes one message per
generated answer:

```json
{
  "type": "answer",
  "data": {
    "question": {
      "transcript": {
        "speaker_id": "B",
        "text": "What's the tradeoff between Kafka and Redis here?",
        "start_time": 12.4,
        "end_time": 14.9,
        "language": "en"
      },
      "matched_keywords": ["kafka", "redis", "tradeoff"],
      "ends_with_question_mark": true
    },
    "text": "| | Kafka | Redis |\n|---|---|---|\n...",
    "format_type": "table",
    "confidence": 0.88,
    "low_confidence": false
  }
}
```

The client currently only receives; anything sent from the client on `/ws`
is read and ignored (reserved for future overlay -> backend commands).
