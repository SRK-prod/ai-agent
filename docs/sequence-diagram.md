# Sequence: one question, end to end

```mermaid
sequenceDiagram
    participant Mic as BlackHole/Mic (CoreAudio)
    participant Cap as AudioCapture
    participant VAD as SileroVAD
    participant Spk as SpeakerDiarizer
    participant STT as SttStage (Faster-Whisper)
    participant QD as QuestionDetector
    participant Ret as HybridSearcher (Qdrant)
    participant Cache as RedisCache
    participant LLM as ClaudeClient (CLI or API)
    participant Opt as AnswerOptimizer
    participant WS as FastAPI /ws
    participant UI as PySide6 Overlay

    Mic->>Cap: raw PCM blocks
    Cap->>VAD: AudioFrame stream
    VAD->>VAD: buffer until speech start/end
    VAD->>Spk: SpeechSegment (one utterance)
    Spk->>Spk: embed + cosine similarity vs enrolled "me"
    alt is_me
        Spk-->>Spk: drop (never transcribed)
    else other speaker
        Spk->>STT: DiarizedSegment
        STT->>QD: Transcript
        alt not a question (denylist / no keyword+"?")
            QD-->>QD: drop
        else question detected
            QD->>Ret: DetectedQuestion
            Ret->>Ret: embed question, Qdrant search, keyword-overlap fuse
            Ret->>Cache: check llm_response cache (question + chunk hash)
            alt cache hit
                Cache-->>Opt: cached raw text
            else cache miss
                Ret->>LLM: system+user prompt
                LLM-->>Opt: raw completion incl. CONFIDENCE line
                Opt->>Cache: store raw text
            end
            Opt->>Opt: parse confidence, detect format, low-confidence gate
            Opt->>WS: Answer
            WS->>UI: {"type":"answer","data":{...}}
            UI->>UI: render in floating overlay
        end
    end
```
