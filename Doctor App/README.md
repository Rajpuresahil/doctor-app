# 🩺 Doctor App — speech-gated clinical scribe

Mic → **VAD (in the browser)** → **Whisper** → live transcript → **LLM** → structured clinical record + summary.

## Run
```bash
./run.sh            # creates venv, installs, serves http://localhost:8000
```
Open http://localhost:8000 (mic needs `localhost` or HTTPS). Optional: copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.
No keys? It still works: local Whisper + an offline rule-based extractor. **▶ Demo consult** runs the whole UI without a mic.

## What makes it different
| Idea | Why it matters |
|---|---|
| **VAD runs client-side** (adaptive noise floor, hysteresis, 300 ms pre-roll, 700 ms hangover, 12 s cap) | Silence never leaves the browser. The UI shows *"Silence skipped %"*: bandwidth, latency and ASR compute saved, and fewer Whisper hallucinations. |
| **Context-primed Whisper** | Each segment is decoded with a medical vocabulary primer plus the transcript tail, which improves drug names and keeps phrasing consistent across chunks. |
| **Hallucination filter** | Drops typical Whisper phantom outputs ("thank you", "you") on very short clips. |
| **Evidence-grounded extraction** | The LLM must return a verbatim quote for each finding. The server verifies each quote exists in the transcript and drops the rest. The UI shows `grounded N/M`, and **hovering a summary section highlights its source words** in the transcript. Clinicians can verify in one glance. |
| **No-invention schema** | Positive vs. explicitly denied symptoms kept separate; assessment only as stated by the clinician. |
| **Graceful degradation** | Claude → OpenAI → offline heuristic. The UI states which engine produced the result. |
| **Ordered async queue** | Segments are transcribed in order, with a "… transcribing" indicator. |

## Model selection (lightweight, reproducible)
Default **`faster-whisper base.en`, int8, CPU, beam 1**. Rationale: `tiny.en` is fastest but weakest on drug names; `small.en` is more accurate but roughly doubles latency on CPU. `base.en` is the usual latency/accuracy sweet spot for VAD-chunked live use. I did **not** measure numbers for this submission. Run the included script on your own clips to choose with data:
```bash
python backend/benchmark.py consult1.wav "reference transcript" ...   # 16 kHz mono wav
```
It prints WER and real-time factor for tiny/base/small. Switch with `WHISPER_MODEL=small.en` or `ASR_BACKEND=openai` (whisper-1).

## Structure
```
backend/main.py       FastAPI: /api/transcribe, /api/analyze, /api/health, serves UI
backend/asr.py        faster-whisper / OpenAI whisper-1
backend/extract.py    LLM extraction, schema normalisation, quote grounding, offline fallback
backend/benchmark.py  WER + RTF model comparison
frontend/             index.html, style.css, app.js (VAD, mic, UI; no build step)
```
Extracted schema: patient, chief complaint, HPI (onset/duration/progression/aggravating/relieving), symptoms (positive/negative), PMH, medications + allergies, observations (vitals/exam), assessment, plan (investigations/prescriptions/advice/follow-up), clinical summary.

## Limits / next steps
Energy VAD is lightweight but not as robust as a neural VAD (Silero via onnxruntime-web is the drop-in upgrade). No speaker diarisation. Output is a draft for clinician review, not a medical record on its own.
