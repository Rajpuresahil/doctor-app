# 🩺 Doctor App — Speech-Gated Clinical Scribe

> **Author:** Sahil Rajpure (M.Sc. AI & Data Science)
> 
> 
> **Overview:** A full-stack, speech-gated clinical scribe web application designed to eliminate administrative burden for clinicians while maintaining strict data grounding and safety.
> 
> 

---

## 🚀 Run the Application

```bash
./run.sh            # creates venv, installs dependencies, and serves http://localhost:8000

```

Open **http://localhost:8000** in your browser (microphone access requires `localhost` or HTTPS).

* **Optional Configuration:** Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.
* **No Keys? No Problem:** The app features a graceful offline fallback utilizing local Whisper and offline rule-based extraction. Click **▶ Demo consult** to run the complete UI workflow instantly without a microphone.



---

## ✨ What Makes It Different

| Innovation | Why It Matters |
| --- | --- |
| **Client-Side VAD** (Adaptive noise floor, hysteresis, 300 ms pre-roll, 700 ms hangover, 12s cap)

 | Silence never leaves the browser. Shows *"Silence skipped %"* to save bandwidth, reduce latency, cut ASR compute costs, and prevent Whisper hallucinations.

 |
| **Context-Primed Whisper**

 | Each audio segment is decoded with a medical vocabulary primer plus transcript tail context, significantly improving drug name accuracy.

 |
| **Hallucination Filter**

 | Automatically drops typical Whisper phantom outputs (e.g., "thank you", "you") on short clips.

 |
| **Evidence-Grounded Extraction**

 | The LLM must return a verbatim quote for every finding. The server verifies quotes against the transcript and discards ungrounded data. Hovering any summary section highlights its exact source words in the transcript.

 |
| **No-Invention Schema**

 | Strictly separates positive symptoms from explicitly denied symptoms; assessments are captured only as stated by the clinician.

 |
| **Graceful Degradation**

 | Automatically routes from Claude → OpenAI → Offline heuristic, displaying the active engine status directly in the UI.

 |

---

## 🖥️ Live Demo Walkthrough

Clicking **▶ Demo Consult** in the application loads a pre-recorded consultation for **Tushar Sharma** (30-year-old male presenting with fever and sore throat):

* **Transcript Generated:** Full conversational dialogue between doctor and patient, capturing symptoms, duration, medical history (type 2 diabetes, appendectomy in 2015), and medications (metformin, penicillin allergy).


* **AI Medical Summary:** Fully populated clinical record featuring **23/23 verified grounded quotes** (`heuristic` engine active).


* **Extracted Schema Categories:**
* *Clinical Summary & Chief Complaint* (fever, sore throat)


* *History of Present Illness (HPI)* (Duration: 3 days; Aggravating: worse at night / swallowing; Relieving: paracetamol)


* *Symptoms* (Positive & Negative toggles)


* *Past Medical History & Medications/Allergies* (metformin 500mg, penicillin allergy)


* *Examinations & Vitals* (BP 120/80 mmHg, Temp 101°F, red throat with swollen tonsils)


* *Plan & Follow-Up* (Throat swab, CBC, azithromycin prescription, advice, and 1-week follow-up)





---

## 📊 Model Selection & Benchmarking

Default configuration uses **`faster-whisper base.en`**, `int8`, `CPU`, `beam 1`.

* *Rationale:* `tiny.en` is fast but weak on complex drug names; `small.en` improves accuracy but roughly doubles CPU latency. `base.en` provides the optimal latency/accuracy balance for live VAD-chunked use.



To benchmark models on your own audio clips (`16 kHz mono wav`), run:

```bash
python backend/benchmark.py consult1.wav "reference transcript"

```

This outputs Word Error Rate (WER) and Real-Time Factor (RTF). Switch models easily using environment variables: `WHISPER_MODEL=small.en` or `ASR_BACKEND=openai`.

---

## 📂 Project Directory Structure

```text
doctor-app/
├── backend/
│   ├── main.py        # FastAPI server: /api/transcribe, /api/analyze, /api/health
│   ├── asr.py         # faster-whisper / OpenAI whisper-1 integration
│   ├── extract.py     # LLM extraction, schema normalisation, quote grounding
│   └── benchmark.py   # WER & RTF model comparison script
├── frontend/
│   ├── index.html     # Single-page UI layout
│   ├── style.css      # Styling & hover-highlight styles
│   └── app.js         # Client-side VAD, mic streaming, & interactive viewer
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
└── run.sh             # Automated setup and run script

```

---

## ⚠️ Limitations & Next Steps

* **VAD:** Uses a lightweight energy VAD; upgrading to a neural VAD (Silero via `onnxruntime-web`) is planned as a drop-in enhancement.


* **Diarisation:** Speaker diarisation is currently unsupported.


* **Clinical Disclaimer:** Generated records serve as a draft for clinician review and are not standalone medical records.
