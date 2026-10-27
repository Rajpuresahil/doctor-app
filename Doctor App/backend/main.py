import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import asr, extract

app = FastAPI(title="Doctor App")


class AnalyzeIn(BaseModel):
    transcript: str


@app.get("/api/health")
def health():
    llm = "claude" if os.getenv("ANTHROPIC_API_KEY") else "openai" if os.getenv("OPENAI_API_KEY") else "heuristic (offline)"
    return {"asr": f"{asr.BACKEND}:{asr.MODEL_NAME}", "llm": llm}


@app.post("/api/transcribe")
async def transcribe(request: Request):
    pcm = await request.body()  # raw 16 kHz mono int16, already VAD-gated by the browser
    ctx = request.headers.get("x-context", "")[-300:]
    try:
        text = await run_in_threadpool(asr.transcribe, pcm, ctx)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=503)
    return {"text": text}


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    return extract.analyze(body.transcript)


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True), name="ui")
