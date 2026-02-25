import os
import io
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, ConfigDict
from scipy import signal as scipy_signal

from schema import FunctionCallResponse, MediaBinItem, TimelineState

# Directory for uploaded CSV files
CSV_UPLOAD_DIR = Path(__file__).parent / "csv_uploads"
CSV_UPLOAD_DIR.mkdir(exist_ok=True)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# Gemini client is optional – ha nincs API kulcs, az /ai endpoint 503-mal válaszol
gemini_api = None
if GEMINI_API_KEY:
    gemini_api = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("⚠️  GEMINI_API_KEY nincs beállítva – az AI asszisztens kikapcsolva.")

# Serve uploaded CSV files statically
app.mount("/csv", StaticFiles(directory=str(CSV_UPLOAD_DIR)), name="csv_files")


@app.post("/api/signal/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file, return its server URL."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files accepted")
    safe_name = Path(file.filename).name
    dest = CSV_UPLOAD_DIR / safe_name
    # If name collides, add suffix
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = CSV_UPLOAD_DIR / f"{stem}_{os.urandom(3).hex()}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file_url = f"http://127.0.0.1:3000/csv/{dest.name}"
    return {"filename": dest.name, "url": file_url, "path": str(dest)}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    # Be permissive with incoming payloads from the frontend
    model_config = ConfigDict(extra="ignore")

    message: str  # the full user message
    mentioned_scrubber_ids: list[str] | None = None  # scrubber ids mentioned via '@'
    # Accept any shape for resilience; backend does not mutate these
    timeline_state: dict[str, Any] | None = None  # current timeline state
    mediabin_items: list[dict[str, Any]] | None = None  # current media bin
    chat_history: list[dict[str, Any]] | None = None  # prior turns: [{"role":"user"|"assistant","content":"..."}]


@app.post("/ai")
async def process_ai_message(request: Message) -> FunctionCallResponse:
    if not gemini_api:
        raise HTTPException(status_code=503, detail="AI asszisztens nem elérhető: GEMINI_API_KEY nincs beállítva.")
    print(FunctionCallResponse)
    try:
        response = gemini_api.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
            You are Kimu, an AI assistant inside a video editor. You can decide to either:
            - call ONE tool from the provided schema when the user explicitly asks for an editing action, or
            - return a short friendly assistant_message when no concrete action is needed (e.g., greetings, small talk, clarifying questions).

            Strictly follow:
            - If the user's message does not clearly request an editing action, set function_call to null and include an assistant_message.
            - Only produce a function_call when it is safe and unambiguous to execute.

            Inference rules:
            - Assume a single active timeline; do NOT require a timeline_id.
            - Tracks are named like "track-1", but when the user says "track 1" they mean number 1.
            - Use pixels_per_second=100 by default if not provided.
            - When the user names media like "twitter" or "twitter header", map that to the closest media in the media bin by name substring match.
            - Prefer LLMAddScrubberByName when the user specifies a name, track number, and time in seconds.
            - If the user asks to remove scrubbers in a specific track, call LLMDeleteScrubbersInTrack with that track number.

            Conversation so far (oldest first): {request.chat_history}

            User message: {request.message}
            Mentioned scrubber ids: {request.mentioned_scrubber_ids}
            Timeline state: {request.timeline_state}
            Media bin items: {request.mediabin_items}
            """,
            config={
                "response_mime_type": "application/json",
                "response_schema": FunctionCallResponse,
            },
        )
        print(response)

        return FunctionCallResponse.model_validate(response.parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ── Signal / CSV endpoints ─────────────────────────────────────────────────────

class SignalColumnsRequest(BaseModel):
    csv_path: str  # absolute local path or http URL


class SignalDataRequest(BaseModel):
    csv_path: str
    column: str
    points: int = 800
    sample_rate: float = 200.0


class SignalCorrelateRequest(BaseModel):
    csv_path_1: str
    column_1: str
    csv_path_2: str
    column_2: str
    sample_rate: float = 200.0
    max_offset_seconds: float = 120.0


def _read_csv(csv_path: str) -> pd.DataFrame:
    import urllib.request
    if csv_path.startswith("http://") or csv_path.startswith("https://"):
        with urllib.request.urlopen(csv_path) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(content), sep=None, engine="python")
    return pd.read_csv(csv_path, sep=None, engine="python")


@app.post("/api/signal/columns")
async def signal_columns(req: SignalColumnsRequest):
    """Return column names from a CSV file."""
    try:
        df = _read_csv(req.csv_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return {"columns": df.columns.tolist(), "numeric_columns": numeric_cols}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/signal/data")
async def signal_data(req: SignalDataRequest):
    """Return downsampled signal data (envelope min/max) for waveform display."""
    try:
        df = _read_csv(req.csv_path)
        df.columns = [c.strip() for c in df.columns]
        col = req.column.strip()
        if col not in df.columns:
            raise HTTPException(400, f"Column '{col}' not found. Available: {df.columns.tolist()}")

        raw = pd.to_numeric(df[col], errors="coerce").to_numpy()
        n = len(raw)
        duration_s = n / req.sample_rate
        target = max(2, min(req.points, n))
        bucket = max(1, n // target)
        n_buckets = n // bucket
        trimmed = raw[:n_buckets * bucket].reshape(n_buckets, bucket)

        return {
            "x": (np.arange(n_buckets) * bucket / req.sample_rate).tolist(),
            "y_min": np.nanmin(trimmed, axis=1).tolist(),
            "y_max": np.nanmax(trimmed, axis=1).tolist(),
            "y_mean": np.nanmean(trimmed, axis=1).tolist(),
            "global_min": float(np.nanmin(raw)),
            "global_max": float(np.nanmax(raw)),
            "duration_s": duration_s,
            "n_samples": n,
            "sample_rate": req.sample_rate,
            "column": col,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/signal/correlate")
async def signal_correlate(req: SignalCorrelateRequest):
    """Cross-correlate two CSV signals, return best time offset (seconds)."""
    try:
        df1 = _read_csv(req.csv_path_1)
        df2 = _read_csv(req.csv_path_2)
        df1.columns = [c.strip() for c in df1.columns]
        df2.columns = [c.strip() for c in df2.columns]

        col1, col2 = req.column_1.strip(), req.column_2.strip()
        if col1 not in df1.columns:
            raise HTTPException(400, f"Column '{col1}' not in first CSV")
        if col2 not in df2.columns:
            raise HTTPException(400, f"Column '{col2}' not in second CSV")

        def _prep(df, col):
            s = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
            s = s - np.nanmean(s)
            std = np.nanstd(s)
            if std > 0: s = s / std
            sos = scipy_signal.butter(4, 2.0, btype="low", fs=req.sample_rate, output="sos")
            return scipy_signal.sosfiltfilt(sos, s)

        f1 = _prep(df1, col1)
        f2 = _prep(df2, col2)

        corr = scipy_signal.correlate(f1, f2, mode="full")
        lags_s = scipy_signal.correlation_lags(len(f1), len(f2), mode="full") / req.sample_rate
        valid = np.abs(lags_s) <= req.max_offset_seconds
        corr_v, lags_v = corr[valid], lags_s[valid]

        w = 21
        smoothed = np.convolve(corr_v, np.ones(w) / w, mode="same")
        best_idx = int(np.argmax(smoothed))

        top_idxs = np.argsort(smoothed)[-5:][::-1]
        return {
            "best_offset_s": float(lags_v[best_idx]),
            "score": float(smoothed[best_idx] / len(f1)),
            "candidates": [
                {"offset_s": float(lags_v[i]), "score": float(smoothed[i] / len(f1))}
                for i in top_idxs
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=3000)
