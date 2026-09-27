"""OpenAI-compatible TTS service for GPT-SoVITS on the QCS8550 NPU.

    POST /v1/audio/speech   {"model": "...", "input": "...", "voice": "<voice>", "response_format": "wav", ...}
    GET  /v1/models         GET /v1/audio/voices         GET /health

Everything is loaded when the service starts (6 QNN contexts on the NPU, CPU tables, all reference voices) and a
warm-up synthesis touches every lazy load, so requests never read models from disk. There is one NPU, so requests
run one at a time on a dedicated worker thread (others wait in line).

Request fields (OpenAI): input (<= 4096 chars), voice, response_format (wav | pcm | mp3 | opus | aac | flac;
default mp3 like OpenAI; mp3/opus/aac/flac need `ffmpeg` on PATH), speed (only 1.0 is supported), model (ignored).
Extensions: stream (bool: send each sentence as soon as it is synthesized; always on for pcm), seed, top_k,
temperature, hf_filter (off | notch | lowpass), presence_db (high-shelf boost above ~4 kHz). pcm is 24 kHz 16-bit mono little-endian as in the OpenAI API; the other formats are 32 kHz.

Run:  source setup_env.sh && python serve_tts.py --host 0.0.0.0 --port 8000 [--api-key KEY]
Test: curl http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \\
        -d '{"model":"gpt-sovits","input":"今天天气不错。","response_format":"wav"}' -o out.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import shutil
import struct
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import gsv_runtime as R

log = logging.getLogger("gsv-tts")
OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"}
MAX_INPUT = 4096
PCM_SR = 24000  # OpenAI's pcm format
MIME = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac",
        "flac": "audio/flac"}
FFMPEG_ARGS = {"mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"],
               "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "48k", "-ar", "48000"],
               "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
               "flac": ["-f", "flac", "-c:a", "flac"]}


# ------------------------------------------------------------------ audio helpers
def to_int16(a):
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


def wav_bytes(a, sr=R.SR):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(to_int16(a))
    return buf.getvalue()


def wav_stream_header(sr=R.SR):
    """WAV header with unknown length (0xFFFFFFFF sizes), for streamed wav."""
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


class Resampler32to24:
    """32 kHz -> 24 kHz (x3/4) polyphase windowed-sinc FIR in numpy; stateless per call (segments end in silence)."""

    def __init__(self, taps_per_phase=32, beta=8.0):
        up, down = 3, 4
        n = taps_per_phase * up
        t = np.arange(n) - (n - 1) / 2
        cutoff = 0.5 / down * 0.95  # of the upsampled rate
        h = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(n, beta)
        self.h, self.up, self.down = (h / h.sum() * up).astype(np.float32), up, down

    def __call__(self, x):
        u = np.zeros(len(x) * self.up, np.float32)
        u[:: self.up] = x
        y = np.convolve(u, self.h, mode="same")
        return y[:: self.down]


def ffmpeg_encode(a, fmt):
    ff = shutil.which("ffmpeg")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(R.SR), "-ac", "1", "-i", "pipe:0",
           *FFMPEG_ARGS[fmt], "pipe:1"]
    p = subprocess.run(cmd, input=to_int16(a), capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.decode(errors='replace')[:300]}")
    return p.stdout


def error(status, message, param=None, etype="invalid_request_error"):
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": etype, "param": param,
                                                               "code": None}})


# ------------------------------------------------------------------ service
class Service:
    def __init__(self, args):
        man = json.load(open(os.path.join(R.HERE, "manifest.json")))
        R.configure(man)
        self.model_name = man.get("config", {}).get("model_name", "gpt-sovits")
        t = time.perf_counter()
        if args.backend == "qnn":
            self.backend = R.QnnBackend(man, args.models_dir, args.qnn_lib, lib_path=args.gsv_lib, burst=not args.no_burst)
        else:
            self.backend = R.OrtBackend(man, args.onnx_dir)
        self.tts = R.GPTSoVITS(self.backend, args.voices)
        t_load = time.perf_counter() - t
        t = time.perf_counter()
        self.tts.warmup()
        log.info("loaded %d graphs in %.1fs, warm-up %.1fs, voices %s (default %s)", len(R.graphs_of(man)), t_load,
                 time.perf_counter() - t, list(self.tts.refs), self.tts.voice)
        self.resample = Resampler32to24()
        self.ready = True

    def close(self):
        if hasattr(self.backend, "close"):
            self.backend.close()


def create_app(args):
    state = {}

    # All NPU work (model loading included) runs on one dedicated thread; requests are serialized by an asyncio
    # lock held for the whole request (a threading lock held across a streamed response would deadlock the
    # single worker thread as soon as a second request queues on it).
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npu")
    alock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        state["svc"] = await asyncio.get_running_loop().run_in_executor(pool, Service, args)
        yield
        svc = state.pop("svc")
        await asyncio.get_running_loop().run_in_executor(pool, svc.close)  # free the NPU before exit (see QnnBackend)
        pool.shutdown(wait=True)

    app = FastAPI(title="GPT-SoVITS TTS (QCS8550 NPU)", lifespan=lifespan)

    def authorized(req):
        return not args.api_key or req.headers.get("authorization", "") == f"Bearer {args.api_key}"

    @app.get("/health")
    async def health():
        svc = state.get("svc")
        return {"status": "ok" if svc else "loading", "voices": list(svc.tts.refs) if svc else []}

    @app.get("/v1/models")
    async def models(req: Request):
        if not authorized(req):
            return error(401, "invalid api key", etype="authentication_error")
        name = state["svc"].model_name if "svc" in state else "gpt-sovits"
        return {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "local"}]}

    @app.get("/v1/audio/voices")
    async def voices(req: Request):
        if not authorized(req):
            return error(401, "invalid api key", etype="authentication_error")
        svc = state["svc"]
        return {"voices": list(svc.tts.refs), "default": svc.tts.voice}

    @app.post("/v1/audio/speech")
    async def speech(req: Request):
        if not authorized(req):
            return error(401, "invalid api key", etype="authentication_error")
        svc = state.get("svc")
        if svc is None:
            return error(503, "model is still loading", etype="server_error")
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            return error(400, "request body must be JSON")
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            return error(400, "'input' must be a non-empty string", "input")
        if len(text) > MAX_INPUT:
            return error(400, f"'input' is longer than {MAX_INPUT} characters", "input")
        fmt = body.get("response_format", "mp3")
        if fmt not in MIME:
            return error(400, f"unsupported response_format {fmt!r}; use one of {list(MIME)}", "response_format")
        if fmt in FFMPEG_ARGS and not shutil.which("ffmpeg"):
            return error(400, f"response_format {fmt!r} needs ffmpeg on the server (apt install ffmpeg); "
                              "wav and pcm are always available", "response_format")
        speed = float(body.get("speed", 1.0))
        if abs(speed - 1.0) > 1e-3:
            return error(400, "only speed=1.0 is supported by this model", "speed")
        voice = body.get("voice") or svc.tts.voice
        if isinstance(voice, dict):  # newer OpenAI SDKs allow {"id": ...}
            voice = voice.get("id", svc.tts.voice)
        if voice not in svc.tts.refs:
            if voice in OPENAI_VOICES:
                voice = svc.tts.voice  # OpenAI preset names map to the default voice
            else:
                return error(400, f"unknown voice {voice!r}; available: {list(svc.tts.refs)}", "voice")
        kw = {k: body[k] for k in ("top_k", "temperature") if k in body}
        kw["hf_filter"] = body.get("hf_filter", args.hf_filter)
        kw["presence_db"] = float(body.get("presence_db", args.presence_db))
        kw["interval"], kw["clause_interval"] = args.pause, args.clause_pause
        if kw["hf_filter"] not in ("off", "notch", "lowpass"):
            return error(400, "hf_filter must be off, notch or lowpass", "hf_filter")
        seed = body.get("seed")
        stream = (bool(body.get("stream", False)) or fmt == "pcm") and fmt != "flac"  # flac: one header, no chunks
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        def seg_iter():
            """Runs on the NPU thread; the caller holds `alock` for the whole request (voice / RNG state)."""
            if seed is not None:
                svc.tts.rng = np.random.default_rng(int(seed))
            yield from svc.tts.segments(text, voice=voice, **kw)

        if not stream:
            def run_all():
                parts, infos = [], []
                for a, i in seg_iter():
                    parts.append(a)
                    infos.append(i)
                return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), infos

            async with alock:
                audio, infos = await loop.run_in_executor(pool, run_all)
            if fmt == "wav":
                data = wav_bytes(audio)
            elif fmt == "pcm":
                data = to_int16(svc.resample(audio))
            else:
                data = await loop.run_in_executor(None, ffmpeg_encode, audio, fmt)
            dt = time.perf_counter() - t0
            log.info("speech %s %d chars -> %.2fs audio in %.2fs (%d segments)", fmt, len(text), len(audio) / R.SR,
                     dt, len(infos))
            return Response(content=data, media_type=MIME[fmt],
                            headers={"X-Audio-Seconds": f"{len(audio) / R.SR:.2f}", "X-Synthesis-Seconds": f"{dt:.2f}"})

        # streaming: one chunk per sentence, produced on the NPU thread
        it = seg_iter()

        def next_seg():
            try:
                return next(it)
            except StopIteration:
                return None

        async def body_iter():
            await alock.acquire()
            n, total = 0, 0.0
            try:
                if fmt == "wav":
                    yield wav_stream_header()
                while True:
                    r = await loop.run_in_executor(pool, next_seg)
                    if r is None:
                        break
                    a, _ = r
                    n, total = n + 1, total + len(a) / R.SR
                    if fmt == "wav":
                        yield to_int16(a)
                    elif fmt == "pcm":
                        yield to_int16(svc.resample(a))
                    else:  # mp3 / aac / opus: each sentence encoded separately (concatenated streams play back-to-back)
                        yield await loop.run_in_executor(None, ffmpeg_encode, a, fmt)
            finally:
                await loop.run_in_executor(pool, it.close)
                alock.release()  # also on client disconnect
                log.info("speech(stream) %s %d chars -> %.2fs audio in %.2fs (%d segments)", fmt, len(text), total,
                         time.perf_counter() - t0, n)

        return StreamingResponse(body_iter(), media_type=MIME[fmt])

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.environ.get("GSV_API_KEY"), help="require 'Authorization: Bearer KEY'")
    ap.add_argument("--backend", choices=["qnn", "ort"], default="qnn")
    ap.add_argument("--voices", default=os.path.join(R.HERE, "assets"), help="dir with ref_<voice>.npz (or one file)")
    ap.add_argument("--models-dir", default=os.path.join(R.HERE, "models"))
    ap.add_argument("--qnn-lib", default=os.environ.get("QNN_LIB", os.path.join(R.HERE, "qnn_libs")))
    ap.add_argument("--gsv-lib", default=os.environ.get("GSV_QNN_LIB"))
    ap.add_argument("--no-burst", action="store_true")
    ap.add_argument("--onnx-dir", default=os.path.join(R.HERE, "onnx"), help="ONNX graphs for --backend ort")
    ap.add_argument("--hf-filter", choices=["off", "notch", "lowpass"], default="off",
                    help="default post filter (request field hf_filter overrides): notch removes only the whine tones")
    ap.add_argument("--presence-db", type=float, default=0.0, help="default high-shelf boost above ~4 kHz")
    ap.add_argument("--pause", type=float, default=0.3, help="silence after a sentence end (s)")
    ap.add_argument("--clause-pause", type=float, default=0.15, help="silence where a long sentence was cut at a comma")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
