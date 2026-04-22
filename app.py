"""FastAPI entrypoint. Serves the SPA and streams the overview via SSE.

Large uploads (hundreds of MB) are streamed straight to a temp file on disk
instead of going through multipart parsing + in-memory buffering.
"""
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from llm import warmup as llm_warmup
from pipeline import process

load_dotenv()

ROOT = Path(__file__).parent
app = FastAPI(title="PDF Overview")


@app.on_event("startup")
async def _warm_llm_pool() -> None:
    # Pre-establish TLS/HTTP2 to OpenRouter so first upload doesn't pay handshake.
    import asyncio as _asyncio
    _asyncio.create_task(llm_warmup())


@app.get("/")
async def index():
    return FileResponse(ROOT / "index.html")


@app.post("/api/process")
async def process_endpoint(request: Request):
    raw_name = request.headers.get("x-filename") or "document.pdf"
    filename = unquote(raw_name)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")
    stem = Path(filename).stem or "document"

    # Stream the raw body directly to a temp file — never load the full PDF in RAM.
    tmp = tempfile.NamedTemporaryFile(prefix="pdfov_", suffix=".pdf", delete=False)
    tmp_path = tmp.name
    total = 0
    try:
        async for chunk in request.stream():
            if chunk:
                tmp.write(chunk)
                total += len(chunk)
    finally:
        tmp.close()

    if total == 0:
        _safe_unlink(tmp_path)
        raise HTTPException(400, "Empty request body")

    async def gen():
        try:
            async for event in process(tmp_path, stem):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'kind': 'error', 'message': str(e)})}\n\n"
        finally:
            _safe_unlink(tmp_path)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        # keep the upload connection alive for slow networks on big files
        timeout_keep_alive=600,
    )
