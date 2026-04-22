# PDF Overview

Drop in a PDF of any length — get back a polished, structured overview of 4–12 sections, streamed live into the UI as each one is ready.

Long documents are distilled with an adaptive **map–reduce** pipeline routed through [OpenRouter](https://openrouter.ai/), so you can mix a cheap model for bulk reading and a stronger one for the final composition.

---

## Usage

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in:

| Var                  | Default                       | Purpose                                |
| -------------------- | ----------------------------- | -------------------------------------- |
| `OPENROUTER_API_KEY` | —                             | required                               |
| `MAP_MODEL`          | `google/gemini-2.5-flash`     | cheap/fast — digests raw chunks        |
| `REDUCE_MODEL`       | `anthropic/claude-sonnet-4.5` | stronger — writes the final overview   |

Any OpenRouter-supported model slug works.

### 3. Run

```bash
python app.py
```

Open `http://localhost:8000`, drag a PDF onto the page, and watch the sections stream in.

---

## How it works

```
  PDF upload
      │
      │  streamed directly to a temp file on disk
      ▼
  PyMuPDF text extraction  (off the event loop)
      │
      ▼
  ┌───────────────────────────────────────────┐
  │ small doc (< 50k chars)?                  │
  │   → skip map, hand raw text to reducer    │
  │ large doc?                                │
  │   → split into N chunks (N ≤ 80)          │
  │   → MAP (parallel, cheap model):          │
  │       each chunk → compact notes          │
  │   → merge notes                           │
  └───────────────────────────────────────────┘
      │
      ▼
  REDUCE (strong model, STREAMING)
      │
      │  emits newline-delimited JSON:
      │    line 1 → {"kind":"meta","category":"..."}
      │    line N → {"kind":"section","subtitle":"...","summary":"..."}
      │
      ▼
  Server parses each completed line
      │
      ▼
  SSE → browser appends a row the moment it arrives
```

**Why NDJSON, not a JSON array.** A JSON array is only valid once it's closed, so you can't render it progressively. NDJSON commits at every newline — a perfect fit for token-by-token LLM streaming and a progressive UI.

**Why map–reduce with two model tiers.** The cheap model sees the bulk of the raw text (inputs are large, outputs are ~180 words per chunk). The strong model only sees the compressed notes plus writes the final structured output. This keeps quality high where it matters and cost low where it doesn't.

**Why stream the upload to disk.** A raw `POST /api/process` body is piped chunk-by-chunk into `NamedTemporaryFile`, bypassing multipart parsing and its default size limits. PyMuPDF then opens from path, reading pages lazily — so server RAM stays proportional to the extracted text (usually a few MB) regardless of whether the PDF is 3 MB or 300 MB.

---

## Efficiency & performance

**Cost (typical).** For a 100-page book (~300k chars of text) with `gemini-2.5-flash` + `claude-sonnet-4.5`:

| Stage   | Tokens in | Tokens out | Who       |
| ------- | --------- | ---------- | --------- |
| Map ×25 | ~75k      | ~5k        | flash     |
| Reduce  | ~7k       | ~0.8k      | sonnet    |

Flash handles the heavy lifting; Sonnet only sees ~7k tokens regardless of document size. Scaling from a 10-page paper to a 500-page book roughly doubles the flash spend while the Sonnet call stays flat.

**Latency.**

* **Time-to-first-section** is bounded by *one* map round (parallel) + reducer prefill. Because the reducer streams NDJSON, the user sees section 1 while sections 2..N are still being generated.
* **Upload** never blocks on multipart parsing; PyMuPDF extraction runs in a threadpool via `asyncio.to_thread` so the SSE event loop stays responsive.
* Map fan-out is capped at **80 chunks** via adaptive chunk sizing — a 1,000-page PDF uses bigger chunks, not more of them, so a worst-case doc still finishes with bounded concurrency.

**Memory.** Independent of PDF size: the body streams to disk, PyMuPDF reads pages on demand, chunks are plain strings in a list. A 300 MB scan-heavy PDF peaks in the low tens of MB on the server.

**Robustness.**

* Scanned PDFs with zero extractable text surface a clear client-side error (OCR is out of scope for this MVP).
* Temp file is always cleaned up on success, error, or client disconnect.
* The reducer is hard-capped at 15 sections server-side even if the model tries to produce more.

---

## Files

```
app.py         # FastAPI app: serves the SPA, streams the overview via SSE
pipeline.py    # PDF extraction + adaptive map-reduce + NDJSON parsing
llm.py         # minimal async OpenRouter client (chat + chat_stream)
index.html     # single-page frontend: fetch + SSE reader + progressive table
```

## Scope

This is an MVP. No persistence, no auth, no OCR, no queue. Each upload is processed in-memory (plus one temp file) and the result is ephemeral — reload the page and the overview is gone. Good enough to demo the idea end-to-end; swap in a real storage layer and job queue before shipping.
