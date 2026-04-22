# PDF Overview

Drop in a PDF, stream back a 4–12-section structured overview. Handles text PDFs and scanned PDFs. Adaptive map–reduce against [OpenRouter](https://openrouter.ai/) — cheap fast model for bulk reading, stronger model for the final composition.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# scanned-PDF support (required if any input is a scan):
brew install tesseract leptonica         # macOS; apt install tesseract-ocr libleptonica-dev on debian
pip install tesserocr                    # optional but recommended — 3-10× faster OCR
```

`tesserocr` is auto-detected at runtime and used when present; otherwise the pipeline falls back to `pytesseract` (subprocess-per-page, already installed via requirements.txt). Status line shows which one is active: `OCR [tesserocr, 8w] …` vs `OCR [pytesseract, 8w] …`.

## Configure

```bash
cp .env.example .env  # fill in OPENROUTER_API_KEY
```

| Var | Default | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | required |
| `MAP_MODEL` | `google/gemini-2.5-flash-lite-preview-09-2025:nitro` | cheap fan-out reader |
| `REDUCE_MODEL` | `anthropic/claude-haiku-4.5:nitro` | final composer (Anthropic ⇒ prompt caching kicks in) |
| `MAP_CONCURRENCY` | `24` | in-flight map requests; lower if rate-limited |
| `OCR_LANG` | `eng` | e.g. `eng+chi_sim` |
| `OCR_DPI` | `200` | bump to 300 for tiny fonts |
| `OCR_WORKERS` | CPU count | parallel OCR workers |
| `OCR_MIN_CONF` | `55` | drop words below this tesseract confidence (0–100) |
| `OCR_BACKEND` | auto | force `tesserocr` / `pytesseract` |
| `TESSDATA_PREFIX` | auto | override if tesserocr can't find tessdata |

## Run

```bash
python app.py   # http://localhost:8000
```

---

## Pipeline

```
upload  ──►  temp file (streamed chunk-by-chunk, no multipart parse)
             │
             ▼
  PyMuPDF extract  ──►  boilerplate strip
             │           (cross-page dedup of repeated short lines
             │            + page-number heuristics)
             ▼
  ≥50% pages empty?
       ├── yes ─► OCR fan-out (tesserocr / pytesseract)
       │          · grayscale pixmap → PIL.frombytes (no PNG round-trip)
       │          · round-robin page batches, one PyTessBaseAPI per worker
       │          · per-word confidence filter
       │          · streaming progress with ETA
       ▼
  adaptive chunking  (target 12k chars, hard cap 80 chunks → chunk size grows with doc)
             │
             ▼
  small doc (<50k chars)? ─ yes ─► skip MAP, hand raw text to reducer
             │ no
             ▼
  MAP fan-out (N chunks × MAP_MODEL, semaphore=MAP_CONCURRENCY)
             │     system prompt marked with cache_control
             ▼
  REDUCE (REDUCE_MODEL, streaming NDJSON)
             │     line 1: {"kind":"meta","category":"…"}
             │     line N: {"kind":"section","subtitle":"…","summary":"…"}
             ▼
  SSE  ──►  browser appends a row per completed NDJSON line
```

**NDJSON, not JSON-array.** A JSON array is only valid once closed — you can't render it progressively. NDJSON commits at every `\n`, so the UI can paint a section the moment the model finishes generating its line.

**Why map–reduce + two tiers.** The cheap model sees the bulk of the raw text (big input, ~180-word output per chunk). The strong model only sees the compressed notes (~7–16k tokens) regardless of source document size.

**Boilerplate strip.** Before chunking we build `line → {page indices}` across the whole doc; any short line appearing on ≥30% of pages is a running header/footer and gets dropped, alongside page-number patterns (`12`, `XII`, `Page 12 / 300`, `— 12 —`). Conservative by design — under-pruning costs the LLM a few tokens; over-pruning costs real content.

---

## Performance levers (what's actually doing work)

- **Streaming upload → temp file.** `request.stream()` pipes the body straight to `NamedTemporaryFile`, bypassing FastAPI's multipart parser and its default size limits. Server RAM stays in the low tens of MB regardless of a 3 MB or 300 MB PDF.
- **Shared httpx AsyncClient, HTTP/2 multiplexing, startup warmup.** One pooled client for the whole process; a HEAD request to OpenRouter fires on FastAPI startup so the first upload skips the ~150ms TLS handshake. Without this, an 80-chunk map fan-out eats 10+ seconds in cold handshakes.
- **Prompt caching via OpenRouter `cache_control` passthrough.** Both `MAP_SYSTEM` and `REDUCE_SYSTEM` are wrapped in `cache_control: ephemeral` blocks. On Anthropic models (current `REDUCE_MODEL`) the system prefix is cached for 5 minutes → ~90% input-token discount on the system portion across repeated runs. Ignored harmlessly by non-Anthropic providers.
- **MAP concurrency 24 (up from 8).** Gemini Flash-Lite:nitro handles it without rate-limit hits; 80-chunk fan-out goes from ~10 waves to ~3.
- **Parallel OCR with per-worker state.** Page indices are round-robin partitioned across `OCR_WORKERS` threads. Each worker opens the PDF once and (for tesserocr) loads one LSTM model — both amortized across its whole batch instead of paid per-page.
- **tesserocr fast path.** The C++ API stays in-process, eliminating pytesseract's ~50–150ms subprocess fork per page. 634-page OCR drops from minutes to ~1–2 minutes on an 8-core laptop.
- **Render tricks.** Grayscale rendering (`csGRAY`) cuts 3× data vs RGB; we skip the PNG round-trip entirely by feeding `pix.samples` straight into `Image.frombytes("L", …)`.
- **OCR confidence filter.** `ResultIterator` walks tesserocr output word-by-word; anything below `OCR_MIN_CONF` is dropped before text even leaves the worker. 5–15% fewer input tokens on scanned documents, plus the reducer stops hallucinating around garbage strings.
- **Adaptive chunking, bounded fan-out.** `MAX_MAP_CHUNKS=80` means a 1000-page book uses bigger chunks, not more of them. Worst-case concurrency is predictable.
- **Time-to-first-section.** Bounded by one MAP wave + REDUCE prefill. Because the reducer streams NDJSON, the user sees section 1 while N..last are still generating.
- **Throttled progress events.** OCR and map fan-outs push `done/total · ~Xs left` status at max 3 Hz; the UI gets steady feedback without flooding SSE.

---

## Files

```
app.py         FastAPI app — streaming upload, SSE out, warmup on startup
pipeline.py    extract → boilerplate strip → OCR fallback → map-reduce
llm.py         pooled httpx+HTTP/2 OpenRouter client, cache_control helper, warmup()
index.html     single-page frontend — fetch + SSE reader + progressive table
```

## Scope

MVP. No auth, no persistence, no queue; each upload is processed in-memory (plus one temp file) and the result is ephemeral. Swap in a job queue and storage layer before shipping.
