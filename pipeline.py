"""PDF → structured overview pipeline. Streams events as a generator."""
import asyncio
import io
import json
import os
import re
import time
from typing import AsyncIterator

import fitz  # PyMuPDF

from llm import cached_system, chat, chat_stream

MAP_MODEL = os.getenv("MAP_MODEL", "google/gemini-2.5-flash")
REDUCE_MODEL = os.getenv("REDUCE_MODEL", "anthropic/claude-sonnet-4.5")

CHUNK_CHARS = 12000
SMALL_DOC_CHARS = 50000  # below this, skip map phase
MAP_CONCURRENCY = int(os.getenv("MAP_CONCURRENCY", "24"))
MAX_MAP_CHUNKS = 80  # cap parallel fan-out for huge PDFs

# Threshold for treating a page as "no extractable text" → candidate for OCR.
OCR_PAGE_MIN_CHARS = 20

MAP_SYSTEM = (
    "You are a precise reader. Given a portion of a larger document, produce a compact "
    "extract of its key ideas so another agent can later stitch a whole-document overview.\n"
    "Include: main claims, narrative beats, named concepts, notable evidence or examples.\n"
    "Keep specific terms, names, numbers when they carry meaning.\n"
    "Output plain bulleted prose under 180 words. No preamble, no headings."
)

REDUCE_SYSTEM = """You are producing a polished structured overview of a document from compressed notes.

OUTPUT: newline-delimited JSON. One JSON object per line. No markdown, no prose, no wrapping array.

Line 1 MUST be the meta line:
{"kind":"meta","category":"<ONE uppercase topical tag, e.g. 'PHILOSOPHY / LOGIC', 'TECHNOLOGY / AI', 'HISTORY / WARFARE', 'BUSINESS / STRATEGY'>"}

Subsequent lines, one section each:
{"kind":"section","subtitle":"<2 to 6 word Title Case heading>","summary":"<1 to 2 sentences, 20 to 40 words, present tense, no fluff>"}

Rules:
- Choose 4 to 12 sections — pick the count that best matches the document's natural structure. Do not pad.
- Each section covers a DISTINCT facet; no overlap, no restatements.
- Together sections MUST cover the document's key content — breadth over trivia.
- Subtitles are evocative and specific, not generic ("Introduction" / "Conclusion" are forbidden).
- Start streaming immediately from line 1. Never emit commentary, backticks, or empty lines."""


_ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)


def _strip_boilerplate(pages: list[str]) -> list[str]:
    """Drop lines that are clearly not substantive content:
      - Short lines that repeat on many pages (running headers/footers).
      - Pure page numbers (arabic or roman).
      - Common bibliographic noise like "bare URLs" / lone DOI lines is left
        to the LLM — only high-precision rules here.

    We're intentionally conservative: under-pruning is fine (LLM handles it);
    over-pruning costs real content. The threshold is occurring on ≥30% of
    pages AND being short (≤80 chars)."""
    n = len(pages)
    if n < 5:
        return pages

    # Build: line → set of page indices (dedup within a page).
    line_pages: dict[str, set[int]] = {}
    for i, p in enumerate(pages):
        seen: set[str] = set()
        for raw in p.splitlines():
            s = raw.strip()
            if 0 < len(s) <= 80 and s not in seen:
                seen.add(s)
                line_pages.setdefault(s, set()).add(i)

    threshold = max(3, int(n * 0.3))
    boilerplate = {s for s, pgs in line_pages.items() if len(pgs) >= threshold}

    def is_page_number(s: str) -> bool:
        if not s:
            return False
        if s.isdigit() and len(s) <= 4:
            return True
        if _ROMAN_RE.match(s) and len(s) <= 6:
            return True
        # "Page 12", "12 / 300", "- 12 -"
        if re.match(r"^[\-–—]?\s*(page\s+)?\d+\s*(/\s*\d+)?\s*[\-–—]?$", s, re.IGNORECASE):
            return True
        return False

    out: list[str] = []
    for p in pages:
        kept: list[str] = []
        for raw in p.splitlines():
            s = raw.strip()
            if not s:
                kept.append(raw)
                continue
            if s in boilerplate:
                continue
            if is_page_number(s):
                continue
            kept.append(raw)
        out.append("\n".join(kept))
    return out


def extract_pages(pdf_path: str) -> tuple[list[str], int, int, list[int], int]:
    """Extract text per page. Returns
      (pages, page_count, word_count, ocr_indices, chars_stripped)
    where `chars_stripped` reports how much boilerplate was removed so the
    pipeline can surface the saving to the user."""
    doc = fitz.open(pdf_path)
    try:
        raw_pages = [p.get_text() or "" for p in doc]
    finally:
        doc.close()
    before = sum(len(p) for p in raw_pages)
    pages = _strip_boilerplate(raw_pages)
    after = sum(len(p) for p in pages)
    ocr_indices = [i for i, p in enumerate(pages) if len(p.strip()) < OCR_PAGE_MIN_CHARS]
    word_count = sum(len(p.split()) for p in pages)
    return pages, len(pages), word_count, ocr_indices, max(0, before - after)


def _parse_psm(config: str) -> int:
    m = re.search(r"--psm\s+(\d+)", config)
    return int(m.group(1)) if m else 6


def _resolve_tessdata() -> str:
    """Find the tessdata directory. tesserocr's built-in default is often './'
    on macOS/homebrew builds, so we look it up explicitly."""
    env = os.environ.get("TESSDATA_PREFIX") or os.environ.get("TESSDATA_DIR")
    candidates = [env] if env else []
    candidates += [
        "/opt/homebrew/share/tessdata",          # macOS Apple Silicon (brew)
        "/usr/local/share/tessdata",             # macOS Intel (brew)
        "/opt/local/share/tessdata",             # macports
        "/usr/share/tessdata",                   # linux (some distros)
        "/usr/share/tesseract-ocr/5/tessdata",   # debian/ubuntu (tess 5)
        "/usr/share/tesseract-ocr/4.00/tessdata",  # debian/ubuntu (tess 4)
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    # Last resort: ask tesserocr, though this is what was failing for the user.
    try:
        import tesserocr
        path, _ = tesserocr.get_languages()
        if path and os.path.isdir(path):
            return path
    except Exception:
        pass
    return ""


def _detect_ocr_backend() -> tuple[str, str]:
    """Returns (backend, tessdata_path). 'tesserocr' (in-process, ~3-10× faster)
    preferred; 'pytesseract' (subprocess) fallback. We actually construct a
    throwaway PyTessBaseAPI to verify tessdata is reachable — if not, silently
    demote so the user still gets the parallel pytesseract path."""
    if os.getenv("OCR_BACKEND") == "pytesseract":
        return "pytesseract", ""
    try:
        import tesserocr
    except Exception:
        return "pytesseract", ""
    tessdata = _resolve_tessdata()
    lang = os.getenv("OCR_LANG", "eng").split("+")[0]
    try:
        api = tesserocr.PyTessBaseAPI(path=tessdata, lang=lang)
        api.End()
        return "tesserocr", tessdata
    except Exception:
        return "pytesseract", ""


def _render_gray(page, dpi: int):
    """Render page as an 8-bit grayscale PIL image without PNG round-trip."""
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def _tesserocr_text_filtered(api, min_conf: float) -> str:
    """Walk tesserocr's ResultIterator, dropping words below `min_conf`. Keeps
    paragraph/line breaks so chunking downstream still sees structure."""
    import tesserocr
    RIL = tesserocr.RIL
    api.Recognize()
    it = api.GetIterator()
    if it is None:
        return api.GetUTF8Text() or ""
    out_lines: list[str] = []
    current: list[str] = []
    while True:
        try:
            word = it.GetUTF8Text(RIL.WORD)
            conf = it.Confidence(RIL.WORD)
        except RuntimeError:
            # iterator exhausted mid-step on some tesseract versions
            break
        if word and conf >= min_conf:
            current.append(word)
        end_of_line = it.IsAtFinalElement(RIL.TEXTLINE, RIL.WORD)
        end_of_para = it.IsAtFinalElement(RIL.PARA, RIL.WORD)
        if end_of_line and current:
            out_lines.append(" ".join(current))
            current = []
        if end_of_para:
            out_lines.append("")  # blank line between paragraphs
        if not it.Next(RIL.WORD):
            break
    if current:
        out_lines.append(" ".join(current))
    return "\n".join(out_lines).strip()


def _pytesseract_text_filtered(img, lang: str, config: str, min_conf: float) -> str:
    """Same filtering for the pytesseract fallback, via image_to_data."""
    import pytesseract
    data = pytesseract.image_to_data(
        img, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    lines: dict[tuple, list[str]] = {}
    for i, word in enumerate(data["text"]):
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not word or not word.strip() or conf < min_conf:
            continue
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
    return "\n".join(" ".join(ws) for ws in lines.values()).strip()


async def ocr_pages_streaming(
    pdf_path: str,
    page_indices: list[int],
) -> AsyncIterator[dict]:
    """Parallel OCR with streaming progress. Yields:
      {"kind":"ocr_progress", "done":N, "total":M, "eta_s":..., "backend":...}
      {"kind":"ocr_result", "pages": {idx: text}}

    Work is partitioned across threads (one batch per worker) so the PDF is
    opened — and the tesseract model loaded — exactly once per worker rather
    than once per page.
    """
    lang = os.getenv("OCR_LANG", "eng")
    dpi = int(os.getenv("OCR_DPI", "200"))
    config = os.getenv("OCR_CONFIG", "--oem 1 --psm 6")
    min_conf = float(os.getenv("OCR_MIN_CONF", "55"))
    backend, tessdata = _detect_ocr_backend()
    n_workers = min(
        int(os.getenv("OCR_WORKERS", str(os.cpu_count() or 4))),
        max(1, len(page_indices)),
    )
    # Round-robin partition keeps worker loads balanced as pages complete.
    batches = [page_indices[i::n_workers] for i in range(n_workers)]
    batches = [b for b in batches if b]

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker(batch: list[int]):
        try:
            doc = fitz.open(pdf_path)
            api = None
            try:
                if backend == "tesserocr":
                    import tesserocr
                    api = tesserocr.PyTessBaseAPI(
                        path=tessdata,
                        lang=lang,
                        psm=_parse_psm(config),
                        oem=tesserocr.OEM.LSTM_ONLY,
                    )
                else:
                    import pytesseract  # noqa
                for i in batch:
                    img = _render_gray(doc[i], dpi)
                    if backend == "tesserocr":
                        api.SetImage(img)
                        text = _tesserocr_text_filtered(api, min_conf)
                    else:
                        text = _pytesseract_text_filtered(img, lang, config, min_conf)
                    loop.call_soon_threadsafe(queue.put_nowait, ("page", (i, text)))
            finally:
                if api is not None:
                    api.End()
                doc.close()
        except BaseException as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", e))

    worker_tasks = [asyncio.create_task(asyncio.to_thread(worker, b)) for b in batches]

    async def sentinel():
        try:
            await asyncio.gather(*worker_tasks)
        finally:
            await queue.put(("done", None))

    sentinel_task = asyncio.create_task(sentinel())

    results: dict[int, str] = {}
    total = len(page_indices)
    done = 0
    started = time.monotonic()
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "error":
                raise payload
            if kind == "done":
                break
            idx, text = payload
            results[idx] = text
            done += 1
            elapsed = time.monotonic() - started
            eta_s = int(elapsed / done * (total - done)) if done else 0
            yield {
                "kind": "ocr_progress",
                "done": done,
                "total": total,
                "eta_s": eta_s,
                "backend": backend,
                "workers": n_workers,
            }
    except BaseException:
        for t in worker_tasks:
            t.cancel()
        sentinel_task.cancel()
        raise
    finally:
        await asyncio.gather(sentinel_task, return_exceptions=True)
    yield {"kind": "ocr_result", "pages": results}


def chunk_pages(pages: list[str], target: int = CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for p in pages:
        if not p.strip():
            continue
        if buf and len(buf) + len(p) > target:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


async def _map_chunk(text: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        return await chat(
            MAP_MODEL,
            [
                {"role": "system", "content": cached_system(MAP_SYSTEM)},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )


async def _parse_ndjson_stream(token_stream: AsyncIterator[str]) -> AsyncIterator[dict]:
    buf = ""
    async for tok in token_stream:
        buf += tok
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip().lstrip("\ufeff")
            if not line or line.startswith("```"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    tail = buf.strip()
    if tail:
        try:
            yield json.loads(tail)
        except json.JSONDecodeError:
            pass


async def process(pdf_path: str, filename: str) -> AsyncIterator[dict]:
    """Yield overview events: document, status, meta, section, done, error."""
    # Run the (potentially slow, blocking) extraction off the event loop.
    pages, page_count, word_count, ocr_indices, stripped = await asyncio.to_thread(extract_pages, pdf_path)
    if stripped > 2000:
        yield {
            "kind": "status",
            "message": f"Stripped {stripped // 1000}k chars of headers/footers/page numbers.",
        }

    # OCR fallback: if a meaningful share of pages came back empty, assume scanned
    # and fill them in. Emit status first so the user isn't staring at a blank UI.
    if ocr_indices and len(ocr_indices) >= max(1, page_count // 2):
        yield {
            "kind": "status",
            "message": f"Scanned PDF — starting OCR on {len(ocr_indices)} pages…",
        }
        ocr_map: dict[int, str] = {}
        last_emit = 0.0
        try:
            async for ev in ocr_pages_streaming(pdf_path, ocr_indices):
                if ev["kind"] == "ocr_progress":
                    now = time.monotonic()
                    # throttle status pushes to ~3/s (plus the final 100% frame)
                    if now - last_emit >= 0.33 or ev["done"] == ev["total"]:
                        last_emit = now
                        eta = ev["eta_s"]
                        eta_txt = f"~{eta // 60}m{eta % 60:02d}s left" if eta >= 60 else f"~{eta}s left"
                        yield {
                            "kind": "status",
                            "message": (
                                f"OCR [{ev['backend']}, {ev['workers']}w] "
                                f"{ev['done']}/{ev['total']} · {eta_txt}"
                            ),
                        }
                else:
                    ocr_map = ev["pages"]
        except ImportError:
            yield {
                "kind": "error",
                "message": "OCR dependencies missing. Install pytesseract + Pillow and the tesseract binary (e.g. `brew install tesseract`).",
            }
            return
        except Exception as e:
            yield {"kind": "error", "message": f"OCR failed: {e}"}
            return
        for i, text in ocr_map.items():
            if text.strip():
                pages[i] = text
        word_count = sum(len(p.split()) for p in pages)

    total_chars = sum(len(p) for p in pages)
    if total_chars == 0:
        yield {
            "kind": "error",
            "message": "No text could be extracted, even with OCR. Check PDF quality or set OCR_LANG for non-English scans.",
        }
        return

    read_minutes = max(1, round(word_count / 220))
    yield {
        "kind": "document",
        "filename": filename,
        "page_count": page_count,
        "word_count": word_count,
        "read_minutes": read_minutes,
    }

    # Pick a chunk size that keeps fan-out under MAX_MAP_CHUNKS even for huge PDFs.
    target_chars = max(CHUNK_CHARS, total_chars // MAX_MAP_CHUNKS + 1)
    chunks = chunk_pages(pages, target=target_chars)

    if total_chars <= SMALL_DOC_CHARS or len(chunks) <= 1:
        notes = "\n\n".join(chunks) if chunks else ""
        yield {"kind": "status", "message": "Drafting overview…"}
    else:
        yield {
            "kind": "status",
            "message": f"Distilling {len(chunks)} segments ({MAP_CONCURRENCY} in flight)…",
        }
        sem = asyncio.Semaphore(MAP_CONCURRENCY)

        async def run(index: int, text: str) -> tuple[int, str]:
            return index, await _map_chunk(text, sem)

        tasks = [asyncio.create_task(run(i, c)) for i, c in enumerate(chunks)]
        results_map: dict[int, str] = {}
        started = time.monotonic()
        last_emit = 0.0
        try:
            for coro in asyncio.as_completed(tasks):
                i, r = await coro
                results_map[i] = r
                done = len(results_map)
                total = len(chunks)
                now = time.monotonic()
                if now - last_emit >= 0.33 or done == total:
                    last_emit = now
                    elapsed = now - started
                    eta = int(elapsed / done * (total - done)) if done else 0
                    eta_txt = f"~{eta // 60}m{eta % 60:02d}s left" if eta >= 60 else f"~{eta}s left"
                    yield {
                        "kind": "status",
                        "message": f"Distilling {done}/{total} segments · {eta_txt}",
                    }
        except BaseException:
            for t in tasks:
                t.cancel()
            raise
        results = [results_map[i] for i in range(len(chunks))]
        notes = "\n\n".join(f"## Part {i + 1}\n{r.strip()}" for i, r in enumerate(results))
        yield {"kind": "status", "message": "Composing structured overview…"}

    reduce_user = (
        f"Document filename: {filename}\n"
        f"Total pages: {page_count}\n\n"
        f"Compressed notes:\n\n{notes}"
    )
    stream = chat_stream(
        REDUCE_MODEL,
        [
            {"role": "system", "content": cached_system(REDUCE_SYSTEM)},
            {"role": "user", "content": reduce_user},
        ],
        temperature=0.4,
    )

    section_count = 0
    async for obj in _parse_ndjson_stream(stream):
        kind = obj.get("kind")
        if kind == "section":
            section_count += 1
            if section_count > 15:
                break
            yield obj
        elif kind == "meta":
            yield obj

    yield {"kind": "done", "section_count": section_count}
