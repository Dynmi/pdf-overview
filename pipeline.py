"""PDF → structured overview pipeline. Streams events as a generator."""
import asyncio
import json
import os
from typing import AsyncIterator

import fitz  # PyMuPDF

from llm import chat, chat_stream

MAP_MODEL = os.getenv("MAP_MODEL", "google/gemini-2.5-flash")
REDUCE_MODEL = os.getenv("REDUCE_MODEL", "anthropic/claude-sonnet-4.5")

CHUNK_CHARS = 12000
SMALL_DOC_CHARS = 50000  # below this, skip map phase
MAP_CONCURRENCY = 8
MAX_MAP_CHUNKS = 80  # cap parallel fan-out for huge PDFs

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


def extract_pages(pdf_path: str) -> tuple[list[str], int, int]:
    doc = fitz.open(pdf_path)
    try:
        pages = [p.get_text() or "" for p in doc]
    finally:
        doc.close()
    word_count = sum(len(p.split()) for p in pages)
    return pages, len(pages), word_count


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
                {"role": "system", "content": MAP_SYSTEM},
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
    pages, page_count, word_count = await asyncio.to_thread(extract_pages, pdf_path)
    total_chars = sum(len(p) for p in pages)
    if total_chars == 0:
        yield {
            "kind": "error",
            "message": "No extractable text (possibly a scanned PDF). OCR is out of scope for this MVP.",
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
        yield {"kind": "status", "message": f"Distilling {len(chunks)} segments in parallel…"}
        sem = asyncio.Semaphore(MAP_CONCURRENCY)
        results = await asyncio.gather(*[_map_chunk(c, sem) for c in chunks])
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
            {"role": "system", "content": REDUCE_SYSTEM},
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
