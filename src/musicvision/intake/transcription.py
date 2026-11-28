"""
Whisper transcription with word-level timestamps.

Uses transformers pipeline with openai/whisper-large-v3.
Loads on the secondary GPU by default, unloads when done.

The word timestamps are critical for scene segmentation —
they determine where scenes can be split.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    word: str
    start: float   # seconds
    end: float     # seconds


@dataclass
class TranscriptionResult:
    text: str
    words: list[WordTimestamp] = field(default_factory=list)
    language: str = "en"


def transcribe(
    audio_path: Path,
    model_name: str = "openai/whisper-large-v3",
    device: str = "cuda:0",
    language: str | None = None,
) -> TranscriptionResult:
    """
    Transcribe audio with word-level timestamps using Whisper.

    Args:
        audio_path: Path to audio file (WAV recommended, any format accepted)
        model_name: HuggingFace model ID or local path
        device: Device for inference (default: secondary GPU)
        language: Force language (None = auto-detect)

    Returns:
        TranscriptionResult with full text and per-word timestamps
    """
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    log.info("Loading Whisper model: %s on %s", model_name, device)

    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_name)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    log.info("Transcribing: %s", audio_path.name)

    result = pipe(
        str(audio_path),
        return_timestamps="word",
        generate_kwargs=generate_kwargs,
    )

    # Extract word-level timestamps from result
    words: list[WordTimestamp] = []
    chunks = result.get("chunks", [])

    for chunk in chunks:
        timestamp = chunk.get("timestamp", (None, None))
        if timestamp and timestamp[0] is not None:
            words.append(WordTimestamp(
                word=chunk["text"].strip(),
                start=float(timestamp[0]),
                end=float(timestamp[1]) if timestamp[1] is not None else float(timestamp[0]) + 0.1,
            ))

    detected_language = "en"  # Whisper pipeline doesn't expose detected language easily

    log.info(
        "Transcription complete: %d words, %.1fs duration",
        len(words),
        words[-1].end if words else 0.0,
    )

    # Unload model to free VRAM
    del pipe, model, processor
    if "cuda" in device:
        torch.cuda.empty_cache()
    log.info("Whisper model unloaded")

    return TranscriptionResult(
        text=result["text"].strip(),
        words=words,
        language=detected_language,
    )


def load_lyrics_file(lyrics_path: Path) -> str:
    """Load lyrics from a plain text file."""
    return lyrics_path.read_text(encoding="utf-8").strip()


_SECTION_TAG_RE = re.compile(r"\[[^\]]*\]")


def _norm_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.replace("’", "'").replace("‘", "'").lower())


def align_lyrics_with_timestamps(
    provided_lyrics: str,
    transcription: TranscriptionResult,
) -> list[WordTimestamp]:
    """
    Align user-provided lyrics with Whisper's word timestamps.

    Section tags such as ``[Verse 1]`` / ``[Chorus]`` are not sung and are
    dropped. The provided and transcribed word sequences are aligned with
    difflib on normalised words: matched words and one-for-one mishearings take
    Whisper's timing, uneven replacements ("and miss" heard as "endless") are
    spread over the span Whisper heard, and provided words Whisper missed
    entirely are spaced evenly between their matched neighbours. An unmatched word never
    consumes a Whisper word, so one mismatch cannot shift the rest of the song
    (the old greedy matcher did exactly that; #30).
    """
    import difflib

    provided_words = _SECTION_TAG_RE.sub(" ", provided_lyrics).split()
    whisper_words = transcription.words

    if not whisper_words:
        log.warning("No Whisper timestamps available for alignment")
        return []
    if not provided_words:
        return []

    p_norm = [_norm_word(w) for w in provided_words]
    w_norm = [_norm_word(w.word) for w in whisper_words]
    timing: list[tuple[float, float] | None] = [None] * len(provided_words)
    matcher = difflib.SequenceMatcher(None, p_norm, w_norm, autojunk=False)
    matched = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            # Same words, or one-for-one mishearings ("goodbye" heard as "goodby").
            for k in range(i2 - i1):
                ww = whisper_words[j1 + k]
                timing[i1 + k] = (ww.start, ww.end)
            matched += (i2 - i1) if tag == "equal" else 0
        elif tag == "replace":
            # e.g. "and miss" heard as "endless": spread over the span Whisper heard.
            lo, hi = whisper_words[j1].start, whisper_words[j2 - 1].end
            step = (hi - lo) / (i2 - i1)
            for k in range(i2 - i1):
                timing[i1 + k] = (lo + step * k, lo + step * (k + 1))
        # "delete": provided words Whisper missed (filled below); "insert": extra Whisper words (ignored)

    # Fill runs of unmatched words between their matched neighbours.
    song_end = whisper_words[-1].end
    i = 0
    while i < len(timing):
        if timing[i] is not None:
            i += 1
            continue
        j = i
        while j < len(timing) and timing[j] is None:
            j += 1
        lo = timing[i - 1][1] if i > 0 else whisper_words[0].start
        hi = timing[j][0] if j < len(timing) else max(song_end, lo + 0.3 * (j - i))
        step = max(hi - lo, 0.0) / (j - i)
        for k in range(i, j):
            timing[k] = (lo + step * (k - i), lo + step * (k - i + 1))
        i = j

    log.info("Aligned provided lyrics: %d/%d words matched to Whisper timings", matched, len(provided_words))
    return [WordTimestamp(word=w, start=t[0], end=t[1]) for w, t in zip(provided_words, timing)]
