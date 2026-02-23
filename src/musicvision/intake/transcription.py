"""
Whisper transcription with word-level timestamps.

Uses transformers pipeline with openai/whisper-large-v3.
Loads on secondary GPU (3080 Ti) by default, unloads when done.

The word timestamps are critical for scene segmentation —
they determine where scenes can be split.
"""

from __future__ import annotations

import logging
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
    device: str = "cuda:1",
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


def align_lyrics_with_timestamps(
    provided_lyrics: str,
    transcription: TranscriptionResult,
) -> list[WordTimestamp]:
    """
    Align user-provided lyrics with Whisper's word timestamps.

    When the user provides their own lyrics (more accurate than Whisper),
    we still need timestamps. This does a best-effort alignment by
    matching words from the provided lyrics to the closest Whisper words.

    For v1, this is a simple sequential match. Could upgrade to DTW later.
    """
    provided_words = provided_lyrics.split()
    whisper_words = transcription.words

    if not whisper_words:
        log.warning("No Whisper timestamps available for alignment")
        return []

    aligned: list[WordTimestamp] = []
    w_idx = 0

    for p_word in provided_words:
        p_clean = p_word.strip(".,!?;:'\"()-").lower()

        # Find closest matching Whisper word
        best_idx = w_idx
        best_score = 0.0

        for i in range(w_idx, min(w_idx + 10, len(whisper_words))):
            w_clean = whisper_words[i].word.strip(".,!?;:'\"()-").lower()
            if w_clean == p_clean:
                best_idx = i
                best_score = 1.0
                break
            elif p_clean.startswith(w_clean) or w_clean.startswith(p_clean):
                if len(w_clean) / max(len(p_clean), 1) > best_score:
                    best_idx = i
                    best_score = len(w_clean) / max(len(p_clean), 1)

        if best_idx < len(whisper_words):
            aligned.append(WordTimestamp(
                word=p_word,
                start=whisper_words[best_idx].start,
                end=whisper_words[best_idx].end,
            ))
            w_idx = best_idx + 1
        else:
            # Past end of Whisper output — estimate timing
            last_end = aligned[-1].end if aligned else 0.0
            aligned.append(WordTimestamp(
                word=p_word,
                start=last_end,
                end=last_end + 0.3,
            ))

    return aligned
