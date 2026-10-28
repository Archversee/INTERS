"""Spoken intent callouts -> timestamped annotations, offline with faster-whisper.

    pip install -e ".[voice]"
    sim2sop transcribe cockpit.wav data/raw/air/<run_id> --offset 12.5

`offset` is the telemetry time (source clock) at which the audio recording starts.
Each Whisper segment becomes one spoken annotation at the segment's start time.
"""

from __future__ import annotations

import json
from pathlib import Path

from sim2sop.schema import Annotation


def transcribe(audio: Path, offset_s: float = 0.0, model_size: str = "small.en",
               language: str = "en") -> list[Annotation]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError('Spoken annotations need faster-whisper: pip install -e ".[voice]"') from e
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, _ = model.transcribe(str(audio), language=language, vad_filter=True)
    return [Annotation(t=round(offset_s + s.start, 2), text=s.text.strip(), source="spoken")
            for s in segments if s.text.strip()]


def merge_into_run(run_dir: Path, notes: list[Annotation]) -> Path:
    """Add annotations to <run_dir>/annotations.json, keeping existing typed ones."""
    path = Path(run_dir) / "annotations.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    merged = existing + [n.model_dump(exclude_none=True) for n in notes]
    merged.sort(key=lambda a: a["t"])
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return path
