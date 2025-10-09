"""Chord, key, and tempo analysis for audio files using madmom."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import madmom

ChordSegment = Tuple[float, float, str]
AnalysisResult = Dict[str, object]


def detect_chords(audio_path: str | Path) -> List[ChordSegment]:
    """Return a list of (start, end, chord_label) tuples for the audio file."""
    feat_processor = madmom.features.chords.CNNChordFeatureProcessor()
    recog_processor = madmom.features.chords.CRFChordRecognitionProcessor()
    feats = feat_processor(str(Path(audio_path)))
    chords = recog_processor(feats)

    formatted_chords: List[ChordSegment] = []
    for start_time, end_time, chord_label in chords:
        if ":maj" in chord_label:
            chord_label = chord_label.replace(":maj", "")
        elif ":min" in chord_label:
            chord_label = chord_label.replace(":min", "m")
        formatted_chords.append((start_time, end_time, chord_label))

    return formatted_chords


def detect_key(audio_path: str | Path) -> str:
    """Return the detected key label for the audio file."""
    try:
        key_processor = madmom.features.key.CNNKeyRecognitionProcessor()
        key_prediction = key_processor(str(Path(audio_path)))
        key = madmom.features.key.key_prediction_to_label(key_prediction)
        return key
    except Exception:
        return "Unknown"


def detect_tempo(audio_path: str | Path) -> int:
    """Return the primary tempo in BPM for the audio file."""
    beat_processor = madmom.features.beats.RNNBeatProcessor()
    beats = beat_processor(str(Path(audio_path)))
    tempo_processor = madmom.features.tempo.TempoEstimationProcessor(fps=200)
    tempos = tempo_processor(beats)

    if tempos:
        tempo = tempos[0][0]
        while tempo < 70:
            tempo *= 2
        while tempo > 190:
            tempo /= 2
        return round(tempo)
    return 0


def analyze(audio_path: str | Path, *, rounding: int = 2) -> AnalysisResult:
    """Run the full analysis pipeline and return a JSON-serializable dict."""
    audio_path = Path(audio_path)
    chords = detect_chords(audio_path)

    return {
        "file": str(audio_path),
        "key": detect_key(audio_path),
        "tempo": detect_tempo(audio_path),
        "chords": [
            {
                "start": round(start, rounding),
                "end": round(end, rounding),
                "chord": chord,
            }
            for start, end, chord in chords
        ],
    }
