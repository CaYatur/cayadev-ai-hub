"""Audio and video for the agent.

Called by the Rust `transcribe` and `video_info` tools. Two subcommands:

    media.py transcribe <file> [language]
    media.py video <file> [frame_count]

`transcribe` uses faster-whisper on CPU with int8 weights — the GPU is already
holding a language model, and taking VRAM from it to save a few seconds of
transcription is a bad trade.

`video` returns metadata plus paths to evenly-spaced extracted frames, which the
Rust side hands to a vision model. Sending frames rather than the file keeps
video support working with any vision model instead of needing one that accepts
video.
"""
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

# Longer than this and a local transcription stops being interactive.
MAX_SECONDS = 45 * 60
DEFAULT_FRAMES = 4


def fail(message):
    print(message, file=sys.stderr)
    return 1


def probe(path):
    """Container metadata, via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size,format_name",
            "-show_entries", "stream=codec_type,codec_name,width,height,channels,sample_rate",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return json.loads(result.stdout)


def describe(meta):
    fmt = meta.get("format", {})
    duration = float(fmt.get("duration", 0) or 0)
    lines = [
        f"duration: {duration:.1f}s",
        f"container: {fmt.get('format_name', 'unknown')}",
    ]
    for stream in meta.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video":
            lines.append(
                f"video: {stream.get('codec_name')} "
                f"{stream.get('width')}x{stream.get('height')}"
            )
        elif kind == "audio":
            lines.append(
                f"audio: {stream.get('codec_name')} "
                f"{stream.get('channels')}ch {stream.get('sample_rate')}Hz"
            )
    return duration, "\n".join(lines)


def transcribe(path, language=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return fail("transcription needs faster-whisper: pip install faster-whisper")

    try:
        duration, summary = describe(probe(path))
    except Exception as exc:  # noqa: BLE001
        return fail(f"cannot read {path.name}: {exc}")

    if duration > MAX_SECONDS:
        return fail(
            f"{path.name} is {duration / 60:.0f} minutes long; "
            f"the limit is {MAX_SECONDS // 60}. Split it first."
        )

    # CPU int8: the GPU is busy holding the language model.
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), language=language, vad_filter=True)

    print(summary)
    print(f"language: {info.language} ({info.language_probability:.0%} confidence)")
    print()
    for segment in segments:
        print(f"[{segment.start:6.1f}s] {segment.text.strip()}")
    return 0


def video(path, frames=DEFAULT_FRAMES):
    try:
        duration, summary = describe(probe(path))
    except Exception as exc:  # noqa: BLE001
        return fail(f"cannot read {path.name}: {exc}")

    out_dir = Path(tempfile.gettempdir()) / f"cayahub-frames-{uuid.uuid4()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    # Evenly spaced, skipping the very start and end where fades live.
    for index in range(frames):
        position = duration * (index + 0.5) / frames if duration else 0
        target = out_dir / f"frame{index:02d}.jpg"
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-ss", f"{position:.2f}", "-i", str(path),
                "-frames:v", "1", "-q:v", "3", str(target),
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0 and target.exists():
            extracted.append({"at": round(position, 2), "path": str(target)})

    if not extracted:
        return fail(f"could not extract any frame from {path.name}")

    # The Rust side parses this, so it is JSON rather than prose.
    print(json.dumps({"summary": summary, "duration": duration, "frames": extracted}))
    return 0


def main():
    if len(sys.argv) < 3:
        return fail("usage: media.py <transcribe|video> <file> [arg]")

    command, raw = sys.argv[1], sys.argv[2]
    path = Path(raw)
    if not path.exists():
        return fail(f"no such file: {path}")

    if command == "transcribe":
        return transcribe(path, sys.argv[3] if len(sys.argv) > 3 else None)
    if command == "video":
        count = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_FRAMES
        return video(path, max(1, min(count, 8)))
    return fail(f"unknown command: {command}")


if __name__ == "__main__":
    sys.exit(main())
