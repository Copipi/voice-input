#!/usr/bin/env python3
"""faster-whisper transcription CLI for RTX 3090."""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _bool_env(value: str) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _resolve_whisper_download_root() -> Path:
    root = os.environ.get("WHISPER_DOWNLOAD_ROOT", "").strip()
    if root:
        return Path(root).expanduser().resolve()
    return (Path(__file__).parent / "models" / "whisper").resolve()


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using faster-whisper")
    parser.add_argument("audio", nargs="?", help="Audio file path")
    parser.add_argument("-m", "--model", default="large-v3-turbo",
                        help="Model size (default: large-v3-turbo)")
    parser.add_argument("-l", "--language", default=None,
                        help="Language code (default: auto-detect)")
    parser.add_argument("-o", "--output", default="text",
                        choices=["text", "srt", "vtt", "json"],
                        help="Output format (default: text)")
    parser.add_argument("-d", "--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device (default: cuda)")
    parser.add_argument("--compute-type", default="float16",
                        help="Compute type (default: float16 for GPU)")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size (default: 5)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models")
    parser.add_argument("--info", action="store_true",
                        help="Show model info and GPU status")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        models = [
            ("tiny",           "~75MB",  "Fastest, lowest quality"),
            ("base",           "~145MB", "Fast, basic quality"),
            ("small",          "~488MB", "Good balance"),
            ("medium",         "~1.5GB", "High quality"),
            ("large-v3",       "~3.1GB", "Highest quality"),
            ("large-v3-turbo", "~1.6GB", "Best speed/quality ratio (recommended)"),
        ]
        for name, size, desc in models:
            marker = " <-- default" if name == "large-v3-turbo" else ""
            print(f"  {name:20s} {size:>8s}  {desc}{marker}")
        return

    if args.info:
        from faster_whisper import WhisperModel
        import ctranslate2
        print(f"CTranslate2 version: {ctranslate2.__version__}")
        print(f"CUDA supported: {ctranslate2.get_cuda_device_count() > 0}")
        if ctranslate2.get_cuda_device_count() > 0:
            print(f"CUDA devices: {ctranslate2.get_cuda_device_count()}")
        return

    if not args.audio:
        parser.error("Audio file path is required")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    from faster_whisper import WhisperModel

    if args.device == "cpu":
        compute_type = "int8"
    else:
        compute_type = args.compute_type

    print(f"Loading model '{args.model}' on {args.device} ({compute_type})...",
          file=sys.stderr)
    t0 = time.time()
    download_root = _resolve_whisper_download_root()
    download_root.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=compute_type,
        download_root=str(download_root),
        local_files_only=_bool_env(os.environ.get("WHISPER_LOCAL_FILES_ONLY", "")),
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    print(f"Transcribing: {audio_path}", file=sys.stderr)
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=args.beam_size,
        language=args.language,
        vad_filter=True,
    )

    segments_list = list(segments)
    elapsed = time.time() - t0
    duration = info.duration
    print(f"Done in {elapsed:.1f}s (audio: {duration:.1f}s, "
          f"speed: {duration/elapsed:.1f}x realtime)", file=sys.stderr)
    if info.language:
        print(f"Detected language: {info.language} "
              f"(probability: {info.language_probability:.2f})", file=sys.stderr)

    if args.output == "text":
        for seg in segments_list:
            print(seg.text.strip())

    elif args.output == "srt":
        for i, seg in enumerate(segments_list, 1):
            print(i)
            print(f"{_format_ts(seg.start)} --> {_format_ts(seg.end)}")
            print(seg.text.strip())
            print()

    elif args.output == "vtt":
        print("WEBVTT\n")
        for seg in segments_list:
            print(f"{_format_ts_vtt(seg.start)} --> {_format_ts_vtt(seg.end)}")
            print(seg.text.strip())
            print()

    elif args.output == "json":
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": duration,
            "transcription_time": elapsed,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
                for seg in segments_list
            ],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ts_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


if __name__ == "__main__":
    main()
