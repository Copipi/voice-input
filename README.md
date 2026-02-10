# voice-input

Local voice input with screen-aware context. Push-to-talk on Mac, transcribed by Whisper, refined by a local LLM — all running on your own GPU.

**No cloud services. Your voice and screen data never leave your network.**

## Features

- **Push-to-talk** — Hold left Option key on Mac to record, release to transcribe
- **Real-time streaming** — Audio streamed every 2s with partial transcription results during recording
- **Screen-aware context** — Screenshots captured at recording start, analyzed by a vision model (llava), and used to inform text refinement
- **LLM text refinement** — Removes filler words, adds punctuation, formats lists as bullet points, and fixes recognition errors
- **Multi-language support** — Language-specific prompts for Japanese, English, Chinese, and Korean (auto-detected or configurable)
- **Floating HUD** — macOS cursor-following status overlay showing recording/transcribing/refining state
- **Auto-paste** — Result automatically copied to clipboard and pasted via Cmd+V

## How it works

```
Mac (Push-to-Talk)              Server (GPU)
─────────────────               ─────────────
[Hold Option key]
  ├─ Capture screenshot ──────→ Vision analysis (llava)  ──┐
  ├─ Record audio                                          │
  ├─ Stream chunks (2s) ─────→ Whisper partial results     │
  │                              ↓ (shown in HUD)          │
[Release key]                                              │
  └─ Send final audio ───────→ Whisper final transcribe ───┤
                                                           │
                                LLM refine (context+text) ←┘
                                         │
  Auto-paste via Cmd+V  ←───── Result ───┘
```

The screenshot analysis runs **in parallel** with recording, so context is ready by the time you stop speaking. Partial transcription results are displayed during recording, giving immediate feedback.

## Quick start

### Server (Linux with NVIDIA GPU)

```bash
# Clone
git clone https://github.com/xuiltul/voice-input
cd voice-input

# Setup Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Install Ollama (https://ollama.com)
ollama pull gpt-oss:20b    # Text refinement (or any model you prefer)
ollama pull llava:latest    # Screenshot analysis (if running vision locally)

# Start WebSocket server
python ws_server.py
# or via voice-input CLI:
./voice-input serve ws
```

> **VRAM note:** Whisper (~3 GB) stays loaded. gpt-oss (~12 GB) loads on demand. For best performance, run the vision model on a separate GPU server via `VISION_SERVERS` to avoid model swapping.

### Docker

```bash
# Build
docker build -t voice-input .

# Run with GPU (vision on local Ollama)
docker run --gpus all -p 8991:8991 \
  -e LLM_MODEL=gpt-oss:20b \
  -e VISION_MODEL=llava:latest \
  voice-input

# Run with remote vision server (avoids VRAM contention)
docker run --gpus all -p 8991:8991 \
  -e LLM_MODEL=gpt-oss:20b \
  -e VISION_MODEL=qwen3-vl:8b \
  -e VISION_SERVERS=http://vision-gpu:11434 \
  voice-input

# Persist Ollama models across restarts
docker run --gpus all -p 8991:8991 \
  -v ollama-data:/root/.ollama \
  voice-input
```

### Mac client

```bash
pip3 install sounddevice numpy websockets pynput
scp your-server:~/voice-input/mac_client.py ~/
python3 mac_client.py --server ws://YOUR_SERVER_IP:8991
```

Grant these permissions in System Settings > Privacy & Security:
- **Microphone** → Terminal
- **Accessibility** → Terminal (for key monitoring + auto-paste)
- **Screen Recording** → Terminal (for screenshot context)

### Auto-start (launchd)

To run the client automatically at login:

```bash
# Copy files
mkdir -p ~/voice-input
cp mac_client.py ~/voice-input/
cp com.voice-input.client.plist ~/Library/LaunchAgents/

# Edit the plist — replace YOUR_USERNAME and YOUR_SERVER_IP
vi ~/Library/LaunchAgents/com.voice-input.client.plist

# Load (starts immediately and on every login)
launchctl load ~/Library/LaunchAgents/com.voice-input.client.plist

# Stop / unload
launchctl unload ~/Library/LaunchAgents/com.voice-input.client.plist

# View logs
tail -f ~/Library/Logs/voice-input.log
```

## Usage

### Push-to-talk

1. **Hold left Option/Alt** — Recording starts, screenshot captured, streaming begins
2. **During recording** — Partial transcription shown in floating HUD at screen bottom
3. **Release** — Final audio transcribed with VAD → LLM refines with screen context → result pasted

### Client options

```
python3 mac_client.py [options]

  -s, --server URL      WebSocket server (default: ws://YOUR_SERVER_IP:8991)
  -l, --language CODE   Language hint for Whisper (default: ja)
  -m, --model NAME      Ollama model for refinement (default: gpt-oss:20b)
  --raw                 Skip LLM refinement, Whisper output only
  -p, --prompt TEXT     Custom refinement instructions
  --no-paste            Copy to clipboard only, don't auto-paste
  --no-screenshot       Disable screenshot context analysis
```

### Server CLI

```bash
# WebSocket server (for Mac client)
voice-input serve ws --port 8991

# HTTP API server
voice-input serve --port 8990

# Transcribe a file directly
voice-input recording.mp3
voice-input recording.mp3 --raw
voice-input recording.mp3 --output json
voice-input recording.mp3 --language en --model qwen3:30b
```

### HTTP API

```bash
# Transcribe + refine
curl -X POST http://localhost:8990/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav

# Whisper only (skip LLM)
curl -X POST "http://localhost:8990/transcribe?raw=true" \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav

# With language hint and custom prompt
curl -X POST "http://localhost:8990/transcribe?language=en&prompt=Format%20as%20meeting%20notes" \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav
```

Response:
```json
{
  "text": "Refined text here",
  "raw_text": "Raw Whisper output",
  "language": "ja",
  "duration": 5.2,
  "processing_time": {
    "transcribe": 0.3,
    "refine": 4.1,
    "total": 4.4
  }
}
```

## Multi-language prompts

Refinement prompts are stored in the `prompts/` directory as JSON files, one per language:

```
prompts/
├── ja.json    # Japanese (default)
├── en.json    # English
├── zh.json    # Chinese
└── ko.json    # Korean
```

The language is determined by:
1. **Client config** — `--language` flag on the client
2. **Whisper auto-detection** — If no language is specified, Whisper detects it and the matching prompt is used

To add a new language, create `prompts/{lang_code}.json` with this structure:

```json
{
  "system_prompt": "Your system prompt here...",
  "user_template": "Please format the following text.\n```\n{text}\n```",
  "few_shot": [
    {
      "user": "Please format the following text.\n```\nraw input example\n```",
      "assistant": "Formatted output example"
    }
  ],
  "context_prefix": "Context information (from screenshot analysis):",
  "custom_prompt_prefix": "Additional instructions:"
}
```

If a language has no matching prompt file, it falls back to English, then Japanese.

## Architecture

| Component | Role | Tech |
|-----------|------|------|
| `voice_input.py` | Core pipeline: Whisper + Ollama LLM refinement + Vision | faster-whisper (CUDA), Ollama API |
| `ws_server.py` | WebSocket server, orchestrates streaming pipeline | Python, websockets |
| `mac_client.py` | Push-to-talk, screenshot, HUD overlay, clipboard paste | Python, pynput, sounddevice, PyObjC |
| `prompts/` | Language-specific refinement prompts | JSON |
| `transcribe.py` | Standalone Whisper CLI tool | faster-whisper |

### Models

| Model | Purpose | VRAM | Lifecycle |
|-------|---------|------|-----------|
| `large-v3-turbo` | Whisper speech recognition | ~3 GB | Loaded once at startup, stays in memory |
| `gpt-oss:20b` | Text refinement (configurable) | ~12 GB | Managed by Ollama (load on demand), `think: "low"` for speed |
| `qwen3-vl:8b` | Screenshot context analysis | ~5 GB | Runs on separate GPU server (no local VRAM usage) |

### WebSocket protocol

**Streaming mode** (recommended):

```
Client → Server: {"type": "stream_start", "screenshot": "<base64>"}
Client → Server: <binary WAV chunks> (every 2 seconds, cumulative audio)
Server → Client: {"type": "partial", "text": "..."}  (after each chunk)
Client → Server: <final binary WAV> → {"type": "stream_end"}
Server → Client: {"type": "status", "stage": "refining"}
Server → Client: {"type": "result", "text": "...", "raw_text": "...", ...}
```

**Legacy mode** (single-shot):

```
Client → Server: {"type": "audio_with_screenshot", "screenshot": "<base64>"}
Client → Server: <binary WAV> (complete recording)
Server → Client: {"type": "result", ...}
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL for text refinement |
| `LLM_MODEL` | `gpt-oss:20b` | Model for text refinement |
| `VISION_MODEL` | `llava:latest` | Model for screenshot analysis |
| `VISION_SERVERS` | *(unset = local Ollama)* | Comma-separated Ollama URLs for remote vision inference |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model name |
| `DEFAULT_LANGUAGE` | `ja` | Default language for transcription |
| `VOICE_INPUT_SERVER` | `ws://localhost:8991` | Mac client: default WebSocket server URL |

**Example: separate vision server** (recommended for single-GPU setups):

```bash
export VISION_MODEL=qwen3-vl:8b
export VISION_SERVERS=http://gpu-server-1:11434,http://gpu-server-2:11434
python ws_server.py
```

### VRAM management

- **Whisper** (~3 GB) is loaded once at server startup as a singleton
- **LLM** (~12 GB for gpt-oss:20b) stays loaded in Ollama with `think: "low"` for fast inference (~0.5s)
- **Vision** runs on local Ollama by default. Set `VISION_SERVERS` to offload to separate GPU(s) and avoid model swapping
- Screenshot analysis runs in parallel with recording; if not ready when refinement starts, proceeds without context
- VAD (Voice Activity Detection) is disabled for streaming chunks but enabled for final transcription

## Why?

Cloud voice input services send your audio and screen content to external servers. This tool does everything locally on your own hardware — your voice, screen data, and all processing stay on your network.

## License

MIT
