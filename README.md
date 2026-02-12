# voice-input

Local voice input with screen-aware context. Push-to-talk on Mac, transcribed by Whisper, refined by a local LLM.

**No cloud services. Your voice and screen data never leave your machine.**

## Features

- **Push-to-talk** — Hold left Option key to record, release to transcribe and paste
- **Real-time streaming** — Partial transcription shown during recording
- **Screen-aware context** — Vision model reads your screen to improve accuracy
- **LLM text refinement** — Removes filler words, adds punctuation, fixes recognition errors
- **Multi-language** — Japanese, English, Chinese, Korean (auto-detected)
- **Auto-paste + Enter** — Result pasted and submitted automatically (hold Ctrl to paste without Enter)

## Quick start (Mac, 16 GB)

Works on any Apple Silicon Mac with 16 GB unified memory. No separate server needed.

### 1. Install prerequisites

- [Python 3.11+](https://www.python.org/)
- [Ollama](https://ollama.com/) — local LLM runtime

### 2. Setup

```bash
git clone https://github.com/xuiltul/voice-input
cd voice-input
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull gemma3:4b
```

### 3. Start

Terminal 1 (server):

```bash
WHISPER_MODEL=small LLM_MODEL=gemma3:4b .venv/bin/python ws_server.py
```

Terminal 2 (client):

```bash
.venv/bin/python mac_client.py --server ws://localhost:8991 --model gemma3:4b
```

### 4. Grant permissions

System Settings > Privacy & Security:

- **Microphone** → Terminal
- **Accessibility** → Terminal
- **Screen Recording** → Terminal

### 5. Use it

**Hold left Option** → speak → **release** → text is pasted.

**Memory usage (~9 GB total):**

| Component | Memory |
|-----------|--------|
| macOS | ~5 GB |
| Whisper `small` (CPU, int8) | ~1 GB |
| gemma3:4b (Ollama) | ~3 GB |

> **32 GB+ Mac:** Use `WHISPER_MODEL=large-v3-turbo LLM_MODEL=qwen2.5:7b` for better accuracy. Add `ollama pull qwen3-vl:8b-instruct` and set `VISION_MODEL=qwen3-vl:8b-instruct` for screen-aware context.

### Auto-start (optional)

Wrap the client in an Automator app so macOS can grant permissions to it:

1. Open **Automator.app** → **Application** → **Run Shell Script**:

```bash
cd ~/voice-input && /usr/bin/python3 mac_client.py --server ws://localhost:8991 --model gemma3:4b
```

2. Save as `~/Applications/VoiceInput.app`
3. Grant permissions (Accessibility, Input Monitoring, Microphone, Screen Recording) to VoiceInput.app
4. Add to **Login Items** for auto-start

## Quick start (Linux + NVIDIA GPU)

For faster inference with a dedicated GPU.

```bash
git clone https://github.com/xuiltul/voice-input
cd voice-input
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

ollama pull gpt-oss:20b
ollama pull qwen3-vl:8b-instruct  # optional: screen context

# LD_LIBRARY_PATH required for pip-installed CUDA libs
PYVER=$(.venv/bin/python -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
LD_LIBRARY_PATH=".venv/lib/$PYVER/site-packages/nvidia/cublas/lib:.venv/lib/$PYVER/site-packages/nvidia/cudnn/lib" \
  .venv/bin/python ws_server.py
```

Mac client connects to the server:

```bash
pip3 install sounddevice numpy websockets pynput
python3 mac_client.py --server ws://YOUR_SERVER_IP:8991
```

## Usage

### Push-to-talk

| Action | Result |
|--------|--------|
| **Hold left Option** → speak → **release** | Transcribe → refine → paste + Enter |
| **Hold left Option + Ctrl** → speak → **release** | Transcribe → refine → paste only (no Enter) |

### Screen context

When you start recording, a screenshot is captured and analyzed by a vision model. If the analysis completes before you stop recording, the HUD turns **green** — the LLM will use your screen content to improve accuracy (e.g., technical terms visible on screen).

### Client options

```
python3 mac_client.py [options]

  -s, --server URL      WebSocket server (default: ws://localhost:8991)
  -l, --language CODE   Language hint (default: ja)
  -m, --model NAME      Ollama model for refinement
  --raw                 Skip LLM refinement, Whisper output only
  --no-screenshot       Disable screen context
```

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-oss:20b` | Ollama model for text refinement |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model (`small`, `medium`, `large-v3-turbo`) |
| `WHISPER_DEVICE` | `auto` | `auto` (CUDA if available, else CPU), `cuda`, `cpu` |
| `WHISPER_COMPUTE_TYPE` | `default` | `float16` (CUDA) or `int8` (CPU) |
| `VISION_MODEL` | `qwen3-vl:8b-instruct` | Vision model for screen context |
| `VISION_SERVERS` | *(local Ollama)* | Remote Ollama URLs for vision (comma-separated) |
| `DEFAULT_LANGUAGE` | `ja` | Default language |

### Recommended models by hardware

| Hardware | Whisper | LLM | Vision | Total memory |
|----------|---------|-----|--------|-------------|
| Mac 16 GB | `small` | `gemma3:4b` | *(skip)* | ~9 GB |
| Mac 32 GB | `large-v3-turbo` | `qwen2.5:7b` | `qwen3-vl:8b-instruct` | ~20 GB |
| Linux GPU 24 GB | `large-v3-turbo` | `gpt-oss:20b` | *(remote)* | ~15 GB VRAM |

### Multi-language prompts

Refinement prompts are in `prompts/` as JSON:

```
prompts/
├── ja.json    # Japanese (default)
├── en.json    # English
├── zh.json    # Chinese
└── ko.json    # Korean
```

To add a new language, create `prompts/{code}.json`:

```json
{
  "system_prompt": "...",
  "user_template": "Format this:\n```\n{text}\n```",
  "few_shot": [
    { "user": "...", "assistant": "..." }
  ]
}
```

> **Prompt design note:** Keep the system prompt concise (~400 chars). Small models (7B-20B) with `think: "low"` degrade when the prompt is too long — they drop content instead of formatting it. Use max 2 few-shot examples.

## Advanced

<details>
<summary>Docker</summary>

```bash
docker build -t voice-input .
docker run --gpus all -p 8991:8991 \
  -e LLM_MODEL=gpt-oss:20b \
  -v ollama-data:/root/.ollama \
  voice-input
```

</details>

<details>
<summary>HTTP API</summary>

```bash
# Transcribe + refine
curl -X POST http://localhost:8990/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav

# Whisper only
curl -X POST "http://localhost:8990/transcribe?raw=true" \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav
```

Response:
```json
{
  "text": "Refined text",
  "raw_text": "Raw Whisper output",
  "language": "ja",
  "duration": 5.2,
  "processing_time": { "transcribe": 0.3, "refine": 4.1 }
}
```

</details>

<details>
<summary>Architecture</summary>

| Component | Role |
|-----------|------|
| `voice_input.py` | Core pipeline: Whisper + Ollama LLM + Vision |
| `ws_server.py` | WebSocket server (port 8991) |
| `mac_client.py` | Push-to-talk client with HUD overlay |
| `prompts/` | Language-specific refinement prompts |

Whisper auto-detects CUDA/CPU. On CUDA, uses float16; on CPU, uses int8.

</details>

<details>
<summary>Voice slash commands (Claude Code)</summary>

Say "コマンド" followed by a command name to input a slash command:

- 「コマンド ヘルプ」→ `/help`
- 「コマンド コミット」→ `/commit`
- "command compact" → `/compact`

Commands auto-loaded from `~/.claude/skills/*/SKILL.md` at startup.

</details>

## Why?

Cloud voice input sends your audio and screen to external servers. This tool runs everything locally — your data never leaves your machine.

---

## 日本語ガイド

### これは何？

Macのプッシュトゥートークで音声入力し、Whisperで文字起こし、ローカルLLMで整形するツールです。クラウドサービス不要、音声もスクリーンデータもネットワーク外に出ません。

### MacBook (16 GB) でのセットアップ

**必要なもの:**
- Apple Silicon Mac (M1/M2/M3/M4, 16 GB以上)
- Python 3.11+
- [Ollama](https://ollama.com/)

**手順:**

```bash
# クローン＆セットアップ
git clone https://github.com/xuiltul/voice-input
cd voice-input
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull gemma3:4b

# サーバー起動（ターミナル1）
WHISPER_MODEL=small LLM_MODEL=gemma3:4b .venv/bin/python ws_server.py

# クライアント起動（ターミナル2）
.venv/bin/python mac_client.py --server ws://localhost:8991 --model gemma3:4b
```

**macOSの権限設定** — システム設定 > プライバシーとセキュリティ:
- マイク → Terminal
- アクセシビリティ → Terminal
- 画面収録 → Terminal

### 使い方

- **左Optionキーを押しながら話す** → 離すと文字起こし → 整形 → 自動ペースト＋Enter
- **左Option + Ctrl を押しながら話す** → ペーストのみ（Enter送信なし）

### メモリ使用量

| コンポーネント | メモリ |
|------------|--------|
| macOS | ~5 GB |
| Whisper small (CPU) | ~1 GB |
| gemma3:4b (Ollama) | ~3 GB |
| **合計** | **~9 GB** |

32 GB以上のMacでは `WHISPER_MODEL=large-v3-turbo LLM_MODEL=qwen2.5:7b` でより高精度になります。

## License

MIT
