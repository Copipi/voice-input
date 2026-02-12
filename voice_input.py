#!/usr/bin/env python3
"""voice-input：Whisper 轉寫 → Ollama LLM 整理管線。

在 RTX 3090 上串接 faster-whisper（large-v3-turbo）→ gpt-oss:20b，
用音訊檔快速產生已整理的文字。

用法：
    voice-input audio.mp3                    # 轉寫 + 整理
    voice-input audio.mp3 --raw              # 只轉寫（不做 LLM 整理）
    voice-input audio.mp3 --model qwen3:30b  # 用 qwen3 進行整理
    voice-input audio.mp3 --prompt "整理成會議紀錄"  # 自訂指示
    voice-input serve                        # HTTP server 模式
    voice-input serve --port 8990            # 指定連接埠
"""

import argparse
import json
import re
import sys
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3-vl:8b-instruct")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE_TYPE", "default")
# Where to download/cache faster-whisper models.
# Default: keep it inside this repo so the folder can be copied to another machine.
WHISPER_DOWNLOAD_ROOT = os.environ.get("WHISPER_DOWNLOAD_ROOT", "")
# If set ("1"/"true"), do not hit the network; only use already-downloaded files.
WHISPER_LOCAL_FILES_ONLY = os.environ.get("WHISPER_LOCAL_FILES_ONLY", "")
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "ja")

# Vision servers: comma-separated Ollama URLs for remote vision inference.
# If set, vision runs on these servers (no local VRAM usage).
# If unset, vision runs on the local Ollama (OLLAMA_URL).
_vision_servers_env = os.environ.get("VISION_SERVERS", "")
VISION_SERVERS = (
    [s.strip() for s in _vision_servers_env.split(",") if s.strip()]
    if _vision_servers_env
    else [OLLAMA_URL]
)

PROMPTS_DIR = Path(__file__).parent / "prompts"

VISION_ANALYZE_PROMPT = """請辨識使用者目前準備輸入文字的「前景」分頁/窗格，並讀出其中的文字內容。

規則：
- 第 1 行：應用程式名稱與前景分頁標題
- 將前景分頁/窗格的正文文字原樣、完整抄寫（不要省略）
- 不需要非前景分頁、側邊欄、工具列、圖示或 UI 描述
- 若有游標/輸入框，請標明位置與周邊文字
- 以最大化文字量為目標；不需要裝飾或結構化"""

# --- 依語言的提示詞 ---
_prompt_cache: dict[str, dict] = {}


def _load_prompt(lang: str) -> dict:
    """載入語言碼對應的提示詞（含快取）。"""
    if lang in _prompt_cache:
        return _prompt_cache[lang]

    # 語言碼正規化（例如："ja-JP" → "ja"、"zh-cn" → "zh"）
    lang_base = lang.split("-")[0].lower() if lang else DEFAULT_LANGUAGE

    prompt_file = PROMPTS_DIR / f"{lang_base}.json"
    if not prompt_file.exists():
        # 回退：en → DEFAULT_LANGUAGE
        for fallback in ("en", DEFAULT_LANGUAGE):
            prompt_file = PROMPTS_DIR / f"{fallback}.json"
            if prompt_file.exists():
                break

    with open(prompt_file, encoding="utf-8") as f:
        data = json.load(f)

    _prompt_cache[lang] = data
    _prompt_cache[lang_base] = data
    return data


_whisper_model = None


def _bool_env(value: str) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _resolve_whisper_download_root() -> Path:
    if WHISPER_DOWNLOAD_ROOT.strip():
        return Path(WHISPER_DOWNLOAD_ROOT).expanduser().resolve()
    # repo-local cache
    return (Path(__file__).parent / "models" / "whisper").resolve()


def _get_whisper_model():
    """以 singleton 方式取得 Whisper model（僅首次載入）。"""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        device = WHISPER_DEVICE
        if device == "auto":
            import shutil
            device = "cuda" if shutil.which("nvidia-smi") else "cpu"
        compute = WHISPER_COMPUTE
        if compute == "default":
            compute = "float16" if device == "cuda" else "int8"
        download_root = _resolve_whisper_download_root()
        download_root.mkdir(parents=True, exist_ok=True)
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute,
            download_root=str(download_root),
            local_files_only=_bool_env(WHISPER_LOCAL_FILES_ONLY),
        )
    return _whisper_model


def transcribe(audio_path: str, language: str | None = None,
               vad_filter: bool = True) -> dict:
    """使用 Whisper 將音訊轉寫成文字。"""
    t0 = time.time()
    model = _get_whisper_model()
    load_time = time.time() - t0

    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        vad_filter=vad_filter,
    )
    segments_list = list(segments)
    transcribe_time = time.time() - t0

    raw_text = " ".join(seg.text.strip() for seg in segments_list)

    return {
        "raw_text": raw_text,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "speed": info.duration / transcribe_time if transcribe_time > 0 else 0,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments_list
        ],
    }


def analyze_screenshot(screenshot_b64: str, vision_model: str = VISION_MODEL) -> dict:
    """從截圖判定情境。

    會依序嘗試 `VISION_SERVERS` 指定的 server。
    若是本機 Ollama，會用 `keep_alive=0` 讓 VRAM 盡快釋放。
    instruct 類模型（qwen3-vl:8b-instruct）會直接輸出到 message.content。
    """
    import logging
    import requests

    log = logging.getLogger("voice_input.vision")

    t0 = time.time()
    is_local = len(VISION_SERVERS) == 1 and VISION_SERVERS[0] == OLLAMA_URL

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": VISION_ANALYZE_PROMPT,
                "images": [screenshot_b64],
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024, "num_ctx": 4096},
    }
    if is_local:
        payload["keep_alive"] = "0"

    last_err = None
    for server_url in VISION_SERVERS:
        try:
            resp = requests.post(
                f"{server_url}/api/chat", json=payload, timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            msg = data.get("message", {})
            content = msg.get("content", "")

            analysis_time = time.time() - t0
            log.info(f"Vision done in {analysis_time:.1f}s "
                     f"({len(content)} chars):\n{content}")
            return {
                "analysis": content,
                "analysis_time": analysis_time,
            }
        except Exception as e:
            elapsed = time.time() - t0
            log.error(f"Vision server {server_url} failed ({elapsed:.1f}s): "
                      f"{type(e).__name__}: {e}")
            last_err = e
            continue

    raise RuntimeError(f"All vision servers failed: {last_err}")


def refine_with_llm(
    raw_text: str,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    custom_prompt: str | None = None,
    context_hint: str | None = None,
) -> dict:
    """透過 Ollama API 整理文字（使用依語言提示詞）。"""
    import requests

    prompt_data = _load_prompt(language)
    system_prompt = prompt_data["system_prompt"]

    if context_hint:
        prefix = prompt_data.get("context_prefix", "Context:")
        system_prompt = f"{system_prompt}\n\n{prefix}\n{context_hint}"
    if custom_prompt:
        prefix = prompt_data.get("custom_prompt_prefix", "Additional instructions:")
        system_prompt = f"{system_prompt}\n\n{prefix} {custom_prompt}"

    # Few-shot examples + 模板，明確指定這是「整理任務」
    messages = [{"role": "system", "content": system_prompt}]
    for shot in prompt_data.get("few_shot", []):
        messages.append({"role": "user", "content": shot["user"]})
        messages.append({"role": "assistant", "content": shot["assistant"]})

    user_template = prompt_data.get("user_template", "```\n{text}\n```")
    messages.append({"role": "user", "content": user_template.format(text=raw_text)})

    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "think": "low",
            "options": {"temperature": 0.1, "num_predict": 8192, "num_ctx": 16384},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    refine_time = time.time() - t0

    refined = data["message"]["content"]

    # Guard: if refined text is drastically shorter, the LLM likely hallucinated
    import logging
    log = logging.getLogger("voice_input.refine")
    raw_len = len(raw_text)
    refined_len = len(refined)
    if raw_len > 0 and refined_len < raw_len * 0.4:
        log.warning(
            "Refinement too short (%d -> %d chars), falling back to raw text",
            raw_len,
            refined_len,
        )
        refined = raw_text

    return {
        "refined_text": refined,
        "model": model,
        "refine_time": refine_time,
    }


# --- Slash command 偵測/匹配 ---

SLASH_PREFIXES = [
    r"^コマンド[&＆\s]*",                     # Japanese "command" (primary)
    r"^[Cc][Oo][Mm][Mm][Aa][Nn][Dd][&＆\s]+", # English "command"
]


def detect_slash_prefix(raw_text: str) -> tuple[bool, str]:
    """偵測 raw_text 是否以 slash command 前綴開頭。

    Returns:
        (is_slash, remaining_text) — 偵測到前綴時回傳 True 與剩餘文字
    """
    text = raw_text.strip()
    for pattern in SLASH_PREFIXES:
        m = re.match(pattern, text)
        if m:
            return True, text[m.end():].strip()
    return False, text


def match_slash_command(
    spoken_text: str,
    commands: list[dict],
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
) -> dict:
    """用 LLM 將口述文字匹配成 slash command。

    Returns:
        {"matched": True/False, "command": "/name args", "match_time": float}
    """
    import requests

    cmd_list = "\n".join(
        f"- /{c['name']}"
        + (f" {c['args']}" if c.get("args") else "")
        + f"  -- {c.get('description', '')[:80]}"
        for c in commands
    )

    system_prompt = f"""你是語音指令匹配器。請把使用者口述的內容，從下列指令清單中匹配到最合適的指令。

指令清單：
{cmd_list}

規則：
1. 從輸入文字找出指令名稱與參數
2. 輸出只能是「/指令名 參數」格式（不要任何說明或評論）
3. 若指令名因口述不清楚，選擇最接近的指令
4. 若有口述參數就原樣包含（數字、URL、關鍵字等）
5. 沒有參數就只輸出指令名
6. 若無法匹配任何指令，只輸出「NO_MATCH」
7. 技術用語若被口述成外來語/音譯，請還原為常見英文拼寫（例如：issue、research、resume、commit 等）

例：
- "issue to pr 123" → /issue-to-pr 123
- "research to issue authentication problems" → /research-to-issue authentication problems
- "resume session 30 minutes" → /resume-session 30m
- "help" → /help
- "commit" → /commit
- "pdf" → /pdf
- "compact" → /compact
- "issue to pr number 45 skip review" → /issue-to-pr 45 --skip-review"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": spoken_text},
    ]

    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256, "num_ctx": 4096},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    match_time = time.time() - t0

    result_text = data["message"]["content"].strip()
    # 若有多行，只取第一行
    result_text = result_text.split("\n")[0].strip()

    if result_text == "NO_MATCH" or not result_text.startswith("/"):
        return {"matched": False, "command": "", "match_time": match_time}

    return {"matched": True, "command": result_text, "match_time": match_time}


def process_audio(
    audio_path: str,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    raw_only: bool = False,
    custom_prompt: str | None = None,
    output_format: str = "text",
    quiet: bool = False,
) -> dict:
    """完整管線：音訊檔轉寫 + 整理。"""
    if not quiet:
        print(f"[1/2] Transcribing: {audio_path}", file=sys.stderr)

    whisper_result = transcribe(audio_path, language=language)

    if not quiet:
        print(
            f"  → {whisper_result['duration']:.1f}s audio, "
            f"{whisper_result['speed']:.1f}x realtime, "
            f"lang={whisper_result['language']}",
            file=sys.stderr,
        )

    result = {**whisper_result, "refined_text": None, "refine_time": 0}

    if not raw_only and whisper_result["raw_text"].strip():
        if not quiet:
            print(f"[2/2] Refining with {model}...", file=sys.stderr)

        llm_result = refine_with_llm(
            whisper_result["raw_text"],
            model=model,
            language=whisper_result.get("language", language or DEFAULT_LANGUAGE),
            custom_prompt=custom_prompt,
        )
        result["refined_text"] = llm_result["refined_text"]
        result["refine_time"] = llm_result["refine_time"]
        result["refine_model"] = llm_result["model"]

        if not quiet:
            total = (
                whisper_result["load_time"]
                + whisper_result["transcribe_time"]
                + llm_result["refine_time"]
            )
            print(
                f"  → Refine: {llm_result['refine_time']:.1f}s, "
                f"Total: {total:.1f}s",
                file=sys.stderr,
            )

    return result


# --- HTTP Server Mode ---


class VoiceInputHandler(BaseHTTPRequestHandler):
    """HTTP handler for voice-input API."""

    server_version = "voice-input/1.0"

    def do_GET(self):
        """Health check / usage info."""
        if self.path == "/health":
            self._json_response({"status": "ok"})
            return
        self._json_response({
            "service": "voice-input",
            "usage": "POST /transcribe with audio file",
            "params": {
                "language": "Language code (optional)",
                "model": f"Ollama model (default: {DEFAULT_MODEL})",
                "raw": "true to skip LLM refinement",
                "prompt": "Custom refinement instruction",
            },
        })

    def do_POST(self):
        """Process uploaded audio."""
        if self.path.split("?")[0] != "/transcribe":
            self._json_response({"error": "Use POST /transcribe"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._json_response({"error": "No audio data"}, status=400)
            return

        # Parse query params
        from urllib.parse import urlparse, parse_qs

        params = parse_qs(urlparse(self.path).query)
        language = params.get("language", [None])[0]
        model = params.get("model", [DEFAULT_MODEL])[0]
        raw_only = params.get("raw", ["false"])[0].lower() == "true"
        custom_prompt = params.get("prompt", [None])[0]

        # Save uploaded audio to temp file
        content_type = self.headers.get("Content-Type", "")
        ext = ".wav"
        if "mp3" in content_type or "mpeg" in content_type:
            ext = ".mp3"
        elif "ogg" in content_type:
            ext = ".ogg"
        elif "webm" in content_type:
            ext = ".webm"
        elif "m4a" in content_type or "mp4" in content_type:
            ext = ".m4a"

        audio_data = self.rfile.read(content_length)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        try:
            result = process_audio(
                tmp_path,
                language=language,
                model=model,
                raw_only=raw_only,
                custom_prompt=custom_prompt,
                quiet=True,
            )
            output = {
                "text": result["refined_text"] or result["raw_text"],
                "raw_text": result["raw_text"],
                "language": result["language"],
                "duration": result["duration"],
                "processing_time": {
                    "transcribe": round(result["transcribe_time"], 2),
                    "refine": round(result["refine_time"], 2),
                    "total": round(
                        result["load_time"]
                        + result["transcribe_time"]
                        + result["refine_time"],
                        2,
                    ),
                },
            }
            self._json_response(output)
        except Exception as e:
            self._json_response({"error": str(e)}, status=500)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}", file=sys.stderr)


def serve(host: str, port: int):
    """啟動 HTTP server。"""
    server = HTTPServer((host, port), VoiceInputHandler)
    print(f"voice-input server listening on http://{host}:{port}", file=sys.stderr)
    print(f"  POST /transcribe  - Upload audio for transcription", file=sys.stderr)
    print(f"  GET  /health      - Health check", file=sys.stderr)
    print(f"  GET  /             - Usage info", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
        server.shutdown()


# --- CLI ---


def main():
    # "serve" subcommand detection
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        # "serve ws" → WebSocket server
        if len(sys.argv) >= 3 and sys.argv[2] == "ws":
            parser = argparse.ArgumentParser(description="voice-input WebSocket server")
            parser.add_argument("_cmd", metavar="serve")
            parser.add_argument("_mode", metavar="ws")
            parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
            parser.add_argument("--port", type=int, default=8991, help="Port (default: 8991)")
            args = parser.parse_args()
            from ws_server import main as ws_main
            import asyncio
            asyncio.run(ws_main(args.host, args.port))
            return

        # "serve" → HTTP server
        parser = argparse.ArgumentParser(description="voice-input HTTP server")
        parser.add_argument("_cmd", metavar="serve")
        parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
        parser.add_argument("--port", type=int, default=8990, help="Port (default: 8990)")
        args = parser.parse_args()
        serve(args.host, args.port)
        return

    parser = argparse.ArgumentParser(
                description="voice-input: 音訊 → Whisper 轉寫 → LLM 整理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    voice-input meeting.mp3                          # 轉寫 + 整理
    voice-input meeting.mp3 --raw                    # 只跑 Whisper
    voice-input meeting.mp3 --model qwen3:30b        # 用 Qwen3 整理
    voice-input meeting.mp3 --prompt "用條列整理"     # 自訂指示
  voice-input meeting.mp3 --output json            # JSON出力
    voice-input serve --port 8990                    # HTTP server
        """,
    )
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("-l", "--language", default=None, help="Language code (e.g., ja, en)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--raw", action="store_true", help="Skip LLM refinement")
    parser.add_argument("-p", "--prompt", default=None, help="Custom refinement prompt")
    parser.add_argument("-o", "--output", default="text", choices=["text", "json"], help="Output format")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    result = process_audio(
        str(audio_path),
        language=args.language,
        model=args.model,
        raw_only=args.raw,
        custom_prompt=args.prompt,
        quiet=args.quiet,
    )

    if args.output == "json":
        output = {
            "text": result["refined_text"] or result["raw_text"],
            "raw_text": result["raw_text"],
            "language": result["language"],
            "duration": result["duration"],
            "transcribe_time": round(result["transcribe_time"], 2),
            "refine_time": round(result["refine_time"], 2),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        final = result["refined_text"] or result["raw_text"]
        print(final)


if __name__ == "__main__":
    main()
