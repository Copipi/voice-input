#!/usr/bin/env python3
"""voice-input WebSocket server.

Mac等のクライアントから音声データ（+スクリーンショット）を受信し、
Whisper文字起こし → コンテキスト判定 → LLM整形 → テキスト返却する。

Protocol:
  Client → Server: JSON制御メッセージ or バイナリ音声データ
  Server → Client: JSON応答

ストリーミングモード（推奨）:
  1. Client: {"type": "stream_start", "screenshot": "<base64>"}  ← 録音開始
  2. Server: Vision分析を即座に開始
  3. Client: バイナリ音声データ（2秒ごとに累積WAV送信）
  4. Server: {"type": "partial", "text": "..."}  ← 逐次文字起こし結果
  5. Client: 最終バイナリ音声データ → {"type": "stream_end"}
  6. Server: LLM整形 → {"type": "result", "text": "..."}

レガシーモード:
  1. Client: {"type": "audio_with_screenshot", "screenshot": "<base64>"}
  2. Client: バイナリ音声データ（録音完了後に一括送信）
  3. Server: transcribe → refine → {"type": "result"}
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import websockets

from voice_input import (
    transcribe,
    refine_with_llm,
    analyze_screenshot,
    DEFAULT_MODEL,
    VISION_MODEL,
    OLLAMA_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ws_server")

# Vision contextの最大文字数
MAX_CONTEXT_LEN = 2000

# クライアント別設定
client_configs: dict[str, dict] = {}

# レガシーモード用（audio_with_screenshot + binary）
pending_vision_tasks: dict[str, asyncio.Task] = {}
vision_start_times: dict[str, float] = {}


class StreamState:
    """ストリーミングモードのクライアント状態."""
    __slots__ = (
        "active", "latest_audio", "latest_text", "vision_task",
        "vision_start", "processing", "has_newer",
    )

    def __init__(self):
        self.active = False
        self.latest_audio: bytes | None = None
        self.latest_text = ""
        self.vision_task: asyncio.Task | None = None
        self.vision_start = 0.0
        self.processing = False  # Whisperが実行中か
        self.has_newer = False   # 処理中に新しい音声が到着したか


stream_states: dict[str, StreamState] = {}


async def handle_client(websocket):
    """WebSocketクライアントを処理."""
    addr = websocket.remote_address
    client_id = f"{addr[0]}:{addr[1]}"
    log.info(f"Client connected: {client_id}")

    client_configs[client_id] = {
        "language": "ja",
        "model": DEFAULT_MODEL,
        "raw": False,
        "prompt": None,
    }
    state = StreamState()
    stream_states[client_id] = state

    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await send_json(websocket, {"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = data.get("type", "")

                if msg_type == "config":
                    cfg = client_configs[client_id]
                    for key in ("language", "model", "prompt"):
                        if key in data:
                            cfg[key] = data[key]
                    if "raw" in data:
                        cfg["raw"] = bool(data["raw"])
                    await send_json(websocket, {"type": "config_ack", **cfg})
                    log.info(f"Config updated for {client_id}: {cfg}")

                elif msg_type == "stream_start":
                    await handle_stream_start(websocket, client_id, data)

                elif msg_type == "stream_end":
                    await handle_stream_end(websocket, client_id)

                elif msg_type == "audio_with_screenshot":
                    # レガシーモード: 録音開始時のスクリーンショット送信
                    screenshot_b64 = data.get("screenshot", "")
                    if screenshot_b64:
                        cfg = client_configs.get(client_id, {})
                        if not cfg.get("raw", False):
                            loop = asyncio.get_event_loop()
                            vision_start_times[client_id] = time.time()
                            pending_vision_tasks[client_id] = asyncio.ensure_future(
                                loop.run_in_executor(None, analyze_screenshot, screenshot_b64)
                            )
                            log.info(f"[legacy] Vision STARTED for {client_id} "
                                     f"({len(screenshot_b64) // 1024}KB)")
                        await send_json(websocket, {"type": "screenshot_ack"})

                elif msg_type == "ping":
                    await send_json(websocket, {"type": "pong", "time": time.time()})

            elif isinstance(message, bytes):
                if state.active:
                    # ストリーミングモード: 逐次チャンク
                    await handle_stream_chunk(websocket, client_id, message)
                else:
                    # レガシーモード: 録音完了後の一括送信
                    vision_task = pending_vision_tasks.pop(client_id, None)
                    vision_start = vision_start_times.pop(client_id, None)
                    await handle_audio(websocket, client_id, message,
                                       vision_task, vision_start)

    except websockets.exceptions.ConnectionClosed:
        log.info(f"Client disconnected: {client_id}")
    finally:
        client_configs.pop(client_id, None)
        # ストリーミング状態のクリーンアップ
        st = stream_states.pop(client_id, None)
        if st and st.vision_task and not st.vision_task.done():
            st.vision_task.cancel()
        # レガシー状態のクリーンアップ
        task = pending_vision_tasks.pop(client_id, None)
        if task and not task.done():
            task.cancel()
        vision_start_times.pop(client_id, None)


# =============================================================================
#  ストリーミングモード
# =============================================================================

async def handle_stream_start(websocket, client_id: str, data: dict):
    """ストリーミング開始: Vision分析を即座に開始."""
    state = stream_states[client_id]
    state.active = True
    state.latest_audio = None
    state.latest_text = ""
    state.processing = False
    state.has_newer = False

    # 前回のvisionタスクが残っていればキャンセル
    if state.vision_task and not state.vision_task.done():
        state.vision_task.cancel()
        log.info(f"Cancelled previous vision task for {client_id}")

    # Vision分析
    screenshot_b64 = data.get("screenshot", "")
    cfg = client_configs.get(client_id, {})
    if screenshot_b64 and not cfg.get("raw", False):
        loop = asyncio.get_event_loop()
        state.vision_start = time.time()
        state.vision_task = asyncio.ensure_future(
            loop.run_in_executor(None, analyze_screenshot, screenshot_b64)
        )
        log.info(f"Stream started for {client_id} "
                 f"(vision: {len(screenshot_b64) // 1024}KB)")
    else:
        state.vision_task = None
        log.info(f"Stream started for {client_id}")

    await send_json(websocket, {"type": "stream_ack"})


async def handle_stream_chunk(websocket, client_id: str, audio_data: bytes):
    """ストリーミング中の音声チャンクを処理."""
    state = stream_states.get(client_id)
    if not state or not state.active:
        return

    state.latest_audio = audio_data
    size_kb = len(audio_data) / 1024
    log.info(f"Stream chunk from {client_id}: {size_kb:.0f}KB")

    if state.processing:
        # Whisper実行中 → 新しい音声が来たことをマーク
        state.has_newer = True
        return

    # Whisper開始
    state.processing = True
    asyncio.create_task(_stream_whisper_loop(websocket, client_id))


async def _stream_whisper_loop(websocket, client_id: str):
    """ストリーミングWhisperループ: 最新の音声を処理し続ける."""
    state = stream_states.get(client_id)
    if not state:
        return

    loop = asyncio.get_event_loop()
    cfg = client_configs.get(client_id, {})

    while True:
        audio = state.latest_audio
        state.has_newer = False

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            tmp_path = f.name

        try:
            t0 = time.time()
            # ストリーミング中はVAD無効（短いチャンクだとVADが全削除する）
            result = await loop.run_in_executor(
                None, lambda: transcribe(tmp_path, cfg.get("language"),
                                         vad_filter=False)
            )
            elapsed = time.time() - t0
            state.latest_text = result["raw_text"]

            if state.latest_text.strip():
                await send_json(websocket, {
                    "type": "partial",
                    "text": state.latest_text,
                    "duration": result.get("duration", 0),
                })
                log.info(f"Stream partial ({elapsed:.1f}s): "
                         f"{state.latest_text[:60]}")
        except Exception as e:
            log.error(f"Stream Whisper error: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # 処理中に新しい音声が届いていたら再度処理
        if not state.has_newer or not state.active:
            state.processing = False
            break


async def handle_stream_end(websocket, client_id: str):
    """ストリーミング終了: 最終Whisper結果でLLM整形."""
    state = stream_states.get(client_id)
    if not state:
        return

    state.active = False
    log.info(f"Stream end for {client_id}")

    # 実行中のWhisperが終わるのを待つ
    while state.processing:
        await asyncio.sleep(0.05)

    cfg = client_configs.get(client_id, {})

    # 最終音声を必ずVAD有効で再処理（完全な録音に対してはVADが正しく動く）
    raw_text = ""
    transcribe_time = 0
    duration = 0
    detected_lang = cfg.get("language", "ja")
    if state.latest_audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(state.latest_audio)
            tmp_path = f.name
        try:
            loop = asyncio.get_event_loop()
            t0 = time.time()
            result = await loop.run_in_executor(
                None, lambda: transcribe(tmp_path, cfg.get("language"),
                                         vad_filter=True)
            )
            transcribe_time = time.time() - t0
            raw_text = result["raw_text"]
            duration = result.get("duration", 0)
            detected_lang = result.get("language", detected_lang)
            log.info(f"Final transcribe: {duration:.1f}s audio → "
                     f"{len(raw_text)} chars in {transcribe_time:.1f}s "
                     f"(lang={detected_lang})")
        except Exception as e:
            log.error(f"Final transcribe error: {e}")
            # フォールバック: ストリーミング中のpartialテキストを使う
            raw_text = state.latest_text
            log.info(f"Falling back to partial text: {raw_text[:60]}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        raw_text = state.latest_text

    if not raw_text.strip():
        await send_json(websocket, {
            "type": "result", "text": "", "raw_text": "",
            "duration": 0, "transcribe_time": 0, "analysis_time": 0,
            "refine_time": 0, "context": "",
        })
        return

    # Vision結果（完了していれば使う、未完了なら待たずに進む）
    context_hint = ""
    analysis_time = 0
    if state.vision_task:
        if state.vision_task.done():
            try:
                vision_result = state.vision_task.result()
                analysis_time = time.time() - state.vision_start if state.vision_start else 0
                context_hint = vision_result.get("analysis", "")
                if len(context_hint) > MAX_CONTEXT_LEN:
                    context_hint = context_hint[:MAX_CONTEXT_LEN] + "..."
                log.info(f"Vision ready ({analysis_time:.1f}s, "
                         f"{len(context_hint)} chars):\n{context_hint}")
            except Exception as e:
                log.error(f"Vision error: {e}")
        else:
            elapsed = time.time() - state.vision_start if state.vision_start else 0
            log.info(f"Vision not ready ({elapsed:.0f}s elapsed), proceeding without context")

    # LLM整形
    refined_text = raw_text
    refine_time = 0

    if not cfg.get("raw", False):
        await send_json(websocket, {"type": "status", "stage": "refining"})
        loop = asyncio.get_event_loop()
        t0 = time.time()
        try:
            llm_result = await loop.run_in_executor(
                None,
                lambda: refine_with_llm(
                    raw_text,
                    model=cfg.get("model", DEFAULT_MODEL),
                    language=detected_lang,
                    custom_prompt=cfg.get("prompt"),
                    context_hint=context_hint if context_hint else None,
                ),
            )
            refine_time = time.time() - t0
            refined_text = llm_result["refined_text"]
            log.info(f"Refined in {refine_time:.1f}s (lang={detected_lang}): "
                     f"raw={raw_text[:60]} → refined={refined_text[:60]}")
        except Exception as e:
            log.error(f"Refine error: {e}")
            refine_time = time.time() - t0

    await send_json(websocket, {
        "type": "result",
        "text": refined_text,
        "raw_text": raw_text,
        "duration": duration,
        "transcribe_time": round(transcribe_time, 2),
        "analysis_time": round(analysis_time, 2),
        "refine_time": round(refine_time, 2),
        "context": context_hint,
    })


# =============================================================================
#  レガシーモード（stream_start/stream_end非対応クライアント用）
# =============================================================================

async def handle_audio(websocket, client_id: str, audio_data: bytes,
                       vision_task: asyncio.Task | None = None,
                       vision_start: float | None = None):
    """レガシー: バイナリ音声データを一括処理."""
    cfg = client_configs.get(client_id, {})
    size_kb = len(audio_data) / 1024
    log.info(f"[legacy] Audio from {client_id}: {size_kb:.1f}KB")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        tmp_path = f.name

    try:
        loop = asyncio.get_event_loop()
        context_hint = None
        analysis_time = 0

        await send_json(websocket, {"type": "status", "stage": "transcribing"})
        t0 = time.time()
        whisper_result = await loop.run_in_executor(
            None, transcribe, tmp_path, cfg.get("language")
        )
        transcribe_time = time.time() - t0
        raw_text = whisper_result["raw_text"]
        log.info(f"Transcribed: {whisper_result['duration']:.1f}s → "
                 f"{len(raw_text)} chars in {transcribe_time:.1f}s")

        # Vision結果（完了していれば使う、未完了なら待たずに進む）
        if vision_task is not None:
            if vision_task.done():
                try:
                    vision_result = vision_task.result()
                    analysis_time = time.time() - vision_start if vision_start else 0
                    context_hint = vision_result.get("analysis", "")
                    if len(context_hint) > MAX_CONTEXT_LEN:
                        context_hint = context_hint[:MAX_CONTEXT_LEN] + "..."
                    preview = context_hint.replace("\n", " ")[:80]
                    log.info(f"Vision ready: {analysis_time:.1f}s → {preview}")
                    log.info(f"Vision full context ({len(context_hint)} chars):\n{context_hint}")
                except Exception as e:
                    log.error(f"Vision error: {e}")
            else:
                log.info("[legacy] Vision not ready yet, proceeding without context")

        if not raw_text.strip():
            await send_json(websocket, {
                "type": "result", "text": "", "raw_text": "",
                "language": whisper_result.get("language", ""),
                "duration": whisper_result.get("duration", 0),
                "transcribe_time": round(transcribe_time, 2),
                "analysis_time": round(analysis_time, 2),
                "refine_time": 0, "context": context_hint,
            })
            return

        # partial送信
        await send_json(websocket, {
            "type": "partial",
            "text": raw_text,
            "language": whisper_result.get("language", ""),
            "duration": whisper_result.get("duration", 0),
            "transcribe_time": round(transcribe_time, 2),
        })

        # LLM整形
        detected_lang = whisper_result.get("language", cfg.get("language", "ja"))
        refined_text = raw_text
        refine_time = 0
        if not cfg.get("raw", False):
            await send_json(websocket, {"type": "status", "stage": "refining"})
            t0 = time.time()
            llm_result = await loop.run_in_executor(
                None,
                lambda: refine_with_llm(
                    raw_text,
                    model=cfg.get("model", DEFAULT_MODEL),
                    language=detected_lang,
                    custom_prompt=cfg.get("prompt"),
                    context_hint=context_hint,
                ),
            )
            refine_time = time.time() - t0
            refined_text = llm_result["refined_text"]
            log.info(f"Refined in {refine_time:.1f}s (lang={detected_lang})")

        await send_json(websocket, {
            "type": "result",
            "text": refined_text,
            "raw_text": raw_text,
            "language": whisper_result.get("language", ""),
            "duration": whisper_result.get("duration", 0),
            "transcribe_time": round(transcribe_time, 2),
            "analysis_time": round(analysis_time, 2),
            "refine_time": round(refine_time, 2),
            "context": context_hint,
        })

    except Exception as e:
        log.error(f"Processing error: {e}")
        await send_json(websocket, {"type": "error", "message": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# =============================================================================
#  共通ユーティリティ
# =============================================================================

async def send_json(websocket, data: dict):
    """JSON応答を送信."""
    await websocket.send(json.dumps(data, ensure_ascii=False))


async def main(host: str = "0.0.0.0", port: int = 8991):
    """WebSocketサーバーを起動."""
    log.info(f"Starting WebSocket server on ws://{host}:{port}")
    log.info(f"Protocol: streaming (stream_start/end) + legacy (audio_with_screenshot)")
    from voice_input import VISION_SERVERS
    vision_loc = "local" if len(VISION_SERVERS) == 1 and VISION_SERVERS[0] == OLLAMA_URL else f"remote: {','.join(VISION_SERVERS)}"
    log.info(f"Vision model: {VISION_MODEL} ({vision_loc})")

    # Whisperモデルを事前ロード
    from voice_input import _get_whisper_model
    log.info("Pre-loading Whisper model...")
    t0 = time.time()
    _get_whisper_model()
    log.info(f"Whisper model loaded in {time.time() - t0:.1f}s")

    async with websockets.serve(
        handle_client,
        host,
        port,
        max_size=50 * 1024 * 1024,  # 50MB
        ping_interval=30,
        ping_timeout=10,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="voice-input WebSocket server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8991)
    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        log.info("Shutdown")
