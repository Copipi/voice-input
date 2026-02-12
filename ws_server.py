#!/usr/bin/env python3
"""voice-input WebSocket server.

從 Mac 等 client 接收音訊資料（+ 截圖），
Whisper 轉寫 → 判定情境 → LLM 整理 → 回傳文字。

Protocol:
    Client → Server: JSON 控制訊息 or 二進位音訊資料
    Server → Client: JSON 回應

串流模式（建議）:
  1. Client: {"type": "stream_start", "screenshot": "<base64>"}  ← 録音開始
    2. Server: 立即開始 Vision 分析
    3. Client: 二進位音訊資料（每 2 秒送出累積 WAV）
    4. Server: {"type": "partial", "text": "..."}  ← 即時轉寫結果
    5. Client: 最後一包二進位音訊資料 → {"type": "stream_end"}
  6. Server: LLM整形 → {"type": "result", "text": "..."}

傳統模式（legacy）:
  1. Client: {"type": "audio_with_screenshot", "screenshot": "<base64>"}
    2. Client: 二進位音訊資料（錄音結束後一次送出）
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
    convert_to_traditional_chinese,
    detect_slash_prefix,
    match_slash_command,
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

# Vision 情境文字最大長度
MAX_CONTEXT_LEN = 2000

# 各 client 的設定
client_configs: dict[str, dict] = {}

# 傳統模式用（audio_with_screenshot + binary）
pending_vision_tasks: dict[str, asyncio.Task] = {}
vision_start_times: dict[str, float] = {}


class StreamState:
    """串流模式的 client 狀態。"""
    __slots__ = (
        "active", "latest_audio", "latest_text", "vision_task",
        "vision_start", "processing", "has_newer", "vision_notified",
        "text_context",
    )

    def __init__(self):
        self.active = False
        self.latest_audio: bytes | None = None
        self.latest_text = ""
        self.vision_task: asyncio.Task | None = None
        self.vision_start = 0.0
        self.processing = False  # Whisper 是否正在執行
        self.has_newer = False   # 處理中是否收到更新的音訊
        self.vision_notified = False  # 是否已通知 Vision 完成
        self.text_context: str | None = None  # AX 文字擷取結果


stream_states: dict[str, StreamState] = {}


async def handle_client(websocket):
    """處理 WebSocket client。"""
    addr = websocket.remote_address
    client_id = f"{addr[0]}:{addr[1]}"
    log.info(f"Client connected: {client_id}")

    client_configs[client_id] = {
        "language": "ja",
        "model": DEFAULT_MODEL,
        "raw": False,
        "prompt": None,
        "output_language": None,
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
                    for key in ("language", "model", "prompt", "output_language"):
                        if key in data:
                            cfg[key] = data[key]
                    if "raw" in data:
                        cfg["raw"] = bool(data["raw"])
                    if "slash_commands" in data:
                        cfg["slash_commands"] = data["slash_commands"]
                        log.info(f"Loaded {len(data['slash_commands'])} slash commands for {client_id}")
                    # config_ack 不包含 slash_commands（內容可能很大）
                    ack = {k: v for k, v in cfg.items() if k != "slash_commands"}
                    await send_json(websocket, {"type": "config_ack", **ack})
                    log.info(f"Config updated for {client_id}: {ack}")

                elif msg_type == "stream_start":
                    await handle_stream_start(websocket, client_id, data)

                elif msg_type == "stream_end":
                    await handle_stream_end(websocket, client_id)

                elif msg_type == "audio_with_screenshot":
                    # 傳統模式：錄音開始時送出截圖
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
                    # 串流模式：逐次 chunk
                    await handle_stream_chunk(websocket, client_id, message)
                else:
                    # 傳統模式：錄音結束後一次送出
                    vision_task = pending_vision_tasks.pop(client_id, None)
                    vision_start = vision_start_times.pop(client_id, None)
                    await handle_audio(websocket, client_id, message,
                                       vision_task, vision_start)

    except websockets.exceptions.ConnectionClosed:
        log.info(f"Client disconnected: {client_id}")
    finally:
        client_configs.pop(client_id, None)
        # 清理串流狀態
        st = stream_states.pop(client_id, None)
        if st and st.vision_task and not st.vision_task.done():
            st.vision_task.cancel()
        # 清理傳統模式狀態
        task = pending_vision_tasks.pop(client_id, None)
        if task and not task.done():
            task.cancel()
        vision_start_times.pop(client_id, None)


# =============================================================================
#  串流模式
# =============================================================================

async def handle_stream_start(websocket, client_id: str, data: dict):
    """串流開始：優先使用 text_context，否則非同步啟動 Vision 分析。"""
    state = stream_states[client_id]
    state.active = True
    state.latest_audio = None
    state.latest_text = ""
    state.processing = False
    state.has_newer = False
    state.vision_notified = False
    state.text_context = None

    # 若上一個 vision 任務仍存在則取消
    if state.vision_task and not state.vision_task.done():
        state.vision_task.cancel()
        log.info(f"Cancelled previous vision task for {client_id}")

    cfg = client_configs.get(client_id, {})

    # 優先順序：text_context > screenshot > 無
    text_context = data.get("text_context", "")
    screenshot_b64 = data.get("screenshot", "")
    has_screenshot = bool(screenshot_b64)
    raw_mode = bool(cfg.get("raw", False))
    vision_enabled = has_screenshot and not raw_mode and not bool(text_context)

    log.info(
        "stream_start debug: client=%s vision_enabled=%s vision_model=%s has_screenshot=%s",
        client_id,
        vision_enabled,
        VISION_MODEL,
        has_screenshot,
    )

    if text_context and not raw_mode:
        # AX 文字擷取成功 → 不需要 Vision，情境可立即確定
        if len(text_context) > MAX_CONTEXT_LEN:
            text_context = text_context[:MAX_CONTEXT_LEN] + "..."
        state.text_context = text_context
        state.vision_task = None
        log.info(f"Stream started for {client_id} "
                 f"(text_context: {len(text_context)} chars)")
        await send_json(websocket, {"type": "stream_ack"})
        # AX 文字可立即使用，直接通知 vision_ready
        await send_json(websocket, {"type": "status", "stage": "vision_ready"})
    elif screenshot_b64 and not raw_mode:
        # 截圖 → 維持原本行為：非同步啟動 Vision
        loop = asyncio.get_event_loop()
        state.vision_start = time.time()
        state.vision_task = asyncio.ensure_future(
            loop.run_in_executor(None, analyze_screenshot, screenshot_b64)
        )
        log.info(f"Stream started for {client_id} "
                 f"(vision: {len(screenshot_b64) // 1024}KB)")
        await send_json(websocket, {"type": "stream_ack"})
    else:
        state.vision_task = None
        log.info(f"Stream started for {client_id}")
        await send_json(websocket, {"type": "stream_ack"})


async def handle_stream_chunk(websocket, client_id: str, audio_data: bytes):
    """處理串流中的音訊 chunk。"""
    state = stream_states.get(client_id)
    if not state or not state.active:
        return

    state.latest_audio = audio_data
    size_kb = len(audio_data) / 1024
    log.info(f"Stream chunk from {client_id}: {size_kb:.0f}KB")

    if state.processing:
        # Whisper 執行中 → 標記已收到更新音訊
        state.has_newer = True
        return

    # 開始 Whisper
    state.processing = True
    asyncio.create_task(_stream_whisper_loop(websocket, client_id))


async def _stream_whisper_loop(websocket, client_id: str):
    """串流 Whisper 迴圈：持續處理最新音訊。"""
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
            # 串流時停用 VAD（chunk 太短時 VAD 可能把內容全部濾掉）
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

        # 通知 Vision 完成（只做一次）
        if (state.vision_task and state.vision_task.done()
                and not state.vision_notified):
            state.vision_notified = True
            elapsed = time.time() - state.vision_start if state.vision_start else 0
            try:
                await send_json(websocket, {
                    "type": "status", "stage": "vision_ready",
                })
                log.info(f"Vision ready notified to {client_id} ({elapsed:.1f}s)")
            except Exception:
                pass

        # 若處理期間收到更新音訊則再次處理
        if not state.has_newer or not state.active:
            state.processing = False
            break


async def handle_stream_end(websocket, client_id: str):
    """串流結束：使用最終 Whisper 結果做 LLM 整理。"""
    state = stream_states.get(client_id)
    if not state:
        return

    state.active = False
    log.info(f"Stream end for {client_id}")

    # 等待正在執行的 Whisper 結束
    while state.processing:
        await asyncio.sleep(0.05)

    cfg = client_configs.get(client_id, {})

    # 最終音訊先用 VAD 重新處理；若 VAD 全部濾掉則改用停用 VAD 再試一次
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
            log.info(f"Final transcribe (VAD): {duration:.1f}s audio → "
                     f"{len(raw_text)} chars in {transcribe_time:.1f}s "
                     f"(lang={detected_lang})")

            # 若 VAD 全濾掉但串流 partial 有文字
            # → 停用 VAD 再處理（避免 VAD 誤判）
            if not raw_text.strip() and state.latest_text.strip():
                log.warning(f"VAD filtered all speech, retrying without VAD "
                            f"(partial had: {state.latest_text[:60]})")
                t0 = time.time()
                result = await loop.run_in_executor(
                    None, lambda: transcribe(tmp_path, cfg.get("language"),
                                             vad_filter=False)
                )
                transcribe_time += time.time() - t0
                raw_text = result["raw_text"]
                duration = result.get("duration", 0)
                detected_lang = result.get("language", detected_lang)
                log.info(f"Final transcribe (no VAD): {duration:.1f}s audio → "
                         f"{len(raw_text)} chars")
        except Exception as e:
            log.error(f"Final transcribe error: {e}")
            # 回退：使用串流期間的 partial 文字
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

    # 情境來源：優先 text_context，否則使用 Vision 結果
    context_hint = ""
    analysis_time = 0
    if state.text_context:
        # 直接使用 AX 文字擷取結果（立即可用、不需要 GPU）
        context_hint = state.text_context
        log.info(f"Using text_context ({len(context_hint)} chars)")
    elif state.vision_task:
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

    # Slash command 偵測
    slash_commands = cfg.get("slash_commands", [])
    if slash_commands and raw_text.strip():
        is_slash, remaining = detect_slash_prefix(raw_text)
        if is_slash and remaining:
            log.info(f"Slash command detected: '{remaining}'")
            await send_json(websocket, {"type": "status", "stage": "matching_command"})
            loop = asyncio.get_event_loop()
            t0 = time.time()
            try:
                match_result = await loop.run_in_executor(
                    None,
                    lambda: match_slash_command(
                        remaining,
                        slash_commands,
                        model=cfg.get("model", DEFAULT_MODEL),
                        language=detected_lang,
                    ),
                )
                match_time = time.time() - t0
                if match_result["matched"]:
                    log.info(f"Matched command in {match_time:.1f}s: "
                             f"{match_result['command']}")
                    await send_json(websocket, {
                        "type": "result",
                        "text": match_result["command"],
                        "raw_text": raw_text,
                        "slash_command": True,
                        "duration": duration,
                        "transcribe_time": round(transcribe_time, 2),
                        "analysis_time": round(analysis_time, 2),
                        "refine_time": round(match_time, 2),
                        "context": "",
                    })
                    return
                else:
                    log.info(f"No command match for: '{remaining}', "
                             f"falling through to normal refinement")
            except Exception as e:
                log.error(f"Slash command match error: {e}")

    # LLM 整理
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
                    output_language=cfg.get("output_language"),
                ),
            )
            refine_time = time.time() - t0
            refined_text = llm_result["refined_text"]
            log.info(f"Refined in {refine_time:.1f}s (lang={detected_lang}): "
                     f"raw={raw_text[:60]} → refined={refined_text[:60]}")
        except Exception as e:
            log.error(f"Refine error: {e}")
            refine_time = time.time() - t0

    output_lang = cfg.get("output_language") or detected_lang
    raw_text_out = convert_to_traditional_chinese(raw_text, language=output_lang)
    refined_text_out = convert_to_traditional_chinese(refined_text, language=output_lang)
    context_hint_out = convert_to_traditional_chinese(context_hint, language=output_lang)

    await send_json(websocket, {
        "type": "result",
        "text": refined_text_out,
        "raw_text": raw_text_out,
        "duration": duration,
        "transcribe_time": round(transcribe_time, 2),
        "analysis_time": round(analysis_time, 2),
        "refine_time": round(refine_time, 2),
        "context": context_hint_out,
    })


# =============================================================================
#  傳統模式（給不支援 stream_start/stream_end 的 client）
# =============================================================================

async def handle_audio(websocket, client_id: str, audio_data: bytes,
                       vision_task: asyncio.Task | None = None,
                       vision_start: float | None = None):
    """傳統模式：一次處理整包二進位音訊資料。"""
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

        # Vision 結果（若已完成就使用；若未完成則不等待直接繼續）
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

        # 傳送 partial
        await send_json(websocket, {
            "type": "partial",
            "text": raw_text,
            "language": whisper_result.get("language", ""),
            "duration": whisper_result.get("duration", 0),
            "transcribe_time": round(transcribe_time, 2),
        })

        # Slash command 偵測
        detected_lang = whisper_result.get("language", cfg.get("language", "ja"))
        slash_commands = cfg.get("slash_commands", [])
        if slash_commands and raw_text.strip():
            is_slash, remaining = detect_slash_prefix(raw_text)
            if is_slash and remaining:
                log.info(f"[legacy] Slash command detected: '{remaining}'")
                await send_json(websocket, {"type": "status", "stage": "matching_command"})
                t0 = time.time()
                try:
                    match_result = await loop.run_in_executor(
                        None,
                        lambda: match_slash_command(
                            remaining,
                            slash_commands,
                            model=cfg.get("model", DEFAULT_MODEL),
                            language=detected_lang,
                        ),
                    )
                    match_time = time.time() - t0
                    if match_result["matched"]:
                        log.info(f"[legacy] Matched command in {match_time:.1f}s: "
                                 f"{match_result['command']}")
                        await send_json(websocket, {
                            "type": "result",
                            "text": match_result["command"],
                            "raw_text": raw_text,
                            "slash_command": True,
                            "language": whisper_result.get("language", ""),
                            "duration": whisper_result.get("duration", 0),
                            "transcribe_time": round(transcribe_time, 2),
                            "analysis_time": round(analysis_time, 2),
                            "refine_time": round(match_time, 2),
                            "context": "",
                        })
                        return
                    else:
                        log.info(f"[legacy] No command match for: '{remaining}'")
                except Exception as e:
                    log.error(f"[legacy] Slash command match error: {e}")

        # LLM 整理
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
                    output_language=cfg.get("output_language"),
                ),
            )
            refine_time = time.time() - t0
            refined_text = llm_result["refined_text"]
            log.info(f"Refined in {refine_time:.1f}s (lang={detected_lang})")

        output_lang = cfg.get("output_language") or detected_lang
        raw_text_out = convert_to_traditional_chinese(
            raw_text,
            language=output_lang,
        )
        refined_text_out = convert_to_traditional_chinese(
            refined_text,
            language=output_lang,
        )
        context_hint_out = convert_to_traditional_chinese(
            context_hint or "",
            language=output_lang,
        )

        await send_json(websocket, {
            "type": "result",
            "text": refined_text_out,
            "raw_text": raw_text_out,
            "language": whisper_result.get("language", ""),
            "duration": whisper_result.get("duration", 0),
            "transcribe_time": round(transcribe_time, 2),
            "analysis_time": round(analysis_time, 2),
            "refine_time": round(refine_time, 2),
            "context": context_hint_out,
        })

    except Exception as e:
        log.error(f"Processing error: {e}")
        await send_json(websocket, {"type": "error", "message": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# =============================================================================
#  共用工具
# =============================================================================

async def send_json(websocket, data: dict):
    """送出 JSON 回應。"""
    await websocket.send(json.dumps(data, ensure_ascii=False))


async def main(host: str = "0.0.0.0", port: int = 8991):
    """啟動 WebSocket server。"""
    log.info(f"Starting WebSocket server on ws://{host}:{port}")
    log.info(f"Protocol: streaming (stream_start/end) + legacy (audio_with_screenshot)")
    from voice_input import VISION_SERVERS
    vision_loc = "local" if len(VISION_SERVERS) == 1 and VISION_SERVERS[0] == OLLAMA_URL else f"remote: {','.join(VISION_SERVERS)}"
    log.info(f"Vision model: {VISION_MODEL} ({vision_loc})")

    # 預先載入 Whisper model
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
