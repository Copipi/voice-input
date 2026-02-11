#!/usr/bin/env python3
"""voice-input Mac client: Push-to-Talk → WebSocket → キーボード入力.

使い方:
  1. サーバー側: voice-input serve ws
  2. Mac側:     python3 mac_client.py --server ws://YOUR_SERVER_IP:8991

操作:
  右Option(Alt)キーを押し続ける → 録音
  離す → サーバーに送信 → 整形テキストをペースト

依存 (Mac側):
  pip3 install sounddevice numpy websockets pynput pyperclip

macOS設定:
  システム設定 > プライバシーとセキュリティ > マイク → ターミナルを許可
  システム設定 > プライバシーとセキュリティ > アクセシビリティ → ターミナルを許可
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from typing import List, Optional

import numpy as np
import sounddevice as sd
import websockets
from pynput import keyboard

# --- 設定 ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
HOTKEY = keyboard.Key.alt_l  # 左Optionキー
STREAM_INTERVAL = 2.0  # ストリーミングチャンク送信間隔（秒）

# --- ステータスオーバーレイ（フローティングHUD） ---
# macOS PyObjC (AppKit) で画面下部にワイドなフローティングHUDを表示
# 電光掲示板スタイル: partialテキストが次々に流れ込む
# AppKitが無い環境ではosascript通知にフォールバック
OVERLAY_SCRIPT = r'''
import sys, threading, queue, time

TEXTS = {
    "recording":    "\U0001f3a4 Recording...",
    "screenshot":   "\U0001f4f7 Capturing...",
    "analyzing":    "\U0001f50d Analyzing...",
    "transcribing": "\u270d\ufe0f Transcribing...",
    "partial":      "\u270d\ufe0f ",
    "refining":     "\U0001f4ad Refining...",
    "done":         "\u2705 Done!",
    "error":        "\u274c Error",
}

# 表示テキストの最大文字数（超過分は先頭をカットして最新部分を表示）
MAX_DISPLAY_CHARS = 200

try:
    from AppKit import (
        NSApplication, NSWindow, NSTextField, NSColor, NSFont,
        NSBackingStoreBuffered, NSEvent, NSScreen,
        NSTimer, NSMakeRect, NSView, NSBezierPath,
    )
    from Foundation import NSObject
    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False

if not HAS_APPKIT:
    # Fallback: osascript notifications
    import subprocess
    prev = None
    for line in sys.stdin:
        cmd = line.strip()
        if not cmd or cmd == "HIDE":
            continue
        parts = cmd.split(":", 1)
        stage = parts[0].strip()
        msg = parts[1].strip() if len(parts) > 1 else TEXTS.get(stage, stage)
        if stage != prev:
            subprocess.Popen(
                ["osascript", "-e",
                 'display notification "' + msg + '" with title "voice-input"'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            prev = stage
    sys.exit(0)

# ---- PyObjC Floating HUD (画面下部バー) ----
_cmd_queue = queue.Queue()
_hud_window = None
_hud_label = None
_hud_bg = None
_hud_visible = False
_hide_at = 0

def _stdin_reader():
    for line in sys.stdin:
        cmd = line.strip()
        if cmd:
            _cmd_queue.put(cmd)
    _cmd_queue.put("EXIT")

_bg_green = False

class _RoundedBG(NSView):
    """角丸の半透明背景（通常ダーク、Vision完了で緑）."""
    def drawRect_(self, rect):
        if _bg_green:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.35, 0.12, 0.88).set()
        else:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.10, 0.10, 0.12, 0.88).set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            self.bounds(), 12, 12,
        ).fill()

class _Poller(NSObject):
    """50msごとにstdinキューをチェック."""
    def tick_(self, timer):
        global _hud_visible, _hide_at
        now = time.time()
        # done/error後に自動非表示
        if _hide_at and now >= _hide_at:
            _hide_at = 0
            if _hud_window:
                _hud_window.orderOut_(None)
                _hud_visible = False
        try:
            while True:
                cmd = _cmd_queue.get_nowait()
                if cmd == "EXIT":
                    NSApplication.sharedApplication().terminate_(None)
                    return
                if cmd == "HIDE":
                    if _hud_window:
                        _hud_window.orderOut_(None)
                        _hud_visible = False
                    continue
                parts = cmd.split(":", 1)
                stage = parts[0].strip()
                msg = parts[1].strip() if len(parts) > 1 else TEXTS.get(stage, stage)
                _set_bg_color(stage)
                if stage == "vision_ready":
                    # 背景色だけ変える（テキストはそのまま）
                    if _hud_bg:
                        _hud_bg.setNeedsDisplay_(True)
                    continue
                _show_hud(msg)
                if stage in ("done", "error"):
                    _hide_at = now + 0.2
        except queue.Empty:
            pass

def _set_bg_color(stage):
    """ステージに応じてHUD背景色を切り替え."""
    global _bg_green
    if stage == "vision_ready":
        _bg_green = True
    elif stage in ("recording", "done", "error"):
        _bg_green = False
    if _hud_bg:
        _hud_bg.setNeedsDisplay_(True)

def _show_hud(text):
    """画面下部中央にHUDを表示（電光掲示板スタイル）."""
    global _hud_visible
    # 長文は末尾（最新部分）のみ表示
    if len(text) > MAX_DISPLAY_CHARS:
        text = "\u2026 " + text[-MAX_DISPLAY_CHARS:]
    scr = NSScreen.mainScreen().frame()
    w = _hud_window.frame().size.width
    h = _hud_window.frame().size.height
    x = scr.origin.x + (scr.size.width - w) / 2
    y = scr.origin.y + 24
    _hud_window.setFrameOrigin_((x, y))
    _hud_label.setStringValue_(text)
    if not _hud_visible:
        _hud_window.orderFront_(None)
        _hud_visible = True

def main():
    global _hud_window, _hud_label, _hud_bg
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(2)  # Prohibited: Dockアイコン非表示

    scr = NSScreen.mainScreen().frame()
    W = min(int(scr.size.width * 0.6), 900)
    H = 64
    x = scr.origin.x + (scr.size.width - W) / 2
    y = scr.origin.y + 24

    _hud_window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, W, H), 0, NSBackingStoreBuffered, False,
    )
    _hud_window.setLevel_(25)  # kCGPopUpMenuWindowLevel
    _hud_window.setOpaque_(False)
    _hud_window.setBackgroundColor_(NSColor.clearColor())
    _hud_window.setIgnoresMouseEvents_(True)
    _hud_window.setHasShadow_(True)

    _hud_bg = _RoundedBG.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
    _hud_window.setContentView_(_hud_bg)

    _hud_label = NSTextField.alloc().initWithFrame_(NSMakeRect(16, 8, W - 32, H - 16))
    _hud_label.setEditable_(False)
    _hud_label.setBezeled_(False)
    _hud_label.setDrawsBackground_(False)
    _hud_label.setTextColor_(NSColor.whiteColor())
    _hud_label.setFont_(NSFont.systemFontOfSize_(14))
    _hud_label.setMaximumNumberOfLines_(3)
    _hud_label.cell().setWraps_(True)
    _hud_label.cell().setLineBreakMode_(0)  # NSLineBreakByWordWrapping
    _hud_bg.addSubview_(_hud_label)

    threading.Thread(target=_stdin_reader, daemon=True).start()
    poller = _Poller.alloc().init()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.05, poller, b"tick:", None, True,
    )
    app.run()

if __name__ == "__main__":
    main()
'''


class VoiceInputClient:
    def __init__(self, server_url: str, language: str = "ja",
                 model: str = "gpt-oss:20b", raw: bool = False,
                 prompt: str | None = None, paste: bool = True,
                 use_screenshot: bool = True):
        self.server_url = server_url
        self.language = language
        self.model = model
        self.raw = raw
        self.prompt = prompt
        self.paste = paste
        self.use_screenshot = use_screenshot

        self.recording = False
        self.audio_chunks: list[np.ndarray] = []
        self.stream = None
        self.ws = None
        self.loop = None
        self._connected = False
        self._overlay_proc = None
        self._overlay_script_path = None
        self._stream_timer = None
        self._ctrl_pressed = False
        self._send_enter = True  # Alt only → Enter付き, Alt+Ctrl → Enterなし

    def start(self):
        """メインループを開始."""
        print(f"voice-input client")
        print(f"  Server:   {self.server_url}")
        print(f"  Language: {self.language}")
        print(f"  Model:    {self.model}")
        print(f"  Paste:    {'clipboard+Cmd+V' if self.paste else 'clipboard only'}")
        print(f"  Screenshot: {'ON (context-aware)' if self.use_screenshot else 'OFF'}")
        print(f"")
        print(f"  [左Alt長押し]        → 録音 → ペースト + Enter")
        print(f"  [左Alt+Ctrl長押し]   → 録音 → ペーストのみ（Enterなし）")
        print(f"  [Ctrl+C] → 終了")
        print()

        # ステータスオーバーレイを起動
        self._start_overlay()

        # WebSocket接続をバックグラウンドで管理
        self.loop = asyncio.new_event_loop()
        ws_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        ws_thread.start()

        # キーリスナーをメインスレッドで実行
        with keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        ) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nShutting down.")
                self._stop_overlay()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._maintain_connection())

    async def _maintain_connection(self):
        """WebSocket接続を維持（再接続あり）."""
        while True:
            try:
                async with websockets.connect(
                    self.server_url,
                    max_size=50 * 1024 * 1024,
                    ping_interval=30,
                ) as ws:
                    self.ws = ws
                    self._connected = True
                    print(f"  ✓ Connected to {self.server_url}")

                    # 設定送信
                    await ws.send(json.dumps({
                        "type": "config",
                        "language": self.language,
                        "model": self.model,
                        "raw": self.raw,
                        "prompt": self.prompt,
                    }))

                    # サーバーからのメッセージを受信し続ける
                    async for msg in ws:
                        data = json.loads(msg)
                        self._handle_server_message(data)

            except (websockets.exceptions.ConnectionClosed, OSError,
                    TimeoutError, asyncio.TimeoutError) as e:
                self._connected = False
                self.ws = None
                print(f"  ✗ Connection failed: {e}. Retrying in 3s...")
                await asyncio.sleep(3)

    def _handle_server_message(self, data: dict):
        """サーバーからの応答を処理."""
        msg_type = data.get("type", "")

        if msg_type == "status":
            stage = data.get("stage", "")
            if stage == "analyzing":
                print("\n  ⟳ Analyzing screen...", end="", flush=True)
                self._update_overlay("analyzing")
            elif stage == "transcribing":
                print("  ⟳ Transcribing...", end="", flush=True)
                self._update_overlay("transcribing")
            elif stage == "vision_ready":
                self._update_overlay("vision_ready")
            elif stage == "refining":
                print(" → Refining...", end="", flush=True)
                self._update_overlay("refining")

        elif msg_type == "partial":
            # Whisper生テキスト（LLM整形前）を即座にHUDに流す
            raw = data.get("text", "")
            t_trans = data.get("transcribe_time", 0)
            if raw:
                preview = raw[:60] + ("..." if len(raw) > 60 else "")
                print(f"\r  \u2248 {preview}  ({t_trans:.1f}s)", end="", flush=True)
                self._update_overlay("partial", raw)

        elif msg_type == "result":
            text = data.get("text", "")
            raw = data.get("raw_text", "")
            t_trans = data.get("transcribe_time", 0)
            t_ref = data.get("refine_time", 0)
            dur = data.get("duration", 0)

            enter_label = "+Enter" if self._send_enter else ""
            print(f"\n  Done ({t_trans + t_ref:.1f}s){' ' + enter_label if enter_label else ''}")
            self._update_overlay("done")

            if text:
                self._output_text(text, send_enter=self._send_enter)
                print(f"  → [{dur:.1f}s audio] {text[:80]}{'...' if len(text) > 80 else ''}")
            else:
                print("  → (empty - no speech detected)")

        elif msg_type == "config_ack":
            pass  # 設定確認、表示不要

        elif msg_type == "error":
            print(f"\n  ✗ Error: {data.get('message', 'unknown')}")
            self._update_overlay("error", f"\u2717 {data.get('message', 'Error')[:40]}")

    def _output_text(self, text: str, send_enter: bool = False):
        """テキストをクリップボード経由でペースト.

        send_enter=True の場合、ペースト後にReturnキーも送信する。
        """
        try:
            # macOS pbcopy でクリップボードに設定
            proc = subprocess.Popen(
                ["pbcopy"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(text.encode("utf-8"))

            if self.paste:
                # Cmd+V でペースト
                time.sleep(0.05)
                subprocess.run([
                    "osascript", "-e",
                    'tell application "System Events" to keystroke "v" using command down'
                ], check=True, capture_output=True)

                if send_enter:
                    time.sleep(0.05)
                    subprocess.run([
                        "osascript", "-e",
                        'tell application "System Events" to key code 36'
                    ], check=True, capture_output=True)
        except FileNotFoundError:
            # pbcopy がない環境（Linux等）→ pyperclip フォールバック
            try:
                import pyperclip
                pyperclip.copy(text)
                print("  (clipboard only - paste manually with Cmd+V)")
            except ImportError:
                print(f"  [clipboard unavailable] {text}")

    def _on_key_press(self, key):
        """キー押下時."""
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self._ctrl_pressed = True
        if key == HOTKEY and not self.recording:
            self._start_recording()

    def _on_key_release(self, key):
        """キー離し時."""
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self._ctrl_pressed = False
        if key == HOTKEY and self.recording:
            # Alt離し時にCtrlが押されていればEnter送信しない
            self._send_enter = not self._ctrl_pressed
            self._stop_recording()

    def _start_recording(self):
        """録音開始（ストリーミングモード: スクリーンショット + 2秒ごとにチャンク送信）."""
        if not self._connected:
            print("  ✗ Not connected to server")
            return

        self.recording = True
        self.audio_chunks = []
        print("  ● Recording...", end="", flush=True)
        self._update_overlay("recording")

        # stream_start送信（スクリーンショット付き）
        start_msg = {"type": "stream_start"}
        if self.use_screenshot and self.ws and self._connected:
            screenshot_b64 = self._capture_screenshot()
            if screenshot_b64:
                start_msg["screenshot"] = screenshot_b64
                print(" +screenshot", end="", flush=True)

        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps(start_msg)),
            self.loop,
        )

        # 録音開始
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self.stream.start()

        # ストリーミングタイマー開始（2秒ごとにチャンク送信）
        self._schedule_stream_timer()

    def _schedule_stream_timer(self):
        """STREAM_INTERVAL秒後にチャンク送信をスケジュール."""
        self._stream_timer = threading.Timer(STREAM_INTERVAL, self._on_stream_tick)
        self._stream_timer.daemon = True
        self._stream_timer.start()

    def _on_stream_tick(self):
        """定期的に累積音声をサーバーに送信."""
        if not self.recording or not self.ws or not self._connected:
            return
        self._send_stream_chunk()
        self._schedule_stream_timer()

    def _send_stream_chunk(self):
        """累積音声をWAVでサーバーに送信（ストリーミングチャンク）."""
        if not self.audio_chunks:
            return
        audio_data = np.concatenate(self.audio_chunks)
        duration = len(audio_data) / SAMPLE_RATE
        if duration < 0.5:
            return
        wav_bytes = self._encode_wav(audio_data)
        asyncio.run_coroutine_threadsafe(
            self.ws.send(wav_bytes),
            self.loop,
        )

    def _stop_stream_timer(self):
        """ストリーミングタイマーを停止."""
        if self._stream_timer:
            self._stream_timer.cancel()
            self._stream_timer = None

    def _stop_recording(self):
        """録音停止 → 最終チャンク送信 → stream_end."""
        if not self.recording:
            return

        self.recording = False
        self._stop_stream_timer()

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_chunks:
            print(" (empty)")
            if self.ws and self._connected:
                asyncio.run_coroutine_threadsafe(
                    self.ws.send(json.dumps({"type": "stream_end"})),
                    self.loop,
                )
            return

        audio_data = np.concatenate(self.audio_chunks)
        duration = len(audio_data) / SAMPLE_RATE
        print(f" {duration:.1f}s", end="", flush=True)

        if duration < 0.3:
            print(" (too short, skipped)")
            if self.ws and self._connected:
                asyncio.run_coroutine_threadsafe(
                    self.ws.send(json.dumps({"type": "stream_end"})),
                    self.loop,
                )
            return

        wav_bytes = self._encode_wav(audio_data)
        print(f" ({len(wav_bytes) // 1024}KB)", end="", flush=True)

        if self.ws and self._connected:
            # 最終音声を送信後にstream_end
            async def _send_final():
                await self.ws.send(wav_bytes)
                await self.ws.send(json.dumps({"type": "stream_end"}))
            asyncio.run_coroutine_threadsafe(_send_final(), self.loop)
            print(" → Sent.", end="", flush=True)
            self._update_overlay("refining")
        else:
            print(" ✗ Not connected")
            self._update_overlay("error", "\u2717 Not connected")

    @staticmethod
    def _capture_screenshot():
        """macOS screencaptureでキーボードフォーカスウィンドウのスクリーンショットを取得しbase64で返す.

        osascript (System Events) でフォーカスウィンドウの位置・サイズを取得し、
        screencapture -R でその領域のみキャプチャ。
        スレッドセーフ（subprocess経由）、追加パッケージ不要。
        """
        import base64
        import tempfile
        import os

        tmp_path = tempfile.mktemp(suffix=".png")
        try:
            capture_args = ["screencapture", "-x", "-o"]  # -x=無音, -o=影なし
            window_name = None

            # osascriptでフォーカスウィンドウの位置・サイズを取得（thread-safe）
            try:
                r = subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events"\n'
                     '  set fp to first application process whose frontmost is true\n'
                     '  set fw to first window of fp\n'
                     '  set {px, py} to position of fw\n'
                     '  set {sx, sy} to size of fw\n'
                     '  return (name of fp) & "|" & px & "," & py & "," & sx & "," & sy\n'
                     'end tell'],
                    capture_output=True, text=True, timeout=2,
                )
                if r.returncode == 0 and "|" in r.stdout:
                    name, rect = r.stdout.strip().split("|", 1)
                    capture_args.extend(["-R", rect])
                    window_name = name
            except Exception:
                pass

            # osascript失敗時: 全スクリーンにフォールバック
            if "-R" not in capture_args:
                capture_args.append("-C")

            capture_args.append(tmp_path)
            result = subprocess.run(
                capture_args,
                capture_output=True, timeout=3,
            )
            if result.returncode != 0 or not os.path.exists(tmp_path):
                return None

            if window_name:
                print(f" [{window_name}]", end="", flush=True)

            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice録音コールバック."""
        if status:
            print(f"\n  Audio warning: {status}", file=sys.stderr)
        if self.recording:
            self.audio_chunks.append(indata.copy())

    @staticmethod
    def _encode_wav(audio: np.ndarray) -> bytes:
        """numpy配列をWAVバイト列にエンコード."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    # --- ステータスオーバーレイ管理 ---

    def _start_overlay(self):
        """フローティングステータスオーバーレイを起動."""
        try:
            fd, path = tempfile.mkstemp(suffix=".py", prefix="voice_overlay_")
            with os.fdopen(fd, "w") as f:
                f.write(OVERLAY_SCRIPT)
            self._overlay_script_path = path

            self._overlay_proc = subprocess.Popen(
                [sys.executable, path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("  ✓ Status overlay started")
        except Exception as e:
            print(f"  (overlay unavailable: {e})")
            self._overlay_proc = None

    def _update_overlay(self, stage: str, custom_msg: str | None = None):
        """オーバーレイのステータスを更新."""
        if not self._overlay_proc or not self._overlay_proc.stdin:
            return
        try:
            if custom_msg:
                line = f"{stage}:{custom_msg}\n"
            else:
                line = f"{stage}\n"
            self._overlay_proc.stdin.write(line.encode("utf-8"))
            self._overlay_proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._overlay_proc = None

    def _hide_overlay(self):
        """オーバーレイを非表示にする."""
        if not self._overlay_proc or not self._overlay_proc.stdin:
            return
        try:
            self._overlay_proc.stdin.write(b"HIDE\n")
            self._overlay_proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _stop_overlay(self):
        """オーバーレイプロセスを終了."""
        if self._overlay_proc:
            try:
                self._overlay_proc.terminate()
                self._overlay_proc.wait(timeout=2)
            except Exception:
                pass
        if self._overlay_script_path:
            try:
                os.unlink(self._overlay_script_path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="voice-input Mac client: Push-to-Talk voice input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup (Mac):
  pip3 install sounddevice numpy websockets pynput pyperclip

  # Grant permissions in System Settings:
  #   Privacy & Security > Microphone > Terminal
  #   Privacy & Security > Accessibility > Terminal

Usage:
  python3 mac_client.py --server ws://YOUR_SERVER_IP:8991
  python3 mac_client.py --server ws://your-gpu-server:8991 --language en
        """,
    )
    default_server = os.environ.get("VOICE_INPUT_SERVER", "ws://localhost:8991")
    parser.add_argument(
        "-s", "--server",
        default=default_server,
        help=f"WebSocket server URL (default: {default_server})",
    )
    parser.add_argument("-l", "--language", default="ja", help="Language (default: ja)")
    parser.add_argument("-m", "--model", default="gpt-oss:20b", help="Ollama model")
    parser.add_argument("--raw", action="store_true", help="Skip LLM refinement")
    parser.add_argument("-p", "--prompt", default=None, help="Custom refinement prompt")
    parser.add_argument(
        "--no-paste",
        action="store_true",
        help="Clipboard only, don't auto-paste with Cmd+V",
    )
    parser.add_argument(
        "--no-screenshot",
        action="store_true",
        help="Disable screenshot context analysis",
    )
    args = parser.parse_args()

    client = VoiceInputClient(
        server_url=args.server,
        language=args.language,
        model=args.model,
        raw=args.raw,
        prompt=args.prompt,
        paste=not args.no_paste,
        use_screenshot=not args.no_screenshot,
    )
    client.start()


if __name__ == "__main__":
    main()
