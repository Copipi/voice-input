#!/usr/bin/env python3
"""voice-input Mac client：Push-to-Talk → WebSocket → 鍵盤輸入。

用法：
    1. server 端：voice-input serve ws
    2. Mac 端：  python3 mac_client.py --server ws://YOUR_SERVER_IP:8991

操作：
    按住右 Option(Alt) 鍵 → 錄音
    放開 → 傳送到 server → 貼上整理後文字

依賴（Mac 端）：
    pip3 install sounddevice numpy websockets pynput pyperclip

macOS 設定：
    系統設定 > 隱私權與安全性 > 麥克風 → 允許 Terminal
    系統設定 > 隱私權與安全性 > 輔助使用 → 允許 Terminal
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
HOTKEY = keyboard.Key.alt_l  # 左 Option 鍵
STREAM_INTERVAL = 2.0  # 串流 chunk 傳送間隔（秒）
MIN_AX_TEXT_LEN = 20   # AX 文字擷取的最小字數（少於此值就回退到 Vision）

# --- 狀態覆蓋層（浮動 HUD） ---
# 在 macOS 上用 PyObjC（AppKit）於螢幕底部顯示寬版浮動 HUD
# 跑馬燈風格：partial 文字會持續流入
# 若環境沒有 AppKit，則回退為 osascript 通知
OVERLAY_SCRIPT = r'''
import sys, threading, queue, time

TEXTS = {
    "recording":         "\U0001f3a4 Recording...",
    "screenshot":        "\U0001f4f7 Capturing...",
    "analyzing":         "\U0001f50d Analyzing...",
    "transcribing":      "\u270d\ufe0f Transcribing...",
    "partial":           "\u270d\ufe0f ",
    "refining":          "\U0001f4ad Refining...",
    "matching_command":  "\u2318 Matching command...",
    "done":              "\u2705 Done!",
    "error":             "\u274c Error",
}

# 顯示文字最大長度（超過時會裁掉前段，只顯示最新尾端）
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

# ---- PyObjC Floating HUD（螢幕底部橫條） ----
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
    """圓角半透明背景（平時深色、Vision 完成時變綠）。"""
    def drawRect_(self, rect):
        if _bg_green:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.35, 0.12, 0.88).set()
        else:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.10, 0.10, 0.12, 0.88).set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            self.bounds(), 12, 12,
        ).fill()

class _Poller(NSObject):
    """每 50ms 檢查一次 stdin 佇列。"""
    def tick_(self, timer):
        global _hud_visible, _hide_at
        now = time.time()
        # done/error 後自動隱藏
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
                    # 只改背景色（文字維持不變）
                    if _hud_bg:
                        _hud_bg.setNeedsDisplay_(True)
                    continue
                _show_hud(msg)
                if stage in ("done", "error"):
                    _hide_at = now + 0.2
        except queue.Empty:
            pass

def _set_bg_color(stage):
    """依 stage 切換 HUD 背景色。"""
    global _bg_green
    if stage == "vision_ready":
        _bg_green = True
    elif stage in ("recording", "done", "error"):
        _bg_green = False
    if _hud_bg:
        _hud_bg.setNeedsDisplay_(True)

def _show_hud(text):
    """在螢幕下方中央顯示 HUD（跑馬燈風格）。"""
    global _hud_visible
    # 長文只顯示尾端（最新部分）
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
    app.setActivationPolicy_(2)  # Prohibited：不顯示 Dock 圖示

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
                 model: str = "gpt-oss:20b", output_language: str | None = None,
                 raw: bool = False,
                 prompt: str | None = None, paste: bool = True,
                 use_screenshot: bool = True):
        self.server_url = server_url
        self.language = language
        self.model = model
        self.output_language = output_language
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
        self._send_enter = True  # 只按 Alt → 會送 Enter；Alt+Ctrl → 不送 Enter

    def start(self):
        """啟動主迴圈。"""
        print(f"voice-input client")
        print(f"  Server:   {self.server_url}")
        print(f"  Language: {self.language}")
        if self.output_language:
            print(f"  Output:   {self.output_language} (forced)")
        print(f"  Model:    {self.model}")
        print(f"  Paste:    {'clipboard+Cmd+V' if self.paste else 'clipboard only'}")
        print(f"  Screenshot: {'ON (context-aware)' if self.use_screenshot else 'OFF'}")
        print(f"")
        print(f"  [按住左 Alt]        → 錄音 → 貼上 + Enter")
        print(f"  [按住左 Alt+Ctrl]   → 錄音 → 只貼上（不送 Enter）")
        print(f"  [Ctrl+C] → 結束")
        print()

        # 啟動狀態 HUD
        self._start_overlay()

        # 在背景管理 WebSocket 連線
        self.loop = asyncio.new_event_loop()
        ws_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        ws_thread.start()

        # 在主執行緒啟動按鍵監聽
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
        """維持 WebSocket 連線（含自動重連）。"""
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

                    # 傳送設定（包含 slash command 清單）
                    config_msg = {
                        "type": "config",
                        "language": self.language,
                        "model": self.model,
                        "output_language": self.output_language,
                        "raw": self.raw,
                        "prompt": self.prompt,
                        "slash_commands": self._scan_slash_commands(),
                    }
                    await ws.send(json.dumps(config_msg))

                    # 持續接收 server 訊息
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
        """處理 server 回應。"""
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
            elif stage == "matching_command":
                print(" → Matching command...", end="", flush=True)
                self._update_overlay("matching_command")

        elif msg_type == "partial":
            # 立即把 Whisper 原始文字（尚未經 LLM 整理）送到 HUD
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
            is_slash = data.get("slash_command", False)

            # Slash command 不送 Enter（方便確認）
            send_enter = False if is_slash else self._send_enter

            if is_slash:
                print(f"\n  ⌘ Command ({t_ref:.1f}s)")
            else:
                enter_label = "+Enter" if send_enter else ""
                print(f"\n  Done ({t_trans + t_ref:.1f}s){' ' + enter_label if enter_label else ''}")
            self._update_overlay("done")

            if text:
                self._output_text(text, send_enter=send_enter)
                if is_slash:
                    print(f"  → {text}")
                else:
                    print(f"  → [{dur:.1f}s audio] {text[:80]}{'...' if len(text) > 80 else ''}")
            else:
                print("  → (empty - no speech detected)")

        elif msg_type == "config_ack":
            pass  # 設定確認，不需要顯示

        elif msg_type == "error":
            print(f"\n  ✗ Error: {data.get('message', 'unknown')}")
            self._update_overlay("error", f"\u2717 {data.get('message', 'Error')[:40]}")

    def _output_text(self, text: str, send_enter: bool = False):
        """透過剪貼簿貼上文字。

        當 send_enter=True 時，貼上後也會送出 Return。
        """
        try:
            # 使用 macOS pbcopy 設定剪貼簿
            proc = subprocess.Popen(
                ["pbcopy"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(text.encode("utf-8"))

            if self.paste:
                # Cmd+V 貼上
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
            # 沒有 pbcopy 的環境（例如 Linux）→ 回退到 pyperclip
            try:
                import pyperclip
                pyperclip.copy(text)
                print("  (clipboard only - paste manually with Cmd+V)")
            except ImportError:
                print(f"  [clipboard unavailable] {text}")

    def _on_key_press(self, key):
        """按鍵按下時。"""
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self._ctrl_pressed = True
        if key == HOTKEY and not self.recording:
            self._start_recording()

    def _on_key_release(self, key):
        """按鍵放開時。"""
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self._ctrl_pressed = False
        if key == HOTKEY and self.recording:
            # 放開 Alt 時若仍按住 Ctrl，則不送出 Enter
            self._send_enter = not self._ctrl_pressed
            self._stop_recording()

    def _start_recording(self):
        """開始錄音（串流模式：AX 文字 or 截圖 + 每 2 秒送一次 chunk）。"""
        if not self._connected:
            print("  ✗ Not connected to server")
            return

        self.recording = True
        self.audio_chunks = []
        print("  ● Recording...", end="", flush=True)
        self._update_overlay("recording")

        # 傳送 stream_start（優先 AX 文字，不足則使用截圖）
        start_msg = {"type": "stream_start"}
        if self.use_screenshot and self.ws and self._connected:
            # 嘗試擷取 AX 文字
            ax_text, ax_app = self._extract_ax_text()
            if ax_text and len(ax_text) >= MIN_AX_TEXT_LEN:
                start_msg["text_context"] = ax_text
                app_label = f" [{ax_app}]" if ax_app else ""
                print(f" +ax_text({len(ax_text)}ch){app_label}", end="", flush=True)
            else:
                # AX 不足 → 回退到截圖
                screenshot_b64 = self._capture_screenshot()
                if screenshot_b64:
                    start_msg["screenshot"] = screenshot_b64
                    print(" +screenshot", end="", flush=True)

        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps(start_msg)),
            self.loop,
        )

        # 開始錄音
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self.stream.start()

        # 啟動串流計時器（每 2 秒送一次 chunk）
        self._schedule_stream_timer()

    def _schedule_stream_timer(self):
        """在 STREAM_INTERVAL 秒後排程送出 chunk。"""
        self._stream_timer = threading.Timer(STREAM_INTERVAL, self._on_stream_tick)
        self._stream_timer.daemon = True
        self._stream_timer.start()

    def _on_stream_tick(self):
        """定期把累積音訊送到 server。"""
        if not self.recording or not self.ws or not self._connected:
            return
        self._send_stream_chunk()
        self._schedule_stream_timer()

    def _send_stream_chunk(self):
        """把累積音訊以 WAV 送到 server（串流 chunk）。"""
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
        """停止串流計時器。"""
        if self._stream_timer:
            self._stream_timer.cancel()
            self._stream_timer = None

    def _stop_recording(self):
        """停止錄音 → 傳送最後 chunk → stream_end。"""
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
            # 送出最後音訊後再送 stream_end
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
    def _scan_slash_commands() -> list[dict]:
        """從 ~/.claude/skills/ 收集 slash command 清單。"""
        from pathlib import Path

        # Claude Code 內建指令
        commands = [
            {"name": "help", "description": "Show help information", "args": ""},
            {"name": "clear", "description": "Clear conversation history", "args": ""},
            {"name": "compact", "description": "Compact conversation context", "args": "[instructions]"},
            {"name": "cost", "description": "Show token usage and cost", "args": ""},
            {"name": "doctor", "description": "Check Claude Code setup", "args": ""},
            {"name": "init", "description": "Initialize project CLAUDE.md", "args": ""},
            {"name": "login", "description": "Login to Anthropic", "args": ""},
            {"name": "logout", "description": "Logout from Anthropic", "args": ""},
            {"name": "fast", "description": "Toggle fast mode", "args": ""},
        ]

        skills_dir = Path.home() / ".claude" / "skills"
        if not skills_dir.is_dir():
            return commands

        for skill_dir in sorted(skills_dir.iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                continue
            try:
                text = skill_file.read_text(encoding="utf-8")
                if not text.startswith("---"):
                    continue
                # 解析 YAML front matter（被 --- 包起來的區段）
                end_idx = text.index("---", 3)
                frontmatter = text[3:end_idx].strip()
                meta = {}
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, _, val = line.partition(":")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key in ("name", "description", "argument-hint"):
                            meta[key] = val

                commands.append({
                    "name": meta.get("name", skill_dir.name),
                    "description": meta.get("description", "")[:100],
                    "args": meta.get("argument-hint", ""),
                })
            except Exception:
                continue

        return commands

    @staticmethod
    def _capture_screenshot():
        """用 macOS screencapture 擷取鍵盤焦點視窗的截圖並回傳 base64。

        會用 osascript（System Events）取得焦點視窗的位置/尺寸，
        再用 screencapture -R 只截取該區域。
        走 subprocess，因此相對 thread-safe，且不需要額外套件。
        """
        import base64
        import tempfile
        import os

        tmp_path = tempfile.mktemp(suffix=".png")
        try:
            capture_args = ["screencapture", "-x", "-o"]  # -x=靜音，-o=不含陰影
            window_name = None

            # 透過 osascript 取得焦點視窗的位置/尺寸（thread-safe）
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

            # osascript 失敗時：回退到全螢幕
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

    @staticmethod
    def _extract_ax_text() -> tuple[str | None, str | None]:
        """用 macOS Accessibility API 擷取焦點視窗的顯示文字。

        Returns:
            (text, app_name) — 成功時回傳文字與 app 名稱；失敗時回傳 (None, None)
        """
        MAX_DEPTH = 15
        MAX_ELEMENTS = 500
        TIMEOUT_MS = 200

        try:
            from AppKit import NSWorkspace
            from ApplicationServices import (
                AXUIElementCreateApplication,
                AXUIElementCopyAttributeValue,
            )
            from CoreFoundation import CFEqual
            import Quartz
        except ImportError:
            return (None, None)

        try:
            t0 = time.time()
            deadline = t0 + TIMEOUT_MS / 1000.0

            # 取得前景 app 的 PID
            frontmost = NSWorkspace.sharedWorkspace().frontmostApplication()
            if not frontmost:
                return (None, None)
            pid = frontmost.processIdentifier()
            app_name = frontmost.localizedName() or "Unknown"

            app_ref = AXUIElementCreateApplication(pid)

            # 取得前景視窗
            err, focused_window = AXUIElementCopyAttributeValue(
                app_ref, "AXFocusedWindow", None
            )
            if err != 0 or focused_window is None:
                return (None, None)

            # 文字擷取目標 role
            text_roles = {
                "AXStaticText", "AXTextField", "AXTextArea",
                "AXLink", "AXHeading", "AXCell",
            }
            # 遞迴走訪目標容器 role
            container_roles = {
                "AXGroup", "AXScrollArea", "AXWebArea", "AXWindow",
                "AXSplitGroup", "AXTabGroup", "AXList", "AXTable",
                "AXRow", "AXColumn", "AXOutline", "AXBrowser",
                "AXLayoutArea", "AXSheet", "AXDrawer",
            }

            texts = []
            element_count = [0]

            def _walk(element, depth):
                if depth > MAX_DEPTH:
                    return
                if element_count[0] >= MAX_ELEMENTS:
                    return
                if time.time() > deadline:
                    return

                element_count[0] += 1

                # 取得 role
                err, role = AXUIElementCopyAttributeValue(element, "AXRole", None)
                if err != 0 or role is None:
                    return
                role = str(role)

                # 略過密碼欄位
                err2, subrole = AXUIElementCopyAttributeValue(element, "AXSubrole", None)
                if err2 == 0 and subrole and str(subrole) == "AXSecureTextField":
                    return

                # 從文字元素讀取 AXValue
                if role in text_roles:
                    err_v, value = AXUIElementCopyAttributeValue(element, "AXValue", None)
                    if err_v == 0 and value and isinstance(value, str) and value.strip():
                        texts.append(value.strip())
                    else:
                        # AXTitle fallback
                        err_t, title = AXUIElementCopyAttributeValue(element, "AXTitle", None)
                        if err_t == 0 and title and isinstance(title, str) and title.strip():
                            texts.append(title.strip())

                # 容器或文字元素：走訪子節點
                if role in container_roles or role in text_roles:
                    err_c, children = AXUIElementCopyAttributeValue(
                        element, "AXChildren", None
                    )
                    if err_c == 0 and children:
                        for child in children:
                            if element_count[0] >= MAX_ELEMENTS:
                                break
                            if time.time() > deadline:
                                break
                            _walk(child, depth + 1)

            _walk(focused_window, 0)

            if not texts:
                return (None, None)

            combined = "\n".join(texts)
            elapsed_ms = (time.time() - t0) * 1000
            # 去除重複行並保留順序
            seen = set()
            unique_lines = []
            for line in combined.split("\n"):
                stripped = line.strip()
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    unique_lines.append(stripped)
            result = "\n".join(unique_lines)

            return (result, app_name) if result else (None, None)

        except Exception:
            return (None, None)

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 錄音 callback。"""
        if status:
            print(f"\n  Audio warning: {status}", file=sys.stderr)
        if self.recording:
            self.audio_chunks.append(indata.copy())

    @staticmethod
    def _encode_wav(audio: np.ndarray) -> bytes:
        """把 numpy 陣列編碼成 WAV 位元組。"""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    # --- 狀態 HUD 管理 ---

    def _start_overlay(self):
        """啟動浮動狀態 HUD。"""
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
        """更新 HUD 狀態。"""
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
        """隱藏 HUD。"""
        if not self._overlay_proc or not self._overlay_proc.stdin:
            return
        try:
            self._overlay_proc.stdin.write(b"HIDE\n")
            self._overlay_proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _stop_overlay(self):
        """結束 HUD 程序。"""
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
    parser.add_argument("--output-language", default=None, help="Force final output language (e.g., en, zh, ja, ko)")
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
        output_language=args.output_language,
        raw=args.raw,
        prompt=args.prompt,
        paste=not args.no_paste,
        use_screenshot=not args.no_screenshot,
    )
    client.start()


if __name__ == "__main__":
    main()
