#!/usr/bin/env python3
"""voice-input Windows client: Push-to-Talk → WebSocket → paste.

This is a Windows-oriented rewrite of mac_client.py.

Protocol (client -> server):
  - JSON control messages: {"type": "config"}, {"type": "stream_start"}, {"type": "stream_end"}
  - Binary messages: WAV bytes (int16, 16kHz, mono)

Requires (client side):
  pip install sounddevice numpy websockets pynput pyperclip
Optional (screen context):
  pip install pillow

Notes:
  - Default hotkey is F8 (to avoid Alt/menu conflicts on Windows).
  - Hold hotkey to record, release to transcribe/refine, then paste.
  - Hold Ctrl while releasing hotkey to paste without Enter.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import queue
import sys
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
from pynput import keyboard

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
STREAM_INTERVAL = 2.0
MAX_DISPLAY_CHARS = 200


# -------------------------
# Hotkey mapping
# -------------------------

def _parse_hotkey(name: str) -> keyboard.Key | keyboard.KeyCode:
    name = (name or "").strip().lower()
    mapping: dict[str, keyboard.Key | keyboard.KeyCode] = {
        "f8": keyboard.Key.f8,
        "f9": keyboard.Key.f9,
        "f10": keyboard.Key.f10,
        "alt_l": keyboard.Key.alt_l,
        "alt_r": keyboard.Key.alt_r,
        "shift_l": keyboard.Key.shift_l,
        "shift_r": keyboard.Key.shift_r,
    }
    if name in mapping:
        return mapping[name]
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(f"Unsupported hotkey: {name}")


# -------------------------
# Simple HUD (tkinter)
# -------------------------

@dataclass
class HudMessage:
    stage: str
    text: str


class StatusHud:
    """A tiny always-on-top status bar.

    Uses tkinter (bundled with CPython on Windows). If tkinter is unavailable,
    this becomes a no-op and the client prints to console only.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._q: "queue.Queue[HudMessage]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._has_tk = False

        self._bg_normal = "#1a1a1f"
        self._bg_green = "#135a2a"
        self._fg = "#ffffff"

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import tkinter as tk  # noqa: F401
            self._has_tk = True
        except Exception:
            self._has_tk = False
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(HudMessage("exit", ""))
        except Exception:
            pass

    def update(self, stage: str, text: str) -> None:
        if not self.enabled or not self._has_tk:
            return
        if text and len(text) > MAX_DISPLAY_CHARS:
            text = "… " + text[-MAX_DISPLAY_CHARS:]
        try:
            self._q.put_nowait(HudMessage(stage, text))
        except Exception:
            pass

    def _run(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        # Slight transparency if supported
        try:
            root.attributes("-alpha", 0.92)
        except Exception:
            pass

        label = tk.Label(
            root,
            text="",
            bg=self._bg_normal,
            fg=self._fg,
            font=("Segoe UI", 11),
            justify="left",
            anchor="w",
            padx=12,
            pady=10,
        )
        label.pack(fill="both", expand=True)

        def place_bottom_center() -> None:
            root.update_idletasks()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            w = min(int(screen_w * 0.62), 900)
            h = 58
            x = int((screen_w - w) / 2)
            y = int(screen_h - h - 32)
            root.geometry(f"{w}x{h}+{x}+{y}")

        place_bottom_center()

        def poll() -> None:
            if self._stop.is_set():
                try:
                    root.destroy()
                except Exception:
                    pass
                return

            try:
                while True:
                    msg = self._q.get_nowait()
                    if msg.stage == "exit":
                        self._stop.set()
                        continue

                    if msg.stage == "vision_ready":
                        label.configure(bg=self._bg_green)
                        continue

                    if msg.stage in ("recording", "done", "error"):
                        label.configure(bg=self._bg_normal)

                    label.configure(text=msg.text)
            except queue.Empty:
                pass

            root.after(50, poll)

        root.after(50, poll)
        root.mainloop()


# -------------------------
# Windows client
# -------------------------


class VoiceInputWinClient:
    def __init__(
        self,
        server_url: str,
        language: str = "ja",
        model: str = "gpt-oss:20b",
        raw: bool = False,
        prompt: str | None = None,
        paste: bool = True,
        use_screenshot: bool = True,
        hotkey: keyboard.Key | keyboard.KeyCode = keyboard.Key.f8,
        overlay: bool = True,
    ):
        self.server_url = server_url
        self.language = language
        self.model = model
        self.raw = raw
        self.prompt = prompt
        self.paste = paste
        self.use_screenshot = use_screenshot
        self.hotkey = hotkey

        self.recording = False
        self.audio_chunks: list[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False

        self._stream_timer: Optional[threading.Timer] = None
        self._ctrl_pressed = False
        self._send_enter = True

        self.hud = StatusHud(enabled=overlay)

    def start(self) -> None:
        print("voice-input (Windows client)")
        print(f"  Server:     {self.server_url}")
        print(f"  Language:   {self.language}")
        print(f"  Model:      {self.model}")
        print(f"  Hotkey:     {self._hotkey_label()}")
        print(f"  Paste:      {'Ctrl+V' if self.paste else 'clipboard only'}")
        print(f"  Screenshot: {'ON' if self.use_screenshot else 'OFF'}")
        print("")
        print(f"  [hold {self._hotkey_label()}]        → record → paste + Enter")
        print(f"  [hold {self._hotkey_label()} + Ctrl] → record → paste only (no Enter)")
        print("  [Ctrl+C] → quit")
        print("")

        self.hud.start()

        self.loop = asyncio.new_event_loop()
        ws_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        ws_thread.start()

        with keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        ) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nShutting down.")
            finally:
                self.hud.stop()

    def _run_event_loop(self) -> None:
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._maintain_connection())

    async def _maintain_connection(self) -> None:
        while True:
            try:
                async with websockets.connect(
                    self.server_url,
                    max_size=50 * 1024 * 1024,
                    ping_interval=30,
                ) as ws:
                    self.ws = ws
                    self._connected = True
                    print(f"   Connected: {self.server_url}")

                    cfg = {
                        "type": "config",
                        "language": self.language,
                        "model": self.model,
                        "raw": self.raw,
                        "prompt": self.prompt,
                        "slash_commands": self._scan_slash_commands(),
                    }
                    await ws.send(json.dumps(cfg))

                    async for msg in ws:
                        data = json.loads(msg)
                        self._handle_server_message(data)

            except (websockets.exceptions.ConnectionClosed, OSError, TimeoutError, asyncio.TimeoutError) as e:
                self._connected = False
                self.ws = None
                print(f"   Connection failed: {e}. Retrying in 3s...")
                await asyncio.sleep(3)

    def _handle_server_message(self, data: dict) -> None:
        msg_type = data.get("type", "")

        if msg_type == "status":
            stage = data.get("stage", "")
            if stage == "analyzing":
                self._hud_stage("analyzing", " Analyzing screen...")
            elif stage == "transcribing":
                self._hud_stage("transcribing", " Transcribing...")
            elif stage == "vision_ready":
                self.hud.update("vision_ready", "")
            elif stage == "refining":
                self._hud_stage("refining", " Refining...")
            elif stage == "matching_command":
                self._hud_stage("matching_command", " Matching command...")

        elif msg_type == "partial":
            raw = data.get("text", "")
            if raw:
                preview = raw[:60] + ("..." if len(raw) > 60 else "")
                print(f"\r   {preview}", end="", flush=True)
                self.hud.update("partial", raw)

        elif msg_type == "result":
            text = data.get("text", "")
            dur = data.get("duration", 0)
            is_slash = data.get("slash_command", False)
            send_enter = False if is_slash else self._send_enter

            print("\n   Done")
            self.hud.update("done", " Done")

            if text:
                self._output_text(text, send_enter=send_enter)
                if is_slash:
                    print(f"   {text}")
                else:
                    t = text[:80] + ("..." if len(text) > 80 else "")
                    print(f"   [{dur:.1f}s audio] {t}")
            else:
                print("   (empty - no speech detected)")

        elif msg_type == "error":
            message = data.get("message", "unknown")
            print(f"\n   Error: {message}")
            self.hud.update("error", f" {message}")

    def _hud_stage(self, stage: str, text: str) -> None:
        self.hud.update(stage, text)

    def _on_key_press(self, key) -> None:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_pressed = True
        if key == self.hotkey and not self.recording:
            self._start_recording()

    def _on_key_release(self, key) -> None:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_pressed = False
        if key == self.hotkey and self.recording:
            self._send_enter = not self._ctrl_pressed
            self._stop_recording()

    def _start_recording(self) -> None:
        if not self._connected or not self.ws or not self.loop:
            print("   Not connected to server")
            return

        self.recording = True
        self.audio_chunks = []
        print("   Recording...", end="", flush=True)
        self.hud.update("recording", " Recording...")

        start_msg: dict = {"type": "stream_start"}

        if self.use_screenshot and not self.raw:
            screenshot_b64 = self._capture_screenshot_b64()
            if screenshot_b64:
                start_msg["screenshot"] = screenshot_b64
                print(" +screenshot", end="", flush=True)
            else:
                print(" +screenshot(FAILED)", end="", flush=True)

        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps(start_msg)),
            self.loop,
        )

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self.stream.start()
        self._schedule_stream_timer()

    def _stop_recording(self) -> None:
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
            self._send_stream_end_only()
            return

        audio_data = np.concatenate(self.audio_chunks)
        duration = len(audio_data) / SAMPLE_RATE
        print(f" {duration:.1f}s", end="", flush=True)

        if duration < 0.3:
            print(" (too short, skipped)")
            self._send_stream_end_only()
            return

        wav_bytes = self._encode_wav(audio_data)
        print(f" ({len(wav_bytes) // 1024}KB)", end="", flush=True)

        if not self.ws or not self.loop or not self._connected:
            print("  Not connected")
            self.hud.update("error", " Not connected")
            return

        async def _send_final() -> None:
            assert self.ws is not None
            await self.ws.send(wav_bytes)
            await self.ws.send(json.dumps({"type": "stream_end"}))

        asyncio.run_coroutine_threadsafe(_send_final(), self.loop)
        print("  Sent.", end="", flush=True)
        self.hud.update("refining", " Processing...")

    def _schedule_stream_timer(self) -> None:
        self._stream_timer = threading.Timer(STREAM_INTERVAL, self._on_stream_tick)
        self._stream_timer.daemon = True
        self._stream_timer.start()

    def _stop_stream_timer(self) -> None:
        if self._stream_timer:
            self._stream_timer.cancel()
            self._stream_timer = None

    def _on_stream_tick(self) -> None:
        if not self.recording or not self.ws or not self.loop or not self._connected:
            return
        self._send_stream_chunk()
        self._schedule_stream_timer()

    def _send_stream_chunk(self) -> None:
        if not self.audio_chunks or not self.ws or not self.loop:
            return
        audio_data = np.concatenate(self.audio_chunks)
        duration = len(audio_data) / SAMPLE_RATE
        if duration < 0.5:
            return
        wav_bytes = self._encode_wav(audio_data)
        asyncio.run_coroutine_threadsafe(self.ws.send(wav_bytes), self.loop)

    def _send_stream_end_only(self) -> None:
        if self.ws and self.loop and self._connected:
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps({"type": "stream_end"})),
                self.loop,
            )

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"\n  Audio warning: {status}", file=sys.stderr)
        if self.recording:
            self.audio_chunks.append(indata.copy())

    @staticmethod
    def _encode_wav(audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    @staticmethod
    def _capture_screenshot_b64() -> Optional[str]:
        """Capture a screenshot and return raw base64 PNG (no data: prefix)."""
        try:
            from PIL import ImageGrab
        except Exception:
            return None

        try:
            img = ImageGrab.grab(all_screens=True)
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            return base64.b64encode(bio.getvalue()).decode("ascii")
        except Exception:
            return None

    @staticmethod
    def _scan_slash_commands() -> list[dict]:
        from pathlib import Path

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
                end_idx = text.index("---", 3)
                frontmatter = text[3:end_idx].strip()
                meta: dict[str, str] = {}
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, _, val = line.partition(":")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key in ("name", "description", "argument-hint"):
                            meta[key] = val

                commands.append(
                    {
                        "name": meta.get("name", skill_dir.name),
                        "description": meta.get("description", "")[:100],
                        "args": meta.get("argument-hint", ""),
                    }
                )
            except Exception:
                continue

        return commands

    def _output_text(self, text: str, send_enter: bool) -> None:
        """Copy to clipboard and optionally paste with Ctrl+V (+ Enter)."""
        try:
            import pyperclip

            pyperclip.copy(text)
        except Exception as e:
            print(f"   Clipboard failed: {e}")
            print(text)
            return

        if not self.paste:
            return

        controller = keyboard.Controller()
        time.sleep(0.03)
        with controller.pressed(keyboard.Key.ctrl):
            controller.press("v")
            controller.release("v")
        if send_enter:
            time.sleep(0.02)
            controller.press(keyboard.Key.enter)
            controller.release(keyboard.Key.enter)

    def _hotkey_label(self) -> str:
        hk = self.hotkey
        if isinstance(hk, keyboard.Key):
            return hk.name.upper() if hk.name else str(hk)
        if isinstance(hk, keyboard.KeyCode):
            return (hk.char or "?").upper()
        return str(hk)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="voice-input Windows client: Push-to-Talk voice input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dependencies:
  pip install sounddevice numpy websockets pynput pyperclip
Optional (screen context):
  pip install pillow

Example:
  python win_client.py --server ws://localhost:8991 --model gemma3:4b --language zh
""",
    )

    default_server = os.environ.get("VOICE_INPUT_SERVER", "ws://localhost:8991")
    parser.add_argument("-s", "--server", default=default_server, help=f"WebSocket server URL (default: {default_server})")
    parser.add_argument("-l", "--language", default="zh", help="Language hint (default: ja)")
    parser.add_argument("-m", "--model", default="gpt-oss:20b", help="Ollama model for refinement")
    parser.add_argument("--raw", action="store_true", help="Skip LLM refinement")
    parser.add_argument("-p", "--prompt", default=None, help="Custom refinement prompt")
    parser.add_argument("--no-paste", action="store_true", help="Clipboard only; don't auto-paste")
    parser.add_argument("--no-screenshot", action="store_true", help="Disable screen context screenshot")
    parser.add_argument("--no-overlay", action="store_true", help="Disable the small status HUD")
    parser.add_argument(
        "--hotkey",
        default="f8",
        help="Hotkey name: f8|f9|f10|alt_l|alt_r|shift_l|shift_r or a single character (default: f8)",
    )

    args = parser.parse_args()

    try:
        hotkey = _parse_hotkey(args.hotkey)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    client = VoiceInputWinClient(
        server_url=args.server,
        language=args.language,
        model=args.model,
        raw=args.raw,
        prompt=args.prompt,
        paste=not args.no_paste,
        use_screenshot=not args.no_screenshot,
        hotkey=hotkey,
        overlay=not args.no_overlay,
    )
    client.start()


if __name__ == "__main__":
    main()
