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
    - Default hold-to-talk hotkey is F9.
    - Default toggle hotkey is F10 (continuous dictation ON/OFF).
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
from pynput import keyboard as pynput_keyboard

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
STREAM_INTERVAL = 2.0
MAX_DISPLAY_CHARS = 200


# -------------------------
# Hotkey mapping
# -------------------------

def _is_combo_hotkey(name: str) -> bool:
    return "+" in (name or "")


def _parse_hotkey(name: str) -> pynput_keyboard.Key | pynput_keyboard.KeyCode:
    name = (name or "").strip().lower()
    mapping: dict[str, pynput_keyboard.Key | pynput_keyboard.KeyCode] = {
        "f8": pynput_keyboard.Key.f8,
        "f9": pynput_keyboard.Key.f9,
        "f10": pynput_keyboard.Key.f10,
        "f11": pynput_keyboard.Key.f11,
        "alt_l": pynput_keyboard.Key.alt_l,
        "alt_r": pynput_keyboard.Key.alt_r,
        "alt": pynput_keyboard.Key.alt_l,
        "ctrl_l": pynput_keyboard.Key.ctrl_l,
        "ctrl_r": pynput_keyboard.Key.ctrl_r,
        "ctrl": pynput_keyboard.Key.ctrl_l,
        "shift_l": pynput_keyboard.Key.shift_l,
        "shift_r": pynput_keyboard.Key.shift_r,
        "shift": pynput_keyboard.Key.shift_l,
    }
    if name in mapping:
        return mapping[name]
    if len(name) == 1:
        return pynput_keyboard.KeyCode.from_char(name)
    raise ValueError(f"Unsupported hotkey: {name}")


def _normalize_hotkey_token_for_keyboard(token: str) -> str:
    t = (token or "").strip().lower()
    aliases = {
        "control": "ctrl",
        "ctl": "ctrl",
        "cmd": "win",
        "windows": "win",
    }
    t = aliases.get(t, t)

    # Keep function keys and single characters as-is.
    if len(t) == 1:
        return t
    if t.startswith("f") and t[1:].isdigit():
        return t

    mapping = {
        "alt": "alt",
        "alt_l": "left alt",
        "alt_r": "right alt",
        "ctrl": "ctrl",
        "ctrl_l": "left ctrl",
        "ctrl_r": "right ctrl",
        "shift": "shift",
        "shift_l": "left shift",
        "shift_r": "right shift",
    }
    return mapping.get(t, t)


def _normalize_hotkey_for_keyboard(name: str) -> str:
    parts = [p.strip() for p in (name or "").split("+") if p.strip()]
    if not parts:
        raise ValueError("Empty hotkey")
    return "+".join(_normalize_hotkey_token_for_keyboard(p) for p in parts)


def _hotkey_contains_ctrl(name: str) -> bool:
    tokens = [t.strip().lower() for t in (name or "").split("+") if t.strip()]
    return any(tok in ("ctrl", "ctrl_l", "ctrl_r", "control", "ctl") for tok in tokens)


def _is_modifier_key_name_for_keyboard(name: str) -> bool:
    n = (name or "").strip().lower()
    return n in {
        "ctrl",
        "left ctrl",
        "right ctrl",
        "shift",
        "left shift",
        "right shift",
        "alt",
        "left alt",
        "right alt",
        "win",
        "left win",
        "right win",
        "left windows",
        "right windows",
    }


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
        output_language: str | None = None,
        raw: bool = False,
        prompt: str | None = None,
        paste: bool = True,
        use_screenshot: bool = True,
        hotkey: pynput_keyboard.Key | pynput_keyboard.KeyCode = pynput_keyboard.Key.f8,
        hotkey_name: str = "f9",
        no_enter_hotkey: pynput_keyboard.Key | pynput_keyboard.KeyCode | None = pynput_keyboard.Key.f8,
        no_enter_hotkey_name: str | None = "f8",
        toggle_hotkey: pynput_keyboard.Key | pynput_keyboard.KeyCode | None = pynput_keyboard.Key.f10,
        toggle_hotkey_name: str | None = "f10",
        pause_hotkey: pynput_keyboard.Key | pynput_keyboard.KeyCode | None = pynput_keyboard.Key.f11,
        pause_hotkey_name: str | None = "f11",
        auto_segment_min_sec: float = 10.0,
        auto_segment_max_sec: float = 20.0,
        auto_segment_silence_sec: float = 1.2,
        auto_segment_voice_threshold: float = 0.012,
        overlay: bool = True,
    ):
        self.server_url = server_url
        self.language = language
        self.model = model
        self.output_language = output_language
        self.raw = raw
        self.prompt = prompt
        self.paste = paste
        self.use_screenshot = use_screenshot
        self.hotkey = hotkey
        self.hotkey_name = hotkey_name
        self.no_enter_hotkey = no_enter_hotkey
        self.no_enter_hotkey_name = (no_enter_hotkey_name or "").strip().lower() if no_enter_hotkey_name else None
        self.toggle_hotkey = toggle_hotkey
        self.toggle_hotkey_name = (toggle_hotkey_name or "").strip().lower() if toggle_hotkey_name else None
        self.pause_hotkey = pause_hotkey
        self.pause_hotkey_name = (pause_hotkey_name or "").strip().lower() if pause_hotkey_name else None

        self.recording = False
        self.audio_chunks: list[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
        self.continuous_mode = False
        self.continuous_paused = False
        self._audio_lock = threading.Lock()
        self._segment_started_at = 0.0
        self._last_voice_at = 0.0
        self._last_toggle_press_at = 0.0
        self._last_pause_press_at = 0.0

        self.auto_segment_min_sec = auto_segment_min_sec
        self.auto_segment_max_sec = auto_segment_max_sec
        self.auto_segment_silence_sec = auto_segment_silence_sec
        self.auto_segment_voice_threshold = auto_segment_voice_threshold

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False

        self._stream_timer: Optional[threading.Timer] = None
        self._ctrl_pressed = False
        self._alt_pressed = False
        self._send_enter = True

        # When the hotkey itself contains Ctrl, using Ctrl as the "no-enter"
        # modifier no longer makes sense (it would always be pressed). In that
        # case we switch the modifier to Alt.
        self._no_enter_modifier = "alt" if _hotkey_contains_ctrl(self.hotkey_name) else "ctrl"

        # Optional Windows hotkey backend that can suppress the key event so
        # apps like Word won't see F8 (Word uses F8 for Extend Selection).
        self._keyboard_backend = None

        self.hud = StatusHud(enabled=overlay)

    def start(self) -> None:
        print("voice-input (Windows client)")
        print(f"  Server:     {self.server_url}")
        print(f"  Language:   {self.language}")
        if self.output_language:
            print(f"  Output:     {self.output_language} (forced)")
        print(f"  Model:      {self.model}")
        print(f"  Hotkey:     {self._hotkey_label()}")
        print(f"  Paste:      {'Ctrl+V' if self.paste else 'clipboard only'}")
        print(f"  Screenshot: {'ON' if self.use_screenshot else 'OFF'}")
        print("")
        print(f"  [hold {self._hotkey_label()}]        → record → paste + Enter")
        if self.no_enter_hotkey_name:
            print(
                f"  [hold {self._format_hotkey_label(self.no_enter_hotkey_name)}]        "
                "→ record → paste only (no Enter)"
            )
        else:
            mod_label = "Alt" if self._no_enter_modifier == "alt" else "Ctrl"
            print(f"  [hold {self._hotkey_label()} + {mod_label}] → record → paste only (no Enter)")
        if self.toggle_hotkey_name:
            print(
                f"  [press {self._format_hotkey_label(self.toggle_hotkey_name)}]      "
                f"→ toggle continuous dictation ON/OFF (auto send {self.auto_segment_min_sec:.0f}-{self.auto_segment_max_sec:.0f}s)"
            )
        if self.pause_hotkey_name:
            print(
                f"  [press {self._format_hotkey_label(self.pause_hotkey_name)}]      "
                "→ pause/resume continuous dictation"
            )
        print("  [Ctrl+C] → quit")
        print("")

        self.hud.start()

        self.loop = asyncio.new_event_loop()
        ws_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        ws_thread.start()

        # Prefer `keyboard` library if available, because it can suppress the
        # key event (prevents Word's F8 Extend Selection and other app hotkeys).
        if self._try_start_keyboard_backend():
            return

        # Fallback: pynput can monitor keys but cannot reliably suppress them.
        if _is_combo_hotkey(self.hotkey_name) or (
            self.no_enter_hotkey_name is not None and _is_combo_hotkey(self.no_enter_hotkey_name)
        ) or (
            self.toggle_hotkey_name is not None and _is_combo_hotkey(self.toggle_hotkey_name)
        ) or (
            self.pause_hotkey_name is not None and _is_combo_hotkey(self.pause_hotkey_name)
        ):
            print("  Error: Combo hotkeys (e.g. ctrl+shift) require the `keyboard` package.")
            print("         Install it with: pip install keyboard")
            return

        print("  Hotkey backend: pynput (key is NOT suppressed)")
        if self._hotkey_label() == "F8":
            print("  Note: Microsoft Word uses F8 for Extend Selection.")
            print("        If Word keeps selecting text, install `keyboard` (recommended) or use --hotkey f9.")

        with pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        ) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nShutting down.")
            finally:
                self.hud.stop()

    def _try_start_keyboard_backend(self) -> bool:
        """Start hotkey listener using the `keyboard` module if available.

        Returns True if started (and this method blocks until exit).
        """
        try:
            import keyboard as kb  # type: ignore
        except Exception:
            return False

        try:
            key_name = _normalize_hotkey_for_keyboard(self.hotkey_name)
        except Exception:
            return False

        toggle_key_name: str | None = None
        if self.toggle_hotkey_name:
            try:
                toggle_key_name = _normalize_hotkey_for_keyboard(self.toggle_hotkey_name)
            except Exception:
                print("  Warning: invalid --toggle-hotkey. Toggle mode disabled.")
                toggle_key_name = None

        no_enter_key_name: str | None = None
        if self.no_enter_hotkey_name:
            try:
                no_enter_key_name = _normalize_hotkey_for_keyboard(self.no_enter_hotkey_name)
            except Exception:
                print("  Warning: invalid --no-enter-hotkey. No-enter hotkey disabled.")
                no_enter_key_name = None

        pause_key_name: str | None = None
        if self.pause_hotkey_name:
            try:
                pause_key_name = _normalize_hotkey_for_keyboard(self.pause_hotkey_name)
            except Exception:
                print("  Warning: invalid --pause-hotkey. Pause mode disabled.")
                pause_key_name = None

        if toggle_key_name and toggle_key_name == key_name:
            print("  Warning: --toggle-hotkey is same as --hotkey. Toggle mode disabled.")
            toggle_key_name = None

        if no_enter_key_name and no_enter_key_name in {key_name, toggle_key_name}:
            print("  Warning: --no-enter-hotkey conflicts with other hotkeys. No-enter hotkey disabled.")
            no_enter_key_name = None

        if pause_key_name and pause_key_name in {key_name, toggle_key_name, no_enter_key_name}:
            print("  Warning: --pause-hotkey conflicts with other hotkeys. Pause mode disabled.")
            pause_key_name = None

        self._keyboard_backend = kb
        print("  Hotkey backend: keyboard (suppressed)")

        if self._no_enter_modifier == "alt":
            print("  Note: Hotkey includes Ctrl, so 'paste without Enter' modifier is Alt.")

        def _on_press() -> None:
            if not self.recording:
                self._start_recording()

        def _on_release() -> None:
            if not self.recording:
                return
            mod_pressed = bool(kb.is_pressed(self._no_enter_modifier))
            self._send_enter = not mod_pressed
            self._stop_recording()

        def _on_no_enter_press() -> None:
            if not self.recording:
                self._send_enter = False
                self._start_recording(continuous=False)

        def _on_no_enter_release() -> None:
            if self.recording and not self.continuous_mode:
                self._send_enter = False
                self._stop_recording()

        def _on_toggle() -> None:
            now = time.time()
            if now - self._last_toggle_press_at < 0.35:
                return
            self._last_toggle_press_at = now
            if self.recording and self.continuous_mode:
                self._send_enter = True
                self._stop_recording()
                return
            if not self.recording:
                self._send_enter = True
                self._start_recording(continuous=True)

        def _on_pause_toggle() -> None:
            now = time.time()
            if now - self._last_pause_press_at < 0.35:
                return
            self._last_pause_press_at = now
            if not self.recording or not self.continuous_mode:
                return
            self.continuous_paused = not self.continuous_paused
            if self.continuous_paused:
                self.hud.update("recording", "ߟ Continuous paused")
                print("\n  ߟ Continuous paused", end="", flush=True)
            else:
                self._last_voice_at = time.time()
                self.hud.update("recording", "ߟ Recording (continuous)...")
                print("\n  ߟ Continuous resumed", end="", flush=True)

        combo_parts = [p.strip() for p in key_name.split("+") if p.strip()]
        modifier_only_combo = len(combo_parts) > 1 and all(
            _is_modifier_key_name_for_keyboard(p) for p in combo_parts
        )

        if modifier_only_combo:
            combo_active = False

            def _all_combo_pressed() -> bool:
                return all(bool(kb.is_pressed(k)) for k in combo_parts)

            def _combo_press(_event) -> None:
                nonlocal combo_active
                if combo_active:
                    return
                if _all_combo_pressed():
                    combo_active = True
                    _on_press()

            def _combo_release(_event) -> None:
                nonlocal combo_active
                if not combo_active:
                    return
                if not _all_combo_pressed():
                    _on_release()
                    combo_active = False

            for k in sorted(set(combo_parts)):
                kb.on_press_key(k, _combo_press, suppress=True)
                kb.on_release_key(k, _combo_release, suppress=True)
        else:
            kb.add_hotkey(key_name, _on_press, suppress=True, trigger_on_release=False)
            kb.add_hotkey(key_name, _on_release, suppress=True, trigger_on_release=True)

        if no_enter_key_name:
            kb.add_hotkey(no_enter_key_name, _on_no_enter_press, suppress=True, trigger_on_release=False)
            kb.add_hotkey(no_enter_key_name, _on_no_enter_release, suppress=True, trigger_on_release=True)

        if toggle_key_name:
            kb.add_hotkey(toggle_key_name, _on_toggle, suppress=True, trigger_on_release=False)
        if pause_key_name:
            kb.add_hotkey(pause_key_name, _on_pause_toggle, suppress=True, trigger_on_release=False)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            try:
                kb.unhook_all()
            except Exception:
                pass
            self.hud.stop()
        return True

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
                        "output_language": self.output_language,
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
        if key in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
            self._ctrl_pressed = True
        if key in (pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r):
            self._alt_pressed = True
        if self.toggle_hotkey is not None and key == self.toggle_hotkey:
            now = time.time()
            if now - self._last_toggle_press_at > 0.35:
                self._last_toggle_press_at = now
                if self.recording and self.continuous_mode:
                    self._send_enter = True
                    self._stop_recording()
                elif not self.recording:
                    self._send_enter = True
                    self._start_recording(continuous=True)
            return
        if self.pause_hotkey is not None and key == self.pause_hotkey:
            now = time.time()
            if now - self._last_pause_press_at > 0.35:
                self._last_pause_press_at = now
                if self.recording and self.continuous_mode:
                    self.continuous_paused = not self.continuous_paused
                    if self.continuous_paused:
                        self.hud.update("recording", "ߟ Continuous paused")
                        print("\n  ߟ Continuous paused", end="", flush=True)
                    else:
                        self._last_voice_at = time.time()
                        self.hud.update("recording", "ߟ Recording (continuous)...")
                        print("\n  ߟ Continuous resumed", end="", flush=True)
            return
        if self.no_enter_hotkey is not None and key == self.no_enter_hotkey and not self.recording:
            self._send_enter = False
            self._start_recording(continuous=False)
            return
        if key == self.hotkey and not self.recording:
            self._start_recording(continuous=False)

    def _on_key_release(self, key) -> None:
        if key in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
            self._ctrl_pressed = False
        if key in (pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r):
            self._alt_pressed = False
        if self.no_enter_hotkey is not None and key == self.no_enter_hotkey and self.recording and not self.continuous_mode:
            self._send_enter = False
            self._stop_recording()
            return
        if key == self.hotkey and self.recording and not self.continuous_mode:
            mod_pressed = self._alt_pressed if self._no_enter_modifier == "alt" else self._ctrl_pressed
            self._send_enter = not mod_pressed
            self._stop_recording()

    def _start_recording(self, continuous: bool = False) -> None:
        if not self._connected or not self.ws or not self.loop:
            print("   Not connected to server")
            return

        self.continuous_mode = continuous
        self.continuous_paused = False
        self.recording = True
        with self._audio_lock:
            self.audio_chunks = []
        self._segment_started_at = time.time()
        self._last_voice_at = self._segment_started_at
        prefix = "  ߟ Recording (continuous)..." if self.continuous_mode else "  ߟ Recording..."
        print(prefix, end="", flush=True)
        self.hud.update("recording", "ߟ Recording (continuous)..." if self.continuous_mode else "ߟ Recording...")

        self._send_stream_start(include_screenshot=self.use_screenshot and not self.raw)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self.stream.start()
        self._schedule_stream_timer()

    def _send_stream_start(self, include_screenshot: bool) -> None:
        if not self.ws or not self.loop:
            return

        start_msg: dict = {"type": "stream_start"}

        if include_screenshot:
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

    def _stop_recording(self) -> None:
        if not self.recording:
            return

        was_continuous = self.continuous_mode
        self.recording = False
        self.continuous_mode = False
        self.continuous_paused = False
        self._stop_stream_timer()

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self._flush_current_segment(final=True)
        if was_continuous:
            print("  (continuous OFF)", end="", flush=True)

    def _flush_current_segment(self, final: bool) -> None:
        if not self.ws or not self.loop or not self._connected:
            print(" ߟ Not connected")
            self.hud.update("error", "ߟ Not connected")
            return

        with self._audio_lock:
            if not self.audio_chunks:
                audio_data = None
            else:
                audio_data = np.concatenate(self.audio_chunks)
            self.audio_chunks = []

        if audio_data is None:
            print(" (empty)")
            self._send_stream_end_only()
        else:
            duration = len(audio_data) / SAMPLE_RATE
            print(f" {duration:.1f}s", end="", flush=True)

            if duration < 0.3:
                print(" (too short, skipped)")
                self._send_stream_end_only()
            else:
                wav_bytes = self._encode_wav(audio_data)
                print(f" ({len(wav_bytes) // 1024}KB)", end="", flush=True)

                async def _send_final() -> None:
                    assert self.ws is not None
                    await self.ws.send(wav_bytes)
                    await self.ws.send(json.dumps({"type": "stream_end"}))

                asyncio.run_coroutine_threadsafe(_send_final(), self.loop)
                print(" ߟ Sent.", end="", flush=True)
                self.hud.update("refining", "ߟ Processing...")

        if self.recording and self.continuous_mode and not final:
            self._segment_started_at = time.time()
            self._last_voice_at = self._segment_started_at
            self._send_stream_start(include_screenshot=False)

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
        if self.continuous_mode and self.continuous_paused:
            self._schedule_stream_timer()
            return
        self._send_stream_chunk()
        if self.continuous_mode:
            self._auto_flush_if_needed()
        self._schedule_stream_timer()

    def _auto_flush_if_needed(self) -> None:
        now = time.time()
        with self._audio_lock:
            if not self.audio_chunks:
                duration = 0.0
            else:
                duration = len(np.concatenate(self.audio_chunks)) / SAMPLE_RATE

        if duration < self.auto_segment_min_sec:
            return

        silence_sec = now - self._last_voice_at
        if duration >= self.auto_segment_max_sec or silence_sec >= self.auto_segment_silence_sec:
            reason = "max" if duration >= self.auto_segment_max_sec else "silence"
            print(f"\n  ߟ Auto-send ({reason})", end="", flush=True)
            self._flush_current_segment(final=False)

    def _send_stream_chunk(self) -> None:
        if not self.ws or not self.loop:
            return
        with self._audio_lock:
            if not self.audio_chunks:
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
            if self.continuous_mode and self.continuous_paused:
                return
            chunk = indata.copy()
            with self._audio_lock:
                self.audio_chunks.append(chunk)
            if self.continuous_mode:
                energy = np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
                if float(energy) >= self.auto_segment_voice_threshold:
                    self._last_voice_at = time.time()

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

        # If we are using the `keyboard` backend (suppressed global hotkeys),
        # the user's modifier keys can still be physically down at release time
        # (e.g. Ctrl+Shift). Release modifiers first, then emit paste.
        kb = self._keyboard_backend
        if kb is not None:
            try:
                for key_name in (
                    "left ctrl",
                    "right ctrl",
                    "ctrl",
                    "left shift",
                    "right shift",
                    "shift",
                    "left alt",
                    "right alt",
                    "alt",
                ):
                    try:
                        kb.release(key_name)
                    except Exception:
                        pass

                time.sleep(0.04)
                kb.press_and_release("ctrl+v")
                if send_enter:
                    time.sleep(0.02)
                    kb.press_and_release("enter")
                return
            except Exception:
                # Fall back to pynput path below.
                pass

        controller = pynput_keyboard.Controller()
        time.sleep(0.03)
        with controller.pressed(pynput_keyboard.Key.ctrl):
            controller.press("v")
            controller.release("v")
        if send_enter:
            time.sleep(0.02)
            controller.press(pynput_keyboard.Key.enter)
            controller.release(pynput_keyboard.Key.enter)

    def _hotkey_label(self) -> str:
        name = (self.hotkey_name or "").strip()
        return self._format_hotkey_label(name)

    @staticmethod
    def _format_hotkey_label(name: str) -> str:
        name = (name or "").strip()
        return name.upper() if name else "?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="voice-input Windows client: Push-to-Talk voice input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dependencies:
  pip install sounddevice numpy websockets pynput pyperclip
Optional (screen context):
  pip install pillow
Recommended (Windows hotkey suppression / combo hotkeys):
    pip install keyboard

Example:
  python win_client.py --server ws://localhost:8991 --model gemma3:4b --language zh
""",
    )

    default_server = os.environ.get("VOICE_INPUT_SERVER", "ws://localhost:8991")
    parser.add_argument("-s", "--server", default=default_server, help=f"WebSocket server URL (default: {default_server})")
    parser.add_argument("-l", "--language", default="zh", help="Language hint (default: ja)")
    parser.add_argument("-m", "--model", default="gpt-oss:20b", help="Ollama model for refinement")
    parser.add_argument("--output-language", default=None, help="Force final output language (e.g., en, zh, ja, ko)")
    parser.add_argument("--raw", action="store_true", help="Skip LLM refinement")
    parser.add_argument("-p", "--prompt", default=None, help="Custom refinement prompt")
    parser.add_argument("--no-paste", action="store_true", help="Clipboard only; don't auto-paste")
    parser.add_argument("--no-screenshot", action="store_true", help="Disable screen context screenshot")
    parser.add_argument("--no-overlay", action="store_true", help="Disable the small status HUD")
    parser.add_argument(
        "--hotkey",
        default="f9",
        help=(
            "Hotkey name: f8|f9|f10|f11|ctrl|ctrl_l|ctrl_r|shift|shift_l|shift_r|alt|alt_l|alt_r, "
            "a single character, or a combo like ctrl+shift (default: f9)"
        ),
    )
    parser.add_argument(
        "--no-enter-hotkey",
        default="f8",
        help="Push-to-talk hotkey that pastes without Enter. Use 'none' to disable (default: f8)",
    )
    parser.add_argument(
        "--toggle-hotkey",
        default="f10",
        help="Press once to start/stop continuous dictation. Use 'none' to disable (default: f10)",
    )
    parser.add_argument(
        "--pause-hotkey",
        default="f11",
        help="Continuous mode pause/resume hotkey. Use 'none' to disable (default: f11)",
    )
    parser.add_argument(
        "--auto-segment-min",
        type=float,
        default=10.0,
        help="Continuous mode: minimum segment seconds before auto-send can trigger (default: 10)",
    )
    parser.add_argument(
        "--auto-segment-max",
        type=float,
        default=20.0,
        help="Continuous mode: force auto-send at this segment length (default: 20)",
    )
    parser.add_argument(
        "--auto-segment-silence",
        type=float,
        default=1.2,
        help="Continuous mode: auto-send when silence reaches this many seconds after min duration (default: 1.2)",
    )

    args = parser.parse_args()

    hotkey_name = (args.hotkey or "").strip().lower()
    if not hotkey_name:
        print("Error: --hotkey cannot be empty")
        sys.exit(2)

    no_enter_hotkey_name = (args.no_enter_hotkey or "").strip().lower()
    if no_enter_hotkey_name in ("none", "off", "no", "disable", "disabled", "0"):
        no_enter_hotkey_name = ""

    toggle_hotkey_name = (args.toggle_hotkey or "").strip().lower()
    if toggle_hotkey_name in ("none", "off", "no", "disable", "disabled", "0"):
        toggle_hotkey_name = ""

    pause_hotkey_name = (args.pause_hotkey or "").strip().lower()
    if pause_hotkey_name in ("none", "off", "no", "disable", "disabled", "0"):
        pause_hotkey_name = ""

    if args.auto_segment_min <= 0 or args.auto_segment_max <= 0:
        print("Error: --auto-segment-min/max must be positive")
        sys.exit(2)
    if args.auto_segment_min >= args.auto_segment_max:
        print("Error: --auto-segment-min must be smaller than --auto-segment-max")
        sys.exit(2)
    if args.auto_segment_silence <= 0:
        print("Error: --auto-segment-silence must be positive")
        sys.exit(2)

    # Combo hotkeys are handled by the `keyboard` backend. We still parse
    # single-key hotkeys for the pynput fallback.
    hotkey: pynput_keyboard.Key | pynput_keyboard.KeyCode
    if _is_combo_hotkey(hotkey_name):
        hotkey = pynput_keyboard.Key.f8
    else:
        try:
            hotkey = _parse_hotkey(hotkey_name)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(2)

    no_enter_hotkey: pynput_keyboard.Key | pynput_keyboard.KeyCode | None = None
    if no_enter_hotkey_name:
        if _is_combo_hotkey(no_enter_hotkey_name):
            no_enter_hotkey = pynput_keyboard.Key.f8
        else:
            try:
                no_enter_hotkey = _parse_hotkey(no_enter_hotkey_name)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(2)

    toggle_hotkey: pynput_keyboard.Key | pynput_keyboard.KeyCode | None = None
    if toggle_hotkey_name:
        if _is_combo_hotkey(toggle_hotkey_name):
            toggle_hotkey = pynput_keyboard.Key.f8
        else:
            try:
                toggle_hotkey = _parse_hotkey(toggle_hotkey_name)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(2)

    pause_hotkey: pynput_keyboard.Key | pynput_keyboard.KeyCode | None = None
    if pause_hotkey_name:
        if _is_combo_hotkey(pause_hotkey_name):
            pause_hotkey = pynput_keyboard.Key.f8
        else:
            try:
                pause_hotkey = _parse_hotkey(pause_hotkey_name)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(2)

    client = VoiceInputWinClient(
        server_url=args.server,
        language=args.language,
        model=args.model,
        output_language=args.output_language,
        raw=args.raw,
        prompt=args.prompt,
        paste=not args.no_paste,
        use_screenshot=not args.no_screenshot,
        hotkey=hotkey,
        hotkey_name=hotkey_name,
        no_enter_hotkey=no_enter_hotkey,
        no_enter_hotkey_name=no_enter_hotkey_name or None,
        toggle_hotkey=toggle_hotkey,
        toggle_hotkey_name=toggle_hotkey_name or None,
        pause_hotkey=pause_hotkey,
        pause_hotkey_name=pause_hotkey_name or None,
        auto_segment_min_sec=args.auto_segment_min,
        auto_segment_max_sec=args.auto_segment_max,
        auto_segment_silence_sec=args.auto_segment_silence,
        overlay=not args.no_overlay,
    )
    client.start()


if __name__ == "__main__":
    main()
