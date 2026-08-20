"""Global hotkeys (Hide/Pin/Expand/Copy) via pynput.

Qt's own QShortcut only fires while the app is focused, which defeats the
purpose of a Hide/Pin/Expand overlay meant to be toggled while the meeting
app has focus. pynput's GlobalHotKeys runs its own OS-level listener thread;
its callbacks emit Qt Signals (thread-safe to emit across threads, queued
onto the Qt main thread automatically) rather than touching widgets directly.
"""

from __future__ import annotations

import sys

from pynput import keyboard
from PySide6.QtCore import QObject, Signal

from meeting_copilot.config import OverlayConfig, get_config
from meeting_copilot.desktop.overlay import OverlayWindow
from meeting_copilot.utils.logging import get_logger

logger = get_logger()


class _HotkeyBridge(QObject):
    hide_triggered = Signal()
    pin_triggered = Signal()
    expand_triggered = Signal()
    copy_triggered = Signal()


class HotkeyManager:
    def __init__(self, overlay: OverlayWindow, config: OverlayConfig | None = None):
        self._cfg = config or get_config().overlay
        self._bridge = _HotkeyBridge()
        self._bridge.hide_triggered.connect(overlay.toggle_hidden)
        self._bridge.pin_triggered.connect(overlay.toggle_pin)
        self._bridge.expand_triggered.connect(overlay.toggle_expand)
        self._bridge.copy_triggered.connect(overlay.copy_answer)

        self._listener = keyboard.GlobalHotKeys(
            {
                self._cfg.hotkeys.hide: self._bridge.hide_triggered.emit,
                self._cfg.hotkeys.pin: self._bridge.pin_triggered.emit,
                self._cfg.hotkeys.expand: self._bridge.expand_triggered.emit,
                self._cfg.hotkeys.copy_answer: self._bridge.copy_triggered.emit,
            }
        )

    def start(self) -> None:
        self._warn_if_cmd_on_non_macos()
        self._listener.start()
        logger.info(
            "Global hotkeys active: "
            f"hide={self._cfg.hotkeys.hide} pin={self._cfg.hotkeys.pin} "
            f"expand={self._cfg.hotkeys.expand} copy={self._cfg.hotkeys.copy_answer}"
        )

    def _warn_if_cmd_on_non_macos(self) -> None:
        if sys.platform == "darwin":
            return
        hotkeys = self._cfg.hotkeys
        if any("<cmd>" in hk for hk in (hotkeys.hide, hotkeys.pin, hotkeys.expand, hotkeys.copy_answer)):
            logger.warning(
                "Configured hotkeys use <cmd>, which maps to the Windows key on this "
                "platform -- Win+Shift+* combos are OS-reserved and often won't register "
                "reliably. Consider <ctrl>+<alt>+... instead in configs/settings.yaml "
                "(see docs/installation-windows.md)."
            )

    def stop(self) -> None:
        self._listener.stop()
