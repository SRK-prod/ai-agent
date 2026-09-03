"""Floating, always-on-top, transparent overlay window.

Deliberately thin: it only renders Answer payloads pushed over the backend
WebSocket and reacts to hotkeys (see desktop/hotkeys.py). No ML/audio work
happens here, so the Qt event loop never blocks.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from meeting_copilot.config import OverlayConfig, get_config

_COLLAPSED_SIZE = (600, 520)
_EXPANDED_SIZE = (680, 820)
_PANEL_COLOR = QColor(20, 20, 24, 235)
_PANEL_RADIUS = 12.0

# How many recent question/answer pairs stay on screen at once, newest first. The
# candidate often gets a follow-up that only makes sense against the previous answer
# ("and how would that change if..."), so the prior pair needs to stay readable.
_HISTORY_SIZE = 2

# Small, unobtrusive inline indicator -- normal operation, never meant to draw the eye.
# AUDIO_SILENT is deliberately NEUTRAL (white/gray, not a caution color) -- it means "the
# interviewer isn't talking right now" (routinely true for the entire time the candidate is
# answering), not "something might be wrong". A yellow/amber tone here would still read as
# a low-grade warning, which is exactly the false-alarm impression this is meant to avoid.
_AUDIO_STATUS_STYLE = {
    "AUDIO_ACTIVE": ("color: #6fcf6f; font-size: 11px;", "\U0001f7e2 Audio"),
    "AUDIO_SILENT": ("color: #aaaaaa; font-size: 11px;", "\U000026aa Listening"),
}
# Deliberately the opposite -- AUDIO_INPUT_LOST means the CAPTURE PIPELINE itself stopped
# working (not "the interviewer is quiet"), which needs action.
_AUDIO_LOST_TEXT = "\U0001f534 AUDIO INPUT LOST\nCheck audio capture"
_AUDIO_LOST_STYLE = (
    "background-color: rgba(180, 40, 40, 0.92); color: white; font-weight: bold; "
    "font-size: 12px; padding: 8px 12px; border-radius: 6px; border: 1px solid #ff6b6b;"
)


class OverlayWindow(QWidget):
    def __init__(self, config: OverlayConfig | None = None):
        super().__init__()
        self._cfg = config or get_config().overlay
        self._pinned = self._cfg.always_on_top
        self._expanded = False
        # Cursor-to-window offset held while dragging; None when not dragging.
        self._drag_offset: QPoint | None = None

        # Recent Q&A pairs, newest at index 0, capped at _HISTORY_SIZE. Each entry is
        # {"q": str, "a": str, "confidence": float, "low_confidence": bool}.
        self._history: list[dict] = []
        # True while the newest entry is a partial still streaming in -- the next partial
        # updates it in place; the first partial after a completed answer starts a new one.
        self._streaming = False

        self._apply_window_flags()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(self._cfg.opacity)
        # The dark panel background is painted directly in paintEvent() below, not via a QSS
        # background-color -- on macOS, a frameless "Tool" window with WA_TranslucentBackground
        # gets an OS-level translucent "vibrancy"/blur material behind it that washes out a
        # QSS-painted background, making the panel look light instead of the intended dark.
        # Painting pixels ourselves guarantees the actual color regardless of that effect.
        self.setStyleSheet(
            "QLabel { color: white; padding: 6px 12px; font-size: 13px; background: transparent; }"
        )

        layout = QVBoxLayout(self)

        # Audio health banner: hidden unless the stream is actually lost -- see
        # show_audio_health(). Sits above everything else so it's impossible to miss when
        # it does appear, but takes zero space otherwise.
        self._audio_lost_banner = QLabel(_AUDIO_LOST_TEXT)
        self._audio_lost_banner.setStyleSheet(_AUDIO_LOST_STYLE)
        self._audio_lost_banner.setWordWrap(True)
        self._audio_lost_banner.hide()
        layout.addWidget(self._audio_lost_banner)

        header_row = QHBoxLayout()
        self._header_label = QLabel("meeting-copilot: listening…")
        self._header_label.setStyleSheet("font-weight: bold; color: #8ab4f8;")
        # Small inline audio-health indicator (🟢/🟡) -- normal-operation state, deliberately
        # subtle. Starts blank: no health report has arrived yet at window construction.
        self._audio_status_label = QLabel("")
        header_row.addWidget(self._header_label, 1)
        header_row.addWidget(self._audio_status_label)
        layout.addLayout(header_row)

        self._answer_label = QLabel("")
        self._answer_label.setWordWrap(True)
        self._answer_label.setTextFormat(Qt.TextFormat.MarkdownText)
        self._answer_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Structured answers (Answer/Detail/Example/Comparison/...) run long -- scroll
        # instead of clipping. Transparent so the painted panel shows through.
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._answer_label)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; }")
        self._scroll.viewport().setStyleSheet("background: transparent;")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._confidence_label = QLabel("")
        self._confidence_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")

        layout.addWidget(self._scroll, stretch=1)
        layout.addWidget(self._confidence_label)

        self.resize(*_COLLAPSED_SIZE)
        self._move_to_top_right_corner()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), _PANEL_RADIUS, _PANEL_RADIUS)
        painter.fillPath(path, _PANEL_COLOR)
        super().paintEvent(event)

    def _apply_window_flags(self) -> None:
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if self._pinned:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def _move_to_top_right_corner(self) -> None:
        screen = QGuiApplication.primaryScreen().availableGeometry()
        margin = 24
        self.move(screen.width() - self.width() - margin, margin)

    # --- drag to reposition -------------------------------------------------
    # The window is deliberately frameless (no title bar to grab) and a Qt "Tool" window
    # (no taskbar button, so no minimize control either). That is right for an overlay --
    # it must not look like an app window during a screen share -- but it left the window
    # physically stuck wherever it opened, which is a problem when it covers the thing the
    # candidate needs to see. Reported live 2026-09-01: "not able to move/minimize the
    # overlay". Dragging the panel body is the standard substitute for a title bar.
    # Ctrl+Alt+H already hides it; that is the "minimize".

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Offset between the cursor and the window origin, so the window keeps its
            # grab point under the cursor instead of jumping its top-left corner there.
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click snaps back to the default corner -- a way out of having dragged the
        panel somewhere unhelpful (or off a screen that has since been disconnected)."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._move_to_top_right_corner()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def show_partial_answer(self, text_so_far: str) -> None:
        """Called as the answer streams in, before the final formatted/confidence-gated
        version replaces it via show_answer() -- gets text on screen within ~seconds
        instead of waiting for the whole answer. May briefly show the trailing
        `CONFIDENCE: N` marker right before show_answer() strips and replaces it.

        The partial carries no question text (see api.py broadcast_partial_answer), so the
        newest entry shows an "answering…" placeholder until show_answer() backfills it."""
        if self._streaming and self._history:
            self._history[0]["a"] = text_so_far
        else:
            self._history.insert(
                0, {"q": "", "a": text_so_far, "confidence": 0.0, "low_confidence": False}
            )
            del self._history[_HISTORY_SIZE:]
            self._streaming = True
        self._render()
        self.show()
        self.raise_()

    def show_answer(self, answer: dict) -> None:
        text = answer.get("text", "")
        confidence = answer.get("confidence", 0.0)
        low_confidence = answer.get("low_confidence", False)
        question = (
            ((answer.get("question") or {}).get("transcript") or {}).get("text") or ""
        )
        entry = {
            "q": question,
            "a": text,
            "confidence": confidence,
            "low_confidence": low_confidence,
        }

        if self._streaming and self._history:
            self._history[0] = entry
        else:
            self._history.insert(0, entry)
            del self._history[_HISTORY_SIZE:]
        self._streaming = False

        self._render()
        self._scroll.verticalScrollBar().setValue(0)
        self.show()
        self.raise_()

    def _render(self) -> None:
        """Rebuild the answer panel from _history -- newest pair on top, older pairs below
        a divider."""
        blocks: list[str] = []
        for idx, entry in enumerate(self._history):
            if entry["q"]:
                heading = f"**Q: {entry['q']}**"
            elif idx == 0 and self._streaming:
                heading = "**Q: …**"
            else:
                heading = ""
            parts = [p for p in (heading, entry["a"]) if p]
            blocks.append("\n\n".join(parts))
        self._answer_label.setText("\n\n---\n\n".join(blocks))

        newest_entry = self._history[0] if self._history else None
        if self._streaming:
            self._header_label.setText("meeting-copilot — answering…")
            self._confidence_label.setText("")
        elif newest_entry and newest_entry["low_confidence"]:
            self._header_label.setText("meeting-copilot — low confidence")
            self._confidence_label.setText(
                f"confidence: {newest_entry['confidence'] * 100:.0f}%"
            )
        elif newest_entry:
            self._header_label.setText("meeting-copilot")
            self._confidence_label.setText(
                f"confidence: {newest_entry['confidence'] * 100:.0f}%"
            )

    def show_audio_health(self, data: dict) -> None:
        """Reflects an AudioHealth state transition (see audio/capture.py, pipeline/
        orchestrator.py _watchdog_loop) -- called only when the state actually changes,
        not on every watchdog poll. AUDIO_ACTIVE/AUDIO_SILENT are a small inline indicator
        that never forces the window visible -- routine states shouldn't interrupt a
        candidate who intentionally hid the overlay. AUDIO_INPUT_LOST is the one state that
        needs action, so it gets a hard-to-miss banner and forces the window visible even
        if it was hidden or unpinned.
        """
        state = data.get("state", "AUDIO_ACTIVE")
        if state == "AUDIO_INPUT_LOST":
            self._audio_status_label.setText("")
            self._audio_lost_banner.show()
            self.show()
            self.raise_()
            return

        self._audio_lost_banner.hide()
        style, text = _AUDIO_STATUS_STYLE.get(state, _AUDIO_STATUS_STYLE["AUDIO_ACTIVE"])
        self._audio_status_label.setStyleSheet(style)
        self._audio_status_label.setText(text)

    def toggle_hidden(self) -> None:
        self.hide() if self.isVisible() else self.show()

    def toggle_pin(self) -> None:
        self._pinned = not self._pinned
        was_visible = self.isVisible()
        self._apply_window_flags()
        if was_visible:
            self.show()  # changing window flags requires re-showing on macOS

    def toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self.resize(*(_EXPANDED_SIZE if self._expanded else _COLLAPSED_SIZE))

    def copy_answer(self) -> None:
        """Copy only the newest answer -- the one the candidate is speaking to now."""
        newest = self._history[0]["a"] if self._history else ""
        QGuiApplication.clipboard().setText(newest)
