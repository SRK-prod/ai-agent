"""Floating, always-on-top, transparent overlay window.

Deliberately thin: it only renders Answer payloads pushed over the backend
WebSocket and reacts to hotkeys (see desktop/hotkeys.py). No ML/audio work
happens here, so the Qt event loop never blocks.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from meeting_copilot.config import OverlayConfig, get_config

_COLLAPSED_SIZE = (600, 520)
_EXPANDED_SIZE = (680, 820)
_PANEL_COLOR = QColor(20, 20, 24, 235)
_PANEL_RADIUS = 12.0

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

    def show_partial_answer(self, text_so_far: str) -> None:
        """Called as the answer streams in, before the final formatted/confidence-gated
        version replaces it via show_answer() -- gets text on screen within ~seconds
        instead of waiting for the whole answer. May briefly show the trailing
        `CONFIDENCE: N` marker right before show_answer() strips and replaces it."""
        self._header_label.setText("meeting-copilot — answering…")
        self._answer_label.setText(text_so_far)
        self._confidence_label.setText("")
        self.show()
        self.raise_()

    def show_answer(self, answer: dict) -> None:
        text = answer.get("text", "")
        confidence = answer.get("confidence", 0.0)
        low_confidence = answer.get("low_confidence", False)

        self._header_label.setText(
            "meeting-copilot — low confidence" if low_confidence else "meeting-copilot"
        )
        self._answer_label.setText(text)
        self._confidence_label.setText(f"confidence: {confidence * 100:.0f}%")
        self.show()
        self.raise_()

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
        QGuiApplication.clipboard().setText(self._answer_label.text())
