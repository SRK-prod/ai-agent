"""Floating, always-on-top, transparent overlay window.

Deliberately thin: it only renders Answer payloads pushed over the backend
WebSocket and reacts to hotkeys (see desktop/hotkeys.py). No ML/audio work
happens here, so the Qt event loop never blocks.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPaintEvent
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from meeting_copilot.config import OverlayConfig, get_config

_COLLAPSED_SIZE = (600, 520)
_EXPANDED_SIZE = (680, 820)
_PANEL_COLOR = QColor(20, 20, 24, 235)
_PANEL_RADIUS = 12.0


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
        self._header_label = QLabel("meeting-copilot: listening…")
        self._header_label.setStyleSheet("font-weight: bold; color: #8ab4f8;")
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

        layout.addWidget(self._header_label)
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
