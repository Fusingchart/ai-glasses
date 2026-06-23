"""PyQt6 transparent frameless overlay — fake AR view on Mac screen."""
import sys
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import QApplication, QWidget

from card_synthesizer import Card

CARD_WIDTH = 420
CARD_HEIGHT = 130
CARD_MARGIN = 24
DISPLAY_MS = 6000
FADE_STEPS = 20
FADE_INTERVAL_MS = 50


class CardWidget(QWidget):
    def __init__(self, card: Card):
        super().__init__()
        self._card = card
        self._opacity = 1.0
        self._setup_window()
        self._start_fade_timer()

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        screen = QApplication.primaryScreen().geometry()
        x = screen.right() - CARD_WIDTH - CARD_MARGIN
        y = screen.top() + CARD_MARGIN
        self.setGeometry(x, y, CARD_WIDTH, CARD_HEIGHT)

    def _start_fade_timer(self) -> None:
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.timeout.connect(self._begin_fade)
        self._display_timer.start(DISPLAY_MS)

    def _begin_fade(self) -> None:
        self._fade_step = 0
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._fade_tick)
        self._fade_timer.start(FADE_INTERVAL_MS)

    def _fade_tick(self) -> None:
        self._fade_step += 1
        self._opacity = max(0.0, 1.0 - self._fade_step / FADE_STEPS)
        self.update()
        if self._opacity <= 0:
            self._fade_timer.stop()
            self.close()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)

        # Background
        path = QPainterPath()
        path.addRoundedRect(0, 0, CARD_WIDTH, CARD_HEIGHT, 16, 16)
        painter.fillPath(path, QColor(10, 10, 30, 210))

        # Border
        painter.setPen(QColor(80, 130, 255, 180))
        painter.drawPath(path)

        # Title
        title_font = QFont("SF Pro Display", 13, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(200, 220, 255))
        painter.drawText(16, 28, self._card.title)

        # Body
        body_font = QFont("SF Pro Text", 11)
        painter.setFont(body_font)
        painter.setPen(QColor(170, 190, 230))
        body_rect = self.rect().adjusted(16, 38, -16, -30)
        painter.drawText(body_rect, Qt.TextFlag.TextWordWrap, self._card.body)

        # Source
        src_font = QFont("SF Mono", 9)
        painter.setFont(src_font)
        painter.setPen(QColor(90, 120, 180))
        painter.drawText(16, CARD_HEIGHT - 10, f"↗ {self._card.source}")


class OverlayBridge(QObject):
    card_received = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._widgets: list[CardWidget] = []
        self.card_received.connect(self._show_card)

    def push_card(self, card: Card) -> None:
        self.card_received.emit(card)

    def _show_card(self, card: Card) -> None:
        w = CardWidget(card)
        self._widgets.append(w)
        w.show()


def run_overlay(bridge: OverlayBridge) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    bridge.setParent(app)
    app.exec()
