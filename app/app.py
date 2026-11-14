"""Native PySide6 + PyQtGraph Desktop Client for Workplace Risk Detection (liveleak)."""

from __future__ import annotations

import csv
import sys
import time
import tomllib
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, Slot
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.db import EventHistoryStore, SessionSummary
from app.worker import StreamWorker

# Configure PyQtGraph global aesthetics
pg.setConfigOptions(antialias=True, background="#14171F", foreground="#94A3B8")

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #0F1117;
    color: #E2E8F0;
    font-family: 'Inter', 'Segoe UI', 'Noto Sans', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #262B38;
    background-color: #0F1117;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #161922;
    color: #94A3B8;
    padding: 9px 22px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #262B38;
    border-bottom: none;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #1E2433;
    color: #38BDF8;
    border-bottom: 2px solid #38BDF8;
}
QFrame#Card {
    background-color: #161922;
    border: 1px solid #262B38;
    border-radius: 8px;
}
QLabel#CardTitle {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QLabel#MonoValue {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-weight: 700;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
    background-color: #0F1117;
    border: 1px solid #2E3446;
    border-radius: 5px;
    padding: 5px 8px;
    color: #F8FAFC;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus {
    border: 1px solid #38BDF8;
}
QPushButton {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 6px 14px;
    color: #F1F5F9;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #334155;
}
QPushButton#StartBtn {
    background-color: #059669;
    border: 1px solid #10B981;
    color: #FFFFFF;
}
QPushButton#StartBtn:hover {
    background-color: #10B981;
}
QPushButton#StopBtn {
    background-color: #DC2626;
    border: 1px solid #EF4444;
    color: #FFFFFF;
}
QPushButton#StopBtn:hover {
    background-color: #EF4444;
}
QProgressBar {
    background-color: #0F1117;
    border: 1px solid #262B38;
    border-radius: 5px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    border-radius: 4px;
    background-color: #10B981;
}
QTableWidget {
    background-color: #12151E;
    alternate-background-color: #161A25;
    gridline-color: #222736;
    border: 1px solid #262B38;
    border-radius: 6px;
}
QTableWidget::item:selected {
    background-color: #1E3A5F;
    color: #FFFFFF;
}
QHeaderView::section {
    background-color: #1A1F2E;
    color: #94A3B8;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #262B38;
    font-weight: 700;
    font-size: 11px;
}
"""

RULE_LABELS = {
    "rule_ppe_compliance": "PPE Compliance",
    "rule_vectorized_proximity": "Hazard Proximity",
    "rule_kinematic_ttc": "Kinematic TTC",
    "rule_cone_zone": "Cone Zone Breach",
}


def make_card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("Card")
    return frame


def risk_color_hex(score: float | None) -> str:
    if score is None:
        return "#64748B"
    if score >= 0.70:
        return "#EF4444"  # Critical Crimson
    if score >= 0.40:
        return "#F59E0B"  # Elevated Amber
    return "#10B981"  # Nominal Emerald


def draw_entity_overlays(
        frame: np.ndarray,
        vectors: np.ndarray,
        ema_score: float | None,
        raw_score: float | None,
        rtt_ms: float,
        draw_entities: bool = True,
        draw_velocities: bool = True,
        draw_hud: bool = True,
) -> np.ndarray:
    """
    Render client-side entity markers from (N, 13) vectors and optional telemetry HUD.
    Columns (SSOT §4.2):
    0:id, 1:person, 2:machine, 3:vehicle, 4:cone, 5:px, 6:py, 7:vx, 8:vy, 9:hh, 10:mask, 11:vest, 12:na
    """
    canvas = frame.copy()

    if draw_entities and vectors is not None and vectors.ndim == 2 and vectors.shape[0] > 0:
        for row in vectors:
            track_id = int(row[0])
            is_person = row[1] > 0.5
            is_machine = row[2] > 0.5
            is_vehicle = row[3] > 0.5
            is_cone = row[4] > 0.5

            px, py = int(row[5]), int(row[6])
            vx, vy = float(row[7]), float(row[8])
            hh, mask, vest = int(row[9] > 0.5), int(row[10] > 0.5), int(row[11] > 0.5)

            if is_person:
                # Missing hardhat or vest turns person marker amber/red
                ppe_ok = (hh == 1) and (vest == 1)
                color = (129, 185, 16) if ppe_ok else (68, 68, 239)  # BGR
                label = f"P#{track_id} [{'H' if hh else '-'}{'V' if vest else '-'}]"
                radius = 9
            elif is_machine:
                color = (11, 158, 245)  # BGR Amber
                label = f"MCH#{track_id}"
                radius = 11
            elif is_vehicle:
                color = (246, 130, 59)  # BGR Blue
                label = f"VEH#{track_id}"
                radius = 11
            elif is_cone:
                color = (0, 140, 255)  # BGR Orange
                label = "CONE"
                radius = 6
            else:
                continue

            # Draw velocity vector arrow (scaled 5x for visibility)
            if draw_velocities and not is_cone and (abs(vx) > 0.25 or abs(vy) > 0.25):
                end_pt = (int(px + vx * 5.0), int(py + vy * 5.0))
                cv2.arrowedLine(canvas, (px, py), end_pt, (255, 230, 0), 2, tipLength=0.25)

            cv2.circle(canvas, (px, py), radius, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                label,
                (px + 12, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )

    if draw_hud:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (12, 12), (310, 96), (15, 17, 23), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        ema_str = f"{ema_score:.3f}" if ema_score is not None else "---"
        raw_str = f"{raw_score:.3f}" if raw_score is not None else "---"
        fps = 1000.0 / max(rtt_ms, 1e-3)

        hud_color = (80, 230, 80)
        if ema_score is not None and ema_score >= 0.70:
            hud_color = (68, 68, 239)
        elif ema_score is not None and ema_score >= 0.40:
            hud_color = (11, 185, 245)

        cv2.putText(
            canvas,
            f"EMA RISK: {ema_str} (RAW: {raw_str})",
            (22, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            hud_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"RTT: {rtt_ms:5.1f} ms | {fps:4.1f} FPS | ENTITIES: {len(vectors)}",
            (22, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 215, 225),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════════


class LiveMonitorTab(QWidget):
    def __init__(self, store: EventHistoryStore, cfg: dict) -> None:
        super().__init__()
        self.store = store
        self.cfg = cfg

        self.worker: StreamWorker | None = None
        self.active_session_id: str | None = None

        # EMA state
        self.ema_alpha: float = float(cfg.get("telemetry", {}).get("ema_alpha", 0.22))
        self.ema_score: float | None = None

        # Rolling history buffers (~15s)
        self.history_sec = float(cfg.get("telemetry", {}).get("history_seconds", 15.0))
        self.t0 = time.monotonic()
        self.ts_buf: deque[float] = deque(maxlen=600)
        self.ema_buf: deque[float] = deque(maxlen=600)
        self.raw_buf: deque[float] = deque(maxlen=600)
        self.model_buf: deque[float] = deque(maxlen=600)
        self.rule_buf: deque[float] = deque(maxlen=600)

        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(12)

        # ── Left Column: Video Stage (68% stretch) ─────────────────────────
        video_card = make_card()
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(10, 10, 10, 10)

        header_row = QHBoxLayout()
        title_lbl = QLabel("LIVE WORKSITE FEED")
        title_lbl.setObjectName("CardTitle")
        self.status_badge = QLabel("● DISCONNECTED")
        self.status_badge.setStyleSheet("color: #64748B; font-weight: 700; font-size: 12px;")
        self.session_lbl = QLabel("Session: None")
        self.session_lbl.setStyleSheet("color: #94A3B8; font-family: monospace;")
        header_row.addWidget(title_lbl)
        header_row.addStretch()
        header_row.addWidget(self.session_lbl)
        header_row.addSpacing(14)
        header_row.addWidget(self.status_badge)
        video_layout.addLayout(header_row)

        self.video_label = QLabel("Stream Inactive — Configure source and click Start Session")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setStyleSheet(
            "background-color: #090B10; border: 1px solid #1E2433; border-radius: 6px; color: #64748B;"
        )
        video_layout.addWidget(self.video_label, stretch=1)

        # Overlay Controls Bar
        overlay_bar = QHBoxLayout()
        self.chk_entities = QCheckBox("Draw Tracked Entities & PPE")
        self.chk_entities.setChecked(True)
        self.chk_vel = QCheckBox("Draw Velocity Vectors")
        self.chk_vel.setChecked(True)
        self.chk_hud = QCheckBox("Canvas HUD Overlay")
        self.chk_hud.setChecked(True)

        self.fps_lbl = QLabel("RTT: -- ms | FPS: --")
        self.fps_lbl.setObjectName("MonoValue")
        self.fps_lbl.setStyleSheet("color: #94A3B8;")

        overlay_bar.addWidget(self.chk_entities)
        overlay_bar.addWidget(self.chk_vel)
        overlay_bar.addWidget(self.chk_hud)
        overlay_bar.addStretch()
        overlay_bar.addWidget(self.fps_lbl)
        video_layout.addLayout(overlay_bar)

        root_layout.addWidget(video_card, stretch=68)

        # ── Right Column: Controls, Gauges, Strip Chart (32% stretch) ──────
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # 1. Connection & Source Config Card
        conn_card = make_card()
        conn_layout = QGridLayout(conn_card)
        conn_layout.setContentsMargins(12, 10, 12, 10)
        conn_layout.setVerticalSpacing(8)

        conn_title = QLabel("STREAM & SESSION CONFIG")
        conn_title.setObjectName("CardTitle")
        conn_layout.addWidget(conn_title, 0, 0, 1, 3)

        conn_layout.addWidget(QLabel("Operator ID:"), 1, 0)
        self.user_input = QLineEdit(self.cfg.get("session", {}).get("default_user", "operator_01"))
        conn_layout.addWidget(self.user_input, 1, 1, 1, 2)

        conn_layout.addWidget(QLabel("WebSocket URI:"), 2, 0)
        self.uri_input = QLineEdit(self.cfg.get("connection", {}).get("uri", "ws://localhost:5020"))
        conn_layout.addWidget(self.uri_input, 2, 1, 1, 2)

        conn_layout.addWidget(QLabel("Video Source:"), 3, 0)
        self.source_input = QLineEdit(str(self.cfg.get("connection", {}).get("source", "0")))
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_video)
        conn_layout.addWidget(self.source_input, 3, 1)
        conn_layout.addWidget(self.browse_btn, 3, 2)

        conn_layout.addWidget(QLabel("JPEG Quality:"), 4, 0)
        self.qual_spin = QSpinBox()
        self.qual_spin.setRange(20, 100)
        self.qual_spin.setValue(int(self.cfg.get("connection", {}).get("jpeg_quality", 85)))
        self.qual_spin.setSuffix(" %")
        conn_layout.addWidget(self.qual_spin, 4, 1)

        self.toggle_btn = QPushButton("▶ Start Session")
        self.toggle_btn.setObjectName("StartBtn")
        self.toggle_btn.clicked.connect(self._toggle_stream)
        conn_layout.addWidget(self.toggle_btn, 4, 2)

        right_col.addWidget(conn_card)

        # 2. Primary EMA Risk Gauge & Component Breakdown Card
        gauge_card = make_card()
        gauge_layout = QVBoxLayout(gauge_card)
        gauge_layout.setContentsMargins(12, 10, 12, 10)
        gauge_layout.setSpacing(8)

        g_head = QHBoxLayout()
        g_title = QLabel("REAL-TIME RISK ASSESSMENT (EMA SMOOTHED)")
        g_title.setObjectName("CardTitle")
        self.reason_badge = QLabel("STATUS: IDLE")
        self.reason_badge.setFixedWidth(185)  # Fits "STATUS: NO_DETECTIONS" without shifting
        self.reason_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reason_badge.setStyleSheet(
            "background-color: #1E293B; color: #94A3B8; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;"
        )
        g_head.addWidget(g_title)
        g_head.addStretch()
        g_head.addWidget(self.reason_badge)
        gauge_layout.addLayout(g_head)

        # Big Numeric Readout Row
        readout_row = QHBoxLayout()
        self.ema_big_lbl = QLabel("0.000")
        self.ema_big_lbl.setObjectName("MonoValue")
        self.ema_big_lbl.setStyleSheet("font-size: 36px; color: #10B981;")
        self.ema_big_lbl.setFixedWidth(120)

        sub_metrics = QVBoxLayout()
        self.raw_fused_lbl = QLabel("Raw Fused:  0.000")
        self.raw_fused_lbl.setObjectName("MonoValue")
        self.raw_fused_lbl.setFixedWidth(155)
        self.cov_lbl = QLabel("Kalman Cov: 0.0000")
        self.cov_lbl.setObjectName("MonoValue")
        self.cov_lbl.setFixedWidth(155)
        sub_metrics.addWidget(self.raw_fused_lbl)
        sub_metrics.addWidget(self.cov_lbl)

        readout_row.addWidget(self.ema_big_lbl)
        readout_row.addStretch()
        readout_row.addLayout(sub_metrics)
        gauge_layout.addLayout(readout_row)

        # Primary EMA Progress Bar (0..1000)
        self.ema_bar = QProgressBar()
        self.ema_bar.setRange(0, 1000)
        self.ema_bar.setValue(0)
        self.ema_bar.setFixedHeight(16)
        gauge_layout.addWidget(self.ema_bar)

        # EMA Alpha Slider Row
        ema_ctrl_row = QHBoxLayout()
        ema_ctrl_row.addWidget(QLabel("EMA Smoothing (α):"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(5, 100)
        self.alpha_slider.setValue(int(self.ema_alpha * 100))
        self.alpha_val_lbl = QLabel(f"{self.ema_alpha:.2f}")
        self.alpha_val_lbl.setObjectName("MonoValue")
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        ema_ctrl_row.addWidget(self.alpha_slider, stretch=1)
        ema_ctrl_row.addWidget(self.alpha_val_lbl)
        gauge_layout.addLayout(ema_ctrl_row)

        # Component sub-bars: ST-GNN (0..1) and DetNG (0..0.632)
        comp_grid = QGridLayout()
        comp_grid.addWidget(QLabel("ST-GNN (Forecast):"), 0, 0)
        self.stgnn_bar = QProgressBar()
        self.stgnn_bar.setRange(0, 1000)
        self.stgnn_bar.setFixedHeight(10)
        self.stgnn_val_lbl = QLabel("0.000")
        self.stgnn_val_lbl.setObjectName("MonoValue")
        self.stgnn_val_lbl.setFixedWidth(48)
        self.stgnn_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        comp_grid.addWidget(self.stgnn_bar, 0, 1)
        comp_grid.addWidget(self.stgnn_val_lbl, 0, 2)

        comp_grid.addWidget(QLabel("DetNG (Rule Engine):"), 1, 0)
        self.detng_bar = QProgressBar()
        self.detng_bar.setRange(0, 1000)
        self.detng_bar.setFixedHeight(10)
        self.detng_val_lbl = QLabel("0.000")
        self.detng_val_lbl.setObjectName("MonoValue")
        self.detng_val_lbl.setFixedWidth(48)
        self.detng_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        comp_grid.addWidget(self.detng_bar, 1, 1)
        comp_grid.addWidget(self.detng_val_lbl, 1, 2)
        gauge_layout.addLayout(comp_grid)

        # Rule Violation Pills
        viol_title = QLabel("ACTIVE DETNG RULE VIOLATIONS")
        viol_title.setObjectName("CardTitle")
        gauge_layout.addWidget(viol_title)

        viol_grid = QGridLayout()
        self.rule_pills: dict[str, QLabel] = {}
        for idx, (r_key, r_name) in enumerate(RULE_LABELS.items()):
            pill = QLabel(r_name)
            pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pill.setFixedHeight(26)
            pill.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            pill.setStyleSheet(
                "background-color: #12151E; color: #475569; border: 1px solid #262B38; "
                "border-radius: 4px; padding: 4px 6px; font-weight: 600; font-size: 11px;"
            )
            self.rule_pills[r_key] = pill
            viol_grid.addWidget(pill, idx // 2, idx % 2)
        gauge_layout.addLayout(viol_grid)

        # Tracked Entity Counters
        cnt_row = QHBoxLayout()
        self.cnt_labels: dict[str, QLabel] = {}
        for key, label in [("person", "Persons"), ("machine", "Machinery"), ("vehicle", "Vehicles"), ("cone", "Cones")]:
            box = QLabel(f"{label}: 0")
            box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.setFixedHeight(28)
            box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            box.setStyleSheet(
                "background-color: #12151E; border: 1px solid #262B38; border-radius: 4px; "
                "padding: 4px; font-family: monospace; font-weight: 700; color: #CBD5E1;"
            )
            self.cnt_labels[key] = box
            cnt_row.addWidget(box)
        gauge_layout.addLayout(cnt_row)

        right_col.addWidget(gauge_card)

        # 3. Live Rolling Telemetry Strip Chart
        chart_card = make_card()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        chart_title = QLabel("ROLLING TELEMETRY (EMA vs RAW / COMPONENTS)")
        chart_title.setObjectName("CardTitle")
        chart_layout.addWidget(chart_title)

        self.strip_plot = pg.PlotWidget()
        self.strip_plot.setYRange(0.0, 1.05, padding=0.02)
        self.strip_plot.showGrid(x=True, y=True, alpha=0.2)
        self.strip_plot.setLabel("left", "Risk")
        self.strip_plot.setLabel("bottom", "Time (s)")
        self.strip_plot.addLegend(offset=(10, 10), labelTextSize="8pt")

        self.curve_raw = self.strip_plot.plot(
            pen=pg.mkPen(color=(148, 163, 184, 90), width=1, style=Qt.PenStyle.DotLine),
            name="Raw Fused",
        )
        self.curve_model = self.strip_plot.plot(
            pen=pg.mkPen(color="#38BDF8", width=1.5), name="ST-GNN"
        )
        self.curve_rule = self.strip_plot.plot(
            pen=pg.mkPen(color="#F59E0B", width=1.5), name="DetNG"
        )
        self.curve_ema = self.strip_plot.plot(
            pen=pg.mkPen(color="#10B981", width=2.8), name="EMA Risk"
        )

        chart_layout.addWidget(self.strip_plot, stretch=1)
        right_col.addWidget(chart_card, stretch=1)

        root_layout.addLayout(right_col, stretch=32)

    @Slot(int)
    def _on_alpha_changed(self, value: int) -> None:
        self.ema_alpha = value / 100.0
        self.alpha_val_lbl.setText(f"{self.ema_alpha:.2f}")

    @Slot()
    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)",
        )
        if path:
            self.source_input.setText(path)

    @Slot()
    def _toggle_stream(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.toggle_btn.setEnabled(False)
            return

        # Reset EMA & rolling chart buffers
        self.ema_score = None
        self.t0 = time.monotonic()
        self.ts_buf.clear()
        self.ema_buf.clear()
        self.raw_buf.clear()
        self.model_buf.clear()
        self.rule_buf.clear()

        # Open a new SQLite session in EventHistoryStore
        user_id = self.user_input.text().strip() or "operator_01"
        self.active_session_id = self.store.open_session(user_id)
        self.session_lbl.setText(f"Session: {self.active_session_id}")

        self.worker = StreamWorker(
            uri=self.uri_input.text(),
            source=self.source_input.text(),
            jpeg_quality=self.qual_spin.value(),
        )
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

        self.toggle_btn.setText("■ Stop Session")
        self.toggle_btn.setObjectName("StopBtn")
        self.toggle_btn.style().unpolish(self.toggle_btn)
        self.toggle_btn.style().polish(self.toggle_btn)

    @Slot(str, str)
    def _on_status_changed(self, level: str, msg: str) -> None:
        colors = {
            "connecting": ("#F59E0B", "● CONNECTING"),
            "monitoring": ("#10B981", "● LIVE MONITORING"),
            "stopped": ("#64748B", "● STOPPED"),
            "error": ("#EF4444", "● ERROR"),
        }
        col, badge = colors.get(level, ("#64748B", "● IDLE"))
        self.status_badge.setText(badge)
        self.status_badge.setStyleSheet(f"color: {col}; font-weight: 700; font-size: 12px;")
        if level in ("stopped", "error"):
            self.video_label.setText(msg)

    @Slot()
    def _on_worker_finished(self) -> None:
        self.toggle_btn.setEnabled(True)
        self.toggle_btn.setText("▶ Start Session")
        self.toggle_btn.setObjectName("StartBtn")
        self.toggle_btn.style().unpolish(self.toggle_btn)
        self.toggle_btn.style().polish(self.toggle_btn)

    @Slot(np.ndarray, dict, float)
    def _on_frame_ready(self, bgr_frame: np.ndarray, data: dict, rtt_ms: float) -> None:
        if "error" in data:
            self.reason_badge.setText(f"ERR: {data['error'][:24]}")
            return

        fused = data["fused_score"]
        model = data["model_score"]
        rule = data["rule_score"]
        cov = data["covariance"]
        reason = data["reason"]
        violations = set(data["violations"])
        vectors = data["vectors"]
        counts = data["counts"]

        # Update EMA filter when fused score is valid
        if fused is not None:
            if self.ema_score is None:
                self.ema_score = fused
            else:
                self.ema_score = self.ema_alpha * fused + (1.0 - self.ema_alpha) * self.ema_score

            # Persist estimate to SQLite via non-blocking writer queue
            if self.active_session_id:
                self.store.record_estimate(self.active_session_id, fused, cov)

        # Update Gauges
        hex_col = risk_color_hex(self.ema_score)
        self.ema_big_lbl.setText(f"{self.ema_score:.3f}" if self.ema_score is not None else "---")
        self.ema_big_lbl.setStyleSheet(f"font-size: 36px; color: {hex_col};")

        self.raw_fused_lbl.setText(f"Raw Fused:  {fused:5.3f}" if fused is not None else "Raw Fused:    ---")
        self.cov_lbl.setText(f"Kalman Cov: {cov:.4f}")
        self.reason_badge.setText(f"STATUS: {reason.upper()}")

        ema_int = int(np.clip((self.ema_score or 0.0) * 1000, 0, 1000))
        self.ema_bar.setValue(ema_int)
        self.ema_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {hex_col}; border-radius: 4px; }}")

        model_val = model or 0.0
        rule_val = rule or 0.0
        self.stgnn_bar.setValue(int(np.clip(model_val * 1000, 0, 1000)))
        self.stgnn_bar.setStyleSheet("QProgressBar::chunk { background-color: #38BDF8; }")
        self.stgnn_val_lbl.setText(f"{model:5.3f}" if model is not None else "  ---")

        self.detng_bar.setValue(int(np.clip(rule_val * 1000, 0, 1000)))
        self.detng_bar.setStyleSheet("QProgressBar::chunk { background-color: #F59E0B; }")
        self.detng_val_lbl.setText(f"{rule:5.3f}" if rule is not None else "  ---")

        # Update DetNG Violation Pills
        # Update DetNG Violation Pills with live per-rule scores
        rule_scores = data.get("rule_breakdown", [])
        for idx, (r_key, pill) in enumerate(self.rule_pills.items()):
            base_name = RULE_LABELS[r_key]
            sub_score = rule_scores[idx] if idx < len(rule_scores) else 0.0
            pill.setText(f"{base_name} ({sub_score:.2f})")

            if r_key in violations or sub_score > 0.0:
                pill.setStyleSheet(
                    "background-color: #451A1C; color: #FCA5A5; border: 1px solid #EF4444; "
                    "border-radius: 4px; padding: 4px 6px; font-weight: 700; font-size: 11px;"
                )
            else:
                pill.setStyleSheet(
                    "background-color: #12151E; color: #475569; border: 1px solid #262B38; "
                    "border-radius: 4px; padding: 4px 6px; font-weight: 600; font-size: 11px;"
                )

        # Update Entity Counts
        for k, title in [("person", "Persons"), ("machine", "Machinery"), ("vehicle", "Vehicles"), ("cone", "Cones")]:
            self.cnt_labels[k].setText(f"{title}: {counts.get(k, 0)}")

        # Update Rolling Strip Chart
        t_rel = time.monotonic() - self.t0
        self.ts_buf.append(t_rel)
        self.ema_buf.append(self.ema_score if self.ema_score is not None else 0.0)
        self.raw_buf.append(fused if fused is not None else 0.0)
        self.model_buf.append(model_val)
        self.rule_buf.append(rule_val)

        ts_arr = np.asarray(self.ts_buf)
        mask = ts_arr >= (t_rel - self.history_sec)
        t_win = ts_arr[mask]
        self.curve_raw.setData(t_win, np.asarray(self.raw_buf)[mask])
        self.curve_model.setData(t_win, np.asarray(self.model_buf)[mask])
        self.curve_rule.setData(t_win, np.asarray(self.rule_buf)[mask])
        self.curve_ema.setData(t_win, np.asarray(self.ema_buf)[mask])
        self.strip_plot.setXRange(max(0.0, t_rel - self.history_sec), max(self.history_sec, t_rel), padding=0)

        # Update RTT/FPS label
        fps = 1000.0 / max(rtt_ms, 1e-3)
        self.fps_lbl.setText(f"RTT: {rtt_ms:5.1f} ms | FPS: {fps:4.1f}")

        # Draw Overlays and Blit Frame
        rendered = draw_entity_overlays(
            bgr_frame,
            vectors,
            self.ema_score,
            fused,
            rtt_ms,
            draw_entities=self.chk_entities.isChecked(),
            draw_velocities=self.chk_vel.isChecked(),
            draw_hud=self.chk_hud.isChecked(),
        )
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def shutdown(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: RETROACTIVE DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════


class RetroactiveDataTab(QWidget):
    def __init__(self, store: EventHistoryStore, cfg: dict) -> None:
        super().__init__()
        self.store = store
        self.cfg = cfg
        self.sessions: list[SessionSummary] = []
        self.selected_session_id: str | None = None

        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Top Filter Bar ────────────────────────────────────────────────
        filter_card = make_card()
        f_layout = QHBoxLayout(filter_card)
        f_layout.setContentsMargins(12, 8, 12, 8)

        f_layout.addWidget(QLabel("User:"))
        self.user_combo = QComboBox()
        self.user_combo.setMinimumWidth(140)
        f_layout.addWidget(self.user_combo)

        f_layout.addSpacing(10)
        f_layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit(QDate.currentDate().addDays(-14))
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        f_layout.addWidget(self.start_date)

        f_layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        f_layout.addWidget(self.end_date)

        f_layout.addSpacing(10)
        f_layout.addWidget(QLabel("Incident Threshold:"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.05, 0.99)
        self.thresh_spin.setSingleStep(0.05)
        self.thresh_spin.setValue(float(self.cfg.get("telemetry", {}).get("risk_threshold", 0.65)))
        f_layout.addWidget(self.thresh_spin)

        f_layout.addStretch()

        self.refresh_btn = QPushButton("↻ Query Database")
        self.refresh_btn.clicked.connect(self.refresh_data)
        self.export_btn = QPushButton("⤓ Export CSV")
        self.export_btn.clicked.connect(self._export_csv)
        f_layout.addWidget(self.refresh_btn)
        f_layout.addWidget(self.export_btn)

        root.addWidget(filter_card)

        # ── KPI Summary Strip ─────────────────────────────────────────────
        kpi_row = QHBoxLayout()
        self.kpi_sessions = self._make_kpi_card("TOTAL SESSIONS", "0")
        self.kpi_samples = self._make_kpi_card("LOGGED ESTIMATES", "0")
        self.kpi_peak = self._make_kpi_card("PEAK RISK SCORE", "0.000")
        self.kpi_incidents = self._make_kpi_card("HIGH-RISK SPIKES", "0")
        for card, _ in (self.kpi_sessions, self.kpi_samples, self.kpi_peak, self.kpi_incidents):
            kpi_row.addWidget(card)
        root.addLayout(kpi_row)

        # ── Main Splitter: Left Sessions Table | Right Charts & Spike Log ─
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Panel: Sessions Browser
        sess_card = make_card()
        sess_layout = QVBoxLayout(sess_card)
        sess_head = QHBoxLayout()
        sess_title = QLabel("RECORDED SESSIONS (CLICK TO INSPECT)")
        sess_title.setObjectName("CardTitle")
        self.clear_sel_btn = QPushButton("Show All Sessions")
        self.clear_sel_btn.clicked.connect(self._clear_session_selection)
        sess_head.addWidget(sess_title)
        sess_head.addStretch()
        sess_head.addWidget(self.clear_sel_btn)
        sess_layout.addLayout(sess_head)

        self.sess_table = QTableWidget(0, 6)
        self.sess_table.setHorizontalHeaderLabels(
            ["Session ID", "User", "Date", "Samples", "Mean Risk", "Peak Risk"]
        )
        self.sess_table.setAlternatingRowColors(True)
        self.sess_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.sess_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sess_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sess_table.itemSelectionChanged.connect(self._on_session_selected)
        sess_layout.addWidget(self.sess_table)
        splitter.addWidget(sess_card)

        # Right Panel: Session Timeline + Daily Histogram + Incident Log
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(10)

        # Plot 1: Session Timeline with ±sqrt(Covariance) band
        tl_card = make_card()
        tl_layout = QVBoxLayout(tl_card)
        self.tl_title = QLabel("SESSION TIMELINE (SCORE ± √COVARIANCE BAND)")
        self.tl_title.setObjectName("CardTitle")
        tl_layout.addWidget(self.tl_title)

        self.timeline_plot = pg.PlotWidget()
        self.timeline_plot.setYRange(0.0, 1.05)
        self.timeline_plot.showGrid(x=True, y=True, alpha=0.2)
        self.timeline_plot.setLabel("left", "Risk Score")
        self.timeline_plot.setLabel("bottom", "Sample Index")
        tl_layout.addWidget(self.timeline_plot)
        r_layout.addWidget(tl_card, stretch=38)

        # Plot 2: Daily Incident Histogram (ReportAggregator style)
        hist_card = make_card()
        hist_layout = QVBoxLayout(hist_card)
        hist_title = QLabel("DAILY HIGH-RISK INCIDENT HISTOGRAM")
        hist_title.setObjectName("CardTitle")
        hist_layout.addWidget(hist_title)

        self.hist_plot = pg.PlotWidget()
        self.hist_plot.showGrid(x=False, y=True, alpha=0.2)
        self.hist_plot.setLabel("left", "Spike Count")
        self.hist_plot.setLabel("bottom", "Day Bucket")
        hist_layout.addWidget(self.hist_plot)
        r_layout.addWidget(hist_card, stretch=30)

        # Bottom Table: High-Risk Event Spikes
        ev_card = make_card()
        ev_layout = QVBoxLayout(ev_card)
        ev_title = QLabel("HIGH-RISK ESTIMATE LOG (SCORE ≥ THRESHOLD)")
        ev_title.setObjectName("CardTitle")
        ev_layout.addWidget(ev_title)

        self.ev_table = QTableWidget(0, 4)
        self.ev_table.setHorizontalHeaderLabels(["Session ID", "Timestamp", "Fused Score", "Covariance (P)"])
        self.ev_table.setAlternatingRowColors(True)
        self.ev_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        ev_layout.addWidget(self.ev_table)
        r_layout.addWidget(ev_card, stretch=32)

        splitter.addWidget(right_panel)
        splitter.setSizes([460, 740])
        root.addWidget(splitter, stretch=1)

    def _make_kpi_card(self, title: str, initial: str) -> tuple[QFrame, QLabel]:
        card = make_card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        t_lbl = QLabel(title)
        t_lbl.setObjectName("CardTitle")
        v_lbl = QLabel(initial)
        v_lbl.setObjectName("MonoValue")
        v_lbl.setStyleSheet("font-size: 24px; color: #F8FAFC;")
        lay.addWidget(t_lbl)
        lay.addWidget(v_lbl)
        return card, v_lbl

    @Slot()
    def _clear_session_selection(self) -> None:
        self.selected_session_id = None
        self.sess_table.clearSelection()
        self._update_details()

    @Slot()
    def refresh_data(self) -> None:
        # Populate User dropdown while preserving selection
        curr_user = self.user_combo.currentText()
        users = ["All Users"] + self.store.list_users()
        self.user_combo.blockSignals(True)
        self.user_combo.clear()
        self.user_combo.addItems(users)
        if curr_user in users:
            self.user_combo.setCurrentText(curr_user)
        self.user_combo.blockSignals(False)

        user_filter = self.user_combo.currentText()
        s_date = self.start_date.date().toString("yyyy-MM-dd")
        e_date = self.end_date.date().toString("yyyy-MM-dd")

        self.sessions = self.store.query_sessions(user_filter, s_date, e_date)
        self.sess_table.setRowCount(len(self.sessions))

        total_samples = 0
        global_peak = 0.0

        for row_idx, s in enumerate(self.sessions):
            total_samples += s.sample_count
            global_peak = max(global_peak, s.peak_score)

            items = [
                QTableWidgetItem(s.session_id),
                QTableWidgetItem(s.user_id),
                QTableWidgetItem(s.session_date),
                QTableWidgetItem(str(s.sample_count)),
                QTableWidgetItem(f"{s.mean_score:.3f}"),
                QTableWidgetItem(f"{s.peak_score:.3f}"),
            ]
            items[5].setForeground(QColor(risk_color_hex(s.peak_score)))
            for col_idx, it in enumerate(items):
                self.sess_table.setItem(row_idx, col_idx, it)

        self.kpi_sessions[1].setText(str(len(self.sessions)))
        self.kpi_samples[1].setText(f"{total_samples:,}")
        self.kpi_peak[1].setText(f"{global_peak:.3f}")
        self.kpi_peak[1].setStyleSheet(f"font-size: 24px; color: {risk_color_hex(global_peak)};")

        self._update_details()

    @Slot()
    def _on_session_selected(self) -> None:
        sel = self.sess_table.selectedItems()
        if not sel:
            self.selected_session_id = None
        else:
            self.selected_session_id = sel[0].text()
        self._update_details()

    def _update_details(self) -> None:
        user_filter = self.user_combo.currentText()
        s_date = self.start_date.date().toString("yyyy-MM-dd")
        e_date = self.end_date.date().toString("yyyy-MM-dd")
        thresh = self.thresh_spin.value()

        # 1. High-Risk Spike Table
        spikes = self.store.get_filtered_estimates(
            user_id=user_filter,
            start_date=s_date,
            end_date=e_date,
            session_id=self.selected_session_id,
            min_score=thresh,
            limit=500,
        )
        self.kpi_incidents[1].setText(str(len(spikes)))
        self.ev_table.setRowCount(len(spikes))
        for r_idx, est in enumerate(spikes):
            row_items = [
                QTableWidgetItem(est.session_id),
                QTableWidgetItem(est.timestamp),
                QTableWidgetItem(f"{est.score:.3f}"),
                QTableWidgetItem(f"{est.covariance:.4f}"),
            ]
            row_items[2].setForeground(QColor(risk_color_hex(est.score)))
            for c_idx, item in enumerate(row_items):
                self.ev_table.setItem(r_idx, c_idx, item)

        # 2. Timeline Plot (Selected session or most recent session)
        self.timeline_plot.clear()
        target_session = self.selected_session_id or (self.sessions[0].session_id if self.sessions else None)
        if target_session:
            self.tl_title.setText(f"SESSION TIMELINE: {target_session} (SCORE ± √COVARIANCE)")
            rows = self.store.get_session_estimates(target_session)
            if rows:
                x = np.arange(len(rows), dtype=np.float32)
                scores = np.array([r.score for r in rows], dtype=np.float32)
                std = np.sqrt(np.clip([r.covariance for r in rows], 0.0, 1.0))
                upper = np.clip(scores + std, 0.0, 1.0)
                lower = np.clip(scores - std, 0.0, 1.0)

                c_up = pg.PlotDataItem(x, upper, pen=pg.mkPen(None))
                c_lo = pg.PlotDataItem(x, lower, pen=pg.mkPen(None))
                fill = pg.FillBetweenItem(c_lo, c_up, brush=pg.mkBrush(56, 189, 248, 45))
                self.timeline_plot.addItem(fill)
                self.timeline_plot.plot(x, scores, pen=pg.mkPen("#38BDF8", width=2.2))

                # Threshold line
                t_line = pg.InfiniteLine(
                    pos=thresh,
                    angle=0,
                    pen=pg.mkPen("#EF4444", width=1.5, style=Qt.PenStyle.DashLine),
                )
                self.timeline_plot.addItem(t_line)
        else:
            self.tl_title.setText("SESSION TIMELINE (NO SESSIONS RECORDED)")

        # 3. Daily Incident Histogram
        self.hist_plot.clear()
        buckets = self.store.get_daily_incident_buckets(thresh, user_filter, s_date, e_date)
        if buckets:
            days = [b[0][5:] for b in buckets]  # MM-DD
            counts = [b[1] for b in buckets]
            x_idx = np.arange(len(buckets))
            bg = pg.BarGraphItem(x=x_idx, height=counts, width=0.55, brush="#F59E0B", pen="#D97706")
            self.hist_plot.addItem(bg)
            ax = self.hist_plot.getAxis("bottom")
            ax.setTicks([list(zip(x_idx, days))])

    @Slot()
    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Filtered Estimates to CSV", "risk_estimates_export.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        rows = self.store.get_filtered_estimates(
            user_id=self.user_combo.currentText(),
            start_date=self.start_date.date().toString("yyyy-MM-dd"),
            end_date=self.end_date.date().toString("yyyy-MM-dd"),
            session_id=self.selected_session_id,
            min_score=0.0,
            limit=100_000,
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["SessionID", "Timestamp", "Score", "Covariance"])
            for r in rows:
                writer.writerow([r.session_id, r.timestamp, f"{r.score:.6f}", f"{r.covariance:.6f}"])

        QMessageBox.information(self, "Export Complete", f"Exported {len(rows)} rows to:\n{path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ═══════════════════════════════════════════════════════════════════════════


class LiveLeakDesktopApp(QMainWindow):
    def __init__(self, config_path: Path = Path("app/config.toml")) -> None:
        super().__init__()
        self.cfg = self._load_config(config_path)
        db_path = self.cfg.get("session", {}).get("db_path", "risk.db")
        self.store = EventHistoryStore(db_path)

        self.setWindowTitle("Workplace Risk Detection — Live Monitor & Retroactive Telemetry")
        self.resize(1480, 880)

        self.tabs = QTabWidget()
        self.live_tab = LiveMonitorTab(self.store, self.cfg)
        self.data_tab = RetroactiveDataTab(self.store, self.cfg)

        self.tabs.addTab(self.live_tab, "◉  Live Worksite Monitor")
        self.tabs.addTab(self.data_tab, "▤  Retroactive Data & DB Analytics")
        self.tabs.currentChanged.connect(self._on_tab_switched)

        self.setCentralWidget(self.tabs)

    @staticmethod
    def _load_config(path: Path) -> dict:
        if path.is_file():
            with open(path, "rb") as f:
                return tomllib.load(f)
        return {}

    @Slot(int)
    def _on_tab_switched(self, idx: int) -> None:
        # Automatically refresh SQLite analytics when switching to the Data tab
        if idx == 1:
            self.data_tab.refresh_data()

    def closeEvent(self, event) -> None:
        self.live_tab.shutdown()
        self.store.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = LiveLeakDesktopApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
