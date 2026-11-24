"""Google OAuth + TOTP 2FA Dialog and User Profile Pill for PySide6."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pyotp
import qrcode
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.db import EventHistoryStore, UserRecord

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


def make_circular_pixmap(source: QPixmap, size: int = 28) -> QPixmap:
    """Crop/scale a QPixmap into a smooth anti-aliased circle."""
    scaled = source.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    out = QPixmap(size, size)
    out.fill(Qt.GlobalColor.transparent)

    painter = QPainter(out)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    painter.setClipPath(path)

    x_off = (scaled.width() - size) // 2
    y_off = (scaled.height() - size) // 2
    painter.drawPixmap(-x_off, -y_off, scaled)
    painter.end()
    return out


def make_initials_pixmap(name: str, size: int = 28) -> QPixmap:
    """Fallback circular avatar with user initials."""
    parts = [p[0].upper() for p in name.strip().split() if p]
    initials = "".join(parts[:2]) or "?"

    out = QPixmap(size, size)
    out.fill(Qt.GlobalColor.transparent)

    painter = QPainter(out)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#0284C7"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, size, size)

    painter.setPen(QColor("#F8FAFC"))
    font = QFont("Inter", max(9, size // 3), QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(out.rect(), Qt.AlignmentFlag.AlignCenter, initials)
    painter.end()
    return out


class AvatarLoaderThread(QThread):
    """Fetches a Google profile picture URL off the UI thread."""

    loaded = Signal(QPixmap)

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url

    def run(self) -> None:
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "LiveLeakClient/1.0"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                raw = resp.read()
            pix = QPixmap()
            if pix.loadFromData(raw):
                self.loaded.emit(pix)
        except Exception:
            pass


class GoogleOAuthWorker(QThread):
    """Runs InstalledAppFlow local server without freezing the Qt event loop."""

    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, secret_path: Path) -> None:
        super().__init__()
        self.secret_path = secret_path

    def run(self) -> None:
        try:
            if not self.secret_path.is_file():
                self.failed.emit(f"OAuth secret file not found: {self.secret_path}")
                return

            with open(self.secret_path, "r", encoding="utf-8") as f:
                oauth_secrets: dict = json.load(f)
            oauth_secrets.pop("otp", None)

            flow = InstalledAppFlow.from_client_config(oauth_secrets, SCOPES)
            creds = flow.run_local_server(port=0)
            service = build("oauth2", "v2", credentials=creds)
            user_info = service.userinfo().get().execute()

            self.succeeded.emit(
                {
                    "id": str(user_info.get("id", "")),
                    "email": user_info.get("email"),
                    "name": user_info.get("name") or user_info.get("email") or "Operator",
                    "picture": user_info.get("picture"),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class OTPDialog(QDialog):
    """
    Modal popup for TOTP QR code setup & 6-digit verification.
    Includes a '?' button to reveal the raw Base32 secret key if QR scanning fails.
    """

    def __init__(
            self,
            user: UserRecord,
            secret: str,
            is_setup: bool,
            parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.user = user
        self.secret = secret
        self.is_setup = is_setup
        self.totp = pyotp.TOTP(secret)

        self.setWindowTitle("Two-Factor Authentication (TOTP)")
        self.setModal(True)
        self.setFixedWidth(390)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Set Up Google Authenticator" if self.is_setup else "Verify Authenticator Code")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #F8FAFC;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle_txt = (
            f"Scan this QR code in Google Authenticator for\n{self.user.email or self.user.name}, "
            "then enter the 6-digit code below."
            if self.is_setup
            else f"Enter the 6-digit Google Authenticator code for\n{self.user.email or self.user.name}."
        )
        subtitle = QLabel(subtitle_txt)
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #94A3B8; font-size: 12px;")
        layout.addWidget(subtitle)

        # QR Code Frame (shown during setup, or toggled via '?' if needed)
        self.qr_container = QFrame()
        self.qr_container.setStyleSheet(
            "background-color: #FFFFFF; border-radius: 8px; padding: 8px;"
        )
        qr_lay = QVBoxLayout(self.qr_container)
        qr_lay.setContentsMargins(6, 6, 6, 6)

        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr_label.setPixmap(self._render_qr_pixmap())
        qr_lay.addWidget(self.qr_label)

        layout.addWidget(self.qr_container, alignment=Qt.AlignmentFlag.AlignCenter)
        self.qr_container.setVisible(self.is_setup)

        # Code Entry + '?' Help Button Row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("000000")
        self.code_input.setMaxLength(6)
        self.code_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.code_input.setStyleSheet(
            "font-family: 'JetBrains Mono', 'Consolas', monospace; "
            "font-size: 20px; font-weight: 700; letter-spacing: 6px; padding: 8px;"
        )
        self.code_input.returnPressed.connect(self._verify_code)
        input_row.addWidget(self.code_input, stretch=1)

        self.help_btn = QPushButton("?")
        self.help_btn.setFixedSize(38, 38)
        self.help_btn.setToolTip("Show manual setup key if QR code isn't working")
        self.help_btn.setStyleSheet(
            "QPushButton { background-color: #1E293B; border: 1px solid #38BDF8; "
            "border-radius: 19px; color: #38BDF8; font-weight: 800; font-size: 15px; }"
            "QPushButton:hover { background-color: #0284C7; color: #FFFFFF; }"
        )
        self.help_btn.clicked.connect(self._toggle_manual_key)
        input_row.addWidget(self.help_btn)

        layout.addLayout(input_row)

        # Collapsible Manual Key Box (toggled by '?' button)
        self.key_box = QFrame()
        self.key_box.setStyleSheet(
            "background-color: #161922; border: 1px solid #2E3446; border-radius: 6px;"
        )
        key_lay = QVBoxLayout(self.key_box)
        key_lay.setContentsMargins(10, 8, 10, 8)
        key_lay.setSpacing(6)

        key_hint = QLabel("Can't scan the QR? Enter this setup key manually:")
        key_hint.setStyleSheet("color: #94A3B8; font-size: 11px;")
        key_lay.addWidget(key_hint)

        key_copy_row = QHBoxLayout()
        self.key_field = QLineEdit(self.secret)
        self.key_field.setReadOnly(True)
        self.key_field.setStyleSheet(
            "font-family: 'JetBrains Mono', 'Consolas', monospace; "
            "font-size: 12px; color: #38BDF8; background-color: #0F1117;"
        )
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(60)
        copy_btn.clicked.connect(self._copy_secret)
        key_copy_row.addWidget(self.key_field)
        key_copy_row.addWidget(copy_btn)
        key_lay.addLayout(key_copy_row)

        self.key_box.setVisible(False)
        layout.addWidget(self.key_box)

        # Error feedback label
        self.error_lbl = QLabel("")
        self.error_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_lbl.setStyleSheet("color: #EF4444; font-size: 12px; font-weight: 600;")
        self.error_lbl.setVisible(False)
        layout.addWidget(self.error_lbl)

        # Action Buttons
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        verify_btn = QPushButton("Verify & Continue")
        verify_btn.setObjectName("StartBtn")
        verify_btn.clicked.connect(self._verify_code)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(verify_btn, stretch=1)
        layout.addLayout(btn_row)

        self.code_input.setFocus()

    def _render_qr_pixmap(self) -> QPixmap:
        account_label = self.user.email or self.user.name or self.user.id
        uri = self.totp.provisioning_uri(name=account_label, issuer_name="Team Werkzeug")
        qr_img = qrcode.make(uri)
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        pix = QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")
        return pix.scaled(
            210,
            210,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    @Slot()
    def _toggle_manual_key(self) -> None:
        show = not self.key_box.isVisible()
        self.key_box.setVisible(show)
        self.adjustSize()

    @Slot()
    def _copy_secret(self) -> None:
        QApplication.clipboard().setText(self.secret)
        self.error_lbl.setStyleSheet("color: #10B981; font-size: 12px; font-weight: 600;")
        self.error_lbl.setText("Setup key copied to clipboard.")
        self.error_lbl.setVisible(True)

    @Slot()
    def _verify_code(self) -> None:
        code = self.code_input.text().strip().replace(" ", "")
        if len(code) != 6 or not code.isdigit():
            self._show_error("Please enter a valid 6-digit numeric code.")
            return

        if self.totp.verify(code, valid_window=1):
            self.accept()
        else:
            self._show_error("Invalid or expired code. Please try again.")
            self.code_input.selectAll()
            self.code_input.setFocus()

    def _show_error(self, msg: str) -> None:
        self.error_lbl.setStyleSheet("color: #EF4444; font-size: 12px; font-weight: 600;")
        self.error_lbl.setText(msg)
        self.error_lbl.setVisible(True)


class UserProfilePill(QFrame):
    """
    Top-bar pill displaying either a 'Sign in with Google' button (when logged out)
    or the authenticated user's circular avatar + name + account menu (when logged in).
    """

    login_requested = Signal()
    logout_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_user: UserRecord | None = None
        self._avatar_loader: AvatarLoaderThread | None = None

        self.setObjectName("UserPill")
        self.setStyleSheet(
            """
            QFrame#UserPill {
                background-color: #161922;
                border: 1px solid #2E3446;
                border-radius: 17px;
                margin: 3px 8px 3px 0px;
            }
            QFrame#UserPill:hover {
                border: 1px solid #38BDF8;
                background-color: #1E2433;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 12, 3)
        layout.setSpacing(8)

        self.avatar_lbl = QLabel()
        self.avatar_lbl.setFixedSize(26, 26)
        layout.addWidget(self.avatar_lbl)

        self.name_lbl = QLabel("Sign in")
        self.name_lbl.setStyleSheet("color: #F8FAFC; font-weight: 600; font-size: 12px;")
        layout.addWidget(self.name_lbl)

        self.caret_lbl = QLabel("▾")
        self.caret_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
        layout.addWidget(self.caret_lbl)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_logged_out()

    def set_logged_out(self) -> None:
        self.current_user = None
        self.avatar_lbl.setPixmap(make_initials_pixmap("?", 26))
        self.name_lbl.setText("Sign in with Google")
        self.caret_lbl.setVisible(False)
        self.setToolTip("Click to sign in with Google + 2FA")

    def set_loading(self, text: str = "Signing in…") -> None:
        self.name_lbl.setText(text)
        self.setEnabled(False)

    def set_user(self, user: UserRecord) -> None:
        self.setEnabled(True)
        self.current_user = user
        display_name = user.name or user.email or user.id
        self.name_lbl.setText(display_name)
        self.caret_lbl.setVisible(True)
        self.setToolTip(f"{display_name}\n{user.email or ''}\nClick for account options")

        # Immediately render initials fallback, then fetch picture async if available
        self.avatar_lbl.setPixmap(make_initials_pixmap(display_name, 26))
        if user.picture:
            self._avatar_loader = AvatarLoaderThread(user.picture)
            self._avatar_loader.loaded.connect(self._on_avatar_loaded)
            self._avatar_loader.start()

    @Slot(QPixmap)
    def _on_avatar_loaded(self, pix: QPixmap) -> None:
        self.avatar_lbl.setPixmap(make_circular_pixmap(pix, 26))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_user is None:
                self.login_requested.emit()
            else:
                menu = QMenu(self)
                menu.setStyleSheet(
                    """
                    QMenu {
                        background-color: #161922;
                        border: 1px solid #2E3446;
                        color: #E2E8F0;
                        padding: 4px;
                    }
                    QMenu::item {
                        padding: 6px 16px;
                        border-radius: 4px;
                    }
                    QMenu::item:selected {
                        background-color: #1E3A5F;
                    }
                    """
                )
                info_act = menu.addAction(self.current_user.email or self.current_user.id)
                info_act.setEnabled(False)
                menu.addSeparator()
                switch_act = menu.addAction("Switch Account…")
                logout_act = menu.addAction("Sign Out")

                chosen = menu.exec(self.mapToGlobal(self.rect().bottomLeft()))
                if chosen == switch_act:
                    self.login_requested.emit()
                elif chosen == logout_act:
                    self.logout_requested.emit()
        super().mousePressEvent(event)


def run_interactive_login(
        store: EventHistoryStore,
        profile: dict,
        parent: QWidget | None = None,
) -> UserRecord | None:
    """
    Upserts the Google profile into USERS, launches the OTPDialog (QR setup if `otp` is NULL,
    or verification if already enrolled), and returns the authenticated UserRecord on success.
    """
    user = store.upsert_user_profile(
        user_id=profile["id"],
        email=profile.get("email"),
        name=profile.get("name"),
        picture=profile.get("picture"),
    )

    is_setup = not bool(user.otp)
    secret = pyotp.random_base32() if is_setup else str(user.otp)

    dlg = OTPDialog(user=user, secret=secret, is_setup=is_setup, parent=parent)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        if is_setup:
            store.set_user_otp_secret(user.id, secret)
            user = UserRecord(
                id=user.id,
                email=user.email,
                name=user.name,
                picture=user.picture,
                otp=secret,
            )
        return user

    return None
