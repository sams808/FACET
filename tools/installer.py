"""The FACET installer.

A single executable that a collaborator can run without administrator rights.
It unpacks the application into the user's local application data, writes Start
Menu and optional desktop shortcuts, registers an uninstaller with Windows, and
offers to associate .cif files.

Per-user rather than machine-wide, deliberately: a research group is exactly the
setting where people cannot install software on their own machines, and an
installer that needs an administrator is an installer that does not get used.

Built by ``tools/build_exe.py``, which carries the zipped application inside
this executable as bundled data.
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "FACET"
VERSION = "0.1.0"


def resource(relative: str) -> Path:
    """Locate bundled data, whether frozen by PyInstaller or run from source."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


def default_target() -> Path:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "Programs" / APP_NAME


def payload_zip() -> Path | None:
    folder = resource("payload")
    if folder.is_dir():
        for item in folder.iterdir():
            if item.suffix.lower() == ".zip":
                return item
    return None


# ---------------------------------------------------------------------------

class InstallWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(Path)
    failed = Signal(str)

    def __init__(self, target: Path, desktop: bool, start_menu: bool,
                 associate: bool):
        super().__init__()
        self.target = target
        self.desktop = desktop
        self.start_menu = start_menu
        self.associate = associate

    def run(self) -> None:
        try:
            archive = payload_zip()
            if archive is None:
                raise RuntimeError("this installer carries no application payload")

            self.progress.emit(5, "Preparing…")
            if self.target.exists():
                self.progress.emit(10, "Removing the previous version…")
                shutil.rmtree(self.target, ignore_errors=True)
            self.target.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(archive) as z:
                members = z.namelist()
                total = max(len(members), 1)
                for i, member in enumerate(members):
                    # the archive holds a top-level FACET/ folder; strip it so
                    # the target directory is the application, not its parent
                    parts = Path(member).parts
                    if len(parts) <= 1:
                        continue
                    destination = self.target / Path(*parts[1:])
                    if member.endswith("/"):
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, open(destination, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if i % 40 == 0:
                        self.progress.emit(10 + int(80 * i / total),
                                           f"Installing… {Path(member).name}")

            exe = self.target / f"{APP_NAME}.exe"
            if not exe.exists():
                raise RuntimeError(f"{APP_NAME}.exe is missing from the payload")

            self.progress.emit(92, "Creating shortcuts…")
            if self.start_menu:
                self._shortcut(self._start_menu_dir() / f"{APP_NAME}.lnk", exe)
            if self.desktop:
                self._shortcut(Path.home() / "Desktop" / f"{APP_NAME}.lnk", exe)

            self.progress.emit(96, "Registering…")
            self._write_uninstaller(exe)
            self._register()
            if self.associate:
                self._associate_cif(exe)

            self.progress.emit(100, "Done")
            self.finished_ok.emit(exe)
        except Exception as exc:                          # pragma: no cover
            self.failed.emit(str(exc))

    # -- Windows integration ----------------------------------------------
    @staticmethod
    def _start_menu_dir() -> Path:
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        folder = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _shortcut(self, link: Path, target: Path) -> None:
        """Write a .lnk through the Windows scripting host.

        No third-party dependency: every Windows machine has wscript, and a
        four-line VBScript is the shortest reliable way to create a shortcut
        from a frozen Python application.
        """
        import subprocess
        import tempfile

        link.parent.mkdir(parents=True, exist_ok=True)
        script = f'''Set s = CreateObject("WScript.Shell")
Set l = s.CreateShortcut("{link}")
l.TargetPath = "{target}"
l.WorkingDirectory = "{target.parent}"
l.IconLocation = "{target}, 0"
l.Description = "{APP_NAME} - coordination analysis, cut by bond valence"
l.Save
'''
        with tempfile.NamedTemporaryFile("w", suffix=".vbs", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(script)
            path = handle.name
        try:
            subprocess.run(["wscript.exe", "//nologo", path],
                           check=False, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _write_uninstaller(self, exe: Path) -> None:
        script = self.target / "uninstall.bat"
        start_menu = self._start_menu_dir() / f"{APP_NAME}.lnk"
        desktop = Path.home() / "Desktop" / f"{APP_NAME}.lnk"
        script.write_text(f"""@echo off
echo Removing {APP_NAME}...
reg delete "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}" /f >nul 2>&1
del "{start_menu}" >nul 2>&1
del "{desktop}" >nul 2>&1
cd /d "%TEMP%"
rmdir /s /q "{self.target}"
echo Done.
""", encoding="utf-8")

    def _register(self) -> None:
        """Add an entry to Add/Remove Programs, under HKCU so no admin rights
        are needed."""
        try:
            import winreg

            key_path = (r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
                        f"\\{APP_NAME}")
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, APP_NAME)
                winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, VERSION)
                winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "FACET")
                winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ,
                                  str(self.target))
                winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ,
                                  str(self.target / f"{APP_NAME}.exe"))
                winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ,
                                  f'"{self.target / "uninstall.bat"}"')
                winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
        except Exception:
            pass          # an unregistered install still works

    def _associate_cif(self, exe: Path) -> None:
        try:
            import winreg

            prog_id = f"{APP_NAME}.cif"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                                  rf"Software\Classes\{prog_id}") as key:
                winreg.SetValueEx(key, None, 0, winreg.REG_SZ,
                                  "Crystallographic Information File")
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                                  rf"Software\Classes\{prog_id}\DefaultIcon") as key:
                winreg.SetValueEx(key, None, 0, winreg.REG_SZ, f"{exe},0")
            with winreg.CreateKey(
                    winreg.HKEY_CURRENT_USER,
                    rf"Software\Classes\{prog_id}\shell\open\command") as key:
                winreg.SetValueEx(key, None, 0, winreg.REG_SZ, f'"{exe}" "%1"')
            with winreg.CreateKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Classes\.cif\OpenWithProgids") as key:
                winreg.SetValueEx(key, prog_id, 0, winreg.REG_NONE, b"")
        except Exception:
            pass


# ---------------------------------------------------------------------------

class InstallerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Install {APP_NAME}")
        self.setFixedSize(560, 420)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.target = default_target()
        self.worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        banner = QLabel()
        banner.setAlignment(Qt.AlignCenter)
        splash = resource("payload/splash.png")
        if splash.exists():
            banner.setPixmap(QPixmap(str(splash)).scaledToWidth(
                560, Qt.SmoothTransformation))
        else:
            banner.setText(APP_NAME)
            f = QFont(banner.font())
            f.setPointSize(30)
            f.setBold(True)
            banner.setFont(f)
            banner.setFixedHeight(150)
            banner.setStyleSheet("background:#14161b; color:#e8ecf4;")
        layout.addWidget(banner)

        body = QWidget()
        form = QVBoxLayout(body)
        form.setContentsMargins(26, 18, 26, 18)
        form.setSpacing(10)

        intro = QLabel(
            f"<b>{APP_NAME} {VERSION}</b> — coordination analysis, cut by "
            "bond valence.<br><span style='color:#777'>Installs for you only, "
            "so no administrator rights are needed.</span>")
        intro.setWordWrap(True)
        form.addWidget(intro)

        self.path_label = QLabel(str(self.target))
        self.path_label.setStyleSheet("color:#555; font-size:11px;")
        self.path_label.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(QLabel("Install to:"))
        row.addWidget(self.path_label, 1)
        change = QPushButton("Change…")
        change.clicked.connect(self._choose)
        row.addWidget(change)
        form.addLayout(row)

        self.start_menu = QCheckBox("Add to the Start Menu")
        self.start_menu.setChecked(True)
        self.desktop = QCheckBox("Create a desktop shortcut")
        self.desktop.setChecked(True)
        self.associate = QCheckBox("Open .cif files with FACET")
        self.associate.setChecked(False)
        for box in (self.start_menu, self.desktop, self.associate):
            form.addWidget(box)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        form.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#666; font-size:11px;")
        form.addWidget(self.status)
        form.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.close)
        self.install = QPushButton(f"Install {APP_NAME}")
        self.install.setDefault(True)
        self.install.clicked.connect(self._install)
        buttons.addWidget(self.cancel)
        buttons.addWidget(self.install)
        form.addLayout(buttons)

        layout.addWidget(body, 1)

    def _choose(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(
            self, "Install to", str(self.target.parent))
        if folder:
            self.target = Path(folder) / APP_NAME
            self.path_label.setText(str(self.target))

    def _install(self) -> None:
        self.install.setEnabled(False)
        self.cancel.setEnabled(False)
        self.progress.show()
        self.worker = InstallWorker(self.target, self.desktop.isChecked(),
                                    self.start_menu.isChecked(),
                                    self.associate.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self.status.setText(message)

    def _on_done(self, exe: Path) -> None:
        self.status.setText("Installed.")
        answer = QMessageBox.question(
            self, f"{APP_NAME} installed",
            f"{APP_NAME} {VERSION} is installed at:\n{self.target}\n\n"
            "Start it now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            os.startfile(str(exe))            # noqa: S606 - the app we installed
        self.close()

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Installation failed", message)
        self.install.setEnabled(True)
        self.cancel.setEnabled(True)
        self.progress.hide()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(f"{APP_NAME} installer")
    icon = resource("payload/facet.ico")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    window = InstallerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
