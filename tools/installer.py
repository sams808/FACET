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
UNINSTALL_KEY = (r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
                 f"\\{APP_NAME}")


def resource(relative: str) -> Path:
    """Locate bundled data, whether frozen by PyInstaller or run from source."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


def payload_zip() -> Path | None:
    folder = resource("payload")
    if folder.is_dir():
        for item in folder.iterdir():
            if item.suffix.lower() == ".zip":
                return item
    return None


def payload_version() -> str:
    """The version being installed, read from the payload archive's name.

    The build names the archive ``FACET-0.3.1.zip``, so the version is already
    travelling with the payload and does not need to be repeated here. It used
    to be a constant in this file, which went stale immediately and made every
    upgrade register under the version of whichever build first wrote it.
    """
    archive = payload_zip()
    if archive is not None:
        stem = archive.stem
        if "-" in stem:
            candidate = stem.rsplit("-", 1)[-1]
            if candidate and candidate[0].isdigit():
                return candidate
    return "0.0.0"


VERSION = payload_version()


def previous_install() -> tuple[Path | None, str]:
    """Where an earlier FACET is installed, and which version, if any.

    Read from the same registry key this installer writes. Without this the
    installer would default to its own idea of where FACET belongs and, for
    anyone who had once chosen a different folder, install a second copy
    alongside the first rather than replacing it.
    """
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
            location, _ = winreg.QueryValueEx(key, "InstallLocation")
            try:
                version, _ = winreg.QueryValueEx(key, "DisplayVersion")
            except OSError:
                version = ""
        path = Path(location)
        if path.is_dir():
            return path, str(version)
    except Exception:
        pass
    return None, ""


def default_target() -> Path:
    """Where to install: over the existing copy if there is one."""
    found, _version = previous_install()
    if found is not None:
        return found
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "Programs" / APP_NAME


def running_exe(folder: Path) -> Path | None:
    """The application executable in ``folder`` if it is currently running.

    Windows holds a lock on a running executable, so opening it for append
    fails. Detecting that here turns a partial, confusing failure part-way
    through removing the old version into a clear instruction before anything
    has been touched.
    """
    exe = folder / f"{APP_NAME}.exe"
    if not exe.exists():
        return None
    try:
        with open(exe, "ab"):
            return None
    except OSError:
        return exe


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

            # Everywhere a previous copy might be: where it says it is, and
            # where this install is going. Both are cleared, so an upgrade
            # cannot leave two applications behind one registry entry.
            earlier, _version = previous_install()
            doomed = [p for p in {self.target.resolve()
                                  if self.target.exists() else self.target,
                                  earlier.resolve() if earlier else None}
                      if p is not None]

            for folder in doomed:
                locked = running_exe(folder)
                if locked is not None:
                    raise RuntimeError(
                        f"{APP_NAME} is running from {folder}. Close it and "
                        "run this installer again.")

            for folder in doomed:
                if not folder.exists():
                    continue
                self.progress.emit(10, "Removing the previous version…")
                shutil.rmtree(folder, ignore_errors=True)
                stale = folder / f"{APP_NAME}.exe"
                if stale.exists():
                    raise RuntimeError(
                        f"the previous version in {folder} could not be "
                        f"removed. Close {APP_NAME} and any window showing "
                        "that folder, then run this installer again.")

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

            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
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
        self.setWindowTitle(f"Install {APP_NAME}")   # retitled below once a previous install is known
        self.setFixedSize(560, 420)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.previous, self.previous_version = previous_install()
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

        if self.previous is not None:
            was = f" {self.previous_version}" if self.previous_version else ""
            note = (f"<span style='color:#777'>Replaces the copy{was} already "
                    f"installed in {self.previous}. Your settings and themes "
                    "are kept.</span>")
        else:
            note = ("<span style='color:#777'>Installs for you only, so no "
                    "administrator rights are needed.</span>")
        intro = QLabel(
            f"<b>{APP_NAME} {VERSION}</b> — coordination analysis, cut by "
            f"bond valence.<br>{note}")
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
        verb = "Upgrade" if self.previous is not None else "Install"
        self.setWindowTitle(f"{verb} {APP_NAME}")
        self.install = QPushButton(f"{verb} {APP_NAME}")
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
