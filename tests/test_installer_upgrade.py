"""The installer must replace the copy already on the machine.

A collaborator who runs a new installer expects to end up with one FACET, not
two. Three things have to hold for that: the version written to the registry
has to be the version actually being installed, the installer has to default to
wherever the previous copy lives rather than to its own idea of where FACET
belongs, and it has to refuse rather than half-succeed when the application it
is replacing is running.

The registry test creates the real HKCU key, because that is the one the
installer reads, and skips itself entirely if a genuine installation is already
registered so that it can never disturb one.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

installer = pytest.importorskip("installer")
winreg = pytest.importorskip("winreg")


# -- the version that gets registered ---------------------------------------

def test_the_version_comes_from_the_payload(tmp_path, monkeypatch):
    """It was a constant in the file, and the constant went stale.

    Every build after the first registered itself as 0.1.0, so Windows showed
    the wrong version and an upgrade looked like nothing had changed.
    """
    archive = tmp_path / "FACET-9.9.9.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("FACET/FACET.exe", b"not really")
    monkeypatch.setattr(installer, "payload_zip", lambda: archive)
    assert installer.payload_version() == "9.9.9"


def test_a_missing_payload_does_not_raise(monkeypatch):
    monkeypatch.setattr(installer, "payload_zip", lambda: None)
    assert installer.payload_version() == "0.0.0"


def test_an_unversioned_payload_name_does_not_raise(tmp_path, monkeypatch):
    archive = tmp_path / "payload.zip"
    archive.write_bytes(b"")
    monkeypatch.setattr(installer, "payload_zip", lambda: archive)
    assert installer.payload_version() == "0.0.0"


# -- finding the copy already installed -------------------------------------

@pytest.fixture
def no_real_install():
    """Skip rather than touch a real installation's registry entry."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, installer.UNINSTALL_KEY):
            pytest.skip("FACET is installed on this machine; not touching it")
    except FileNotFoundError:
        pass
    yield
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, installer.UNINSTALL_KEY)
    except OSError:
        pass


def test_no_entry_means_the_standard_location(no_real_install):
    assert installer.previous_install() == (None, "")
    target = installer.default_target()
    assert target.name == installer.APP_NAME
    assert "Programs" in target.parts


def test_an_existing_install_is_found_and_installed_over(no_real_install,
                                                         tmp_path):
    """The point of the whole exercise.

    Someone who once chose a different folder must not end up with a second
    copy there and a first one still in place.
    """
    earlier = tmp_path / "SomewhereElse" / "FACET"
    earlier.mkdir(parents=True)
    (earlier / "FACET.exe").write_bytes(b"old")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                          installer.UNINSTALL_KEY) as key:
        winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(earlier))
        winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, "0.2.0")

    found, version = installer.previous_install()
    assert found == earlier
    assert version == "0.2.0"
    assert installer.default_target() == earlier


def test_a_registered_folder_that_has_been_deleted_is_ignored(no_real_install,
                                                              tmp_path):
    """An uninstall by hand leaves the key behind; do not install into nothing."""
    gone = tmp_path / "deleted" / "FACET"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                          installer.UNINSTALL_KEY) as key:
        winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(gone))

    assert installer.previous_install() == (None, "")
    assert installer.default_target() != gone


# -- refusing to overwrite a running application ----------------------------

def test_a_folder_without_the_exe_is_not_running(tmp_path):
    assert installer.running_exe(tmp_path) is None


def test_an_openable_exe_is_not_running(tmp_path):
    (tmp_path / "FACET.exe").write_bytes(b"idle")
    assert installer.running_exe(tmp_path) is None


def test_a_locked_exe_is_reported_as_running(tmp_path):
    """Windows locks a running executable; opening it for append fails.

    Without this check the installer deletes most of the previous version,
    fails on the locked file, and leaves a half-removed application behind.

    The lock is taken the way Windows takes it for a loaded image: a handle
    that permits further readers but no further writers. ``msvcrt.locking``
    will not do, because it locks a byte range rather than the file, and a
    later ``open(..., "ab")`` succeeds straight through it.
    """
    import ctypes
    from ctypes import wintypes

    exe = tmp_path / "FACET.exe"
    exe.write_bytes(b"running")

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    OPEN_EXISTING = 3
    INVALID = wintypes.HANDLE(-1).value

    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.restype = wintypes.HANDLE
    CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                            wintypes.HANDLE]
    handle = CreateFileW(str(exe), GENERIC_READ, FILE_SHARE_READ, None,
                         OPEN_EXISTING, 0, None)
    if handle == INVALID:
        pytest.skip("could not take an image-style lock on this filesystem")
    try:
        assert installer.running_exe(tmp_path) == exe
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)

    # and once the handle is gone it is installable again
    assert installer.running_exe(tmp_path) is None
