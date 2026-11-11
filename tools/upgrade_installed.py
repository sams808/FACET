"""Upgrade the installed FACET from the build in ``dist/``, without the
installer's window.

    py -3.11 tools/upgrade_installed.py            # the version of facet/version.py
    py -3.11 tools/upgrade_installed.py 1.3.0

The Desktop and Start Menu shortcuts point at the installed copy (the
folder the registry names, by default under the user's local application
data), not at ``dist/FACET``, so a build alone leaves them on the previous
version. This runs the installer's own worker -- the code a collaborator's
``FACET-Setup-x.y.z.exe`` runs -- on ``dist/FACET-x.y.z.zip``: it refuses
while FACET is running from the installed folder, replaces that folder,
rewrites the shortcuts that installation owns and the registry entry.
Exit code 1 when the installer reports a failure, 2 when there is nothing
to install from.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv: list[str]) -> int:
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)  # noqa: F841
    import installer
    from facet.version import __version__

    version = argv[1] if len(argv) > 1 else __version__
    payload = ROOT / "dist" / f"FACET-{version}.zip"
    if not payload.is_file():
        print(f"no build to install: {payload} (run tools/build_exe.py)")
        return 2
    installer.payload_zip = lambda: payload
    installer.VERSION = installer.payload_version()
    earlier, was = installer.previous_install()
    target = installer.default_target()
    print(f"installed: {earlier or 'nothing'} {was}  ->  {target} {installer.VERSION}")
    locked = installer.running_exe(target)
    if locked is not None:
        print(f"FACET is running from {target}; close it and run this again")
        return 1
    outcome: dict = {}
    worker = installer.InstallWorker(target, desktop=True, start_menu=True,
                                     associate=False)
    worker.finished_ok.connect(lambda exe: outcome.setdefault("exe", exe))
    worker.failed.connect(lambda message: outcome.setdefault("failed", message))
    worker.run()
    if "failed" in outcome:
        print("not installed:", outcome["failed"])
        return 1
    print("installed", outcome.get("exe"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
