"""File > Report on CIFs…: a set of files or folders in, a report out.

The dialog gathers the files, takes the thresholds and the parameter set the
window is using, and hands everything to ``facet.core.report``, which does
the reading, the analysis and the writing. Every format writes the same five
tables; the dialog only chooses which container and where.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import bv
from ..core import report as report_mod


class ReportDialog(QDialog):
    """Choose the files, the thresholds and the format; write the report."""

    def __init__(self, parent: QWidget | None = None, *, v_bond: float = bv.V_BOND_DEFAULT,
                 v_list: float = bv.V_LIST_DEFAULT, params=None,
                 anions: bool = False, paths: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Report on CIFs")
        self.setMinimumWidth(620)
        self.params = params
        self.written: list[Path] = []

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Every site of every file, at the thresholds below: coordination "
            "number, bond valences and their sums, distances, plateaus, the "
            "distance each pair's cutoff corresponds to, and the charge "
            "balance. The same five tables in whichever format is chosen.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.files = QListWidget()
        self.files.setSelectionMode(QListWidget.ExtendedSelection)
        self.files.setToolTip("CIF files, or folders of them. A folder "
                              "contributes every .cif it holds.")
        layout.addWidget(self.files, 1)
        for p in paths or []:
            self.files.addItem(str(p))

        row = QHBoxLayout()
        add_files = QPushButton("Add files…")
        add_files.clicked.connect(self._add_files)
        add_folder = QPushButton("Add folder…")
        add_folder.clicked.connect(self._add_folder)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove)
        for b in (add_files, add_folder, remove):
            row.addWidget(b)
        row.addStretch(1)
        self.count = QLabel("")
        row.addWidget(self.count)
        layout.addLayout(row)

        form = QFormLayout()
        self.v_bond = QDoubleSpinBox()
        self.v_bond.setDecimals(4)
        self.v_bond.setRange(0.0001, 5.0)
        self.v_bond.setSingleStep(0.005)
        self.v_bond.setValue(v_bond)
        self.v_bond.setSuffix(" v.u.")
        self.v_bond.setToolTip("A contact above this valence counts as a bond.")
        form.addRow("Bond threshold", self.v_bond)
        self.v_list = QDoubleSpinBox()
        self.v_list.setDecimals(4)
        self.v_list.setRange(0.0001, 5.0)
        self.v_list.setSingleStep(0.005)
        self.v_list.setValue(v_list)
        self.v_list.setSuffix(" v.u.")
        self.v_list.setToolTip("Contacts above this valence are listed and "
                               "summed to the listing cut.")
        form.addRow("Listing threshold", self.v_list)
        self.anions = QCheckBox("Analyse the anion sites as well")
        self.anions.setChecked(anions)
        form.addRow("", self.anions)
        self.second = QDoubleSpinBox()
        self.second.setDecimals(4)
        self.second.setRange(0.0, 5.0)
        self.second.setSingleStep(0.005)
        self.second.setValue(0.0)
        self.second.setSpecialValueText("none")
        self.second.setSuffix(" v.u.")
        self.second.setToolTip("A second bond threshold to compare against: "
                               "the sites whose coordination number changes "
                               "between the two come first in the report.")
        form.addRow("Second threshold", self.second)
        self.figures = QCheckBox("With figures (HTML only): the explorer of "
                                 "every site and a picture of each structure")
        form.addRow("", self.figures)
        self.format = QComboBox()
        for key in report_mod.FORMATS:
            self.format.addItem(report_mod.FORMAT_NAMES[key], key)
        form.addRow("Format", self.format)
        layout.addLayout(form)

        buttons = QDialogButtonBox()
        self.templates = QPushButton("Templates…")
        self.templates.setToolTip("Save these settings under a name, or "
                                  "load a template saved before; the "
                                  "command line takes them with --template.")
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self.templates)
        menu.aboutToShow.connect(self._fill_templates)
        self.templates.setMenu(menu)
        buttons.addButton(self.templates, QDialogButtonBox.ActionRole)
        self.run = buttons.addButton("Write report…", QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Close)
        buttons.accepted.connect(self._write)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_count()

    # -- the file list -------------------------------------------------------
    def paths(self) -> list[str]:
        return [self.files.item(i).text() for i in range(self.files.count())]

    def add_paths(self, paths) -> None:
        have = set(self.paths())
        for p in paths:
            p = str(p)
            if p and p not in have:
                self.files.addItem(p)
                have.add(p)
        self._refresh_count()

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "CIF files to report on", "",
            "CIF files (*.cif *.mcif);;All files (*)")
        self.add_paths(paths)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder of CIF files")
        if folder:
            self.add_paths([folder])

    def _remove(self) -> None:
        for item in self.files.selectedItems():
            self.files.takeItem(self.files.row(item))
        self._refresh_count()

    def _refresh_count(self) -> None:
        n = len(report_mod.collect_cifs(self.paths()))
        self.count.setText(f"{n} file{'s' if n != 1 else ''}")
        self.run.setEnabled(n > 0)

    # -- templates -------------------------------------------------------------
    def template(self) -> dict:
        second = self.second.value() if self.second.value() > 0 else None
        return report_mod.template_fields(
            self.format.currentData(), self.v_bond.value(), self.v_list.value(),
            self.anions.isChecked(), second)

    def apply_template(self, fields: dict) -> None:
        i = self.format.findData(fields.get("format"))
        if i >= 0:
            self.format.setCurrentIndex(i)
        if "v_bond" in fields:
            self.v_bond.setValue(float(fields["v_bond"]))
        if "v_list" in fields:
            self.v_list.setValue(float(fields["v_list"]))
        self.anions.setChecked(bool(fields.get("anions", False)))
        other = fields.get("v_bond_other")
        self.second.setValue(float(other) if other else 0.0)

    def _fill_templates(self) -> None:
        menu = self.templates.menu()
        menu.clear()
        save = menu.addAction("Save these settings as…")
        save.triggered.connect(self._save_template)
        names = report_mod.template_names()
        if names:
            menu.addSeparator()
            for name in names:
                act = menu.addAction(name)
                act.triggered.connect(lambda _=False, n=name: self.apply_template(
                    report_mod.load_template(n) or {}))

    def _save_template(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save the report settings",
                                        "A name for this template:")
        if ok and name.strip():
            report_mod.save_template(name.strip(), self.template())

    # -- writing -------------------------------------------------------------
    def _write(self) -> None:
        if self.v_bond.value() < self.v_list.value():
            QMessageBox.warning(
                self, "Thresholds",
                "The bond threshold is below the listing threshold: a bond "
                "must be at least a listed contact.")
            return
        fmt = self.format.currentData()
        suffix = "" if fmt == "csv" else f".{fmt}"
        filters = {"xlsx": "Excel workbook (*.xlsx)", "csv": "CSV stem (*)",
                   "json": "JSON (*.json)", "html": "Web page (*.html)",
                   "md": "Markdown (*.md)", "txt": "Text (*.txt)"}[fmt]
        out, _ = QFileDialog.getSaveFileName(
            self, "Write the report", f"facet_report{suffix}", filters)
        if not out:
            return
        files = report_mod.collect_cifs(self.paths())
        progress = QProgressDialog("Reading…", "Stop", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)

        def step(done: int, total: int, name: str) -> bool:
            progress.setValue(done)
            progress.setLabelText(f"{name}  ({done + 1} of {total})")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            second = self.second.value() if self.second.value() > 0 else None
            report = report_mod.analyse_files(
                files, params=self.params, v_bond=self.v_bond.value(),
                v_list=self.v_list.value(), anions=self.anions.isChecked(),
                progress=step, v_bond_other=second)
            progress.setValue(len(files))
            self.written = report_mod.write_report(report, out, fmt)
            if fmt == "html" and self.figures.isChecked():
                from .report_figures import add_figures
                add_figures(self.written[0], report,
                            getattr(self.parent(), "theme", None))
        except Exception as error:                    # noqa: BLE001
            progress.close()
            QMessageBox.critical(self, "Report not written", str(error))
            return
        progress.close()
        failed = [s for s in report.structures if s.error]
        lines = [f"{len(report.structures) - len(failed)} of "
                 f"{len(report.structures)} files reported."]
        lines += [f"  {p}" for p in self.written]
        if failed:
            lines.append("")
            lines.append("Not read:")
            lines += [f"  {s.name}: {s.error}" for s in failed]
        QMessageBox.information(self, "Report written", "\n".join(lines))
        self.accept()
