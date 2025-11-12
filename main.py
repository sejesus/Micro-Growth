# -*- coding: utf-8 -*-
"""
Pequeno utilitário desenvolvido para análise de curvas de crescimento microbiano da cadeira de Microbiologia Aplicada.
Permite a escolha interativa das fases de crescimento, ajusta regressões lineares simples por mínimos quadrados
(caso particular da aplicação de Monod), e compara estatisticamente com modelos Gompertz, Logístico e Richards.

«If it is a terrifying thought that life is at the mercy of the multiplication of these minute bodies [microbes],
it is a consoling hope that Science will not always remain powerless before such enemies…»
— Louis Pasteur

Autor: Sílvio do Ó, 2025
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QCheckBox, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QSpinBox, QDoubleSpinBox
)

# módulos do projeto
import phases
from phases import (segment_phases, pick_points_within_exp, mu_instantaneo)
from io_data import (
    load_growth_data, load_mu_ref_table, resolve_meta_for_column, resolve_mu_thresholds_for_column
)
from export import (
    save_overview, save_phase_plot, save_linearization,
    export_tables, write_metadata, save_models_and_mu_combo
)
from fits import fit_all_models, SCIPY_OK

# ========================= Defaults (editáveis em Settings) =========================
SPECIES = "Escherichia coli"
STRAIN  = ""
MEDIUM  = "LB"
TEMP_C_GLOBAL = 37.0

MU_REF_CSV_PATH = ""        # caminho para CSV opcional μ_ref
MU_REF_MANUAL = None        # μ_ref manual (float) ou None

MU_MIN_FACTOR  = 0.40
MU_EXIT_FACTOR = 0.20
MU_STAT_FACTOR = 0.20
MU_MIN_DEFAULT = 0.20
MU_STAT_DEFAULT = 0.05

PLOT_FASES_IN_LOG = True
NORMALIZE_FOR_FITS = True

COLUMN_TO_MEDIUM: dict[str, str] = {}
COLUMN_TO_TEMP: dict[str, float] = {}

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "outputs" / datetime.now().strftime("%Y%m%d_%H%M")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================= Settings dialog ================================
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings — Parâmetros de Análise")
        layout = QFormLayout(self)

        self.ed_species = QLineEdit(SPECIES)
        self.ed_strain  = QLineEdit(STRAIN)
        self.ed_medium  = QLineEdit(MEDIUM)
        self.sp_temp    = QDoubleSpinBox(); self.sp_temp.setRange(0, 100); self.sp_temp.setValue(TEMP_C_GLOBAL); self.sp_temp.setSuffix(" °C")

        self.ed_mu_csv    = QLineEdit(str(MU_REF_CSV_PATH) if MU_REF_CSV_PATH else "")
        self.ed_mu_manual = QLineEdit("" if MU_REF_MANUAL is None else str(MU_REF_MANUAL))

        self.sp_mu_min_factor  = QDoubleSpinBox(); self.sp_mu_min_factor.setRange(0, 1); self.sp_mu_min_factor.setDecimals(3); self.sp_mu_min_factor.setSingleStep(0.01); self.sp_mu_min_factor.setValue(MU_MIN_FACTOR)
        self.sp_mu_exit_factor = QDoubleSpinBox(); self.sp_mu_exit_factor.setRange(0, 1); self.sp_mu_exit_factor.setDecimals(3); self.sp_mu_exit_factor.setSingleStep(0.01); self.sp_mu_exit_factor.setValue(MU_EXIT_FACTOR)
        self.sp_mu_stat_factor = QDoubleSpinBox(); self.sp_mu_stat_factor.setRange(0, 1); self.sp_mu_stat_factor.setDecimals(3); self.sp_mu_stat_factor.setSingleStep(0.01); self.sp_mu_stat_factor.setValue(MU_STAT_FACTOR)
        self.sp_mu_min_default = QDoubleSpinBox(); self.sp_mu_min_default.setRange(0, 5); self.sp_mu_min_default.setDecimals(3); self.sp_mu_min_default.setValue(MU_MIN_DEFAULT)
        self.sp_mu_stat_default= QDoubleSpinBox(); self.sp_mu_stat_default.setRange(0, 5); self.sp_mu_stat_default.setDecimals(3); self.sp_mu_stat_default.setValue(MU_STAT_DEFAULT)

        self.sp_deriv_win = QSpinBox(); self.sp_deriv_win.setRange(1, 25); self.sp_deriv_win.setValue(phases.DERIV_SMOOTH_WINDOW)
        self.sp_r2_target = QDoubleSpinBox(); self.sp_r2_target.setRange(0.0, 1.0); self.sp_r2_target.setDecimals(3); self.sp_r2_target.setValue(phases.R2_TARGET)
        self.sp_max_remove= QSpinBox(); self.sp_max_remove.setRange(0, 10); self.sp_max_remove.setValue(phases.MAX_REMOVE)
        self.sp_min_len   = QSpinBox(); self.sp_min_len.setRange(2, 200); self.sp_min_len.setValue(phases.MIN_LEN)
        self.sp_raw_slope = QDoubleSpinBox(); self.sp_raw_slope.setRange(0, 2); self.sp_raw_slope.setDecimals(3); self.sp_raw_slope.setValue(phases.RAW_SLOPE_MAX)
        self.sp_tail_relax= QDoubleSpinBox(); self.sp_tail_relax.setRange(0, 2); self.sp_tail_relax.setDecimals(2); self.sp_tail_relax.setValue(phases.TAIL_RELAX)

        self.chk_models = QCheckBox("Executar modelos empíricos (SciPy)"); self.chk_models.setChecked(True)

        layout.addRow("Espécie:", self.ed_species)
        layout.addRow("Estirpe:", self.ed_strain)
        layout.addRow("Meio (default):", self.ed_medium)
        layout.addRow("Temperatura global:", self.sp_temp)
        layout.addRow("μ_ref CSV (opcional):", self.ed_mu_csv)
        layout.addRow("μ_ref manual (h⁻¹, opc.):", self.ed_mu_manual)
        layout.addRow("MU_MIN_FACTOR:", self.sp_mu_min_factor)
        layout.addRow("MU_EXIT_FACTOR:", self.sp_mu_exit_factor)
        layout.addRow("MU_STAT_FACTOR:", self.sp_mu_stat_factor)
        layout.addRow("MU_MIN_DEFAULT (h⁻¹):", self.sp_mu_min_default)
        layout.addRow("MU_STAT_DEFAULT (h⁻¹):", self.sp_mu_stat_default)
        layout.addRow("Janela suavização derivada (pontos):", self.sp_deriv_win)
        layout.addRow("R² alvo (linearização):", self.sp_r2_target)
        layout.addRow("MAX_REMOVE (pontas):", self.sp_max_remove)
        layout.addRow("MIN_LEN (pontos exp):", self.sp_min_len)
        layout.addRow("RAW_SLOPE_MAX (Abs/h):", self.sp_raw_slope)
        layout.addRow("TAIL_RELAX:", self.sp_tail_relax)
        layout.addRow(self.chk_models)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def values(self):
        return dict(
            species=self.ed_species.text().strip(),
            strain=self.ed_strain.text().strip(),
            medium=self.ed_medium.text().strip(),
            temp=float(self.sp_temp.value()),
            mu_csv=self.ed_mu_csv.text().strip(),
            mu_manual=(None if self.ed_mu_manual.text().strip()=="" else float(self.ed_mu_manual.text().strip())),
            mu_min_factor=float(self.sp_mu_min_factor.value()),
            mu_exit_factor=float(self.sp_mu_exit_factor.value()),
            mu_stat_factor=float(self.sp_mu_stat_factor.value()),
            mu_min_default=float(self.sp_mu_min_default.value()),
            mu_stat_default=float(self.sp_mu_stat_default.value()),
            deriv_window=int(self.sp_deriv_win.value()),
            r2_target=float(self.sp_r2_target.value()),
            max_remove=int(self.sp_max_remove.value()),
            min_len=int(self.sp_min_len.value()),
            raw_slope=float(self.sp_raw_slope.value()),
            tail_relax=float(self.sp_tail_relax.value()),
            run_models=bool(self.chk_models.isChecked()),
        )

# ================================== Main Window ===================================
class PhaseSelectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Micro Growth")
        self.resize(1040, 680)

        # dados
        self.th = None
        self.t_min = None
        self.data_dict = None
        self.meta_medium_row = None
        self.meta_temp_row = None
        self.mu_ref_csv_df = None

        # estado
        self.columns: list[str] = []
        self.idx = 0
        self.phase_order = ["lag", "exp", "stat", "decl"]
        self.colors = {"lag": "#f6c57d", "exp": "#8fc8a9", "stat": "#efb4b4", "decl": "#d58b8b"}
        self.current_phase = "exp"
        self.gui_phases: dict[str, dict[str, tuple[float, float] | None]] = {}
        self.spans_artists = []
        self.selector = None
        self.snap_artists = []  # elementos visuais do snap
        self.meta_info_per_col: dict[str, dict] = {}
        self.flag_run_models = True

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        topbar = QHBoxLayout()
        self.btn_open = QPushButton("Abrir CSV…")
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_save = QPushButton("Exportar tudo")
        self.btn_settings = QPushButton("Settings…")
        self.lbl_phase = QLabel(f"Fase ativa: {self.current_phase}")
        self.btn_lag  = QPushButton("Lag")
        self.btn_exp  = QPushButton("Exp")
        self.btn_stat = QPushButton("Est.")
        self.btn_decl = QPushButton("Decl.")

        for b in [self.btn_open, self.btn_prev, self.btn_next, self.btn_lag, self.btn_exp, self.btn_stat, self.btn_decl, self.btn_save, self.btn_settings]:
            topbar.addWidget(b)
        topbar.addStretch(1)
        topbar.addWidget(self.lbl_phase)
        layout.addLayout(topbar)

        self.fig = Figure(figsize=(8.2, 4.8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # status/summary das fases (monoespaçado e selecionável)
        self.lbl_coords = QLabel("")
        self.lbl_coords.setStyleSheet("font-family: Consolas, 'Fira Code', 'Courier New', monospace;")
        self.lbl_coords.setTextInteractionFlags(self.lbl_coords.textInteractionFlags() | 0x1)  # Texto selecionável
        layout.addWidget(self.lbl_coords)

        self.ax = self.fig.add_subplot(111)

        # ligações
        self.btn_open.clicked.connect(self.on_open)
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next.clicked.connect(self.on_next)
        self.btn_save.clicked.connect(self.on_export)
        self.btn_settings.clicked.connect(self.on_settings)
        self.btn_lag.clicked.connect(lambda: self.set_phase("lag"))
        self.btn_exp.clicked.connect(lambda: self.set_phase("exp"))
        self.btn_stat.clicked.connect(lambda: self.set_phase("stat"))
        self.btn_decl.clicked.connect(lambda: self.set_phase("decl"))
        self.canvas.mpl_connect("key_press_event", self.on_key)

    # ---------------------------- helpers de snap/visual ----------------------------
    def _clear_snap_guides(self):
        if not self.snap_artists:
            return
        for art in self.snap_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.snap_artists = []

    def _draw_snap_guides(self, a: float, b: float, transient: bool = False):
        self._clear_snap_guides()
        if self.th is None or not self.columns:
            return
        col = self.columns[self.idx]
        y = self.data_dict[col]
        y_plot = np.log(np.clip(y, 1e-12, None)) if PLOT_FASES_IN_LOG else y
        ia = int(np.argmin(np.abs(self.th - a)))
        ib = int(np.argmin(np.abs(self.th - b)))
        alpha = 0.45 if transient else 0.85
        lw = 1.2 if transient else 1.8
        self.snap_artists.append(self.ax.axvline(self.th[ia], color="#222", linewidth=lw, alpha=alpha))
        self.snap_artists.append(self.ax.axvline(self.th[ib], color="#222", linewidth=lw, alpha=alpha))
        ms = 7 if transient else 9
        mew = 1.4 if transient else 2.0
        self.snap_artists += self.ax.plot(self.th[ia], y_plot[ia], marker="o", markersize=ms, markeredgewidth=mew, markerfacecolor="none", markeredgecolor="#000")
        self.snap_artists += self.ax.plot(self.th[ib], y_plot[ib], marker="o", markersize=ms, markeredgewidth=mew, markerfacecolor="none", markeredgecolor="#000")
        self.canvas.draw_idle()

    def _on_span_move(self, xmin: float, xmax: float):
        if self.th is None or not self.columns:
            return
        arr = self.th
        a = arr[np.argmin(np.abs(arr - xmin))]
        b = arr[np.argmin(np.abs(arr - xmax))]
        if a > b:
            a, b = b, a
        self._draw_snap_guides(a, b, transient=True)
        self._update_phase_status(preview_override=(self.current_phase, (a, b)))

    # -------------------------- resumo dinâmico das fases ---------------------------
    def _fmt_iv(self, iv):
        """Devolve (string_intervalo, string_duracao) para um intervalo em horas; 'n/d' se None/degenerado."""
        if not iv:
            return "n/d", "n/d"
        a, b = float(iv[0]), float(iv[1])
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return "n/d", "n/d"
        return f"{a:.4f}–{b:.4f}", f"{(b - a):.4f} h"

    def _phase_summary_text(self, phases_dict):
        """Constrói o texto compacto com intervalos e durações de todas as fases."""
        s_lag, d_lag   = self._fmt_iv(phases_dict.get("lag"))
        s_exp, d_exp   = self._fmt_iv(phases_dict.get("exp"))
        s_stat, d_stat = self._fmt_iv(phases_dict.get("stat"))
        s_dec, d_dec   = self._fmt_iv(phases_dict.get("decl"))
        return (
            f"Lag: {s_lag} ({d_lag})  |  "
            f"Exp: {s_exp} ({d_exp})  |  "
            f"Est: {s_stat} ({d_stat})  |  "
            f"Dec: {s_dec} ({d_dec})"
        )

    def _update_phase_status(self, preview_override=None):
        """
        Atualiza a QLabel com o resumo das fases.
        - preview_override: tuplo (phase_name, (a,b)) para pré-visualizar enquanto se arrasta.
        """
        if not self.columns:
            self.lbl_coords.setText("—")
            return
        col = self.columns[self.idx]
        phases_dict = {
            "lag":  self.gui_phases[col].get("lag"),
            "exp":  self.gui_phases[col].get("exp"),
            "stat": self.gui_phases[col].get("stat"),
            "decl": self.gui_phases[col].get("decl"),
        }
        if preview_override is not None:
            p, iv = preview_override
            phases_dict[p] = iv
        self.lbl_coords.setText(self._phase_summary_text(phases_dict))

    # ---------------------------------- UI logic -----------------------------------
    def set_phase(self, p: str):
        self.current_phase = p
        self.lbl_phase.setText(f"Fase ativa: {self.current_phase}")

    def on_key(self, event):
        if event.key in ("1", "2", "3", "4"):
            p = self.phase_order[int(event.key) - 1]
            self.set_phase(p)
        elif event.key == "left":
            self.on_prev()
        elif event.key == "right":
            self.on_next()

    def install_selector(self):
        if self.selector:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass

        def on_select(xmin, xmax):
            arr = self.th
            a = arr[np.argmin(np.abs(arr - xmin))]
            b = arr[np.argmin(np.abs(arr - xmax))]
            if a > b:
                a, b = b, a
            self.replace_or_add_span(self.current_phase, a, b)
            self._draw_snap_guides(a, b, transient=False)

        try:
            minspan = ((self.th[1] - self.th[0]) * 0.05 if len(self.th) > 1 else 1e-3)
        except Exception:
            minspan = 1e-3

        try:
            self.selector = SpanSelector(
                self.ax, on_select, "horizontal",
                interactive=True, drag_from_anywhere=True,
                useblit=False, button=1, minspan=minspan,
                onmove_callback=lambda xmin, xmax: self._on_span_move(xmin, xmax),
                props=dict(alpha=0.28, facecolor="#666", edgecolor="#111", linewidth=1.2)
            )
        except TypeError:
            # compat com versões antigas do matplotlib
            self.selector = SpanSelector(
                self.ax, on_select, "horizontal",
                useblit=False, minspan=minspan,
                onmove_callback=lambda xmin, xmax: self._on_span_move(xmin, xmax),
                rectprops=dict(alpha=0.28, facecolor="#666", edgecolor="#111", linewidth=1.2)
            )
        try:
            self.selector.set_active(True)
        except Exception:
            pass
        self.canvas.draw_idle()

    def replace_or_add_span(self, phase_name: str, a: float, b: float):
        if self.th is None:
            return
        col = self.columns[self.idx]
        self.gui_phases[col][phase_name] = (a, b)

        kept = []
        for p, art in self.spans_artists:
            if p == phase_name:
                try:
                    art.remove()
                except Exception:
                    pass
            else:
                kept.append((p, art))
        self.spans_artists = kept
        r = self.ax.axvspan(a, b, color=self.colors[phase_name], alpha=0.25, lw=0)
        self._update_phase_status()
        self.spans_artists.append((phase_name, r))
        self.canvas.draw_idle()

    def draw_column(self, i: int):
        self.ax.clear()
        self.spans_artists = []
        self._clear_snap_guides()
        if not self.columns:
            self.canvas.draw_idle()
            return
        col = self.columns[i]
        y = self.data_dict[col]
        y_plot = np.log(np.clip(y, 1e-12, None)) if PLOT_FASES_IN_LOG else y
        self.ax.plot(self.th, y_plot, "o-", lw=2)
        medium_display = self.meta_info_per_col[col]["medium_display"]
        temp_c = self.meta_info_per_col[col]["temp_c"]
        y_label = "ln DO (Abs$_{540}$)" if PLOT_FASES_IN_LOG else "DO (Abs$_{540}$)"
        self.ax.set_xlabel("t (h)")
        self.ax.set_ylabel(y_label)
        self.ax.set_title(f"{col} — {medium_display}, {temp_c:.0f} °C — arrasta intervalos (1:Lag,2:Exp,3:Est.,4:Decl.)")
        self.ax.grid(True, ls="--", alpha=0.3)
        for p in self.phase_order:
            iv = self.gui_phases[col].get(p)
            if iv:
                a, b = iv
                r = self.ax.axvspan(a, b, color=self.colors[p], alpha=0.25, lw=0)
                self.spans_artists.append((p, r))
        self.canvas.draw_idle()
        self._update_phase_status()
        self.install_selector()

    # -------------------------------- data / actions --------------------------------
    def on_open(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Abrir CSV de crescimento", str(BASE_DIR), "CSV (*.csv)")
        if not fpath:
            return
        try:
            th, t_min, data_dict, meta_medium_row, meta_temp_row = load_growth_data(fpath)
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Não foi possível ler o CSV:\n{e}")
            return

        self.data_path = Path(fpath)
        self.th, self.t_min, self.data_dict = th, t_min, data_dict
        self.meta_medium_row, self.meta_temp_row = meta_medium_row, meta_temp_row
        self.columns = list(data_dict.keys())
        self.idx = 0
        self.gui_phases = {c: {"lag": None, "exp": None, "stat": None, "decl": None} for c in self.columns}

        self.mu_ref_csv_df = load_mu_ref_table(MU_REF_CSV_PATH) if MU_REF_CSV_PATH else None

        # resolver meta (display + alias) por coluna
        from config.medium_aliases import resolve_medium_alias, canonical_medium_key  # noqa: F401
        self.meta_info_per_col = {}
        for col in self.columns:
            medium_display, medium_alias, temp_c = resolve_meta_for_column(
                col, MEDIUM, TEMP_C_GLOBAL, COLUMN_TO_MEDIUM, COLUMN_TO_TEMP,
                self.meta_medium_row, self.meta_temp_row
            )
            self.meta_info_per_col[col] = {"medium_display": medium_display, "medium_alias": medium_alias, "temp_c": float(temp_c)}

            # thresholds a partir do alias canónico
            mu_ref, MU_MIN, MU_EXIT, MU_STAT, SOURCE, _ = resolve_mu_thresholds_for_column(
                col, SPECIES, STRAIN, MEDIUM, COLUMN_TO_MEDIUM, MU_REF_MANUAL, self.mu_ref_csv_df,
                MU_MIN_FACTOR, MU_EXIT_FACTOR, MU_STAT_FACTOR, MU_MIN_DEFAULT, MU_STAT_DEFAULT,
                medium_alias_override=medium_alias
            )

            phases_auto, _ = segment_phases(self.th, self.data_dict[col], MU_MIN, MU_EXIT, MU_STAT)
            self.gui_phases[col]["lag"]  = phases_auto["lag"]
            self.gui_phases[col]["exp"]  = phases_auto["exp"]
            self.gui_phases[col]["stat"] = phases_auto["stat"]
            self.gui_phases[col]["decl"] = phases_auto["decl"]

        self.draw_column(self.idx)

    def on_prev(self):
        if not self.columns:
            return
        self.idx = (self.idx - 1) % len(self.columns)
        self.draw_column(self.idx)

    def on_next(self):
        if not self.columns:
            return
        self.idx = (self.idx + 1) % len(self.columns)
        self.draw_column(self.idx)

    def on_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            vals = dlg.values()
            global SPECIES, STRAIN, MEDIUM, TEMP_C_GLOBAL, MU_REF_CSV_PATH, MU_REF_MANUAL
            global MU_MIN_FACTOR, MU_EXIT_FACTOR, MU_STAT_FACTOR, MU_MIN_DEFAULT, MU_STAT_DEFAULT
            SPECIES = vals["species"]; STRAIN = vals["strain"]; MEDIUM = vals["medium"]; TEMP_C_GLOBAL = vals["temp"]
            MU_REF_CSV_PATH = vals["mu_csv"]; MU_REF_MANUAL = vals["mu_manual"]
            MU_MIN_FACTOR = vals["mu_min_factor"]; MU_EXIT_FACTOR = vals["mu_exit_factor"]; MU_STAT_FACTOR = vals["mu_stat_factor"]
            MU_MIN_DEFAULT = vals["mu_min_default"]; MU_STAT_DEFAULT = vals["mu_stat_default"]
            self.flag_run_models = vals["run_models"]

            # parâmetros do phases
            phases.DERIV_SMOOTH_WINDOW = vals["deriv_window"]
            phases.R2_TARGET = vals["r2_target"]
            phases.MAX_REMOVE = vals["max_remove"]
            phases.MIN_LEN = vals["min_len"]
            phases.RAW_SLOPE_MAX = vals["raw_slope"]
            phases.TAIL_RELAX = vals["tail_relax"]

            self.mu_ref_csv_df = load_mu_ref_table(MU_REF_CSV_PATH) if MU_REF_CSV_PATH else None

            if self.th is not None:
                for col in self.columns:
                    medium_alias = self.meta_info_per_col[col]["medium_alias"]
                    mu_ref, MU_MIN, MU_EXIT, MU_STAT, SOURCE, _ = resolve_mu_thresholds_for_column(
                        col, SPECIES, STRAIN, MEDIUM, COLUMN_TO_MEDIUM, MU_REF_MANUAL, self.mu_ref_csv_df,
                        MU_MIN_FACTOR, MU_EXIT_FACTOR, MU_STAT_FACTOR, MU_MIN_DEFAULT, MU_STAT_DEFAULT,
                        medium_alias_override=medium_alias
                    )
                    phases_auto, _ = segment_phases(self.th, self.data_dict[col], MU_MIN, MU_EXIT, MU_STAT)
                    self.gui_phases[col]["lag"]  = phases_auto["lag"]
                    self.gui_phases[col]["exp"]  = phases_auto["exp"]
                    self.gui_phases[col]["stat"] = phases_auto["stat"]
                    self.gui_phases[col]["decl"] = phases_auto["decl"]
                self.draw_column(self.idx)

    def on_export(self):
        if self.th is None or not self.columns:
            QMessageBox.information(self, "Info", "Abre primeiro um CSV com dados.")
            return

        save_overview(self.th, self.data_dict, self.meta_info_per_col, OUT_DIR, SPECIES, STRAIN)

        rows_reg, rows_models, rows_summary = [], [], []
        ts = OUT_DIR.name

        for i, col in enumerate(self.columns):
            y = self.data_dict[col]
            medium_display = self.meta_info_per_col[col]["medium_display"]
            medium_alias   = self.meta_info_per_col[col]["medium_alias"]
            temp_c         = self.meta_info_per_col[col]["temp_c"]

            mu_ref, MU_MIN, MU_EXIT, MU_STAT, SOURCE, medium_for_col = resolve_mu_thresholds_for_column(
                col, SPECIES, STRAIN, MEDIUM, COLUMN_TO_MEDIUM, MU_REF_MANUAL, self.mu_ref_csv_df,
                MU_MIN_FACTOR, MU_EXIT_FACTOR, MU_STAT_FACTOR, MU_MIN_DEFAULT, MU_STAT_DEFAULT,
                medium_alias_override=medium_alias
            )

            phases_dict = {
                "lag": self.gui_phases[col]["lag"],
                "exp": self.gui_phases[col]["exp"],
                "stat": self.gui_phases[col]["stat"],
                "decl": self.gui_phases[col]["decl"],
                "mu_series": mu_instantaneo(self.th, y)
            }
            if not phases_dict["exp"]:
                phases_auto, _ = segment_phases(self.th, y, MU_MIN, MU_EXIT, MU_STAT)
                phases_dict["exp"] = phases_auto["exp"]

            def near_idx(t): 
                return int(np.argmin(np.abs(self.th - float(t))))
            i0e = near_idx(phases_dict["exp"][0])
            i1e = near_idx(phases_dict["exp"][1])
            if i1e < i0e:
                i0e, i1e = i1e, i0e

            j0, j1, mu, intercept, r2, trim_s, trim_e = pick_points_within_exp(self.th, y, i0e, i1e)
            td_h  = np.log(2) / mu if np.isfinite(mu) else np.nan
            td_min = td_h * 60 if np.isfinite(td_h) else np.nan

            # Fases (com marcação de removidos)
            save_phase_plot(
                self.th, y, phases_dict, j0, j1, i0e, i1e,
                OUT_DIR / f"{col.replace(' ', '_')}_Fases.png", i,
                SPECIES, STRAIN, col, medium_display, temp_c, plot_log=True
            )

            # Linearização com tabela
            save_linearization(
                self.th, y, j0, j1, mu, intercept, r2,
                OUT_DIR / f"{col.replace(' ', '_')}_Linearizacao.png",
                SPECIES, STRAIN, col, medium_display, temp_c
            )

            # Modelos empíricos + figura combinada (μ(t) + modelos + tabela Δrel)
            fits = []
            if self.flag_run_models and SCIPY_OK:
                fits = fit_all_models(self.th, y, mu_guess=mu, lam_guess=self.th[j0], normalize=True)
                if fits:
                    for name, _yfit, r2m, params in fits:
                        rec = {
                            "Coluna": col, "Modelo": name, "R²_modelo": float(r2m),
                            "A": params["A"], "mu_max (h⁻¹)": params["mu_max (h^-1)"], "λ (h)": params["λ (h)"],
                            "Fonte μ_ref": SOURCE, "Meio_ref": medium_for_col
                        }
                        if name == "Richards" and "v (forma)" in params:
                            rec["v (forma)"] = params["v (forma)"]
                        rows_models.append(rec)

            save_models_and_mu_combo(
                self.th, y, phases_dict["mu_series"], fits, r2,
                OUT_DIR / f"{col.replace(' ', '_')}_Modelos_e_Mu.png",
                SPECIES, STRAIN, col, medium_display, temp_c
            )

            # --- helpers para extrair intervalos ---
            def _iv_tuple(iv):
                if iv is None: 
                    return (np.nan, np.nan, np.nan, "n/d")
                a, b = float(iv[0]), float(iv[1])
                if b <= a: 
                    return (a, b, 0.0, "n/d")
                return (a, b, b - a, f"{a:.3f}–{b:.3f}")
            
            lag_a, lag_b, lag_dur, lag_str   = _iv_tuple(phases_dict["lag"])
            exp_a, exp_b, exp_dur, exp_str   = _iv_tuple(phases_dict["exp"])
            stat_a, stat_b, stat_dur, stat_str = _iv_tuple(phases_dict["stat"])
            dec_a,  dec_b,  dec_dur,  dec_str  = _iv_tuple(phases_dict["decl"])
            
            rows_reg.append({
                "Coluna": col, 
                "Meio_ref": medium_for_col, 
                "Fonte μ_ref": SOURCE,
                # Regressão
                "R²": r2, 
                "μ (h⁻¹)": mu, 
                "t_d (h)": (np.log(2)/mu) if np.isfinite(mu) else np.nan,
                "t_d (min)": (np.log(2)/mu*60) if np.isfinite(mu) else np.nan,
                # Fase Exponencial (string e numérico)
                "Exp (h)": exp_str,
                "Exp_start (h)": exp_a, "Exp_end (h)": exp_b, "Exp_dur (h)": exp_dur,
                "Removidos início": trim_s, "Removidos fim": trim_e,
                # Fase Lag
                "Lag (h)": lag_str,
                "Lag_start (h)": lag_a, "Lag_end (h)": lag_b, "Lag_dur (h)": lag_dur,
                # Estacionária
                "Est (h)": stat_str,
                "Est_start (h)": stat_a, "Est_end (h)": stat_b, "Est_dur (h)": stat_dur,
                # Declínio
                "Dec (h)": dec_str,
                "Dec_start (h)": dec_a, "Dec_end (h)": dec_b, "Dec_dur (h)": dec_dur,
            })

            row_sum = {
                "Coluna": col, "Meio_ref": medium_for_col,
                "μ_ref (fonte)": f"{mu_ref if mu_ref is not None else np.nan} ({SOURCE})",
                "μ regressão (h⁻¹)": mu, "R² regressão": r2, "t_d (min)": td_min
            }
            rows_summary.append(row_sum)

        excel_path = export_tables(rows_reg, rows_models, rows_summary, OUT_DIR, ts)
        write_metadata(
            OUT_DIR, ts, SPECIES, STRAIN, MEDIUM, TEMP_C_GLOBAL,
            getattr(self, "data_path", Path("<escolhido no diálogo>")),
            Path(MU_REF_CSV_PATH) if MU_REF_CSV_PATH else None
        )

        QMessageBox.information(
            self, "Concluído",
            f"Exportação terminada em:\n{OUT_DIR}\n\n"
            f"- {Path(excel_path).name}\n- PNGs por coluna (Fases, Linearização, Modelos+μ)\n"
            f"- Janelas_Fases.csv\n- run_metadata.txt"
        )

def main():
    app = QApplication(sys.argv)
    win = PhaseSelectorWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
