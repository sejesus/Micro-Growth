# -*- coding: utf-8 -*-
"""
Funções de exportação gráfica e tabular.
Gera os gráficos das fases, linearizações e combinações μ(t)+modelos,
bem como ficheiros de saída em Excel e CSV com resultados e metadados.
Inclui também visualização dos pontos removidos e tabelas comparativas de R² (com Δrel face à regressão linear).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def make_title(species: str, strain: str, col: str,
               medium_display: str | None = None,
               temp_c: float | None = None,
               suffix: str = ""):
    """
    Cria um título padronizado para figuras.
    Se medium_display ou temp_c não forem fornecidos, omite-os do título.
    """
    latex_species = species.replace(" ", r"\ ")
    base = rf"$\it{{{latex_species}}}$" + (f" {strain}" if strain else "")
    parts = [base]

    # coluna (ex: "Exp1") ou tipo de gráfico
    if col:
        parts.append(col)

    # meio e temperatura, só se existirem
    if medium_display:
        parts.append(medium_display)
    if temp_c is not None and temp_c > 0:
        parts.append(f"{temp_c:.0f} °C")

    title = " — ".join(parts)
    if suffix:
        title += f" — {suffix}"
    return title


def save_overview(th, data_dict, meta_info_per_col, outdir: Path, species: str, strain: str):
    import numpy as np
    import matplotlib.pyplot as plt

    # evita que a figura apareça no ecrã em ambientes interativos (Spyder, etc.)
    was_interactive = plt.isinteractive()
    try:
        if was_interactive:
            plt.ioff()

        fig, ax = plt.subplots(figsize=(8, 5.6))
        palette = plt.get_cmap("tab10")

        for i, (col, vals) in enumerate(data_dict.items()):
            md = meta_info_per_col[col]["medium_display"]
            tc = meta_info_per_col[col]["temp_c"]
            y_plot = np.log(np.clip(np.asarray(vals, float), 1e-12, None))
            ax.plot(th, y_plot, marker="o", label=f"{col} — {md} ({tc:.0f} °C)", color=palette(i))

        ax.set_xlabel("t (h)")
        ax.set_ylabel(r"ln DO (Abs$_{540}$)")

        # make_title(species, strain, col, medium_display, temp_c, suffix)
        ax.set_title(make_title(species, strain, "Curvas de crescimento (ln)"))

        # legenda fora do eixo, em baixo (para não tapar as curvas)
        leg = ax.legend(
            ncol=2,                        # ou 3, se tiveres muitas séries
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            borderaxespad=0.0,
        )
    
        fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.98])
        fig.savefig(outdir / "Geral_Crescimento.png", bbox_extra_artists=(leg,), bbox_inches="tight")
        plt.close(fig)

        #fig.savefig(outdir / "Geral_Crescimento.png")
    finally:
        plt.close('all')
        if was_interactive:
            plt.ion()


def save_phase_plot(
    th, y, phases, j0, j1, i0e, i1e, out_path: Path, color_idx: int,
    species: str, strain: str, col: str, medium_display: str, temp_c: float, plot_log=True
):
    """
    Gráfico das fases + marcação dos pontos removidos + rodapé com intervalos e durações.
    """
    palette = plt.get_cmap("tab10")

    # figura com rodapé para o resumo dos intervalos
    fig = plt.figure(figsize=(6.8, 5.2))
    gs = GridSpec(2, 1, height_ratios=[4.0, 0.9], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_footer = fig.add_subplot(gs[1])
    ax_footer.axis("off")

    y = np.asarray(y, float)
    y_plot = np.log(np.clip(y, 1e-12, None)) if plot_log else y
    y_removed = (lambda idx: np.log(max(y[idx], 1e-12))) if plot_log else (lambda idx: y[idx])

    # série completa + janela exp usada
    ax.plot(th, y_plot, "o-", color=palette(color_idx), alpha=0.55, label="Original")
    ax.plot(th[j0:j1+1], y_plot[j0:j1+1], "o-", color=palette(color_idx), label="Exponencial (usada)")

    # spans
    if phases["lag"]:  ax.axvspan(*phases["lag"],  alpha=0.15, color="#FFA500", label="Lag")
    if phases["exp"]:  ax.axvspan(*phases["exp"],  alpha=0.18, color="#2E8B57", label="Exponencial")
    if phases["stat"]: ax.axvspan(*phases["stat"], alpha=0.12, color="#B22222", label="Estacionária")
    if phases["decl"]: ax.axvspan(*phases["decl"], alpha=0.12, color="#8B0000", label="Declínio")

    # pontos removidos (pontas)
    removed_idxs = list(range(i0e, max(j0, i0e))) + list(range(min(j1+1, i1e+1), i1e+1))
    for idx in removed_idxs:
        yy = y_removed(idx)
        ax.plot(th[idx], yy, "x", color="#B22222", mew=1.8, ms=8, zorder=5)
        ax.text(th[idx], yy + 0.02, "†", color="#B22222", ha="center", va="bottom", fontsize=11)

    # eixos/título
    ax.set_xlabel("t (h)")
    ax.set_ylabel("ln DO (Abs$_{540}$)" if plot_log else "DO (Abs$_{540}$)")
    ax.set_title(make_title(species, strain, col, medium_display, temp_c, "Fases"))

    # legenda sem duplicados
    H, L = [], []
    hh, ll = ax.get_legend_handles_labels()
    seen = set()
    for h, l in zip(hh, ll):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    ax.legend(H, L, frameon=False, loc="best")

    # ---------- rodapé com intervalos ----------
    def _fmt_iv(iv):
        if not iv:
            return "n/d", "n/d"
        a, b = float(iv[0]), float(iv[1])
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return "n/d", "n/d"
        return f"{a:.4f}–{b:.4f}", f"{(b - a):.4f} h"

    s_lag, d_lag   = _fmt_iv(phases.get("lag"))
    s_exp, d_exp   = _fmt_iv(phases.get("exp"))
    s_stat, d_stat = _fmt_iv(phases.get("stat"))
    s_dec, d_dec   = _fmt_iv(phases.get("decl"))

    footer_text = (
        f"Lag: {s_lag} ({d_lag})   |   "
        f"Exp: {s_exp} ({d_exp})    \n  "
        f"Est: {s_stat} ({d_stat})   |   "
        f"Dec: {s_dec} ({d_dec})"
    )

    ax_footer.text(
        0.01, 0.5, footer_text,
        ha="left", va="center",
        fontsize=11.5,
        family="Consolas, Fira Code, Courier New, monospace",
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_linearization(th, y, j0, j1, mu, intercept, r2, out_path: Path,
                       species: str, strain: str, col: str, medium_display: str, temp_c: float):
    fig = plt.figure(figsize=(6.8, 5.2))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.3], hspace=0.32)
    ax = fig.add_subplot(gs[0]); axT = fig.add_subplot(gs[1]); axT.axis("off")

    x_exp = np.asarray(th[j0:j1+1], float); y_exp = np.asarray(y[j0:j1+1], float)
    ax.plot(x_exp, np.log(y_exp), "o", color="black")
    xx = np.linspace(float(x_exp.min()), float(x_exp.max()), 100)
    ax.plot(xx, mu*xx+intercept, "--", color="black", linewidth=1.6)
    ax.set_xlabel("t (h)"); ax.set_ylabel("ln DO (Abs$_{540}$)")
    ax.set_title(make_title(species, strain, col, medium_display, temp_c, "Linearização (ln)"))

    tdh = (np.log(2)/mu) if np.isfinite(mu) else np.nan
    tbl = [
        ["R²", f"{r2:.4f}"],
        ["μ (h⁻¹)", f"{mu:.4f}"],
        ["t_d (h)", f"{tdh:.4f}" if np.isfinite(tdh) else "—"],
        ["t_d (min)", f"{(tdh*60):.1f}" if np.isfinite(tdh) else "—"],
        ["Exp (h)", f"{th[j0]:.3f}–{th[j1]:.3f}"],
    ]
    table = axT.table(cellText=tbl, colLabels=["Parâmetro","Valor"], cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(11.5); table.scale(1.30,1.30)

    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def save_models_and_mu_combo(th, y, mu_series, fits, r2_linear, out_path: Path,
                             species: str, strain: str, col: str, medium_display: str, temp_c: float):
    """PNG combinado: topo μ(t), meio dados+modelos, base tabela com R² e Δrel face ao R² linear."""
    fig = plt.figure(figsize=(7.4, 7.8))
    gs = GridSpec(3, 1, height_ratios=[2.2, 3.0, 1.6], hspace=0.35)
    ax_mu = fig.add_subplot(gs[0])
    ax_md = fig.add_subplot(gs[1])
    ax_tb = fig.add_subplot(gs[2]); ax_tb.axis("off")

    t = np.asarray(th, float); y = np.asarray(y, float)
    # μ(t)
    ax_mu.plot(t, mu_series, "o-")
    ax_mu.set_xlabel("t (h)"); ax_mu.set_ylabel("μ inst. (h⁻¹)")
    ax_mu.set_title(make_title(species, strain, col, medium_display, temp_c, "μ(t) suavizado"))

    # dados + modelos
    ax_md.plot(t, y, "o", label="Dados")
    rows = [["Regressão linear (ln)", f"{r2_linear:.4f}", "—"]]
    if fits:
        for name, yfit, r2m, params in fits:
            ax_md.plot(t, yfit, "--", label=f"{name} (R²={r2m:.3f})")
            drel = (r2m - r2_linear) / r2_linear * 100.0 if (r2_linear and np.isfinite(r2_linear) and r2_linear != 0) else np.nan
            rows.append([name, f"{r2m:.4f}", (f"{drel:+.2f} %" if np.isfinite(drel) else "—")])
    ax_md.set_xlabel("t (h)"); ax_md.set_ylabel(r"DO (Abs$_{540}$)")
    ax_md.set_title(make_title(species, strain, col, medium_display, temp_c, "Modelos empíricos"))
    ax_md.legend(frameon=False)

    table = ax_tb.table(cellText=rows, colLabels=["Modelo","R²","Δrel vs linear"],
                        cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(11.0); table.scale(1.20,1.20)

    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def export_tables(rows_reg, rows_models, rows_summary, out_dir: Path, ts: str):
    df_reg = pd.DataFrame(rows_reg)
    df_mod = pd.DataFrame(rows_models) if rows_models else pd.DataFrame(
        columns=["Coluna","Modelo","R²_modelo","A","mu_max (h⁻¹)","λ (h)","v (forma)","Fonte μ_ref","Meio_ref"])
    df_sum = pd.DataFrame(rows_summary)

    excel_path = out_dir / f"Resultados_Crescimento_{ts}.xlsx"
    try:
        writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    except ModuleNotFoundError:
        writer = pd.ExcelWriter(excel_path)
    with writer as xw:
        df_reg.to_excel(xw, sheet_name="Fases_e_Regressao", index=False)
        df_mod.to_excel(xw, sheet_name="Modelos", index=False)
        df_sum.to_excel(xw, sheet_name="Resumo", index=False)

    # CSV rápido com janelas de TODAS as fases (string + numérico)
    def _col(df, name):
        return df[name] if name in df.columns else np.nan
    
    janelas_cols = [
        "Coluna",
        # Lag
        "Lag (h)", "Lag_start (h)", "Lag_end (h)", "Lag_dur (h)",
        # Exp
        "Exp (h)", "Exp_start (h)", "Exp_end (h)", "Exp_dur (h)",
        # Estacionária
        "Est (h)", "Est_start (h)", "Est_end (h)", "Est_dur (h)",
        # Declínio
        "Dec (h)", "Dec_start (h)", "Dec_end (h)", "Dec_dur (h)",
    ]
    present = [c for c in janelas_cols if c in df_reg.columns]
    df_reg[present].to_csv(out_dir/"Janelas_Fases.csv", index=False, encoding="utf-8-sig")

    return excel_path

def write_metadata(out_dir: Path, ts: str, species: str, strain: str, medium: str, temp_c: float,
                   data_path: Path, mu_ref_csv_path: Path | None):
    with open(out_dir / "run_metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"Run timestamp: {ts}\n")
        f.write(f"Species: {species}  Strain: {strain}\n")
        f.write(f"Global medium default: {medium}\n")
        f.write(f"Global temperature default: {temp_c} °C\n")
        f.write(f"Data file: {data_path}\n")
        f.write(f"μ_ref CSV: {mu_ref_csv_path if mu_ref_csv_path else 'n/d'}\n")
