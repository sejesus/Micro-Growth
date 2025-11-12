# -*- coding: utf-8 -*-
"""
Módulo responsável pela leitura e pré-processamento dos ficheiros CSV de crescimento (formato 'wide').
Inclui deteção de linhas meta ('Meta_Medium', 'Meta_Temp'), resolução de meios e temperaturas por coluna,
e carregamento das tabelas de referência de μ_ref (CSV externo ou base interna).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config.medium_aliases import resolve_medium_alias, canonical_medium_key

# --- base local de μ_ref (exemplos) ---
growth_db = pd.DataFrame([
    {"species":"Escherichia coli",        "strain":"", "medium":"LB",   "mu_ref_h_inv":0.70, "source":"Neidhardt et al., 1996"},
    {"species":"Bacillus subtilis",       "strain":"", "medium":"LB",   "mu_ref_h_inv":0.55, "source":"Kunst et al., 1997"},
    {"species":"Pseudomonas aeruginosa",  "strain":"", "medium":"M9",   "mu_ref_h_inv":0.30, "source":"Shih et al., 2019"},
    {"species":"Saccharomyces cerevisiae","strain":"", "medium":"YPD",  "mu_ref_h_inv":0.45, "source":"Verduyn et al., 1992"},
])
growth_db["medium_key"] = growth_db["medium"].map(lambda s: canonical_medium_key(resolve_medium_alias(str(s))))

def load_growth_data(path: str):
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV precisa da 1ª coluna = tempo (min) e ≥1 coluna de absorbância.")
    col_time = df.columns[0]

    meta_medium_row = None
    meta_temp_row   = None
    meta_mask = df[col_time].astype(str).str.startswith("Meta_", na=False)
    if meta_mask.any():
        meta_df = df[meta_mask]
        for _, row in meta_df.iterrows():
            key = str(row[col_time]).strip()
            if key == "Meta_Medium":
                meta_medium_row = {c: row[c] for c in df.columns[1:]}
            elif key == "Meta_Temp":
                meta_temp_row = {c: row[c] for c in df.columns[1:]}
        df = df[~meta_mask]

    t_min = pd.to_numeric(df[col_time].astype(str).str.replace(",", "."), errors="coerce").to_numpy()
    if np.any(~np.isfinite(t_min)):
        raise ValueError("Coluna de tempo contém valores inválidos.")
    th = t_min / 60.0

    series = {}
    for c in df.columns[1:]:
        y = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce").to_numpy()
        series[c] = y

    return th, t_min, series, meta_medium_row, meta_temp_row

def load_mu_ref_table(path_csv: str):
    if not path_csv:
        return None
    p = Path(path_csv)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    cols = {c.lower().strip(): c for c in df.columns}
    req = ["species", "medium", "mu_ref_h_inv"]
    for r in req:
        if r not in cols:
            raise ValueError("CSV de μ_ref precisa das colunas: species, medium, mu_ref_h_inv")
    out = pd.DataFrame({
        "species": df[cols["species"]].astype(str),
        "strain":  df[cols["strain"]] if "strain" in cols else "",
        "medium":  df[cols["medium"]].astype(str),
        "mu_ref_h_inv": pd.to_numeric(df[cols["mu_ref_h_inv"]], errors="coerce"),
        "source":  df[cols["source"]] if "source" in cols else "CSV",
    })
    out["medium_key"] = out["medium"].map(lambda s: canonical_medium_key(resolve_medium_alias(str(s))))
    return out

def resolve_meta_for_column(
    col_name: str,
    default_medium: str,
    default_temp_c: float,
    column_to_medium: dict[str, str],
    column_to_temp: dict[str, float],
    meta_medium_row: dict | None = None,
    meta_temp_row: dict | None = None,
):
    """
    Devolve (medium_display, medium_alias, temp_c)
    - medium_display: o texto “humano” (original do CSV Meta_Medium, se existir) ou o default/mapeamento
    - medium_alias: sigla normalizada (via resolve_medium_alias) para lookups
    - temp_c: float
    """
    if meta_medium_row and col_name in meta_medium_row:
        raw_medium = str(meta_medium_row[col_name]).strip()
    else:
        raw_medium = column_to_medium.get(col_name, default_medium)

    medium_display = raw_medium if raw_medium else default_medium
    medium_alias   = resolve_medium_alias(medium_display)

    if meta_temp_row and col_name in meta_temp_row:
        try:
            temp_c = float(str(meta_temp_row[col_name]).replace(",", "."))
        except Exception:
            temp_c = float(default_temp_c)
    else:
        temp_c = float(column_to_temp.get(col_name, default_temp_c))

    return medium_display, medium_alias, temp_c

def resolve_mu_thresholds_for_column(
    col_name: str,
    species: str,
    strain: str,
    global_medium: str,
    column_to_medium: dict[str, str],
    mu_ref_manual: float | None,
    mu_ref_csv_df,  # DataFrame ou None
    MU_MIN_FACTOR: float, MU_EXIT_FACTOR: float, MU_STAT_FACTOR: float,
    MU_MIN_DEFAULT: float, MU_STAT_DEFAULT: float,
    medium_alias_override: str | None = None,
):
    # 1) escolher meio para esta coluna
    medium_for_col = medium_alias_override if medium_alias_override else column_to_medium.get(col_name, global_medium)
    # 2) chave canónica via alias
    medium_key = canonical_medium_key(resolve_medium_alias(medium_for_col))

    # prioridade: manual
    mu_ref = None; src = None
    if mu_ref_manual is not None and np.isfinite(mu_ref_manual) and mu_ref_manual > 0:
        mu_ref = float(mu_ref_manual); src = "Manual"

    # CSV
    if mu_ref is None and mu_ref_csv_df is not None and len(mu_ref_csv_df) > 0:
        m = mu_ref_csv_df[(mu_ref_csv_df["species"].str.lower() == species.lower()) &
                          (mu_ref_csv_df["medium_key"] == medium_key)]
        if strain and str(strain).strip():
            m2 = m[m["strain"].astype(str).str.lower() == str(strain).lower()]
            if not m2.empty: m = m2
        if not m.empty:
            mu_ref = float(m.iloc[0]["mu_ref_h_inv"]); src = f"CSV: {m.iloc[0].get('source','CSV')}"

    # base local
    if mu_ref is None:
        m = growth_db[(growth_db["species"].str.lower() == species.lower()) &
                      (growth_db["medium_key"] == medium_key)]
        if strain and str(strain).strip():
            m2 = m[m["strain"].astype(str).str.lower() == str(strain).lower()]
            if not m2.empty: m = m2
        if not m.empty:
            mu_ref = float(m.iloc[0]["mu_ref_h_inv"]); src = f"Base local: {m.iloc[0]['source']}"

    # limiares
    if mu_ref is None:
        mu_min  = MU_MIN_DEFAULT
        mu_exit = MU_MIN_DEFAULT * 0.8
        mu_stat = MU_STAT_DEFAULT
        src     = "Default heurístico"
    else:
        mu_min  = MU_MIN_FACTOR  * mu_ref
        mu_exit = MU_EXIT_FACTOR * mu_ref
        mu_stat = MU_STAT_FACTOR * mu_ref

    return mu_ref, mu_min, mu_exit, mu_stat, src, medium_for_col
