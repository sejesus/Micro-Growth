# -*- coding: utf-8 -*-
"""
Implementa os modelos empíricos de crescimento Gompertz, Logístico e Richards,
ajustados por mínimos quadrados não lineares (SciPy curve_fit quando disponível).
Retorna os parâmetros estimados e o coeficiente de determinação R² de cada modelo.
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.optimize import curve_fit
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def gompertz(t, A, mu_max, lam):
    return A * np.exp(-np.exp((mu_max * np.e / A) * (lam - t) + 1))

def logistic(t, A, mu_max, lam):
    return A / (1 + np.exp((4 * mu_max / A) * (lam - t) + 2))

def richards(t, A, mu_max, lam, v):
    return A * (1 + v * np.exp(-mu_max * (t - lam))) ** (-1.0 / v)

def r2_calc(y, yfit):
    y = np.asarray(y, float); yfit = np.asarray(yfit, float)
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

def fit_all_models(th, y, mu_guess=None, lam_guess=None, normalize=True):
    """Devolve lista [(name, yfit, r2, params_dict), ...]."""
    if not SCIPY_OK:
        return []
    t = np.asarray(th, float)
    y = np.clip(np.asarray(y, float), 1e-8, None)
    A0_raw = float(np.nanmax(y))
    A0_tail = float(np.nanmedian(y[-max(3, len(y)//4):]))
    A0 = max(A0_raw, A0_tail)

    if normalize:
        yn = y / A0; A_seed = 1.0
    else:
        yn = y.copy(); A_seed = A0

    mu0  = float(mu_guess) if (mu_guess is not None and np.isfinite(mu_guess) and mu_guess > 0) else 0.5
    lam0 = float(lam_guess) if (lam_guess is not None and np.isfinite(lam_guess)) else float(t[0])
    eps  = 1e-8
    fits = []

    def denorm(name, p):
        if name in ("Gompertz", "Logístico"):
            A, mmax, lam = p
            if normalize: A *= A0
            return {"A": float(A), "mu_max (h^-1)": float(mmax), "λ (h)": float(lam)}
        elif name == "Richards":
            A, mmax, lam, v = p
            if normalize: A *= A0
            return {"A": float(A), "mu_max (h^-1)": float(mmax), "λ (h)": float(lam), "v (forma)": float(v)}
        return {}

    # Gompertz
    try:
        popt_g, _ = curve_fit(
            lambda tt, A, mu_max, lam: gompertz(tt, A, mu_max, lam),
            t, yn, p0=[A_seed, mu0, lam0],
            bounds=([eps, eps, t[0]-10.0], [10.0*A_seed, 5.0, t[-1]+10.0]),
            maxfev=50000
        )
        yfit_gn = gompertz(t, *popt_g); yfit_g = (yfit_gn*A0) if normalize else yfit_gn
        fits.append(("Gompertz", yfit_g, r2_calc(y, yfit_g), denorm("Gompertz", popt_g)))
    except Exception:
        pass

    # Logístico
    try:
        popt_l, _ = curve_fit(
            lambda tt, A, mu_max, lam: logistic(tt, A, mu_max, lam),
            t, yn, p0=[A_seed, mu0, lam0],
            bounds=([eps, eps, t[0]-10.0], [10.0*A_seed, 5.0, t[-1]+10.0]),
            maxfev=50000
        )
        yfit_ln = logistic(t, *popt_l); yfit_l = (yfit_ln*A0) if normalize else yfit_ln
        fits.append(("Logístico", yfit_l, r2_calc(y, yfit_l), denorm("Logístico", popt_l)))
    except Exception:
        pass

    # Richards
    try:
        popt_r, _ = curve_fit(
            richards, t, yn, p0=[A_seed, mu0, lam0, 1.0],
            bounds=([eps, eps, t[0]-10.0, 0.05], [10.0*A_seed, 5.0, t[-1]+10.0, 10.0]),
            maxfev=60000
        )
        yfit_rn = richards(t, *popt_r); yfit_r = (yfit_rn*A0) if normalize else yfit_rn
        fits.append(("Richards", yfit_r, r2_calc(y, yfit_r), denorm("Richards", popt_r)))
    except Exception:
        pass

    return fits
