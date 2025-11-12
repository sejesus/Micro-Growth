# -*- coding: utf-8 -*-
"""
Ferramentas de análise temporal para curvas de crescimento microbiano.
Calcula a derivada μ(t) suavizada, identifica automaticamente as fases (lag, exponencial, estacionária, declínio)
com base em limiares e histerese, e seleciona os pontos ótimos para regressão linear na fase exponencial.
"""
from __future__ import annotations
import numpy as np

# parâmetros (ajustáveis via Settings no main.py)
DERIV_SMOOTH_WINDOW = 3
R2_TARGET = 0.98
MAX_REMOVE = 1
MIN_LEN = 6
RAW_SLOPE_MAX = 0.08
TAIL_RELAX = 0.80

def moving_average(x, window):
    if window <= 1: return x.copy()
    k = int(window);  k = k + 1 if k % 2 == 0 else k
    ker = np.ones(k)/k
    return np.convolve(x, ker, mode="same")

def mu_instantaneo(t_h, a):
    a = np.asarray(a, float); a[a<=0] = np.nan
    t = np.asarray(t_h, float)
    ly = np.log(a)
    d  = np.gradient(ly, t)
    m  = ~np.isnan(d)
    out = d.copy()
    if m.any():
        out[m] = moving_average(d[m], DERIV_SMOOTH_WINDOW)
    return out

def linreg_log(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.any(y <= 0): return np.nan, np.nan, np.nan, None
    ly = np.log(y)
    A  = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = m*x + c
    resid = ly - yhat
    ssr = np.sum(resid**2); sst = np.sum((ly - np.mean(ly))**2)
    r2 = 1 - ssr/sst if sst > 0 else np.nan
    return m, c, r2, resid

def segment_phases(th, y, mu_min, mu_exit, mu_stat):
    """Segmentação heurística simples: lag até μ>=mu_min; exp até μ<mu_exit; resto estacionária (ou declínio se μ< -mu_stat)."""
    th = np.asarray(th, float); y = np.asarray(y, float)
    mu = mu_instantaneo(th, y)
    i0 = 0
    # início exp = primeiro ponto com μ >= mu_min (precisa de 2 consecutivos)
    enter = np.where((mu[:-1] >= mu_min) & (mu[1:] >= mu_min))[0]
    if enter.size: i0 = int(enter[0])
    # fim exp = primeiro ponto após i0 com μ < mu_exit
    after = mu[i0+1:]
    if after.size:
        idx = np.where(after < mu_exit)[0]
        i1 = int(i0+1+idx[0]) if idx.size else len(th)-1
    else:
        i1 = len(th)-1

    lag  = (th[0], th[i0])        if i0 > 0 else None
    exp_ = (th[i0], th[i1])       if i1 > i0 else (th[i0], th[i0])
    rest = (th[i1], th[-1])       if i1 < len(th)-1 else None

    stat = rest
    decl = None
    if rest is not None:
        mu_tail = float(np.nanmedian(mu[i1+1:])) if i1+1 < len(mu) else 0.0
        if mu_tail < -mu_stat:
            decl = rest; stat = None

    return {"lag": lag, "exp": exp_, "stat": stat, "decl": decl, "mu_series": mu}, (i0, i1)

def score_candidate(x, y, i0, i1):
    m,c,r2,resid = linreg_log(x[i0:i1+1], y[i0:i1+1])
    if np.isnan(r2) or m <= 0: return None
    end_pen = 0.0
    if resid is not None and len(resid)>=2:
        rms=np.sqrt(np.mean(resid**2))
        if rms>0: end_pen=(abs(resid[0])+abs(resid[-1]))/(2*rms)
    length = i1-i0+1
    return ((r2, length, -end_pen), i0, i1, m, c, r2)

def pick_points_within_exp(th, y, i0e, i1e):
    """Seleciona [j0..j1] dentro de [i0e..i1e] removendo pontas até MAX_REMOVE para maximizar R² e comprimento."""
    best=None
    for trim_s in range(0, MAX_REMOVE+1):
        for trim_e in range(0, MAX_REMOVE-trim_s+1):
            j0=i0e+trim_s; j1=i1e-trim_e
            if j1-j0+1 < MIN_LEN: continue
            sc=score_candidate(th,y,j0,j1)
            if sc is None: continue
            if (best is None) or (sc[0] > best[0]): best=sc
    if best is None:
        m,c,r2,_=linreg_log(th[i0e:i1e+1], y[i0e:i1e+1])
        return i0e,i1e,m,c,r2,0,0
    _,j0,j1,m,c,r2 = best
    trim_s = j0 - i0e; trim_e = i1e - j1
    return j0,j1,m,c,r2,trim_s,trim_e
