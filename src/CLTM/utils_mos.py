# utils_mos.py

import os
import re
import numpy as np
import pandas as pd

def extract_abbr_from_tsv_path(tsv_path):
    base = os.path.basename(tsv_path)
    m = re.match(r'^([A-Za-z0-9\-_]+)\.(train|dev|test)\.tsv$', base)
    if m:
        return m.group(1)
    base = re.sub(r'\.tsv$', '', base)
    base = re.sub(r'\.train$|\.dev$|\.test$', '', base)
    return base

def load_mos_map(dns_csv_path, abbr_normalizer=None):
    df = pd.read_csv(dns_csv_path)
    mos_map = {}
    for _, row in df.iterrows():
        tsv = row.get('tsv_file') or row.get('tsv')
        mean = row.get('mean_dnsmos') or row.get('mean_dnsmos')
        if pd.isna(tsv) or pd.isna(mean):
            continue
        abbr = extract_abbr_from_tsv_path(str(tsv))
        if abbr_normalizer:
            abbr = abbr_normalizer(abbr)
        try:
            mos_map[abbr] = float(mean)
        except Exception:
            continue
    return mos_map

def build_quality_factors_hard(
    mos_map,
    gamma=2.0,
    clip=(0.6, 1.6),
    one_sided=False,
    max_boost=0.25,
    min_valid=1
):
    """
    Versión 'más dura' de build_quality_factors:
      - gamma > 1 amplifica las diferencias: f = (median / mos) ** gamma
      - clip: (min, max) límites para estabilizar
      - one_sided: si True, solo se permite aumentar (f >= 1) y como máximo max_boost
      - max_boost: solo tiene efecto si one_sided=True (ej 0.25 => +25% max)
    """
    vals = np.array(list(mos_map.values()), dtype=float)
    if len(vals) < min_valid or np.all(np.isnan(vals)):
        median = 3.0
    else:
        median = float(np.nanmedian(vals))

    min_clip, max_clip = clip
    # seguridad: no permitir min_clip <= 0
    min_clip = max(min_clip, 1e-6)

    factors = {}
    for abbr, mos in mos_map.items():
        if mos is None or np.isnan(mos) or mos <= 0:
            f = 1.0
        else:
            # cálculo agresivo
            raw = (median / float(mos)) ** float(gamma)
            if one_sided:
                # sólo subir: si raw < 1 -> dejar 1; si raw>1 limitar a 1+max_boost
                if raw <= 1.0:
                    f = 1.0
                else:
                    f = min(1.0 + float(max_boost), float(raw))
            else:
                f = float(raw)

        # clip general (si one_sided=True y clip inferior < 1, respetar 1 como mínimo)
        if one_sided:
            # en one_sided, no bajar nunca por debajo de 1
            f = max(1.0, f)
            if max_clip is not None:
                f = min(f, max_clip)
        else:
            f = float(np.clip(f, min_clip, max_clip))

        factors[abbr] = float(f)

    return factors

# Mantengo la fn original como wrapper para compatibilidad
def build_quality_factors(mos_map, method='median_ratio', clip=(0.75, 1.33), min_valid=1,
                          aggressive=False, **kwargs):
    """
    Compatibilidad:
      - Si aggressive=True -> usar build_quality_factors_hard con kwargs (gamma, one_sided, ...)
      - Si method == 'median_ratio' y not aggressive -> factor = median/mos (clip)
    """
    if aggressive:
        # parámetros por defecto para agresivo
        gamma = kwargs.get('gamma', 2.0)
        one_sided = kwargs.get('one_sided', False)
        max_boost = kwargs.get('max_boost', 0.25)
        return build_quality_factors_hard(mos_map, gamma=gamma, clip=clip,
                                         one_sided=one_sided, max_boost=max_boost, min_valid=min_valid)
    # comportamiento original (suave)
    vals = np.array(list(mos_map.values()), dtype=float)
    if len(vals) < min_valid:
        median = 3.0
    else:
        median = float(np.nanmedian(vals))

    factors = {}
    for abbr, mos in mos_map.items():
        if mos <= 0 or np.isnan(mos):
            f = 1.0
        else:
            f = median / float(mos)
        f = float(np.clip(f, clip[0], clip[1]))
        factors[abbr] = f
    return factors

def apply_mos_adjustment(M, mos_factors, mode='geom', fill_factor=1.0, preserve_mean=False):
    """
    Aplicar mos_factors (dict abbr->f) a la matriz M (DataFrame).
    mode: 'geom' (media geométrica src/target), 'target' (solo fila i), 'source' (solo col j)
    fill_factor: factor por defecto si no existe abbr en mos_factors
    preserve_mean: si True, reescala la matriz resultante para que conserve la media global original.
    Devuelve copia ajustada de M.
    """
    M_adj = M.copy().astype(float)
    idx = list(M_adj.index)
    cols = list(M_adj.columns)

    row_factors = np.array([mos_factors.get(a, fill_factor) for a in idx], dtype=float)
    col_factors = np.array([mos_factors.get(a, fill_factor) for a in cols], dtype=float)

    if mode == 'geom':
        outer = np.outer(row_factors, col_factors)
        fmat = np.sqrt(outer)
    elif mode == 'target':
        fmat = np.repeat(row_factors[:, np.newaxis], len(cols), axis=1)
    elif mode == 'source':
        fmat = np.repeat(col_factors[np.newaxis, :], len(idx), axis=0)
    else:
        raise ValueError("mode debe ser 'geom', 'target' o 'source'")

    orig_mean = np.nanmean(M_adj.values)
    M_adj.iloc[:, :] = M_adj.values * fmat

    if preserve_mean:
        new_mean = np.nanmean(M_adj.values)
        if new_mean != 0 and not np.isnan(new_mean):
            M_adj.iloc[:, :] = M_adj.values * (orig_mean / new_mean)

    return M_adj
