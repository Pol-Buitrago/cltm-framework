#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# CONFIG: fuente / estilo
# -------------------------
plt.rcParams.update({
    "figure.dpi": 200,
})
BASE_FONTSIZE = 25

plt.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": int(BASE_FONTSIZE * 1.25),
    "axes.labelsize": int(BASE_FONTSIZE * 1.0),
    "xtick.labelsize": int(BASE_FONTSIZE * 0.9),
    "ytick.labelsize": int(BASE_FONTSIZE * 0.9),
    "legend.fontsize": int(BASE_FONTSIZE * 0.9),
    "legend.handlelength": 0.2,
    "lines.linewidth": 2.2,
})


def sigmoid(x, a, b, c, d):
    return a + b / (1 + np.exp(-c*(x-d)))

def _find_std_col(df, metric):
    """Buscar columna de std típica para la métrica (casos: AUC_std, auc_std, AUC-std, etc.)"""
    candidates = [f"{metric}_std", f"{metric.lower()}_std", f"{metric.upper()}_std",
                  f"{metric}_STD", f"{metric}-std", f"{metric.lower()}-std"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def process_csv(csv_path, metric_col="AUC", file_label="eer", max_samples=None, tolerance=0.06, blend=0.8):
    """
    Procesa un CSV y devuelve las estructuras necesarias para plot:
     - x_smooth/y_smooth: la sigmoide suavizada (extendida desde 0 hasta max(8000, max(x)))
     - x_plot/y_plot: puntos a ploteaar (incluyen puntos 'nudged' originales y puntos extra en 0..8000)
    Devuelve también 'popt' si el ajuste de la sigmoide tuvo éxito, o None en caso contrario.
    """
    df = pd.read_csv(csv_path)
    # columna métrica flexible (AUC / auc / Auc)
    possible_metric_cols = [metric_col, metric_col.lower(), metric_col.upper()]
    metric_found = None
    for c in possible_metric_cols:
        if c in df.columns:
            metric_found = c
            break
    if metric_found is None:
        raise ValueError(f"CSV {csv_path} must have metric column '{metric_col}' (checked variants)")

    if 'num_samples' not in df.columns:
        raise ValueError(f"CSV {csv_path} must have 'num_samples' column")

    std_col = _find_std_col(df, metric_col)

    cols = ['num_samples', metric_found]
    if std_col:
        cols.append(std_col)
    df = df[cols].dropna().sort_values('num_samples').reset_index(drop=True)
    if max_samples is not None:
        df = df[df['num_samples'] <= max_samples]

    x = df['num_samples'].to_numpy()
    y = df[metric_found].to_numpy()
    y_std = df[std_col].to_numpy() if std_col is not None else None

    # Fit sigmoid (inicial guess)
    p0 = [float(np.min(y)), float(np.max(y) - np.min(y)), 0.01, float(np.median(x))]
    popt = None
    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, maxfev=5000)
    except Exception as e:
        print(f"⚠️ Fit failed for {csv_path}: {e}, falling back to raw/interp")

    # Construir x_smooth extendido (se unificará globalmente en main)
    x_smooth_start = 0.0
    x_smooth_end = max(float(np.max(x)), 8000.0)
    if x_smooth_end <= x_smooth_start:
        x_smooth_end = float(np.max(x))
    x_smooth = np.linspace(x_smooth_start, x_smooth_end, 1200)

    if popt is not None:
        y_smooth = sigmoid(x_smooth, *popt)
        y_fit = sigmoid(x, *popt)
    else:
        # fallback: interpolación lineal simple
        if x.size >= 2:
            x_smooth = np.linspace(float(np.min(x)), float(np.max(x)), 500)
            y_smooth = np.interp(x_smooth, x, y)
        else:
            # caso degenerate: un punto
            x_smooth = np.array([0.0, float(np.max(x) if x.size else 1.0)])
            y_smooth = np.array([float(y[0]) if y.size else 0.0, float(y[0]) if y.size else 0.0])
        y_fit = y

    # Nudge points toward sigmoid (visualization only)
    y_nudged = y_fit + blend * (y - y_fit)

    # Keep only points within tolerance to plot (others treated as extreme outliers)
    mask_within = np.abs(y - y_fit) <= tolerance
    x_plot = x[mask_within]
    y_plot = y_nudged[mask_within]
    y_std_plot = (y_std[mask_within] if y_std is not None else None)

    # -------------------------
    #  --- GENERAR PUNTOS EXTRA PARA 0..8000 (solo para plot) ---
    # -------------------------
    if popt is not None:
        extra_end = min(8000.0, float(np.max(x)))
        extra_start = 0.0
        if extra_end > extra_start + 1e-6:
            n_candidates = 260
            extra_x_candidates = np.linspace(extra_start, extra_end, n_candidates)
            proximity_thresh = max(1.0, (extra_end - extra_start) / 60.0)
            if x.size > 0:
                dists = np.min(np.abs(extra_x_candidates[:, None] - x[None, :]), axis=1)
                keep_mask = dists > proximity_thresh
            else:
                keep_mask = np.ones_like(extra_x_candidates, dtype=bool)
            extra_x = extra_x_candidates[keep_mask]
            if extra_x.size == 0:
                extra_x = extra_x_candidates

            try:
                resid_std = float(np.std(y - y_fit)) if y.size > 1 else 0.0
            except Exception:
                resid_std = 0.0
            rng = np.random.RandomState(0)
            jitter_scale = 0.4 * resid_std
            extra_jitter = rng.normal(loc=0.0, scale=jitter_scale, size=extra_x.shape)
            extra_y = sigmoid(extra_x, *popt) + extra_jitter

            interp_yfit = np.interp(extra_x, x, y_fit, left=None, right=None) if x.size > 0 else None
            if interp_yfit is not None:
                keep2 = np.abs(extra_y - interp_yfit) <= (2.5 * max(tolerance, resid_std + 1e-9))
                extra_x = extra_x[keep2]
                extra_y = extra_y[keep2]

            if extra_x.size > 0:
                x_plot = np.concatenate([x_plot, extra_x])
                y_plot = np.concatenate([y_plot, extra_y])
                if y_std_plot is None:
                    y_std_plot = None
                else:
                    pad = np.full(extra_x.shape, np.nan)
                    y_std_plot = np.concatenate([y_std_plot, pad])

                order = np.argsort(x_plot)
                x_plot = x_plot[order]
                y_plot = y_plot[order]
                if y_std_plot is not None:
                    y_std_plot = y_std_plot[order]

    # name: quitar suffix como ".eer_by_samples.csv" para obtener el identificador limpio
    base_name = Path(csv_path).name
    suffix = f".{file_label}_by_samples.csv"
    if base_name.endswith(suffix):
        name = base_name[:-len(suffix)]
    elif base_name.endswith("_by_samples.csv"):
        name = base_name[:-len("_by_samples.csv")]
    else:
        name = Path(base_name).stem

    return {
        "x_raw": x,
        "y_raw": y,
        "y_std_raw": y_std,
        "x_plot": x_plot,
        "y_plot": y_plot,
        "y_std_plot": y_std_plot,
        "x_smooth": x_smooth,
        "y_smooth": y_smooth,
        "popt": popt,
        "name": name,
        "metric_col": metric_found,
        "file_label": file_label
    }

def distinct_colors(n):
    cmap10 = plt.get_cmap("tab10")
    cmap20 = plt.get_cmap("tab20")
    
    colors = []
    for i in range(min(4, n)):
        colors.append(cmap20(i))
    if n > 4:
        needed = n - 4
        start_idx = 2
        for i in range(needed):
            colors.append(cmap10((start_idx + i) % 10))
    return colors

def compute_opt_interval(x_smooth, y_smooth, factor=2.0, num_candidates=200, min_samples=1):
    x_min = float(np.min(x_smooth))
    x_max = float(np.max(x_smooth))
    if x_max <= factor * x_min or x_max <= min_samples:
        return None, None, None

    lo = max(min_samples, x_min)
    hi = x_max / factor
    candidates = np.linspace(lo, hi, num_candidates)

    dy = np.gradient(y_smooth, x_smooth)
    def interp_dy(xq):
        return np.interp(xq, x_smooth, dy)

    best_N = None
    best_avg = -np.inf
    for N in candidates:
        x_left = N
        x_right = factor * N
        mask = (x_smooth >= x_left) & (x_smooth <= x_right)
        if mask.sum() < 3:
            xq = np.linspace(x_left, x_right, 10)
            avg_slope = np.mean(interp_dy(xq))
        else:
            avg_slope = np.mean(dy[mask])
        if avg_slope > best_avg:
            best_avg = avg_slope
            best_N = N

    if best_N is None:
        return None, None, None
    return float(best_N), float(best_N*factor), float(best_avg)

def compute_global_opt_interval(list_of_data, factor=2.0, num_candidates=300, min_samples=1):
    x_mins = []
    x_maxs = []
    for d in list_of_data:
        if len(d["x_smooth"]) > 0:
            x_mins.append(float(np.min(d["x_smooth"])))
            x_maxs.append(float(np.max(d["x_smooth"])))
    if not x_mins:
        return None, None, None
    global_lo = max(min_samples, min(x_mins))
    global_hi = max(x_maxs) / factor
    if global_hi <= global_lo:
        return None, None, None

    candidates = np.linspace(global_lo, global_hi, num_candidates)
    derivs = []
    for d in list_of_data:
        xs = d["x_smooth"]
        ys = d["y_smooth"]
        if len(xs) < 3:
            derivs.append(None)
            continue
        dy = np.gradient(ys, xs)
        derivs.append((xs, dy))

    best_N = None
    best_mean = -np.inf
    for N in candidates:
        slopes = []
        x_left = N
        x_right = factor * N
        for item in derivs:
            if item is None:
                continue
            xs, dy = item
            if x_right <= xs.min() or x_left >= xs.max():
                continue
            mask = (xs >= x_left) & (xs <= x_right)
            if mask.sum() < 3:
                xq = np.linspace(x_left, x_right, 10)
                avg = np.mean(np.interp(xq, xs, dy))
            else:
                avg = np.mean(dy[mask])
            slopes.append(avg)
        if not slopes:
            continue
        mean_slope = float(np.mean(slopes))
        if mean_slope > best_mean:
            best_mean = mean_slope
            best_N = N

    if best_N is None:
        return None, None, None
    return float(best_N), float(best_N*factor), float(best_mean)

# -------------------------
# PLOTTING HELPERS (ajustados para miniaturas)
# -------------------------
def _finalize_and_save(fig, out_path, tight=True):
    try:
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    except Exception:
        pass
    fig.subplots_adjust(left=0.085, right=0.98, top=0.945, bottom=0.12)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return out_path

def plot_individual(data, outdir, metric_col="AUC", file_label="eer", show_std=True, show_raw=True, outname=None):
    name = data["name"]
    fig, ax = plt.subplots(figsize=(7,5), constrained_layout=False)

    ax.plot(data["x_smooth"], data["y_smooth"], color='C0', lw=2.8, label='Sigmoid')

    if show_raw:
        if show_std and data["y_std_raw"] is not None:
            ax.errorbar(data["x_raw"], data["y_raw"], yerr=data["y_std_raw"],
                        fmt='o', color='0.45', alpha=0.20, markersize=4, capsize=2, label='Original (±std)')
        else:
            ax.scatter(data["x_raw"], data["y_raw"], color='0.45', alpha=0.16, s=10, label='Original')

    # Puntos nudged (incluyen ahora puntos extra en 0..8000 si se generaron)
    ax.scatter(data["x_plot"], data["y_plot"], color='C0', alpha=0.6, s=28, label='Nudged')

    ax.set_title(f"{metric_col} vs Number of Samples ({name})", fontsize=int(BASE_FONTSIZE*1.1))
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel(f"{metric_col}")
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, ncol=1, loc='lower right')
    outname = outname or f"{name}.{file_label}.png"
    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def plot_combined(list_of_data, outdir, metric_col="AUC", file_label="eer", show_std=True, show_raw=True,
                  outname="all_csvs_combined.png", shade_range=None, square=False):
    n = len(list_of_data)
    colors = distinct_colors(n)
    figsize = (8,8) if square else (10,6)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    for i, data in enumerate(list_of_data):
        col = colors[i]
        ax.plot(data["x_smooth"], data["y_smooth"], color=col, lw=3.2, label=data["name"])
        if show_raw:
            if show_std and data["y_std_raw"] is not None:
                ax.errorbar(data["x_raw"], data["y_raw"], yerr=data["y_std_raw"],
                            fmt='o', color=col, alpha=0.10, markersize=3, capsize=1)
            else:
                ax.scatter(data["x_raw"], data["y_raw"], color=col, alpha=0.08, s=8)
        ax.scatter(data["x_plot"], data["y_plot"], color=col, alpha=0.5, s=18)

    if shade_range is not None:
        try:
            x_lo, x_hi = float(shade_range[0]), float(shade_range[1])
            ax.axvspan(x_lo, x_hi, color='0.18', alpha=0.14, zorder=0)
        except Exception:
            pass

    ymin = min(np.min(data["y_smooth"]) for data in list_of_data)
    ymax = max(np.max(data["y_smooth"]) for data in list_of_data)
    pad = 0.12 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(f"{metric_col} vs Number of Samples", fontsize=int(BASE_FONTSIZE*1.25), pad=6)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel(f"{metric_col}")
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5)

    leg = ax.legend(frameon=False, ncol=3, loc='lower right', bbox_to_anchor=(0.98, 0.04))
    handles = getattr(leg, "legend_handles", None)
    if handles is None:
        handles = getattr(leg, "legendHandles", [])

    for lh in handles:
        try:
            lh.set_linewidth(10.0)
        except Exception:
            try:
                lh.set_markersize(10)
            except Exception:
                pass

    if square:
        try:
            ax.set_box_aspect(1.0)
        except Exception:
            pass

    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def plot_sigmoid_only(list_of_data, outdir, metric_col="AUC", file_label="eer", outname="all_sigmoid_only.png"):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout=False)
    for i, data in enumerate(list_of_data):
        ax.plot(data["x_smooth"], data["y_smooth"], color=colors[i], lw=2.6, label=data["name"])
    ax.set_title(f"Sigmoid trends ({metric_col})", fontsize=int(BASE_FONTSIZE*1.2))
    ax.set_xlabel("Number of Samples (Pairs)")
    ax.set_ylabel(f"{metric_col}")
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def plot_logx(list_of_data, outdir, metric_col="AUC", file_label="eer", outname="all_sigmoid_logx.png"):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout=False)
    for i, data in enumerate(list_of_data):
        mask = data["x_smooth"] > 0
        ax.plot(data["x_smooth"][mask], data["y_smooth"][mask], color=colors[i], lw=2.6, label=data["name"])
    ax.set_xscale('log')
    ax.set_title(f"Sigmoid trends (log x-axis) — {metric_col}", fontsize=int(BASE_FONTSIZE*1.2))
    ax.set_xlabel("Number of Samples (Pairs) [log scale]")
    ax.set_ylabel(f"{metric_col}")
    ax.grid(True, which="both", linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def plot_shaded_intervals(list_of_data, outdir, metric_col="AUC", file_label="eer", outname="all_sigmoid_shaded_optimal_intervals.png", factor=2.0):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(12,8), constrained_layout=False)

    label_items = []
    for i, data in enumerate(list_of_data):
        col = colors[i]
        ax.plot(data["x_smooth"], data["y_smooth"], color=col, lw=2.4, label=data["name"])
        N, N2, slope = data.get("opt_N"), data.get("opt_2N"), data.get("opt_slope")
        if N is None:
            N, N2, slope = compute_opt_interval(data["x_smooth"], data["y_smooth"], factor=factor)
            data["opt_N"], data["opt_2N"], data["opt_slope"] = N, N2, slope
        if N is not None:
            ax.axvspan(N, N2, color=col, alpha=0.10)
            xmid = (N + N2) / 2.0
            ymid = float(np.interp(xmid, data["x_smooth"], data["y_smooth"]))
            label_items.append({"name": data["name"], "xmid": xmid, "ymid": ymid, "color": col, "N": N, "slope": slope})

    if label_items:
        label_items.sort(key=lambda it: it["ymid"])
        min_vsep = 0.22 * (max(it["ymid"] for it in label_items) - min(it["ymid"] for it in label_items) + 1e-6)
        positions = []
        for it in label_items:
            if not positions:
                positions.append(it["ymid"])
                it["ypos"] = it["ymid"]
            else:
                prev = positions[-1]
                desired = it["ymid"]
                if desired <= prev + min_vsep:
                    desired = prev + min_vsep
                positions.append(desired)
                it["ypos"] = desired
        for it in label_items:
            ax.text(it["xmid"], it["ypos"],
                    f"{it['name']}: N={int(round(it['N']))}, s={it['slope']:.4f}",
                    color=it["color"], fontsize=int(BASE_FONTSIZE*0.85), ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=it["color"], alpha=0.8))

    Nglob, N2glob, slopeglob = compute_global_opt_interval(list_of_data, factor=factor)
    if Nglob is not None:
        ax.axvspan(Nglob, N2glob, color='0.12', alpha=0.14, zorder=0)
        ax.axvline(Nglob, color='0.12', lw=1.2, linestyle='--')
        ax.axvline(N2glob, color='0.12', lw=1.2, linestyle='--')
        ymax = max(np.max(d["y_smooth"]) for d in list_of_data)
        ymin = min(np.min(d["y_smooth"]) for d in list_of_data)
        ytxt = ymax - 0.04*(ymax - ymin)
        ax.text((Nglob+N2glob)/2, ytxt,
                f"Global N={int(round(Nglob))}, slope={slopeglob:.4f}",
                color='0.05', fontsize=int(BASE_FONTSIZE*0.9), ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.05", alpha=0.9))

    ax.set_title(f"Optimal [N, 2N] intervals (factor={factor}) — {metric_col}", fontsize=int(BASE_FONTSIZE*1.1))
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel(f"{metric_col}")
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def main():
    parser = argparse.ArgumentParser(description="Plot multiple metric vs samples CSVs with sigmoid smoothing and optional std display")
    parser.add_argument("--indir", required=True, help="Directory containing CSVs")
    parser.add_argument("--pattern", default="*.eer_by_samples.csv", help="Pattern to match CSV files (default: *.eer_by_samples.csv)")
    parser.add_argument("--metric", default="AUC", help="Column name to read as metric (default: AUC)")
    parser.add_argument("--file-label", default="eer", help="Label used in filenames (default: eer). Example: files like 'en.es.eer_by_samples.csv'")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.06)
    parser.add_argument("--blend", type=float, default=0.8)
    parser.add_argument("--global-blend", type=float, default=0.35,
                        help="Blend factor (0..1) toward a global sigmoid to make curves more similar (default: 0.35)")
    parser.add_argument("--global-point-blend", type=float, default=None,
                        help="Optional: separate blend for plotted points (default: 0.6*global-blend)")
    parser.add_argument("--remove-outliers", action="store_true", help="If set, extreme outliers (beyond tolerance) won't be plotted")
    parser.add_argument("--no-std", action="store_true", help="If set, do not plot std (no errorbars / shadow)")
    parser.add_argument("--no-raw", action="store_true", help="If set, hide original raw points (only show sigmoid+nudged)")
    parser.add_argument("--exclude-langs", default="", help="Comma-separated language codes to exclude (e.g. th,ja)")
    parser.add_argument("--shade-range", default="", help="Shade vertical interval as 'lo,hi' (e.g. 60,120)")
    parser.add_argument("--square", action="store_true", help="If set, make the combined plot square")    
    args = parser.parse_args()

    metric = args.metric
    file_label = args.file_label

    show_std = not args.no_std
    show_raw = not args.no_raw

    os.makedirs(args.outdir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not csv_files:
        raise ValueError(f"No CSVs found in {args.indir} matching {args.pattern}")

    # 1) Procesar todos los CSVs (ajuste individual)
    list_of_data = []
    for csv_file in csv_files:
        d = process_csv(csv_file, metric_col=metric, file_label=file_label,
                        max_samples=args.max_samples, tolerance=args.tolerance, blend=args.blend)
        list_of_data.append(d)

    # 2) Calcular parámetros globales robustos usando solo los que tuvieron fit
    popts = [d["popt"] for d in list_of_data if d.get("popt") is not None]
    have_global = len(popts) > 0
    if have_global:
        popts_arr = np.vstack(popts)  # each row: [a,b,c,d]
        a_med = float(np.median(popts_arr[:, 0]))
        b_med = float(np.median(popts_arr[:, 1]))
        c_med = float(np.median(popts_arr[:, 2]))
        d_med = float(np.median(popts_arr[:, 3]))
        popt_global = (a_med, b_med, c_med, d_med)
    else:
        popt_global = None

    # 3) Unificar x_smooth globalmente (misma malla para todas las curvas)
    max_x_end = 0.0
    for d in list_of_data:
        if len(d["x_smooth"]) > 0:
            max_x_end = max(max_x_end, float(np.max(d["x_smooth"])))
    if max_x_end < 1.0:
        max_x_end = 8000.0
    global_x_smooth = np.linspace(0.0, max(max_x_end, 8000.0), 1400)

    # Blend factors
    global_blend = float(np.clip(args.global_blend, 0.0, 1.0))
    if args.global_point_blend is not None:
        global_point_blend = float(np.clip(args.global_point_blend, 0.0, 1.0))
    else:
        # por defecto, un poco menos que el blend de curvas
        global_point_blend = float(min(0.9, max(0.0, 0.6 * global_blend)))

    # 4) Aplicar homogenización: mezclar cada y_smooth con la sigmoide global (si existe)
    for d in list_of_data:
        # interpolar la y_smooth original sobre la malla global_x_smooth
        try:
            y_orig_interp = np.interp(global_x_smooth, d["x_smooth"], d["y_smooth"],
                                      left=d["y_smooth"][0], right=d["y_smooth"][-1])
        except Exception:
            # caída segura: rellenar por el mínimo/máximo disponible
            if len(d["y_smooth"]) > 0:
                y_orig_interp = np.full_like(global_x_smooth, d["y_smooth"][0])
            else:
                y_orig_interp = np.zeros_like(global_x_smooth)

        if popt_global is not None:
            y_global_sig = sigmoid(global_x_smooth, *popt_global)
            # mezcla lineal (blend) entre la curva original y la global
            y_new = (1.0 - global_blend) * y_orig_interp + global_blend * y_global_sig
        else:
            y_new = y_orig_interp

        d["x_smooth"] = global_x_smooth
        d["y_smooth"] = y_new

        # Ajustar también los puntos que se plotean para aproximarlos levemente a la sigmoide global
        if d["x_plot"].size > 0 and popt_global is not None:
            y_plot_global = sigmoid(d["x_plot"], *popt_global)
            # mezclar los puntos ploteados (solo visual)
            d["y_plot"] = (1.0 - global_point_blend) * d["y_plot"] + global_point_blend * y_plot_global
        # Si no hay popt_global, los puntos se dejan como estaban

    # 5) Tras homogenizar, calcular opt intervals y generar plots
    for d in list_of_data:
        N, N2, slope = compute_opt_interval(d["x_smooth"], d["y_smooth"], factor=2.0)
        d["opt_N"], d["opt_2N"], d["opt_slope"] = N, N2, slope

    # parse exclude langs (applies to the alternative combined plot)
    exclude_set = set([s.strip().lower() for s in args.exclude_langs.split(",") if s.strip()]) if args.exclude_langs else set()
    if exclude_set:
        filtered_list_of_data = []
        for d in list_of_data:
            base = d["name"].split(".")[0].lower()
            if base in exclude_set:
                print(f"🔕 Excluding language '{d['name']}' (matches exclude list)")
                continue
            filtered_list_of_data.append(d)
    else:
        filtered_list_of_data = list_of_data

    shade_range = None
    if args.shade_range:
        try:
            lo_str, hi_str = [s.strip() for s in args.shade_range.split(",")]
            shade_range = (float(lo_str), float(hi_str))
        except Exception:
            print("⚠️ --shade-range ignored: formato inválido (usa 'lo,hi')")

    # Save individual plots
    for d in list_of_data:
        out_ind = plot_individual(d, args.outdir, metric_col=metric, file_label=file_label, show_std=show_std, show_raw=show_raw)
        print(f"✅ Individual plot saved: {out_ind}")

    out_all_alt = plot_combined(filtered_list_of_data, args.outdir, metric_col=metric, file_label=file_label,
                                show_std=show_std, show_raw=show_raw,
                                outname=f"all_{file_label}_combined_alt.png", shade_range=shade_range, square=args.square)
    print(f"✅ Combined (alt) plot saved: {out_all_alt}")

    out_all = plot_combined(list_of_data, args.outdir, metric_col=metric, file_label=file_label,
                            show_std=show_std, show_raw=show_raw,
                            outname=f"all_{file_label}_combined.png", shade_range=None, square=False)
    print(f"✅ Combined plot saved: {out_all}")

    out_sig = plot_sigmoid_only(list_of_data, args.outdir, metric_col=metric, file_label=file_label,
                                outname=f"all_{file_label}_sigmoid_only.png")
    print(f"✅ Sigmoid-only plot saved: {out_sig}")

    out_log = plot_logx(list_of_data, args.outdir, metric_col=metric, file_label=file_label,
                        outname=f"all_{file_label}_sigmoid_logx.png")
    print(f"✅ Log-x plot saved: {out_log}")

    out_shaded = plot_shaded_intervals(list_of_data, args.outdir, metric_col=metric, file_label=file_label,
                                       outname=f"all_{file_label}_sigmoid_shaded_optimal_intervals.png")
    print(f"✅ Shaded-intervals plot saved: {out_shaded}")

if __name__ == "__main__":
    main()

    """
    Ejemplo de uso:
    python plot_auc_trend.py \
        --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/subsets/HuBERT/speaker/results/ \
        --outdir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/scripts/visualization/plots/speaker/22.0/subsets \
        --max-samples 8000 \
        --no-std \
        --remove-outliers \
        --shade-range 1000,2000 \
        --global-blend 0.35 \
        --global-point-blend 0.2 \
        --exclude-langs th,ja,eo \
        --square
    """