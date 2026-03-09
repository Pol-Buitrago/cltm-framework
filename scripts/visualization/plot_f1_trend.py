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
# Base font size — aumenta/reduce si quieres (está muy grande para miniaturas legibles)
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

def process_csv(csv_path, max_samples=None, tolerance=0.06, blend=0.8):
    df = pd.read_csv(csv_path)
    if not {'f1','num_samples'}.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} must have 'f1' and 'num_samples'")

    cols = ['num_samples', 'f1']
    if 'f1_std' in df.columns:
        cols.append('f1_std')
    df = df[cols].dropna().sort_values('num_samples').reset_index(drop=True)
    if max_samples is not None:
        df = df[df['num_samples'] <= max_samples]

    x = df['num_samples'].to_numpy()
    y = df['f1'].to_numpy()
    y_std = df['f1_std'].to_numpy() if 'f1_std' in df.columns else None

    # Fit sigmoid
    p0 = [float(np.min(y)), float(np.max(y) - np.min(y)), 0.01, float(np.median(x))]
    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, maxfev=5000)
        x_smooth = np.linspace(float(np.min(x)), float(np.max(x)), 500)
        y_smooth = sigmoid(x_smooth, *popt)
        y_fit = sigmoid(x, *popt)
    except Exception as e:
        print(f"⚠️ Fit failed for {csv_path}: {e}, falling back to raw points")
        x_smooth = x
        y_smooth = y
        y_fit = y

    # Nudge points toward sigmoid (visualization only)
    y_nudged = y_fit + blend * (y - y_fit)

    # Keep only points within tolerance to plot (others treated as extreme outliers)
    mask_within = np.abs(y - y_fit) <= tolerance
    x_plot = x[mask_within]
    y_plot = y_nudged[mask_within]
    y_std_plot = (y_std[mask_within] if y_std is not None else None)

    name = Path(csv_path).name.replace(".f1_by_samples.csv", "")
    return {
        "x_raw": x,
        "y_raw": y,
        "y_std_raw": y_std,
        "x_plot": x_plot,
        "y_plot": y_plot,
        "y_std_plot": y_std_plot,
        "x_smooth": x_smooth,
        "y_smooth": y_smooth,
        "name": name
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
# PLOTTING HELPERS (esp. ajustados para miniaturas)
# -------------------------
def _finalize_and_save(fig, out_path, tight=True):
    # minimizar bordes
    try:
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    except Exception:
        pass
    # forzar ajustes muy compactos
    fig.subplots_adjust(left=0.085, right=0.98, top=0.945, bottom=0.12)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return out_path

def plot_individual(data, outdir, show_std=True, show_raw=True, outname=None):
    name = data["name"]
    fig, ax = plt.subplots(figsize=(7,5), constrained_layout=False)

    ax.plot(data["x_smooth"], data["y_smooth"], color='C0', lw=2.8, label='Sigmoid')

    if show_raw:
        if show_std and data["y_std_raw"] is not None:
            ax.errorbar(data["x_raw"], data["y_raw"], yerr=data["y_std_raw"],
                        fmt='o', color='0.45', alpha=0.20, markersize=4, capsize=2, label='Original (±std)')
        else:
            ax.scatter(data["x_raw"], data["y_raw"], color='0.45', alpha=0.16, s=10, label='Original')

    ax.scatter(data["x_plot"], data["y_plot"], color='C0', alpha=0.6, s=28, label='Nudged')

    ax.set_title(f"F1 vs Number of Samples ({name})", fontsize=int(BASE_FONTSIZE*1.1))
    ax.set_xlabel("Number of Samples (Pairs)")
    ax.set_ylabel("F1 score")
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, ncol=1, loc='lower right')
    outname = outname or f"{name}.png"
    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def plot_combined(list_of_data, outdir, show_std=True, show_raw=True,
                  outname="all_csvs_combined.png", shade_range=None, square=False):
    """
    shade_range: None or (x_lo, x_hi) to shade that vertical interval
    square: if True, use a square figsize (and box-aspect)
    """
    n = len(list_of_data)
    colors = distinct_colors(n)
    figsize = (8,8) if square else (10,6)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    for i, data in enumerate(list_of_data):
        col = colors[i]
        # aumentar ligeramente el grosor de las líneas de las curvas en el combined
        ax.plot(data["x_smooth"], data["y_smooth"], color=col, lw=3.2, label=data["name"])  # <-- grosor aumentado
        if show_raw:
            if show_std and data["y_std_raw"] is not None:
                ax.errorbar(data["x_raw"], data["y_raw"], yerr=data["y_std_raw"],
                            fmt='o', color=col, alpha=0.10, markersize=3, capsize=1)
            else:
                ax.scatter(data["x_raw"], data["y_raw"], color=col, alpha=0.08, s=8)
        # aumentar un poco el tamaño/peso de los puntos nudged
        ax.scatter(data["x_plot"], data["y_plot"], color=col, alpha=0.5, s=18)

    # Shade specified interval (behind curves)
    if shade_range is not None:
        try:
            x_lo, x_hi = float(shade_range[0]), float(shade_range[1])
            # sombreado suave, color neutro para no competir con curvas
            ax.axvspan(x_lo, x_hi, color='0.18', alpha=0.14, zorder=0)
        except Exception:
            pass

    # compact y-limits
    ymin = min(np.min(data["y_smooth"]) for data in list_of_data)
    ymax = max(np.max(data["y_smooth"]) for data in list_of_data)
    pad = 0.12 * (ymax - ymin)  # menos padding para maximizar espacio
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title("F1 vs Number of Samples", fontsize=int(BASE_FONTSIZE*1.25), pad=6)
    ax.set_xlabel("Number of Samples (Pairs)")
    ax.set_ylabel("F1 score")
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5)

    # legend compact: dentro a la derecha superior (ocupa menos espacio)
    leg = ax.legend(frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(0.3, 0.99))
    # hacer las líneas de la leyenda mucho más gruesas para destacar
        # compatibilidad con distintas versiones de matplotlib
    handles = getattr(leg, "legend_handles", None)
    if handles is None:
        handles = getattr(leg, "legendHandles", [])

    for lh in handles:
        try:
            lh.set_linewidth(10.0)
        except Exception:
            try:
                # si es un marker, agrandar tamaño/visibilidad
                lh.set_markersize(10)
            except Exception:
                pass

    # remove extra whitespace around plot
    if square:
        # intentar fijar aspecto cuadrado del eje para evitar barras laterales
        try:
            ax.set_box_aspect(1.0)
        except Exception:
            pass

    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def plot_sigmoid_only(list_of_data, outdir, outname="all_sigmoid_only.png"):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout=False)
    for i, data in enumerate(list_of_data):
        ax.plot(data["x_smooth"], data["y_smooth"], color=colors[i], lw=2.6, label=data["name"])
    ax.set_title("Sigmoid trends (clean)", fontsize=int(BASE_FONTSIZE*1.2))
    ax.set_xlabel("Number of Samples (Pairs)")
    ax.set_ylabel("F1 score")
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def plot_logx(list_of_data, outdir, outname="all_sigmoid_logx.png"):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout=False)
    for i, data in enumerate(list_of_data):
        mask = data["x_smooth"] > 0
        ax.plot(data["x_smooth"][mask], data["y_smooth"][mask], color=colors[i], lw=2.6, label=data["name"])
    ax.set_xscale('log')
    ax.set_title("Sigmoid trends (log x-axis)", fontsize=int(BASE_FONTSIZE*1.2))
    ax.set_xlabel("Number of Samples (Pairs) [log scale]")
    ax.set_ylabel("F1 score")
    ax.grid(True, which="both", linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def plot_shaded_intervals(list_of_data, outdir, outname="all_sigmoid_shaded_optimal_intervals.png", factor=2.0):
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

    ax.set_title(f"Optimal [N, 2N] intervals (factor={factor})", fontsize=int(BASE_FONTSIZE*1.1))
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("F1 score")
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    out_path = os.path.join(outdir, outname)
    return _finalize_and_save(fig, out_path)

def main():
    parser = argparse.ArgumentParser(description="Plot multiple F1 vs samples CSVs with sigmoid smoothing and optional std display")
    parser.add_argument("--indir", required=True, help="Directory containing CSVs")
    parser.add_argument("--pattern", default="*.f1_by_samples.csv", help="Pattern to match CSV files")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.06)
    parser.add_argument("--blend", type=float, default=0.8)
    parser.add_argument("--remove-outliers", action="store_true", help="If set, extreme outliers (beyond tolerance) won't be plotted")
    parser.add_argument("--no-std", action="store_true", help="If set, do not plot f1_std (no errorbars / shadow)")
    parser.add_argument("--no-raw", action="store_true", help="If set, hide original raw points (only show sigmoid+nudged)")
    parser.add_argument("--exclude-langs", default="", help="Comma-separated language codes to exclude (e.g. th,ja)")
    parser.add_argument("--shade-range", default="", help="Shade vertical interval as 'lo,hi' (e.g. 60,120)")
    parser.add_argument("--square", action="store_true", help="If set, make the combined plot square")    
    args = parser.parse_args()

    show_std = not args.no_std
    show_raw = not args.no_raw

    os.makedirs(args.outdir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not csv_files:
        raise ValueError(f"No CSVs found in {args.indir} matching {args.pattern}")

    list_of_data = []
    for csv_file in csv_files:
        d = process_csv(csv_file, args.max_samples, args.tolerance, args.blend)
        N, N2, slope = compute_opt_interval(d["x_smooth"], d["y_smooth"], factor=2.0)
        d["opt_N"], d["opt_2N"], d["opt_slope"] = N, N2, slope

        if args.remove_outliers:
            mask_raw_keep = np.isin(d["x_raw"], d["x_plot"])
            d["x_raw"] = d["x_raw"][mask_raw_keep]
            d["y_raw"] = d["y_raw"][mask_raw_keep]
            if d["y_std_raw"] is not None:
                d["y_std_raw"] = d["y_std_raw"][mask_raw_keep]
        list_of_data.append(d)

        out_ind = plot_individual(d, args.outdir, show_std=show_std, show_raw=show_raw)
        print(f"✅ Individual plot saved: {out_ind}")

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

    # parse shade range
    shade_range = None
    if args.shade_range:
        try:
            lo_str, hi_str = [s.strip() for s in args.shade_range.split(",")]
            shade_range = (float(lo_str), float(hi_str))
        except Exception:
            print("⚠️ --shade-range ignored: formato inválido (usa 'lo,hi')")

    # Alternative combined (filtered + shaded + optional square)
    out_all_alt = plot_combined(filtered_list_of_data, args.outdir, show_std=show_std, show_raw=show_raw,
                                outname="all_csvs_combined_alt.png", shade_range=shade_range, square=args.square)
    print(f"✅ Combined (alt) plot saved: {out_all_alt}")

    # Keep also the original combined (full set) with compact layout
    out_all = plot_combined(list_of_data, args.outdir, show_std=show_std, show_raw=show_raw,
                            outname="all_csvs_combined.png", shade_range=None, square=False)
    print(f"✅ Combined plot saved: {out_all}")

    out_sig = plot_sigmoid_only(list_of_data, args.outdir)
    print(f"✅ Sigmoid-only plot saved: {out_sig}")

    out_log = plot_logx(list_of_data, args.outdir)
    print(f"✅ Log-x plot saved: {out_log}")

    out_shaded = plot_shaded_intervals(list_of_data, args.outdir)
    print(f"✅ Shaded-intervals plot saved: {out_shaded}")

if __name__ == "__main__":
    main()
