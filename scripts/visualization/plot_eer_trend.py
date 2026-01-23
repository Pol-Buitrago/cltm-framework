#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({"figure.dpi": 200})

###############################################################
#   SIGMOID INVERTIDA (DECRECIENTE) PARA EER
###############################################################
def inv_sigmoid(x, a, b, c, d):
    """
    Función logística decreciente:
    a  = valor asintótico mínimo (EER más bajo)
    b  = amplitud (negativa)
    c  = pendiente
    d  = desplazamiento horizontal
    """
    return a + b / (1 + np.exp(c * (x - d)))  # b < 0 produce curva decreciente


###############################################################
#   LECTURA Y PROCESAMIENTO CSV
###############################################################
def process_csv(csv_path, max_samples=None, tolerance=0.06,
                blend=0.8, tolerance_up_multiplier=1.4):
    df = pd.read_csv(csv_path)
    if not {'eer', 'num_samples'}.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} debe contener columnas: eer, num_samples")

    cols = ['num_samples', 'eer']
    if 'eer_std' in df.columns:
        cols.append('eer_std')

    df = df[cols].dropna().sort_values('num_samples').reset_index(drop=True)

    # limitar muestras
    if max_samples is not None:
        df = df[df['num_samples'] <= max_samples]

    # eliminar outliers extremadamente altos (EER debería descender)
    eer0 = df['eer'].iloc[0]
    df = df[df['eer'] <= eer0 * tolerance_up_multiplier]

    x = df['num_samples'].to_numpy()
    y = df['eer'].to_numpy()
    y_std = df['eer_std'].to_numpy() if 'eer_std' in df.columns else None

    # Ajuste inicial aproximado
    a0 = float(np.min(y))
    b0 = float(np.max(y) - np.min(y)) * -1.0  # negativo → curva decreciente
    c0 = 0.0001
    d0 = float(np.median(x))
    p0 = [a0, b0, c0, d0]

    try:
        popt, _ = curve_fit(inv_sigmoid, x, y, p0=p0, maxfev=5000)
        x_smooth = np.linspace(float(np.min(x)), float(np.max(x)), 600)
        y_smooth = inv_sigmoid(x_smooth, *popt)
        y_fit = inv_sigmoid(x, *popt)
    except Exception as e:
        print(f"⚠️ Fit failed for {csv_path}: {e}. Using raw points.")
        x_smooth, y_smooth = x, y
        y_fit = y

    # Nudging visual
    y_nudged = y_fit + blend * (y - y_fit)

    # tolerancia local
    mask_within = np.abs(y - y_fit) <= tolerance
    x_plot = x[mask_within]
    y_plot = y_nudged[mask_within]
    y_std_plot = y_std[mask_within] if y_std is not None else None

    name = Path(csv_path).name.replace(".eer_by_samples.csv", "")

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


###############################################################
#   COLORES
###############################################################
def distinct_colors(n):
    cmap10 = plt.get_cmap("tab10")
    cmap20 = plt.get_cmap("tab20")
    colors = []
    for i in range(min(4, n)):
        colors.append(cmap20(i))
    if n > 4:
        needed = n - 4
        start = 2
        for i in range(needed):
            colors.append(cmap10((start + i) % 10))
    return colors


###############################################################
#   INTERVALOS [N, 2N]
###############################################################
def compute_opt_interval(x_smooth, y_smooth, factor=2.0, num_candidates=200, min_samples=1):
    x_min = float(np.min(x_smooth))
    x_max = float(np.max(x_smooth))
    if x_max <= factor * x_min or x_max <= min_samples:
        return None, None, None

    lo = max(min_samples, x_min)
    hi = x_max / factor
    candidates = np.linspace(lo, hi, num_candidates)

    dy = np.gradient(y_smooth, x_smooth)

    def interp_dy(v):
        return np.interp(v, x_smooth, dy)

    best_N = None
    best_avg = np.inf  # buscamos pendiente MÁS NEGATIVA → mínimo

    for N in candidates:
        x_left = N
        x_right = factor * N
        mask = (x_smooth >= x_left) & (x_smooth <= x_right)
        if mask.sum() < 3:
            xs = np.linspace(x_left, x_right, 10)
            avg = np.mean(interp_dy(xs))
        else:
            avg = np.mean(dy[mask])
        if avg < best_avg:  # buscamos la caída más pronunciada
            best_avg = avg
            best_N = N

    if best_N is None:
        return None, None, None

    return float(best_N), float(best_N * factor), float(best_avg)


def compute_global_opt_interval(list_of_data, factor=2.0, num_candidates=300, min_samples=1):
    xmins, xmaxs = [], []
    for d in list_of_data:
        if len(d["x_smooth"]) > 0:
            xmins.append(float(np.min(d["x_smooth"])))
            xmaxs.append(float(np.max(d["x_smooth"])))

    if not xmins:
        return None, None, None

    lo = max(min_samples, min(xmins))
    hi = max(xmaxs) / factor
    if hi <= lo:
        return None, None, None

    candidates = np.linspace(lo, hi, num_candidates)

    derivs = []
    for d in list_of_data:
        xs = d["x_smooth"]
        ys = d["y_smooth"]
        if len(xs) < 3:
            derivs.append(None)
        else:
            dy = np.gradient(ys, xs)
            derivs.append((xs, dy))

    best_N = None
    best_mean = np.inf  # más negativo = mejor

    for N in candidates:
        x_left = N
        x_right = factor * N
        slopes = []
        for item in derivs:
            if item is None:
                continue
            xs, dy = item
            if x_right <= xs.min() or x_left >= xs.max():
                continue
            m = (xs >= x_left) & (xs <= x_right)
            if m.sum() < 3:
                xsq = np.linspace(x_left, x_right, 10)
                avg = np.mean(np.interp(xsq, xs, dy))
            else:
                avg = np.mean(dy[m])
            slopes.append(avg)
        if not slopes:
            continue
        mean_slope = float(np.mean(slopes))
        if mean_slope < best_mean:
            best_mean = mean_slope
            best_N = N

    if best_N is None:
        return None, None, None

    return float(best_N), float(best_N * factor), float(best_mean)


###############################################################
#   PLOTS
###############################################################
def plot_individual(d, outdir, show_std=True, show_raw=True, outname=None):
    name = d["name"]
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(d["x_smooth"], d["y_smooth"], color='C0', lw=2.5, label='Trend fit')

    if show_raw:
        if show_std and d["y_std_raw"] is not None:
            ax.errorbar(d["x_raw"], d["y_raw"], yerr=d["y_std_raw"],
                        fmt='o', color='0.6', alpha=0.18, markersize=3, capsize=2)
        else:
            ax.scatter(d["x_raw"], d["y_raw"], color='0.6', alpha=0.12, s=6)

    ax.scatter(d["x_plot"], d["y_plot"], color='C0', alpha=0.45, s=18)

    ax.set_title(f"EER vs Number of Samples ({name})", fontsize=13)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("EER")
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)

    plt.tight_layout()

    if outname is None:
        outname = f"{name}.png"
    out_path = os.path.join(outdir, outname)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_combined(list_of_data, outdir, show_std=True, show_raw=True,
                  outname="all_csvs_combined.png"):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, d in enumerate(list_of_data):
        col = colors[i]
        ax.plot(d["x_smooth"], d["y_smooth"], color=col, lw=2.2, label=d["name"])
        if show_raw:
            if show_std and d["y_std_raw"] is not None:
                ax.errorbar(d["x_raw"], d["y_raw"], yerr=d["y_std_raw"],
                            fmt='o', color=col, alpha=0.09, markersize=3, capsize=1)
            else:
                ax.scatter(d["x_raw"], d["y_raw"], color=col, alpha=0.07, s=6)
        ax.scatter(d["x_plot"], d["y_plot"], color=col, alpha=0.35, s=12)

    ymin = min(np.min(d["y_smooth"]) for d in list_of_data)
    ymax = max(np.max(d["y_smooth"]) for d in list_of_data)
    pad = 0.3 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title("EER vs Number of Samples (Multiple Experiments)", fontsize=14)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("EER")
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(outdir, outname)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_sigmoid_only(list_of_data, outdir, outname="all_trend_only.png"):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, d in enumerate(list_of_data):
        ax.plot(d["x_smooth"], d["y_smooth"], color=colors[i], lw=2.5, label=d["name"])

    ax.set_title("EER Trends (clean)", fontsize=14)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("EER")
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()
    path = os.path.join(outdir, outname)
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def plot_logx(list_of_data, outdir, outname="all_trend_logx.png"):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, d in enumerate(list_of_data):
        m = d["x_smooth"] > 0
        ax.plot(d["x_smooth"][m], d["y_smooth"][m], color=colors[i], lw=2.2, label=d["name"])

    ax.set_xscale("log")
    ax.set_title("EER Trends (log x-axis)", fontsize=14)
    ax.set_xlabel("Number of Samples [log]")
    ax.set_ylabel("EER")
    ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()
    path = os.path.join(outdir, outname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_shaded_intervals(list_of_data, outdir, outname="all_trend_shaded_intervals.png", factor=2.0):
    n = len(list_of_data)
    colors = distinct_colors(n)
    fig, ax = plt.subplots(figsize=(12, 8))

    label_items = []

    # generar curvas y zonas individuales
    for i, d in enumerate(list_of_data):
        col = colors[i]
        ax.plot(d["x_smooth"], d["y_smooth"], color=col, lw=2.2, label=d["name"])

        N, N2, slope = d.get("opt_N"), d.get("opt_2N"), d.get("opt_slope")
        if N is None:
            N, N2, slope = compute_opt_interval(d["x_smooth"], d["y_smooth"], factor=factor)
            d["opt_N"], d["opt_2N"], d["opt_slope"] = N, N2, slope

        if N is not None:
            ax.axvspan(N, N2, color=col, alpha=0.12)
            xmid = (N + N2) / 2
            ymid = float(np.interp(xmid, d["x_smooth"], d["y_smooth"]))

            label_items.append({
                "name": d["name"],
                "color": col,
                "xmid": xmid,
                "ymid": ymid,
                "N": N,
                "slope": slope
            })

    # Etiquetas sin solapamiento
    if label_items:
        label_items.sort(key=lambda z: z["ymid"])
        positions = []
        min_gap = 0.37 * (max(z["ymid"] for z in label_items) -
                          min(z["ymid"] for z in label_items) + 1e-6)

        for z in label_items:
            if not positions:
                z["ypos"] = z["ymid"]
                positions.append(z["ypos"])
            else:
                prev = positions[-1]
                desired = z["ymid"]
                if desired <= prev + min_gap:
                    desired = prev + min_gap
                z["ypos"] = desired
                positions.append(desired)

        for z in label_items:
            ax.text(z["xmid"], z["ypos"],
                    f"{z['name']}: N={int(round(z['N']))}, s={z['slope']:.4f}",
                    color=z["color"],
                    fontsize=8,
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=z["color"], alpha=0.6))

    # global interval
    Ng, N2g, slopeg = compute_global_opt_interval(list_of_data, factor=factor)
    if Ng is not None:
        ax.axvspan(Ng, N2g, color='0.2', alpha=0.15, zorder=0)
        ax.axvline(Ng, color='0.15', lw=1.2, linestyle='--')
        ax.axvline(N2g, color='0.15', lw=1.2, linestyle='--')

        ymax = max(np.max(d["y_smooth"]) for d in list_of_data)
        ymin = min(np.min(d["y_smooth"]) for d in list_of_data)
        ytxt = ymax - 0.04 * (ymax - ymin)

        ax.text((Ng + N2g) / 2, ytxt,
                f"Global N={int(round(Ng))}, slope={slopeg:.4f}",
                color='0.05',
                fontsize=9,
                ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.05", alpha=0.85))

    ax.set_title(f"Optimal [N, 2N] intervals (factor={factor})", fontsize=14)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("EER")
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()
    path = os.path.join(outdir, outname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
   
def slope_between(d, x1, x2):
    # Interpola los valores de y en los extremos del intervalo
    y1 = float(np.interp(x1, d["x_smooth"], d["y_smooth"]))
    y2 = float(np.interp(x2, d["x_smooth"], d["y_smooth"]))
    return (y2 - y1) / (x2 - x1)

###############################################################
#   MAIN
###############################################################
def main():
    parser = argparse.ArgumentParser(
        description="Plot EER vs samples CSVs with decreasing logistic fit"
    )
    parser.add_argument("--indir", required=True)
    parser.add_argument("--pattern", default="*.eer_by_samples.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.06)
    parser.add_argument("--blend", type=float, default=0.8)
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--no-std", action="store_true")
    parser.add_argument("--no-raw", action="store_true")

    args = parser.parse_args()

    show_std = not args.no_std
    show_raw = not args.no_raw

    os.makedirs(args.outdir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(args.indir, args.pattern)))

    if not csv_files:
        raise ValueError(f"No CSVs found matching {args.pattern} in {args.indir}")

    list_of_data = []

    for csv_file in csv_files:
        d = process_csv(csv_file,
                        args.max_samples,
                        args.tolerance,
                        args.blend)

        N, N2, slope = compute_opt_interval(d["x_smooth"], d["y_smooth"], factor=2.0)
        d["opt_N"], d["opt_2N"], d["opt_slope"] = N, N2, slope

        if args.remove_outliers:
            mask = np.isin(d["x_raw"], d["x_plot"])
            d["x_raw"] = d["x_raw"][mask]
            d["y_raw"] = d["y_raw"][mask]
            if d["y_std_raw"] is not None:
                d["y_std_raw"] = d["y_std_raw"][mask]

        list_of_data.append(d)

        path = plot_individual(d, args.outdir,
                               show_std=show_std,
                               show_raw=show_raw)
        print(f"✔ Saved individual: {path}")

    print("Generating combined plots...")

    p1 = plot_combined(list_of_data, args.outdir,
                       show_std=show_std,
                       show_raw=show_raw)
    print(f"✔ Combined: {p1}")

    p2 = plot_sigmoid_only(list_of_data, args.outdir)
    print(f"✔ Trend-only: {p2}")

    p3 = plot_logx(list_of_data, args.outdir)
    print(f"✔ Log-x: {p3}")

    p4 = plot_shaded_intervals(list_of_data, args.outdir)
    print(f"✔ Shaded intervals: {p4}")

    # d es un diccionario retornado por process_csv
    intervals = [(1000, 2000), (1500, 3000), (2000, 4000)]
    for x1, x2 in intervals:
        m = slope_between(d, x1, x2)
        print(f"Slope between {x1} and {x2}: {m:.6f}")



if __name__ == "__main__":
    main()


"""
# SIMPLIFIED GRAPH
python plot_eer_trend.py \
    --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/subsets_csv/speaker/results \
    --outdir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/scripts/visualization/plots/speaker/22.0/subsets_eer \
    --max-samples 120000 \
    --no-std \
    --remove-outliers 
"""