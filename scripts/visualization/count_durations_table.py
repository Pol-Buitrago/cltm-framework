#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

def seconds_to_hms(seconds):
    """Convierte segundos (float) a H:MM:SS (sin ceros a la izquierda en horas)."""
    if pd.isna(seconds):
        return ""
    s = int(round(float(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"

def compute_durations_from_tsvs(root_path: Path, csv_path: Path):
    """
    Recorre los TSVs en root_path y genera DataFrame con columnas:
    language, subset, num_files, total_duration_s, mean_duration_s
    """
    subsets = ["train", "dev", "test"]
    langs = sorted({f.name.split(".")[0] for f in root_path.glob("*.tsv")})
    results = []
    for lang in langs:
        for subset in subsets:
            tsv_path = root_path / f"{lang}.{subset}.tsv"
            if not tsv_path.exists():
                continue
            try:
                df = pd.read_csv(tsv_path, sep="\t", dtype=str)
            except Exception as e:
                print(f"Warning: could not read {tsv_path}: {e}")
                continue

            durations = []
            for audio_path in df.get("audio", []):
                try:
                    data, sr = sf.read(audio_path)
                    durations.append(len(data) / sr)
                except Exception as e:
                    print(f"Warning: error reading {audio_path}: {e}")
                    continue

            total_dur = sum(durations)
            mean_dur = (total_dur / len(durations)) if durations else 0.0
            results.append({
                "language": lang,
                "subset": subset,
                "num_files": len(durations),
                "total_duration_s": float(total_dur),
                "mean_duration_s": float(mean_dur)
            })
    df_results = pd.DataFrame(results, columns=["language", "subset", "num_files", "total_duration_s", "mean_duration_s"])
    df_results.to_csv(csv_path, index=False)
    return df_results

def render_table_image(df: pd.DataFrame, img_path: Path, title: str = None):
    """
    Dibuja una tabla 'bonita' con pandas/matplotlib a partir de df.
    Se espera que df contenga al menos las columnas:
    language, subset, num_files, total_duration_s, mean_duration_s
    """
    # Preparar df de visualización: convertir segundos a H:MM:SS en columnas nuevas
    disp = df.copy()
    if 'total_duration_s' in disp.columns:
        disp['total_duration'] = disp['total_duration_s'].apply(seconds_to_hms)
    if 'mean_duration_s' in disp.columns:
        disp['mean_duration'] = disp['mean_duration_s'].apply(seconds_to_hms)

    # Reordenar/seleccionar columnas para mostrar (preferible formato humano)
    cols_show = []
    for c in ["language", "subset", "num_files"]:
        if c in disp.columns:
            cols_show.append(c)
    # preferimos mostrar duraciones en H:MM:SS
    if 'total_duration' in disp.columns:
        cols_show.append('total_duration')
    elif 'total_duration_s' in disp.columns:
        cols_show.append('total_duration_s')
    if 'mean_duration' in disp.columns:
        cols_show.append('mean_duration')
    elif 'mean_duration_s' in disp.columns:
        cols_show.append('mean_duration_s')

    disp_show = disp[cols_show].astype(str)

    # Diseño de la figura (ancho = función del nº columnas, alto = función del nº filas)
    n_rows, n_cols = disp_show.shape
    col_width = 0  # ancho por columna
    row_height = 0  # altura por fila
    fig_w = max(8, n_cols * col_width)
    fig_h = max(2.5, (n_rows + 2) * row_height)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    # Cabecera
    header_bg = "#2c3e50"  # azul oscuro
    header_text_color = "white"

    # Zebra rows + highlight train
    even_bg = "#ffffff"
    odd_bg = "#f9f9f9"
    train_bg = "#e6f2ff"  # sutil azul para filas train

    # Construir cellColours 2D: primera fila = header
    cell_colours = []
    header_row = [header_bg] * n_cols
    cell_colours.append(header_row)
    for i, subset in enumerate(disp_show['subset']):
        if subset.lower() == "train":
            row_bg = train_bg
        else:
            row_bg = odd_bg if i % 2 else even_bg
        cell_colours.append([row_bg] * n_cols)

    # Crear tabla: concatenamos header + data
    table_data = [list(disp_show.columns)]
    table_data.extend(disp_show.values.tolist())

    the_table = ax.table(cellText=table_data,
                         cellColours=cell_colours,
                         colLabels=None,
                         cellLoc='center',
                         loc='center')

    # Estilizar celdas
    the_table.auto_set_font_size(False)
    # ajustar tamaño de fuente según filas/cols
    base_font = 10
    if n_rows > 20:
        base_font = max(6, int(10 - (n_rows - 20) * 0.08))
    the_table.set_fontsize(base_font)

    # Recorremos celdas para mejorar estilo: header, cuerpo, bordes
    for (row, col), cell in the_table.get_celld().items():
        # row 0 es header
        if row == 0:
            cell.set_text_props(weight='bold', color=header_text_color)
            cell.set_facecolor(header_bg)
        else:
            # body
            bg = odd_bg if (row - 1) % 2 else even_bg
            cell.set_facecolor(bg)
            cell.set_text_props(color='black')
        # remover bordes pesados: usar borde muy ligero
        cell.set_edgecolor('#e6e6e6')
        cell.set_linewidth(0.4)

    # Ajustes finos: anchos de columna proporcional al contenido
    # la API table no dispone de setting directo de anchuras robusto,
    # hacemos un intento con set_column_width (puede que dependa de backend)
    # No forzamos demasiado para evitar layout roto.
    the_table.auto_set_column_width(col=list(range(n_cols)))

    # Añadir título si se pide
    if title:
        plt.suptitle(title, fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def process_dataset(root_path: Path, out_folder: Path, dataset_name: str, force_recompute: bool = False):
    """
    Si existe CSV -> cargar y solo generar imagen.
    Si no existe o force_recompute=True -> computar CSV (leyendo audios), guardarlo y generar imagen.
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    csv_path = out_folder / f"{dataset_name}_durations_summary.csv"
    img_path = out_folder / f"{dataset_name}_durations_summary.png"

    if csv_path.exists() and not force_recompute:
        print(f"{csv_path} exists, skipping audio computation and using CSV.")
        df_results = pd.read_csv(csv_path)
    else:
        print(f"{csv_path} not found or force_recompute=True, computing durations from TSVs...")
        df_results = compute_durations_from_tsvs(root_path, csv_path)

    # Si el CSV contiene columnas *_s (segundos), mejor conservarlas y renderizar H:MM:SS
    # Aseguramos que las columnas numéricas estén en formato correcto
    for col in ['total_duration_s', 'mean_duration_s']:
        if col in df_results.columns:
            df_results[col] = pd.to_numeric(df_results[col], errors='coerce').fillna(0.0)

    # Render image
    render_table_image(df_results, img_path)

# Paths (ejemplo)
root_age = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/cv-corpus-21.0-2025-03-14__age_id/04_equal_langs/tsv")
root_gender = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/cv-corpus-21.0-2025-03-14__gender_id/04_equal_langs/tsv")
tables_folder = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/scripts/visualization/tables")

# Ejecutar
process_dataset(root_age, tables_folder, "age")
process_dataset(root_gender, tables_folder, "gender")
