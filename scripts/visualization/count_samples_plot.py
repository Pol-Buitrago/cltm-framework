#!/usr/bin/env python3
"""
count_samples_plot.py

Cuenta muestras en TSVs tipo `lang <tab> audio <tab> gender <tab> client_id`
ubicados en una carpeta donde los ficheros tienen patrón '*.*.tsv' (ej. bg.train.tsv)
y genera:
  - una gráfica de barras horizontales del número de muestras por idioma
  - opcionalmente, una gráfica apilada por split (train/dev/test)
  - opcionalmente, una gráfica apilada por categorías de la columna "gender"

Uso:
    python count_samples_plot.py \
      --input-dir "/path/to/tsv" \
      --out "figures/counts_by_lang.png" \
      --by-split \
      --by-gender

Opciones:
    --by-split       -> generar gráfico apilado por split además del total
    --by-gender      -> generar gráfico apilado por gender además del total
    --show           -> mostrar la figura en pantalla (si hay entorno gráfico)
    --pattern        -> patrón glob para localizar TSVs, por defecto "*.*.tsv"
    --output-dir     -> forzar directorio de salida para todas las figuras
    --out            -> ruta de salida para la figura total (archivo .png/.svg/.pdf)
    --out-split      -> ruta explícita para la figura por split, opcional
    --out-gender     -> ruta explícita para la figura por gender, opcional
"""
import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 14,          # tamaño global de fuente
    "axes.titlesize": 14,     # tamaño del título
    "axes.labelsize": 14,     # tamaño de etiquetas de ejes
    "xtick.labelsize": 14,    # tamaño ticks eje x
    "ytick.labelsize": 14,    # tamaño ticks eje y
    "legend.fontsize": 14,    # tamaño leyenda
})


import os
import sys
import argparse
import glob
import csv
from collections import defaultdict, Counter

csv.field_size_limit(sys.maxsize)

import matplotlib.pyplot as plt

def ensure_dir_exists(path):
    """Crear el directorio de salida si no existe."""
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def find_tsv_files(input_dir, pattern="*.*.tsv"):
    p = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(p))
    return files

def infer_split_from_filename(path):
    base = os.path.basename(path)
    parts = base.split('.')
    if len(parts) >= 3:
        return parts[-2]  # ej. xx.train.tsv -> train
    else:
        return "unknown"

def _normalize_gender(g):
    """Normaliza valor de gender: None/'' -> 'unknown', trim + lowercase para consistencia."""
    if g is None:
        return "unknown"
    g = str(g).strip()
    if not g:
        return "unknown"
    return g.lower()

def _normalize_age(a):
    """Normaliza valor de age: None/'' -> 'unknown', trim + lowercase."""
    if a is None:
        return "unknown"
    a = str(a).strip().lower()
    if not a:
        return "unknown"
    return a

def _file_prefix(path):
    """Extrae el prefijo de fichero esperado, por ejemplo:
       /.../ab.train.tsv -> 'ab' (parte antes del primer punto en el nombre base)"""
    base = os.path.basename(path)
    # si el nombre tiene formato <prefix>.<split>.tsv, coger prefix
    parts = base.split('.')
    return parts[0] if parts else base

def build_prefix_lang_map(files, sample_limit_per_file=10000):
    """
    Escanea los ficheros que sí contienen una columna 'lang' y construye
    mapping prefix -> most_common_lang.
    sample_limit_per_file: número máximo de filas a leer por fichero para muestrear.
    """
    from collections import Counter
    prefix_lang_counts = defaultdict(Counter)
    for f in files:
        try:
            with open(f, 'r', newline='') as fh:
                reader = csv.reader(fh, delimiter='\t')
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                header_lower = [h.strip().lower() for h in header]
                # asumimos que si existe 'lang' en header, la columna 0 puede ser otra cosa,
                # pero el usuario indica que en train la lang está en la primera columna.
                # Buscamos explícitamente columna 'lang' si existe
                lang_idx = None
                if 'lang' in header_lower:
                    lang_idx = header_lower.index('lang')
                # si no hay header 'lang', tratamos la primera columna como lang candidate
                use_first_col = lang_idx is None
                count = 0
                for row in reader:
                    if not row:
                        continue
                    if use_first_col:
                        candidate = row[0].strip() if len(row) >= 1 else ""
                    else:
                        candidate = row[lang_idx].strip() if len(row) > lang_idx else ""
                    if candidate:
                        prefix = _file_prefix(f)
                        prefix_lang_counts[prefix][candidate] += 1
                    count += 1
                    if count >= sample_limit_per_file:
                        break
        except Exception:
            # no queremos que un fichero corrupto rompa todo el muestreo
            continue
    # convertir a mapa prefix -> most common language (si existe)
    prefix_lang_map = {}
    for prefix, counter in prefix_lang_counts.items():
        if not counter:
            continue
        most_common_lang, freq = counter.most_common(1)[0]
        # si hay ambigüedad (por ejemplo el top2 tiene frecuencia cercana), dejamos la elección,
        # pero de momento asignamos el más común. Se podría añadir un umbral si se desea.
        prefix_lang_map[prefix] = most_common_lang
    return prefix_lang_map

def count_samples(files):
    """
    Recorre ficheros TSV y devuelve:
      - total_by_lang: Counter(lang -> total)
      - by_lang_split: dict(lang -> Counter(split -> count))
      - by_lang_gender: dict(lang -> Counter(gender -> count))
      - by_lang_age: dict(lang -> Counter(age -> count))
      - file_count: número de ficheros procesados

    Para ficheros que no contienen columna 'lang', intenta asignar lang usando
    el mapa prefix->lang construido a partir de ficheros que sí contienen lang
    (p. ej. los .train.tsv).
    """
    total_by_lang = Counter()
    by_lang_split = defaultdict(Counter)
    by_lang_gender = defaultdict(Counter)
    by_lang_age = defaultdict(Counter)
    file_count = 0

    # construir mapa prefix -> lang a partir de ficheros con lang conocido
    prefix_lang_map = build_prefix_lang_map(files)

    for f in files:
        try:
            with open(f, 'r', newline='') as fh:
                reader = csv.reader(fh, delimiter='\t')
                try:
                    header = next(reader)
                except StopIteration:
                    # fichero vacío
                    continue
                # detectar índices si el header los contiene
                gender_idx = None
                age_idx = None
                set_lang_idx = None
                header_lower = [h.strip().lower() for h in header]
                if 'gender' in header_lower:
                    gender_idx = header_lower.index('gender')
                if 'age' in header_lower:
                    age_idx = header_lower.index('age')
                elif 'age_group' in header_lower:
                    age_idx = header_lower.index('age_group')
                elif 'agegroup' in header_lower:
                    age_idx = header_lower.index('agegroup')
                if 'lang' in header_lower:
                    set_lang_idx = header_lower.index('lang')
                split = infer_split_from_filename(f)
                prefix = _file_prefix(f)
                fallback_lang_for_prefix = prefix_lang_map.get(prefix)

                for row in reader:
                    if not row:
                        continue

                    # intentar obtener lang desde columna explícita si existe
                    lang = None
                    if set_lang_idx is not None and len(row) > set_lang_idx:
                        lang = row[set_lang_idx].strip()
                    # si no hay columna lang, asumimos que la primera columna podría ser lang,
                    # pero sólo si parece un código no vacío
                    if not lang:
                        if len(row) >= 1 and row[0].strip():
                            # Si header no tenía 'lang', pero la primera columna contiene algo
                            # puede que sea el idioma en train. Sin embargo, en test la primera col
                            # puede ser otra cosa; para evitar errores, priorizamos fallback map.
                            # Aquí: si prefix tiene mapping, usarlo; si no, usar row[0].
                            if fallback_lang_for_prefix:
                                lang = fallback_lang_for_prefix
                            else:
                                lang = row[0].strip()
                    if not lang:
                        # si todavía no hay lang, marcar como unknown
                        lang = "unknown"

                    lang = lang.strip()
                    if not lang:
                        continue

                    total_by_lang[lang] += 1
                    by_lang_split[lang][split] += 1

                    # extraer gender si existe esa columna en la fila
                    gender = None
                    if gender_idx is not None and len(row) > gender_idx:
                        gender = row[gender_idx]
                    gender = _normalize_gender(gender)
                    by_lang_gender[lang][gender] += 1

                    # extraer age si existe esa columna en la fila
                    if age_idx is not None and len(row) > age_idx:
                        age = _normalize_age(row[age_idx])
                        by_lang_age[lang][age] += 1
                file_count += 1
        except Exception as e:
            print(f"Warning: could not process '{f}': {e}", file=sys.stderr)
    return total_by_lang, by_lang_split, by_lang_gender, by_lang_age, file_count

def plot_horizontal_counts(total_by_lang, out_path="counts_by_lang.png", show=False, title=None):
    ensure_dir_exists(out_path)
    items = total_by_lang.most_common()
    if not items:
        raise ValueError("No data to plot")
    langs, counts = zip(*items)
    n = len(langs)
    # más apaisado: ancho mayor, altura proporcional al número de idiomas
    width = 14
    height = max(3.5, 0.25 * n)
    fig, ax = plt.subplots(figsize=(width, height))
    y_positions = range(len(langs))
    ax.barh(y_positions, counts)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(langs)
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    if title:
        ax.set_title(title)
    # anotar valores al final de cada barra
    max_count = max(counts)
    for i, v in enumerate(counts):
        ax.text(v + max_count * 0.004, i, str(v), va='center')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved figure: {out_path}")

def plot_stacked_by_split(by_lang_split, out_path="counts_by_lang_by_split.png", show=False, title=None):
    ensure_dir_exists(out_path)
    import numpy as np
    langs = sorted(by_lang_split.keys(), key=lambda l: sum(by_lang_split[l].values()), reverse=True)
    if not langs:
        raise ValueError("No data to plot for splits")
    all_splits = set()
    for c in by_lang_split.values():
        all_splits.update(c.keys())
    preferred = ["train", "dev", "test"]
    other_splits = sorted(s for s in all_splits if s not in preferred)
    splits = [s for s in preferred if s in all_splits] + other_splits
    matrix = np.array([[by_lang_split[l].get(s, 0) for s in splits] for l in langs])
    n = len(langs)
    width = 16
    height = max(3.5, 0.25 * n)
    fig, ax = plt.subplots(figsize=(width, height))
    y = range(n)
    left = np.zeros(n, dtype=int)
    for i, s in enumerate(splits):
        vals = matrix[:, i]
        ax.barh(y, vals, left=left, label=s)
        left = left + vals
    ax.set_yticks(list(y))
    ax.set_yticklabels(langs)
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    if title:
        ax.set_title(title)
    ax.legend(title="split", bbox_to_anchor=(1.02, 1), loc='upper left')
    totals = matrix.sum(axis=1)
    max_total = totals.max() if len(totals) > 0 else 0
    for i, total in enumerate(totals):
        ax.text(total + max_total * 0.004, i, str(int(total)), va='center')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved stacked figure: {out_path}")

def plot_stacked_by_gender(by_lang_gender, out_path="counts_by_lang_by_gender.png", show=False, title=None):
    """
    Gráfico apilado por categorías de gender:
      - male_masculine
      - female_feminine
      - others (cualquier otro valor)
    Colores definidos.
    """
    ensure_dir_exists(out_path)
    import numpy as np

    langs = sorted(by_lang_gender.keys(), key=lambda l: sum(by_lang_gender[l].values()), reverse=True)
    if not langs:
        raise ValueError("No data to plot for gender")

    # Categorías principales
    main_genders = ["male_masculine", "female_feminine"]
    # Detectar si existen en los datos
    existing_main = [g for g in main_genders if any(g in by_lang_gender[l] for l in langs)]
    
    # Determinar si hay otras categorías
    others_exist = any(
        any(g not in main_genders for g in by_lang_gender[l].keys())
        for l in langs
    )
    
    # Construir lista final de columnas
    genders = existing_main.copy()
    if others_exist:
        genders.append("others")

    # Construir la matriz
    matrix = []
    for l in langs:
        row = []
        for g in existing_main:
            row.append(by_lang_gender[l].get(g, 0))
        # Sumar todas las demás categorías en 'others'
        if others_exist:
            others_count = sum(count for cat, count in by_lang_gender[l].items() if cat not in main_genders)
            row.append(others_count)
        matrix.append(row)
    matrix = np.array(matrix)

    # Colores fijos
    gender_colors = {
        "male_masculine": "#1f77b4",   # azul
        "female_feminine": "#ff69b4",  # rosa
        "others": "#7f7f7f"             # gris
    }

    n = len(langs)
    width = 16
    height = max(3.5, 0.25 * n)
    fig, ax = plt.subplots(figsize=(width, height))
    y = range(n)
    left = np.zeros(n, dtype=int)

    for i, g in enumerate(genders):
        vals = matrix[:, i]
        color = gender_colors.get(g, "#7f7f7f")
        ax.barh(y, vals, left=left, label=g, color=color)
        left = left + vals

    ax.set_yticks(list(y))
    ax.set_yticklabels(langs)
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    if title:
        ax.set_title(title)
    ax.legend(title="gender", bbox_to_anchor=(1.02, 1), loc='upper left')

    totals = matrix.sum(axis=1)
    max_total = totals.max() if len(totals) > 0 else 0
    for i, total in enumerate(totals):
        ax.text(total + max_total * 0.004, i, str(int(total)), va='center')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved gender-stacked figure: {out_path}")

def plot_stacked_by_age(by_lang_age, out_path="counts_by_lang_by_age.png", show=False, title=None):
    """
    Gráfico apilado por categorías de edad:
      - young
      - adult
      - senior
    Si aparecen otras categorías en los TSV, se añaden como columnas separadas y se les asigna
    un color (determinístico, usando tab20).
    """
    ensure_dir_exists(out_path)
    import numpy as np
    from matplotlib import colors as mcolors

    langs = sorted(by_lang_age.keys(), key=lambda l: sum(by_lang_age[l].values()), reverse=True)
    if not langs:
        raise ValueError("No data to plot for age")

    # Categorías principales
    main_ages = ["young", "adult", "senior"]
    # Recopilar todas las categorías presentes
    all_age_cats = set()
    for c in by_lang_age.values():
        all_age_cats.update(c.keys())

    # existing main categories (si están presentes)
    existing_main = [a for a in main_ages if any(a in by_lang_age[l] for l in langs)]

    # otras categorías aparte de main (mantener como columnas separadas)
    other_ages = sorted(cat for cat in all_age_cats if cat not in main_ages)

    # construir lista final de edades: main presentes primero, luego otras (excluyendo 'unknown' si se desea incluir)
    ages = existing_main + [a for a in other_ages if a not in existing_main]

    # Construir la matriz (rows = langs, cols = ages)
    matrix = np.array([[by_lang_age[l].get(a, 0) for a in ages] for l in langs])

    # Asignar colores: fijo para main ages, para otras usar tab20 pallete
    age_colors = {}
    fixed_colors = {
        "young": "#2ca02c",   # green
        "adult": "#9467bd",   # purple
        "senior": "#d62728"   # red
    }
    for a in ages:
        if a in fixed_colors:
            age_colors[a] = fixed_colors[a]
    # Paleta para otras (determinística)
    cmap = plt.get_cmap('tab20')
    j = 0
    for a in ages:
        if a not in age_colors:
            age_colors[a] = mcolors.to_hex(cmap(j % 20))
            j += 1

    n = len(langs)
    width = 16
    height = max(3.5, 0.25 * n)
    fig, ax = plt.subplots(figsize=(width, height))
    y = range(n)
    left = np.zeros(n, dtype=int)

    for i, a in enumerate(ages):
        vals = matrix[:, i]
        color = age_colors.get(a)
        ax.barh(y, vals, left=left, label=a, color=color)
        left = left + vals

    ax.set_yticks(list(y))
    ax.set_yticklabels(langs)
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    if title:
        ax.set_title(title)
    ax.legend(title="age", bbox_to_anchor=(1.02, 1), loc='upper left')

    totals = matrix.sum(axis=1)
    max_total = totals.max() if len(totals) > 0 else 0
    for i, total in enumerate(totals):
        ax.text(total + max_total * 0.004, i, str(int(total)), va='center')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved age-stacked figure: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Count samples in TSVs and plot horizontal bars per language")
    parser.add_argument('--input-dir', '-i', required=True, help='Directory with TSV files')
    parser.add_argument('--pattern', default='*.*.tsv', help='Glob pattern to find TSVs (default "*.*.tsv")')
    parser.add_argument('--out', '-o', default='counts_by_lang.png', help='Output path for the total figure (file)')
    parser.add_argument('--out-split', default=None, help='Explicit output path for split figure, optional')
    parser.add_argument('--out-gender', default=None, help='Explicit output path for gender figure, optional')
    parser.add_argument('--out-age', default=None, help='Explicit output path for age figure, optional')
    parser.add_argument('--output-dir', default=None, help='Force output directory for all figures, optional')
    parser.add_argument('--by-split', action='store_true', help='Generate stacked plot by split in addition to totals')
    parser.add_argument('--by-gender', action='store_true', help='Generate stacked plot by gender in addition to totals')
    parser.add_argument('--by-age', action='store_true', help='Generate stacked plot by age in addition to totals')
    parser.add_argument('--show', action='store_true', help='Show figures on screen if graphical environment available')
    args = parser.parse_args()

    files = find_tsv_files(args.input_dir, pattern=args.pattern)
    if not files:
        print(f"No files found with pattern {args.pattern} in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} files...")
    total_by_lang, by_lang_split, by_lang_gender, by_lang_age, file_count = count_samples(files)
    total_samples = sum(total_by_lang.values())
    print(f"Files processed: {file_count}")
    print(f"Total samples counted: {total_samples}")

    # Determinar rutas de salida
    out_main = args.out
    if args.output_dir:
        # forzar directorio de salida
        os.makedirs(args.output_dir, exist_ok=True)
        base_main = os.path.basename(out_main)
        out_main = os.path.join(args.output_dir, base_main)

    # out_split
    if args.out_split:
        out_split = args.out_split
    else:
        out_dir_of_main = os.path.dirname(os.path.abspath(out_main)) or os.getcwd()
        base_split = "counts_by_lang_by_split" + os.path.splitext(out_main)[1]
        out_split = os.path.join(out_dir_of_main, base_split)

    # out_gender
    if args.out_gender:
        out_gender = args.out_gender
    else:
        out_dir_of_main = os.path.dirname(os.path.abspath(out_main)) or os.getcwd()
        base_gender = "counts_by_lang_by_gender" + os.path.splitext(out_main)[1]
        out_gender = os.path.join(out_dir_of_main, base_gender)

    # out_age
    if args.out_age:
        out_age = args.out_age
    else:
        out_dir_of_main = os.path.dirname(os.path.abspath(out_main)) or os.getcwd()
        base_age = "counts_by_lang_by_age" + os.path.splitext(out_main)[1]
        out_age = os.path.join(out_dir_of_main, base_age)

    # Títulos y labels en inglés
    title_main = f"Samples per language (total {total_samples:,d} samples)"
    title_split = "Samples per language and split (stacked)"
    title_gender = "Samples per language and gender (stacked)"
    title_age = "Samples per language and age (stacked)"

    plot_horizontal_counts(total_by_lang, out_path=out_main, show=args.show, title=title_main)
    if args.by_split:
        plot_stacked_by_split(by_lang_split, out_path=out_split, show=args.show, title=title_split)
    if args.by_gender:
        plot_stacked_by_gender(by_lang_gender, out_path=out_gender, show=args.show, title=title_gender)
    if args.by_age:
        # comprobación mínima: si no hay datos de age, avisar
        if not any(by_lang_age.values()):
            print("Warning: no 'age' data found in TSVs (no 'age' or 'age_group' header detected). Skipping age plot.", file=sys.stderr)
        else:
            plot_stacked_by_age(by_lang_age, out_path=out_age, show=args.show, title=title_age)

if __name__ == "__main__":
    main()


"""

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20/01_preprocessed/tsv" \
  --out "plots/gender/22.0/01_preprocessed/counts_by_lang.png" \
  --by-split \
  --by-gender

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/02_filtered_cv_gender/tsv" \
  --out "plots/gender/22.0/02_filtered_cv_gender/counts_by_lang.png" \
  --by-split \
  --by-gender

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/03_balanced_cv_gender/tsv" \
  --out "plots/gender/22.0/03_balanced_cv_gender/counts_by_lang.png" \
  --by-split \
  --by-gender

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/04_balanced_60_cv_gender/tsv" \
  --out "plots/gender/22.0/04_balanced_60/counts_by_lang.png" \
  --by-split \
  --by-gender \
  --pattern "*.train.tsv"

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/05_balanced_120_cv_gender/tsv" \
  --out "plots/gender/22.0/05_balanced_120/counts_by_lang.png" \
  --by-split \
  --by-gender \
  --pattern "*.train.tsv"

----------------------------

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__asr/01_preprocessed/tsv" \
  --out "plots/asr/22.0/01_preprocessed/counts_by_lang.png" \
  --by-split

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__asr/02_subsets_balanced_N/tsv" \
  --out "plots/asr/22.0/02_subsets_balanced_N/counts_by_lang.png" \
  --by-split

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__asr/02_subsets_balanced_2N/tsv" \
  --out "plots/asr/22.0/02_subsets_balanced_2N/counts_by_lang.png" \
  --by-split

----------------------------

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/01_preprocessed/tsv" \
  --out "plots/speaker/22.0/01_preprocessed/counts_by_lang.png" \
  --by-split

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/02_filtered_cv_speaker/tsv" \
  --out "plots/speaker/22.0/02_filtered_cv_speaker/counts_by_lang.png" \
  --by-split

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv" \
  --out "plots/speaker/22.0/03_combined_train/counts_by_lang.png" \
  --by-split

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv" \
  --out "plots/speaker/22.0/05_balanced_1000/counts_by_lang.png" \
  --by-split
python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv" \
  --out "plots/speaker/22.0/06_balanced_2000/counts_by_lang.png" \
  --by-split

----------------------------

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20/01_preprocessed/tsv" \
  --out "plots/age/22.0/01_preprocessed/counts_by_lang.png" \
  --by-split \
  --by-age

python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__age_id/02_filtered_cv_age/tsv" \
  --out "plots/age/22.0/02_filtered_cv_age/counts_by_lang.png" \
  --by-split \
  --by-age

//python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/cv-corpus-21.0-2025-03-14__age_id/03_balanced_cv_age/tsv" \
  --out "plots/age/03_balanced_cv_age/counts_by_lang.png" \
  --by-split \
  --by-age

//python count_samples_plot.py \
  --input-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20/01_preprocessed/tsv" \
  --out "plots/age/04_equal_langs/counts_by_lang.png" \
  --by-split \
  --by-age

"""