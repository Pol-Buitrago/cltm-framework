#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_results_UI.py
Explorador de resultados CSV orientado a monitorizar 'f1' y filtrar experimentos.
VERSIÓN: siempre obvia la columna 'output_dir' (la elimina al cargar).

Requisitos:
    pip install pandas tabulate

Uso:
    python csv_results_UI.py --file /ruta/a/lr_grid_results.csv
"""
import argparse
import pandas as pd
import shlex
import sys
import textwrap
from tabulate import tabulate
import math
import os
from typing import List

# --- Configurables ---
DEFAULT_FILE = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/grid/gender_id/mhubert-base-25hz/ca_251021_1001/lr_grid_results.csv"
DISPLAY_MAX_ROWS = 20
OBVIATED_COLUMN = "output_dir"   # columna que siempre se obvia/elimina
# ----------------------

OPS = [">=", "<=", "==", "!=", ">", "<", "="]  # '=' tratado como '=='

def load_csv(path: str) -> pd.DataFrame:
    """
    Carga el CSV, normaliza nombres, convierte columnas a numéricas cuando procede,
    y ELIMINA la columna 'output_dir' si está presente (siempre obviada).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_csv(path)
    # Normalize column names: strip whitespaces
    df.columns = [c.strip() for c in df.columns]

    # Eliminar la columna obviada si existe
    if OBVIATED_COLUMN in df.columns:
        # Hacemos una copia por seguridad de lo que vamos a eliminar (no se guarda)
        df = df.drop(columns=[OBVIATED_COLUMN])
        print(f"[info] Columna '{OBVIATED_COLUMN}' detectada y eliminada (siempre obviada).")

    # Try to convert obvious numeric columns
    for col in df.columns:
        # if convertible to numeric for majority of entries -> convert
        conv = pd.to_numeric(df[col], errors="coerce")
        non_na_fraction = conv.notna().mean()
        if non_na_fraction >= 0.5:
            df[col] = conv
    # Ensure 'f1' exists and is numeric
    if 'f1' not in df.columns:
        raise KeyError("El CSV no contiene la columna obligatoria 'f1'.")
    df['f1'] = pd.to_numeric(df['f1'], errors='coerce')
    # Reset index
    df = df.reset_index(drop=True)
    return df

def parse_single_expr(expr: str):
    """
    Parse an expression like "lr>1e-4", "scheduler==cosine", "bs=32", "scheduler!='cosine'".
    Returns (col, op, rhs_string).
    """
    expr = expr.strip()
    for op in OPS:
        if op in expr:
            parts = expr.split(op, 1)
            left = parts[0].strip()
            right = parts[1].strip()
            if op == "=":
                op = "=="
            return left, op, right
    # If no op found, treat as equality (col:value or col:value)
    if ":" in expr:
        left, right = expr.split(":", 1)
        return left.strip(), "==", right.strip()
    raise ValueError(f"Expresión inválida (no operator): '{expr}'")

def make_mask(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    Evalúa una expresión y devuelve una máscara booleana.
    """
    col, op, rhs = parse_single_expr(expr)
    if col == OBVIATED_COLUMN:
        raise KeyError(f"La columna '{OBVIATED_COLUMN}' está obviada y no puede usarse como filtro.")
    if col not in df.columns:
        raise KeyError(f"Columna '{col}' no encontrada en el CSV.")
    series = df[col]
    # limpia comillas del rhs si existen
    if (rhs.startswith("'") and rhs.endswith("'")) or (rhs.startswith('"') and rhs.endswith('"')):
        rhs_clean = rhs[1:-1]
    else:
        rhs_clean = rhs
    # Intentar comparaciones numéricas si la columna ya es numérica
    if pd.api.types.is_numeric_dtype(series):
        # Convertir rhs a número
        try:
            rhs_num = float(rhs_clean)
        except Exception:
            # Si rhs no es numérico -> ningún match
            return pd.Series([False]*len(df), index=df.index)
        if op == "==":
            return series == rhs_num
        elif op == "!=":
            return series != rhs_num
        elif op == ">":
            return series > rhs_num
        elif op == "<":
            return series < rhs_num
        elif op == ">=":
            return series >= rhs_num
        elif op == "<=":
            return series <= rhs_num
        else:
            raise ValueError(f"Operador no soportado: {op}")
    else:
        # Cadena: usamos comparación por igualdad
        s = series.astype(str)
        if op == "==":
            return s == rhs_clean
        elif op == "!=":
            return s != rhs_clean
        elif op in (">", "<", ">=", "<="):
            # No tiene sentido numérico para strings -> devolver False
            return pd.Series([False]*len(df), index=df.index)
        else:
            raise ValueError(f"Operador no soportado para strings: {op}")

def apply_filters(df: pd.DataFrame, filter_exprs: List[str]) -> pd.DataFrame:
    if not filter_exprs:
        return df
    mask = pd.Series([True]*len(df), index=df.index)
    for expr in filter_exprs:
        expr = expr.strip()
        if expr == "":
            continue
        m = make_mask(df, expr)
        mask = mask & m
    return df[mask]

def pretty_print(df: pd.DataFrame, n: int = 10, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    display_df = df.head(n)[columns]
    # Truncate large floats
    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            if abs(x) < 1e-6 and x != 0:
                return f"{x:.3e}"
            return f"{x:.6f}".rstrip('0').rstrip('.')
        return str(x)
    display_df = display_df.astype(object).where(pd.notnull(display_df), None)
    print(tabulate(display_df.values, headers=display_df.columns, tablefmt="github", showindex=False, floatfmt=".6f"))

def cmd_top(df: pd.DataFrame, n: int = 5, columns=None):
    if 'f1' not in df.columns:
        print("No existe la columna 'f1' en el dataframe.")
        return
    topdf = df.sort_values('f1', ascending=False).reset_index(drop=True)
    if columns is None:
        columns = df.columns.tolist()
    print(f"\nTop {n} (ordenado por f1 desc):")
    pretty_print(topdf, n=n, columns=columns)

def cmd_best_per(df: pd.DataFrame, by: str, n: int = 5, columns=None):
    if by == OBVIATED_COLUMN:
        print(f"La columna '{OBVIATED_COLUMN}' está obviada y no se puede usar para agrupar.")
        return
    if by not in df.columns:
        print(f"Columna para agrupar no encontrada: {by}")
        return
    if 'f1' not in df.columns:
        print("No existe la columna 'f1' en el dataframe.")
        return
    grouped = df.sort_values('f1', ascending=False).groupby(by, as_index=False).first()
    grouped_sorted = grouped.sort_values('f1', ascending=False).reset_index(drop=True)
    if columns is None:
        columns = grouped_sorted.columns.tolist()
    print(f"\nMejores por '{by}' (top {n}):")
    pretty_print(grouped_sorted, n=n, columns=columns)

def cmd_stats(df: pd.DataFrame, by: str = None):
    if by is None:
        print("\nEstadísticas de 'f1':")
        print(df['f1'].describe())
    else:
        if by == OBVIATED_COLUMN:
            print(f"La columna '{OBVIATED_COLUMN}' está obviada y no se puede usar para agrupar.")
            return
        if by not in df.columns:
            print(f"Columna para agrupar no encontrada: {by}")
            return
        print(f"\nMedia/STD/Conteo de 'f1' por '{by}':")
        g = df.groupby(by)['f1'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        print(tabulate(g.reset_index().values, headers=[by,'mean','std','count'], tablefmt='github', floatfmt=".6f"))

def save_csv(df: pd.DataFrame, path: str):
    # Asegurarse de que la columna obviada no esté presente (por si)
    if OBVIATED_COLUMN in df.columns:
        df = df.drop(columns=[OBVIATED_COLUMN])
    df.to_csv(path, index=False)
    print(f"Guardado: {path}")

def print_help():
    help_text = f"""
    Comandos disponibles (REPL):
      help
          Muestra esta ayuda.
      top N [filters]
          Muestra las N mejores filas ordenadas por f1 descendente.
          Ejemplo: top 5
                   top 10 lr>1e-4;scheduler==cosine
      best_per <column> [N] [filters]
          Muestra la mejor configuración para cada valor único de <column>, ordenadas por f1.
          Ejemplo: best_per lr 5
                   best_per scheduler 10 lr>1e-5
      stats [column] [filters]
          Muestra estadísticas de 'f1'. Si se proporciona columna, agrupa por ella.
          Ejemplo: stats
                   stats scheduler
      show [N] [filters]
          Muestra las primeras N filas (por defecto 20) del conjunto filtrado.
          Ejemplo: show 30 lr>=1e-4
      filter <expressions>
          Fija filtros persistentes para las siguientes llamadas. Reemplaza filtros previos.
          Expresiones separadas por ';' (p.ej. lr>1e-4;scheduler==cosine).
      add_filter <expressions>
          Añade filtros a los ya existentes.
      clear_filters
          Elimina filtros persistentes.
      save <path> [filters]
          Guarda el conjunto (posiblemente filtrado) en CSV.
          Ejemplo: save filtered.csv lr>1e-4
      cols
          Muestra las columnas disponibles (NOTA: '{OBVIATED_COLUMN}' siempre está obviada).
      exit / quit / q
          Salir.
    Operadores soportados: ==, !=, >=, <=, >, <, = (equivale a ==).
    Nota: Para cadenas, puede usar quotes: scheduler=='cosine' o scheduler==cosine.
    """
    print(textwrap.dedent(help_text))

def repl(df: pd.DataFrame):
    persistent_filters: List[str] = []
    print("CSV Results UI - modo interactivo. Escriba 'help' para ver comandos.")
    while True:
        try:
            line = input("csv-ui> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo.")
            break
        if line == "":
            continue
        parts = shlex.split(line)
        cmd = parts[0].lower()
        args = parts[1:]
        try:
            if cmd in ('exit','quit','q'):
                break
            elif cmd == 'help':
                print_help()
            elif cmd == 'cols':
                print("\nColumnas disponibles (la columna obviada no aparece):")
                for c in df.columns:
                    print("  -", c)
            elif cmd == 'filter':
                expr = " ".join(args)
                if expr.strip() == "":
                    persistent_filters = []
                    print("Filtros persistentes vacíos.")
                else:
                    persistent_filters = [e.strip() for e in expr.split(";") if e.strip()!=""]
                    print("Filtros persistentes establecidos:", persistent_filters)
            elif cmd == 'add_filter':
                expr = " ".join(args)
                more = [e.strip() for e in expr.split(";") if e.strip()!=""]
                persistent_filters.extend(more)
                print("Filtros persistentes:", persistent_filters)
            elif cmd == 'clear_filters':
                persistent_filters = []
                print("Filtros persistentes eliminados.")
            elif cmd == 'top':
                n = 5
                extra = []
                if args:
                    # buscar primer argumento numérico
                    if args[0].isdigit():
                        n = int(args[0])
                        extra = args[1:]
                    else:
                        extra = args
                exprs = " ".join(extra).split(";") if extra else []
                exprs = [e for e in exprs if e.strip()!=""]
                combined = persistent_filters + exprs
                subdf = apply_filters(df, combined)
                cmd_top(subdf, n=n)
            elif cmd == 'best_per':
                if len(args) == 0:
                    print("Uso: best_per <column> [N] [filters]")
                    continue
                by = args[0]
                n = 5
                extra = []
                if len(args) >= 2 and args[1].isdigit():
                    n = int(args[1])
                    extra = args[2:]
                else:
                    extra = args[1:]
                exprs = " ".join(extra).split(";") if extra else []
                exprs = [e for e in exprs if e.strip()!=""]
                combined = persistent_filters + exprs
                subdf = apply_filters(df, combined)
                cmd_best_per(subdf, by=by, n=n)
            elif cmd == 'stats':
                if args and args[0] not in ('',):
                    by = args[0]
                    extra = args[1:]
                else:
                    by = None
                    extra = []
                exprs = " ".join(extra).split(";") if extra else []
                exprs = [e for e in exprs if e.strip()!=""]
                combined = persistent_filters + exprs
                subdf = apply_filters(df, combined)
                cmd_stats(subdf, by=by)
            elif cmd == 'show':
                n = DISPLAY_MAX_ROWS
                extra = []
                if args and args[0].isdigit():
                    n = int(args[0])
                    extra = args[1:]
                else:
                    extra = args
                exprs = " ".join(extra).split(";") if extra else []
                exprs = [e for e in exprs if e.strip()!=""]
                combined = persistent_filters + exprs
                subdf = apply_filters(df, combined)
                print(f"\nMostrando primeras {n} filas (total filtrado: {len(subdf)})")
                pretty_print(subdf, n=n)
            elif cmd == 'save':
                if len(args) == 0:
                    print("Uso: save <path> [filters]")
                    continue
                path = args[0]
                extra = args[1:]
                exprs = " ".join(extra).split(";") if extra else []
                exprs = [e for e in exprs if e.strip()!=""]
                combined = persistent_filters + exprs
                subdf = apply_filters(df, combined)
                save_csv(subdf, path)
            else:
                print(f"Comando desconocido: {cmd}. Escriba 'help' para ver ayuda.")
        except Exception as e:
            print(f"Error al ejecutar comando: {e}")

def main():
    parser = argparse.ArgumentParser(description="Explorador CSV orientado a monitorizar 'f1'. (ignora 'output_dir')")
    parser.add_argument("--file", "-f", type=str, default=DEFAULT_FILE, help="Ruta al CSV")
    parser.add_argument("--top", "-t", type=int, default=None, help="Mostrar top N (no interactivo)")
    parser.add_argument("--filter", type=str, default=None, help="Filtros separados por ';' (ej. lr>1e-4;scheduler==cosine)")
    parser.add_argument("--best_per", type=str, default=None, help="Mostrar mejor por valor de esta columna (no interactivo)")
    parser.add_argument("--best_n", type=int, default=10, help="Número de filas a mostrar para best_per")
    parser.add_argument("--no-repl", action="store_true", help="No entrar en modo interactivo tras ejecutar acción")
    parser.add_argument("--show", type=int, default=None, help="Mostrar primeras N filas (no interactivo)")
    parser.add_argument("--save", type=str, default=None, help="Guardar conjunto filtrado a este CSV (no interactivo)")
    args = parser.parse_args()

    try:
        df = load_csv(args.file)
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        sys.exit(1)

    # Aplicar filtro si existe
    filters = []
    if args.filter:
        filters = [e.strip() for e in args.filter.split(";") if e.strip()!=""]
    working_df = apply_filters(df, filters)

    # Ejecutar acciones no interactivas
    if args.top is not None:
        cmd_top(working_df, n=args.top)
    if args.best_per:
        cmd_best_per(working_df, by=args.best_per, n=args.best_n)
    if args.show is not None:
        print(f"\nMostrando primeras {args.show} filas (total filtrado: {len(working_df)})")
        pretty_print(working_df, n=args.show)
    if args.save:
        save_csv(working_df, args.save)

    # Entrar a REPL si procede
    if not args.no_repl:
        repl(df)
    else:
        print("Ejecución finalizada (modo no interactivo).")

if __name__ == "__main__":
    main()
