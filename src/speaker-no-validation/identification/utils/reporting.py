"""
Functions to summarize training runs, save JSON/CSV summaries, and print readable tables.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def _format_seconds(sec: Optional[float]) -> str:
    """Format seconds into human-readable string (ms, s, or m:s)."""
    if sec is None:
        return "N/A"
    if sec < 1.0:
        return f"{sec*1000:.0f} ms"
    m, s = divmod(sec, 60)
    if m >= 1:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.2f}s"


def _sample_list_str(lst: Optional[List[str]], n: int = 10) -> str:
    """Return a truncated string representation of a list."""
    if lst is None:
        return "N/A"
    if len(lst) == 0:
        return "[]"
    if len(lst) <= n:
        return str(lst)
    return f"{lst[:n]}  ...(+{len(lst)-n} more)"


def _simple_table(rows: List[List[str]], title: Optional[str] = None) -> str:
    """Render simple 2-column ASCII table."""
    col1_w = max(len(r[0]) for r in rows) if rows else 0
    col2_w = max(len(r[1]) for r in rows) if rows else 0
    sep = f"+-{'-'*col1_w}-+-{'-'*col2_w}-+"
    lines = [sep]
    if title:
        lines.append(f"| {title.ljust(col1_w+col2_w+3)}|")
        lines.append(sep)
    for a, b in rows:
        lines.append(f"| {a.ljust(col1_w)} | {b.ljust(col2_w)} |")
    lines.append(sep)
    return "\n".join(lines)


def summarize_run_variant(
    output_dir: str,
    variant_name: str,
    args_dict: Dict[str, Any],
    param_counts: Dict[str, Any],
    timings: Dict[str, Optional[float]],
    test_metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    save_json_file: str = "summary.json",
    save_csv_file: str = "summary.csv",
):
    """
    Save and print a summary of a training run for a variant.
    - Saves JSON/CSV in output_dir.
    - Prints human-readable tables of configuration, timings, and metrics.
    This variant reads best_info.json (if present) to obtain the epoch
    corresponding to the best checkpoint and uses that epoch in the report.
    """
    os.makedirs(output_dir, exist_ok=True)
    tstamp = datetime.utcnow().isoformat() + "Z"

    def _norm_metrics(m: Optional[Dict[str, Any]]):
        """Normalize keys starting with 'eval_' or 'test_' and cast to float if possible."""
        if m is None:
            return {}
        out = {}
        for k, v in m.items():
            kn = k
            if kn.startswith("eval_"):
                kn = kn[len("eval_"):]
            elif kn.startswith("test_"):
                kn = kn[len("test_"):]
            out[kn] = float(v) if isinstance(v, (int, float)) else v
        return out

    testm = _norm_metrics(test_metrics)

    # Try to read best_epoch from best_info.json produced by process_test_metrics
    best_info_path = os.path.join(output_dir, "best_info.json")
    test_epoch = None
    if os.path.exists(best_info_path):
        try:
            with open(best_info_path, "r") as fh:
                bi = json.load(fh)
                # accept either numeric or string; keep as-is for reporting
                test_epoch = bi.get("best_epoch", bi.get("best_epoch"))
        except Exception:
            test_epoch = None

    summary = {
        "variant": variant_name,
        "timestamp_utc": tstamp,
        "output_dir": output_dir,
        "args": args_dict,
        "param_counts": param_counts,
        "timings": timings,
        "test_metrics": testm,
        "test_epoch": test_epoch,
        "extra": extra or {},
    }

    # Save JSON summary
    json_path = os.path.join(output_dir, save_json_file)
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                all_summaries = json.load(f)
            if isinstance(all_summaries, list):
                all_summaries.append(summary)
            else:
                all_summaries = [all_summaries, summary]
        else:
            all_summaries = [summary]
        with open(json_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
    except Exception:
        # fallback: save per variant
        with open(os.path.join(output_dir, f"{variant_name}_summary_{tstamp}.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Append (or create) CSV summary, include test_epoch
    csv_path = os.path.join(output_dir, save_csv_file)
    csv_row = {
        "timestamp_utc": tstamp,
        "variant": variant_name,
        "output_dir": output_dir,
        "total_params": param_counts.get("total"),
        "trainable_params": param_counts.get("trainable"),
        "trainable_pct": param_counts.get("trainable_pct"),
        "time_total_s": timings.get("total"),
        "time_train_s": timings.get("train"),
        "test_accuracy": testm.get("accuracy"),
        "test_f1_macro": testm.get("f1_macro"),
        "test_epoch": test_epoch,
    }
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(csv_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row)
    except Exception:
        pass

    # Pretty print summary to console
    print("\n" + f" Experiment summary [{variant_name}] ".center(80, "="))

    conf_rows = [
        ["Output dir", output_dir],
        ["Timestamp (UTC)", tstamp],
        ["Variant", variant_name],
        ["Model total params", str(param_counts.get("total", "N/A"))],
        ["Model trainable params",
         f"{param_counts.get('trainable', 'N/A')} ({param_counts.get('trainable_pct', 'N/A'):.2f}%)"
         if param_counts.get("trainable") is not None else "N/A"],
    ]
    print(_simple_table(conf_rows, title="Configuration / Model"))

    timing_rows = [[k, _format_seconds(v)] for k, v in timings.items()]
    print(_simple_table(timing_rows, title="Timings"))

    if testm:
        # Build a display dict that includes epoch (if available) so epoch appears in the metrics block
        display_metrics = dict(testm)  # copy
        if test_epoch is not None:
            display_metrics["epoch"] = test_epoch

        metric_names = sorted(display_metrics.keys())
        col1_w = max(len(m) for m in metric_names)
        col2_w = max(len(str(display_metrics[m])) for m in metric_names)
        sep = f"+-{'-'*col1_w}-+-{'-'*col2_w}-+"
        print("\n" + sep)
        print(f"| {'Metric'.ljust(col1_w)} | {'Test'.ljust(col2_w)} |")
        print(sep)
        for m in metric_names:
            print(f"| {m.ljust(col1_w)} | {str(display_metrics.get(m,'N/A')).ljust(col2_w)} |")
        print(sep)

    artifacts_rows = [
        ["model_dir", output_dir],
        ["summary_json", json_path],
        ["summary_csv", csv_path],
    ]
    print("\n" + _simple_table(artifacts_rows, title="Artifacts"))

    print("="*80 + "\n")
