#!/usr/bin/env python3
"""
make_speaker_subsets_simple.py

Crear subsets nested de forma eficiente — optimizado para no recomputar trabajo pesado
en cada iteración. Cada subset añade exactamente (s - s_prev) nuevas muestras cuando es posible.

Uso típico:
python make_speaker_subsets_simple.py \
  --train-tsv PATH/xxx.train.tsv \
  --test-tsv PATH/xxx.test.tsv \
  --out-dir /ruta/out \
  --seed 42 \
  --step 50 \
  --max-train 2000 \
  --limit-per-speaker 100 \
  --nested
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timezone
import math
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Crear subsets con restricción speakers >= 0.0125*n (versión optimizada)")
    p.add_argument("--train-tsv", required=True)
    p.add_argument("--test-tsv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--limit-per-speaker", type=int, default=None,
                   help="Máximo muestras por speaker en pool (None = usar todas).")
    p.add_argument("--nested", action="store_true",
                   help="Generar subsets nested reales: cada subset mayor contiene al anterior (mismos speakers).")
    p.add_argument("--no-md5", action="store_true", help="No calcular MD5 para acelerar (sólo para debug).")
    return p.parse_args()


def md5_bytes(b: bytes):
    return hashlib.md5(b).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------- fast builders ----------
def build_per_speaker_fast(train_df, speakers_ordered, limit_per_speaker, rng):
    """Devuelve per_spk_records: dict spk -> list de row-strings (tab-joined) y counts dict."""
    grouped = train_df.groupby("client_id")
    per_spk_records = {}
    counts = {}
    cols = list(train_df.columns)
    for spk in speakers_ordered:
        g = grouped.get_group(spk)
        values = g[cols].astype(str).values
        nrows = values.shape[0]
        idx = np.arange(nrows)
        rng.shuffle(idx)
        if limit_per_speaker is not None:
            take = min(nrows, int(limit_per_speaker))
            idx = idx[:take]
        rows = ["\t".join(values[i].tolist()) for i in idx]
        per_spk_records[spk] = rows
        counts[spk] = len(rows)
    return per_spk_records, counts, cols


def build_interleaved_fast(per_spk_records, speakers_ordered, max_train=None):
    """Interleave by round across speakers. Return list of row-strings (preview)."""
    max_len = 0
    for spk in speakers_ordered:
        L = len(per_spk_records.get(spk, []))
        if L > max_len:
            max_len = L
    out = []
    limit = int(max_train) if max_train is not None else float("inf")
    for r in range(max_len):
        for spk in speakers_ordered:
            recs = per_spk_records.get(spk, [])
            if r < len(recs):
                out.append(recs[r])
                if len(out) >= limit:
                    return out
    return out


def build_subset_with_s_constraint_fast(per_spk_records, counts, speakers_ordered, n, rng):
    """Función auxiliar para modo no-nested (sin optimización incremental)."""
    available_speakers = len(speakers_ordered)
    if n <= 0:
        return [], {"requested_n": int(n), "final_n": 0, "note": "n<=0"}

    S_min_rule = max(1, math.ceil(0.0125 * n))
    S = min(available_speakers, max(1, S_min_rule))
    spk_sorted_by_count = sorted(speakers_ordered, key=lambda s: counts.get(s, 0), reverse=True)

    feasible = False
    chosen_spks = []
    assigned = {}

    while S >= 1:
        base = n // S
        rem = n % S
        quotas = [base + 1] * rem + [base] * (S - rem)
        quotas_sorted = sorted(quotas, reverse=True)
        candidates = spk_sorted_by_count[:S]
        candidates_sorted = sorted(candidates, key=lambda s: counts.get(s, 0), reverse=True)
        ok = True
        for i, q in enumerate(quotas_sorted):
            if i >= len(candidates_sorted) or counts.get(candidates_sorted[i], 0) < q:
                ok = False
                break
        if ok:
            chosen_spks = candidates_sorted
            for spk, q in zip(candidates_sorted, quotas_sorted):
                assigned[spk] = q
            feasible = True
            break
        S -= 1

    if feasible and len(chosen_spks) > 0:
        out = []
        taken = {spk: 0 for spk in chosen_spks}
        while len(out) < n:
            progress = False
            for spk in chosen_spks:
                need = assigned.get(spk, 0)
                t = taken[spk]
                if t < need and t < len(per_spk_records.get(spk, [])):
                    out.append(per_spk_records[spk][t])
                    taken[spk] += 1
                    progress = True
                    if len(out) >= n:
                        break
            if not progress:
                break
        meta = {
            "requested_n": int(n),
            "final_n": int(len(out)),
            "S_used": int(len(chosen_spks)),
            "quota_per_spk": {spk: assigned[spk] for spk in chosen_spks},
            "note": ""
        }
        if len(out) < n:
            meta["note"] = "No se pudieron extraer todas las muestras previstas; subset recortado."
        return out, meta

    interleaved = build_interleaved_fast(per_spk_records, speakers_ordered, max_train=n)
    meta = {
        "requested_n": int(n),
        "final_n": int(len(interleaved)),
        "S_used": int(min(len(speakers_ordered), math.ceil(0.0125 * n))),
        "quota_per_spk": {},
        "note": "Fallback: prefijo interleaved."
    }
    return interleaved, meta


# ---------- main ----------
def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    # leer TSVs una sola vez
    train_df = pd.read_csv(args.train_tsv, sep="\t", dtype=str, keep_default_na=False)
    test_df = pd.read_csv(args.test_tsv, sep="\t", dtype=str, keep_default_na=False)

    for c in ("client_id", "audio"):
        if c not in train_df.columns:
            raise SystemExit(f"ERROR: train.tsv debe contener columna '{c}'")
    if "client_id" not in test_df.columns:
        print("[INFO] test.tsv no tiene 'client_id'; se copiará tal cual en cada subset.")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # speakers ordenados por cantidad (precomputado)
    counts_full = train_df['client_id'].value_counts().sort_values(ascending=False)
    speakers_ordered = counts_full.index.tolist()

    # PRECOMPUTE costy structures ONCE
    per_spk_records, counts_after_limit, cols = build_per_speaker_fast(
        train_df, speakers_ordered, args.limit_per_speaker, rng
    )

    pool_preview_list = build_interleaved_fast(per_spk_records, speakers_ordered, max_train=args.max_train)
    pool_size = len(pool_preview_list)
    if pool_size == 0:
        raise SystemExit("Pool vacío tras aplicar limit-per-speaker / train.tsv incorrecto.")

    desired_max_train = pool_size if args.max_train is None else min(pool_size, int(args.max_train))

    if args.nested:
        max_train_used = desired_max_train
        step = max(1, int(args.step))
        sizes = list(range(1, max_train_used + 1, step))
        # estado incremental
        chosen_speakers = []
        per_spk_taken = {spk: 0 for spk in speakers_ordered}
        next_spk_idx = 0
        cumulative_train_rows = []  # buffer acumulado (list of row-strings)
    else:
        max_train_used = desired_max_train
        step = max(1, int(args.step))
        sizes = list(range(1, max_train_used + 1, step))
        cumulative_train_rows = None

    # metadata base
    base_meta = {
        "source_train": str(Path(args.train_tsv).resolve()),
        "source_test": str(Path(args.test_tsv).resolve()),
        "pool_size": int(pool_size),
        "max_train_used": int(max_train_used),
        "step": step,
        "seed": int(args.seed),
        "limit_per_speaker": args.limit_per_speaker,
        "nested": bool(args.nested),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    write_json(out_root / "pool_meta.json", base_meta)

    # pool preview (precomputed)
    pool_preview_path = out_root / ("pool_preview_nested.tsv" if args.nested else "pool_preview.tsv")
    with open(pool_preview_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        if max_train_used > 0:
            f.write("\n".join(pool_preview_list[:max_train_used]) + "\n")

    # compute test_md5 once (no need to recompute per subset)
    if not args.no_md5:
        test_bytes = test_df.to_csv(sep="\t", index=False).encode("utf-8")
        test_md5_single = md5_bytes(test_bytes)
    else:
        test_md5_single = "(skipped)"

    summary = []
    header_line = "\t".join(cols)

    # MAIN loop over subset sizes (lightweight now)
    for idx, s in enumerate(sizes, start=1):
        subset_name = f"subset_{idx:03d}_n{s}"
        subset_dir = out_root / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        if args.nested:
            # Delta required to grow from previous total to s
            prev_total = len(cumulative_train_rows)
            required_new = max(0, s - prev_total)
            S_req = max(1, math.ceil(0.0125 * s))

            # extend speakers list only if necessary
            while len(chosen_speakers) < S_req and next_spk_idx < len(speakers_ordered):
                chosen_speakers.append(speakers_ordered[next_spk_idx])
                next_spk_idx += 1

            # quotas computed for the whole final set size s
            base_q = s // S_req
            rem = s % S_req
            quotas = [base_q + 1] * rem + [base_q] * (S_req - rem)
            assigned = {spk: q for spk, q in zip(chosen_speakers[:S_req], quotas)}

            out_rows = []
            # First pass: try to take per-speaker delta towards assigned quotas,
            # but never take more than required_new in total.
            for spk in chosen_speakers[:S_req]:
                if len(out_rows) >= required_new:
                    break
                have = per_spk_taken.get(spk, 0)
                need_total_for_spk = assigned.get(spk, 0)
                need_delta = max(0, need_total_for_spk - have)
                if need_delta <= 0:
                    continue
                avail = per_spk_records.get(spk, [])
                can_take = min(need_delta, max(0, len(avail) - have), required_new - len(out_rows))
                if can_take > 0:
                    out_rows.extend(avail[have:have + can_take])
                    per_spk_taken[spk] = have + can_take

            # Second pass (if still short): fill round-robin across chosen speakers until required_new satisfied
            if len(out_rows) < required_new and required_new > 0:
                cursors = {spk: per_spk_taken.get(spk, 0) for spk in chosen_speakers[:S_req]}
                while len(out_rows) < required_new:
                    prog = False
                    for spk in chosen_speakers[:S_req]:
                        cur = cursors[spk]
                        avail = per_spk_records.get(spk, [])
                        if cur < len(avail):
                            out_rows.append(avail[cur])
                            cursors[spk] += 1
                            per_spk_taken[spk] = cursors[spk]
                            prog = True
                            if len(out_rows) >= required_new:
                                break
                    if not prog:
                        break  # no more data available to satisfy required_new

            # append only the newly selected rows
            added = len(out_rows)
            cumulative_train_rows.extend(out_rows)
            train_rows = list(cumulative_train_rows)

            meta = {
                "requested_n": int(s),
                "final_n": int(len(train_rows)),
                "S_used": int(len(chosen_speakers[:S_req])),
                "speakers_used": list(chosen_speakers[:S_req]),
                "note": "nested incremental (delta applied)"
            }

            # quick debug print to ensure delta behavior
            print(f"[{idx}/{len(sizes)}] {subset_name}: prev_total={prev_total}, required_new={required_new}, added={added}", flush=True)

        else:
            # non-nested mode: compute subset from scratch (fast enough since per_spk_records is precomputed)
            train_rows, meta = build_subset_with_s_constraint_fast(
                per_spk_records, counts_after_limit, speakers_ordered, s, rng
            )
            added = len(train_rows)

        # write train file (fast: only writing accumulated content or subset content)
        train_out = subset_dir / Path(args.train_tsv).name
        test_out = subset_dir / Path(args.test_tsv).name

        content_lines = [header_line]
        if train_rows:
            content_lines.extend(train_rows)
        content = "\n".join(content_lines) + "\n"
        content_bytes = content.encode("utf-8")

        if not args.no_md5:
            train_md5 = md5_bytes(content_bytes)
            test_md5 = test_md5_single
        else:
            train_md5 = "(skipped)"
            test_md5 = "(skipped)"

        with open(train_out, "wb") as f:
            f.write(content_bytes)

        # write test only once per subset (small)
        test_df.to_csv(test_out, sep="\t", index=False)

        counts_sub = {}
        if train_rows:
            for r in train_rows:
                spk = r.split('\t', 1)[0]
                counts_sub[spk] = counts_sub.get(spk, 0) + 1
        speakers_in_subset = len(counts_sub)
        top_speakers = dict(sorted(counts_sub.items(), key=lambda kv: kv[1], reverse=True)[:6])

        meta_full = {
            "subset_name": subset_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requested_n": int(s),
            "n_samples_train": int(len(train_rows)),
            "n_speakers": speakers_in_subset,
            "top_speakers_counts": top_speakers,
            "seed": int(args.seed),
            "train_md5": train_md5,
            "test_md5": test_md5,
            "selection_meta": meta
        }
        write_json(subset_dir / "metadata.json", meta_full)

        summary.append({
            "subset_name": subset_name,
            "subset_dir": str(subset_dir.resolve()),
            "n_samples_train": int(len(train_rows)),
            "n_speakers": speakers_in_subset,
            "train_md5": train_md5,
            "test_md5": test_md5,
            "note": meta.get("note", "")
        })

    # summary outputs
    summary_df = pd.DataFrame(summary)
    summary_csv = out_root / "subsets_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    write_json(out_root / 'subsets_summary.json', {"base_meta": base_meta, "subsets": summary})

    print(f"[DONE] Generated {len(summary)} subsets in {out_root}")
    print("Pool preview saved in:", pool_preview_path)
    print("Summary CSV:", summary_csv)


if __name__ == "__main__":
    main()

"""
python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/ca.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/ca.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/ca \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/en.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/en.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/en \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/eo.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/eo.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/eo \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/es.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/es.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/es \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/eu.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/eu.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/eu \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/hu.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/hu.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/hu \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/ja.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/ja.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/ja \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/ka.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/ka.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/ka \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/ru.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/ru.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/ru \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/sw.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/sw.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/sw \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/th.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/th.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/th \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 

python3 make_speaker_subsets.py \
  --train-tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv/zh-CN.train.tsv \
  --test-tsv  /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/zh-CN.test.tsv \
  --out-dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker/zh-CN \
  --seed 42 \
  --step 500 \
  --max-train 50000 \
  --nested 
"""

