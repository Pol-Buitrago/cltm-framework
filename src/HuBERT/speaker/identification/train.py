"""Modified run_variant_continuous with improved training-time visualization callback and
embedding-pairwise statistics for intra / inter-language distances.

Este parche cambia la asignación de colores a un esquema aleatorio por ejecución,
mejora el estilo de las figuras y añade el cálculo de métricas útiles para
verificar si la separación entre idiomas excede la dispersión intra-idioma.

Colocar este archivo en el proyecto y usar la función `run_variant_continuous`
(reemplazando la original en train.py si procede).
"""

import os
import logging
import math
import json
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # asegura backend sin display
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None

from utils.collators import (
    HFDatasetWrapper,
    collate_fn_cont_factory,
    make_collate_fn_quant,
)

from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

def _safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

# Random color assignment per run
_RUNTIME_COLOR_MAP = {}
try:
    _GLOBAL_RNG = np.random.default_rng()
except Exception:
    _GLOBAL_RNG = np.random.RandomState()


def _color_for_label(label):
    key = str(label) if label is not None else 'unk'
    if key in _RUNTIME_COLOR_MAP:
        return _RUNTIME_COLOR_MAP[key]
    try:
        rgb = _GLOBAL_RNG.random(3)
    except Exception:
        rgb = np.random.rand(3)
    rgb = 0.12 + 0.76 * rgb  # ensure contrast
    color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.90)
    _RUNTIME_COLOR_MAP[key] = color
    return color


def _plot_scatter(points, labels, title, path, families=None, figsize=(10, 7)):
    plt.figure(figsize=figsize)
    uniq = sorted(list(set([str(u) for u in labels])))
    color_map = {u: _color_for_label(u) for u in uniq}

    base_s = 28
    n = points.shape[0]
    s = max(8, int(base_s * (512 / max(256, n))))

    for u in uniq:
        idx = [i for i, v in enumerate(labels) if str(v) == u]
        if len(idx) == 0:
            continue
        pts = points[idx]
        plt.scatter(
            pts[:, 0], pts[:, 1],
            label=str(u),
            alpha=0.85,
            s=s,
            c=[color_map[u]],
            edgecolors='k',
            linewidths=0.3,
            rasterized=True,
        )

    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)

    n_uniq = len(uniq)
    if n_uniq > 12:
        cols = 2
        fontsize = 'x-small'
    else:
        cols = 1
        fontsize = 'small'

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=cols, fontsize=fontsize, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

# ---------------------------------------------------------------------------
# Metrics utilities: intra / inter distances and centroid distances
# ---------------------------------------------------------------------------

def _compute_language_stats(X, labels):
    """
    Compute per-language intra-distance stats and pairwise centroid distances.

    Returns a dict with:
      - per_label: {label: {count, centroid (list), intra_euc_mean, intra_euc_std,
                             intra_cos_mean, intra_cos_std}}
      - pairwise_centroid_euclidean: {(a,b): dist}
      - pairwise_centroid_cosine: {(a,b): dist}
      - global: aggregated numbers (mean intra, mean centroid dist)
    """
    labels = list(map(str, labels))
    uniq = sorted(set(labels))
    result = {'per_label': {}, 'pairwise_centroid_euclidean': {}, 'pairwise_centroid_cosine': {}, 'global': {}}

    # Build index lists
    idx_map = {u: [i for i, lab in enumerate(labels) if lab == u] for u in uniq}
    centroids = {}
    for u, idxs in idx_map.items():
        if len(idxs) == 0:
            continue
        Xi = X[idxs]
        centroid = np.mean(Xi, axis=0)
        centroids[u] = centroid
        # Euclidean intra distances
        if Xi.shape[0] > 1:
            d_euc = pairwise_distances(Xi, metric='euclidean')
            iu = np.triu_indices_from(d_euc, k=1)
            vals_euc = d_euc[iu] if iu[0].size > 0 else np.array([])
            d_cos = cosine_distances(Xi)
            vals_cos = d_cos[iu] if iu[0].size > 0 else np.array([])
            intra_euc_mean = float(np.mean(vals_euc)) if vals_euc.size > 0 else 0.0
            intra_euc_std = float(np.std(vals_euc)) if vals_euc.size > 0 else 0.0
            intra_cos_mean = float(np.mean(vals_cos)) if vals_cos.size > 0 else 0.0
            intra_cos_std = float(np.std(vals_cos)) if vals_cos.size > 0 else 0.0
        else:
            intra_euc_mean = 0.0
            intra_euc_std = 0.0
            intra_cos_mean = 0.0
            intra_cos_std = 0.0

        result['per_label'][u] = {
            'count': int(len(idxs)),
            'centroid': centroid.tolist(),
            'intra_euclidean_mean': intra_euc_mean,
            'intra_euclidean_std': intra_euc_std,
            'intra_cosine_mean': intra_cos_mean,
            'intra_cosine_std': intra_cos_std,
        }

    # pairwise centroid distances
    labels_cent = list(centroids.keys())
    for i, a in enumerate(labels_cent):
        for j in range(i + 1, len(labels_cent)):
            b = labels_cent[j]
            ca = centroids[a]
            cb = centroids[b]
            d_euc = float(np.linalg.norm(np.asarray(ca) - np.asarray(cb)))
            # cosine distance between centroids
            denom = (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12)
            d_cos = float(1.0 - np.dot(ca, cb) / denom)
            result['pairwise_centroid_euclidean'][f'{a}__{b}'] = d_euc
            result['pairwise_centroid_cosine'][f'{a}__{b}'] = d_cos

    # global aggregates
    # mean intra (averaged across labels)
    intra_euc_means = [v['intra_euclidean_mean'] for v in result['per_label'].values() if v['count'] > 1]
    intra_cos_means = [v['intra_cosine_mean'] for v in result['per_label'].values() if v['count'] > 1]
    mean_intra_euc = float(np.mean(intra_euc_means)) if len(intra_euc_means) > 0 else 0.0
    mean_intra_cos = float(np.mean(intra_cos_means)) if len(intra_cos_means) > 0 else 0.0

    centroid_euc_vals = list(result['pairwise_centroid_euclidean'].values())
    centroid_cos_vals = list(result['pairwise_centroid_cosine'].values())
    mean_centroid_euc = float(np.mean(centroid_euc_vals)) if len(centroid_euc_vals) > 0 else 0.0
    mean_centroid_cos = float(np.mean(centroid_cos_vals)) if len(centroid_cos_vals) > 0 else 0.0

    result['global'] = {
        'mean_intra_euclidean': mean_intra_euc,
        'mean_intra_cosine': mean_intra_cos,
        'mean_centroid_euclidean': mean_centroid_euc,
        'mean_centroid_cosine': mean_centroid_cos,
    }

    return result


def _dump_stats_json(stats, path):
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.exception("Could not write stats JSON: %s", e)


def _dump_stats_csv(stats, path):
    # Per-label table
    try:
        rows = []
        for lab, v in stats['per_label'].items():
            row = {
                'label': lab,
                'count': v['count'],
                'intra_euclidean_mean': v['intra_euclidean_mean'],
                'intra_euclidean_std': v['intra_euclidean_std'],
                'intra_cosine_mean': v['intra_cosine_mean'],
                'intra_cosine_std': v['intra_cosine_std'],
            }
            rows.append(row)
        if pd is not None:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
        else:
            # fallback write minimal CSV
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write('label,count,intra_euclidean_mean,intra_euclidean_std,intra_cosine_mean,intra_cosine_std\n')
                for r in rows:
                    fh.write(f"{r['label']},{r['count']},{r['intra_euclidean_mean']},{r['intra_euclidean_std']},{r['intra_cosine_mean']},{r['intra_cosine_std']}\n")
    except Exception as e:
        logger.exception("Could not write stats CSV: %s", e)


# ---------------------------------------------------------------------------
# Callback to run visualizations at the end of each epoch
# ---------------------------------------------------------------------------
class EmbeddingVizCallback(TrainerCallback):
    def __init__(self, sample_size=512, perplexity=30, seed=42, enabled=True):
        self.sample_size = int(sample_size)
        self.perplexity = perplexity
        self.seed = int(seed)
        self.enabled = bool(enabled)

    def _select_sample(self, dataset, k):
        n = len(dataset)
        if n <= k:
            return dataset
        idxs = (np.linspace(0, n - 1, k)).astype(int).tolist()
        try:
            return dataset.select(idxs)
        except Exception:
            return [dataset[i] for i in idxs]

    def _get_labels_for_plot(self, ds):
        labels = []
        for ex in ds:
            if isinstance(ex, dict):
                for key in ('language', 'lang', 'locale', 'family'):
                    if key in ex and ex[key] is not None:
                        labels.append(ex[key])
                        break
                else:
                    labels.append('unk')
            else:
                labels.append('unk')
        return labels

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        trainer = kwargs.get('trainer')
        if trainer is None:
            return

        output_dir = getattr(trainer.args, 'output_dir', None) or os.getcwd()
        viz_dir = os.path.join(output_dir, 'embeddings')
        _safe_mkdir(viz_dir)

        ds = None
        if trainer.eval_dataset is not None:
            ds = trainer.eval_dataset
        elif trainer.train_dataset is not None:
            ds = trainer.train_dataset
        else:
            logger.warning('No dataset available for embedding visualization')
            return

        sample_ds = self._select_sample(ds, min(self.sample_size, len(ds)))

        logger.info('EmbeddingVizCallback: predicting sample (n=%d) for epoch=%d', len(sample_ds), state.epoch)
        try:
            pred_out = trainer.predict(sample_ds, metric_key_prefix='viz_pred')
            X = pred_out.predictions
            if X is None:
                logger.warning('No predictions returned by trainer.predict for visualization')
                return
            X = np.asarray(X).astype(np.float32)
            if X.ndim == 3:
                X = X.reshape(X.shape[0], -1)
        except Exception as e:
            logger.exception('EmbeddingVizCallback: trainer.predict failed: %s', e)
            return

        try:
            labels = self._get_labels_for_plot(sample_ds)
        except Exception:
            labels = ['unk'] * X.shape[0]

        # PCA
        try:
            pca = PCA(n_components=2, random_state=self.seed)
            X_pca = pca.fit_transform(X)
            epoch_dir = os.path.join(viz_dir, f'epoch_{int(state.epoch)}')
            _safe_mkdir(epoch_dir)
            pca_path = os.path.join(epoch_dir, f'epoch_{int(state.epoch)}_pca.png')
            _plot_scatter(X_pca, labels, f'PCA embeddings (epoch {int(state.epoch)})', pca_path)
            np.save(os.path.join(epoch_dir, f'epoch_{int(state.epoch)}_pca.npy'), X_pca)
        except Exception as e:
            logger.exception('PCA failed: %s', e)

        # t-SNE
        try:
            perp = min(max(5, int(self.perplexity)), max(5, X.shape[0] // 3))
            ts = TSNE(n_components=2, perplexity=perp, init='pca', random_state=self.seed, max_iter=1000)
            X_tsne = ts.fit_transform(X)
            tsne_path = os.path.join(viz_dir, f'epoch_{int(state.epoch)}_tsne.png')
            _plot_scatter(X_tsne, labels, f't-SNE embeddings (epoch {int(state.epoch)})', tsne_path)
            np.save(os.path.join(viz_dir, f'epoch_{int(state.epoch)}_tsne.npy'), X_tsne)
        except Exception as e:
            logger.exception('t-SNE failed: %s', e)

        # MDS
        try:
            max_mds = 512
            idxs = np.arange(X.shape[0])
            if X.shape[0] > max_mds:
                idxs = np.linspace(0, X.shape[0]-1, max_mds).astype(int)
            X_sub = X[idxs]
            labels_sub = [labels[i] for i in idxs]
            mds = MDS(n_components=2, dissimilarity='euclidean', random_state=self.seed, n_init=4, max_iter=300)
            X_mds = mds.fit_transform(X_sub)
            mds_path = os.path.join(viz_dir, f'epoch_{int(state.epoch)}_mds.png')
            _plot_scatter(X_mds, labels_sub, f'MDS embeddings (epoch {int(state.epoch)})', mds_path)
            np.save(os.path.join(viz_dir, f'epoch_{int(state.epoch)}_mds.npy'), X_mds)
        except Exception as e:
            logger.exception('MDS failed: %s', e)

        # Save raw embeddings
        try:
            np.save(os.path.join(viz_dir, f'epoch_{int(state.epoch)}_raw_preds.npy'), X)
        except Exception:
            pass

        logger.info('EmbeddingVizCallback: visualizations saved under %s', viz_dir)


# ---------------------------------------------------------------------------
# Modified run_variant_continuous: integrates EmbeddingVizCallback and final metrics
# ---------------------------------------------------------------------------
from transformers import Trainer

def run_variant_quantized(*args, **kwargs):
    raise NotImplementedError(
        "run_variant_quantized is not available in this modified train.py"
    )


def run_variant_continuous(encoded_ds, args, feature_extractor, output_dir):
    logger.info("Running continuous variant -> %s", output_dir)

    # build model (re-use the project's builder)
    from utils.model_builders import build_continuous_model, save_json
    from utils.reporting import summarize_run_variant
    from utils.utils import compute_metrics
    from utils.training_helpers import (
        make_training_args,
        build_trainer,
        evaluate_and_save_test,
        save_common_artifacts,
        process_test_metrics,
    )

    # build model
    logger.info('Building continuous model...')
    model, params = build_continuous_model(
        args.hf_model, args.num_labels, args.label2id, args.id2label,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_encoder=args.freeze_encoder,
        freeze_first_n=args.freeze_first_n
    )

    save_json(params, os.path.join(output_dir, "param_counts_continuous.json"))

    # prepare trainer
    collate = collate_fn_cont_factory(args)
    training_args = make_training_args(output_dir, args, fp16=False)
    trainer = build_trainer(
        model,
        training_args,
        encoded_ds["train"],
        encoded_ds.get("validation", None),
        compute_metrics,
        collate,
    )

    # Attach the visualization callback
    viz_cb = EmbeddingVizCallback(sample_size=getattr(args, 'viz_sample_size', 512),
                                  perplexity=getattr(args, 'viz_tsne_perplexity', 30),
                                  seed=getattr(args, 'seed', 42),
                                  enabled=getattr(args, 'viz_enabled', True))
    try:
        trainer.add_callback(viz_cb)
        logger.info('Embedding visualization callback added to trainer')
    except Exception as e:
        logger.warning('Could not add visualization callback to trainer: %s', e)

    # train
    t_train = None
    try:
        import time
        t_train = time.time()
        trainer.train()
        train_elapsed = time.time() - t_train
    except Exception:
        raise

    # Forced final visualization + metric extraction on a deterministic sample
    ds_for_viz = trainer.eval_dataset if trainer.eval_dataset is not None else trainer.train_dataset
    if ds_for_viz is None:
        logger.warning("No dataset available for forced visualizations; skipping.")
    else:
        sample_size = min(512, len(ds_for_viz))
        idxs = np.linspace(0, len(ds_for_viz)-1, sample_size).astype(int).tolist()
        try:
            sample_ds = ds_for_viz.select(idxs)
        except Exception:
            sample_ds = [ds_for_viz[i] for i in idxs]

        logger.info("Forcing embedding extraction on %d examples for final viz", len(sample_ds))
        try:
            pred_out = trainer.predict(sample_ds, metric_key_prefix='viz_forced')
            X = pred_out.predictions
            if X is None:
                logger.warning("trainer.predict returned no predictions for visualization.")
            else:
                X = np.asarray(X).astype(np.float32)
                if X.ndim == 3:
                    X = X.reshape(X.shape[0], -1)

                # obtain labels for coloring
                labels = []
                for ex in (sample_ds if isinstance(sample_ds, list) else sample_ds):
                    if isinstance(ex, dict):
                        labels.append(ex.get('language') or ex.get('lang') or ex.get('locale') or ex.get('family') or 'unk')
                    else:
                        labels.append('unk')

                viz_dir = os.path.join(getattr(trainer.args, 'output_dir', os.getcwd()), 'embeddings_forced')
                _safe_mkdir(viz_dir)

                # PCA
                try:
                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X)
                    _plot_scatter(X_pca, labels, 'PCA embeddings', os.path.join(viz_dir, 'pca.png'))
                    np.save(os.path.join(viz_dir, 'pca.npy'), X_pca)
                except Exception as e:
                    logger.exception("PCA failed: %s", e)

                # t-SNE
                try:
                    perp = min(max(5, 30), max(5, X.shape[0]//3))
                    ts = TSNE(n_components=2, perplexity=perp, init='pca', random_state=42, max_iter=1000)
                    X_tsne = ts.fit_transform(X)
                    _plot_scatter(X_tsne, labels, 't-SNE embeddings', os.path.join(viz_dir, 'tsne.png'))
                    np.save(os.path.join(viz_dir, 'tsne.npy'), X_tsne)
                except Exception as e:
                    logger.exception("t-SNE failed: %s", e)

                # MDS on a smaller subset
                try:
                    max_mds = 300
                    idxs_mds = np.linspace(0, X.shape[0]-1, min(max_mds, X.shape[0])).astype(int)
                    X_sub = X[idxs_mds]
                    labels_sub = [labels[i] for i in idxs_mds]
                    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, n_init=4, max_iter=300)
                    X_mds = mds.fit_transform(X_sub)
                    _plot_scatter(X_mds, labels_sub, 'MDS embeddings', os.path.join(viz_dir, 'mds.png'))
                    np.save(os.path.join(viz_dir, 'mds.npy'), X_mds)
                except Exception as e:
                    logger.exception("MDS failed: %s", e)

                # Save raw embeddings
                np.save(os.path.join(viz_dir, 'raw_preds.npy'), X)

                # --- Compute and save intra/inter-language metrics ---
                try:
                    stats = _compute_language_stats(X, labels)
                    _dump_stats_json(stats, os.path.join(viz_dir, 'embedding_stats.json'))
                    _dump_stats_csv(stats, os.path.join(viz_dir, 'embedding_stats_per_label.csv'))
                    # also save pairwise centroid distances as CSV for quick inspection
                    try:
                        with open(os.path.join(viz_dir, 'pairwise_centroid_euclidean.csv'), 'w', encoding='utf-8') as fh:
                            fh.write('pair,euclidean\n')
                            for k, v in stats['pairwise_centroid_euclidean'].items():
                                fh.write(f'{k},{v}\n')
                        with open(os.path.join(viz_dir, 'pairwise_centroid_cosine.csv'), 'w', encoding='utf-8') as fh:
                            fh.write('pair,cosine_dist\n')
                            for k, v in stats['pairwise_centroid_cosine'].items():
                                fh.write(f'{k},{v}\n')
                    except Exception:
                        pass

                    logger.info("Embedding stats computed and saved under %s", viz_dir)
                    # Log concise summary
                    g = stats.get('global', {})
                    logger.info("Embedding stats summary: mean_intra_euclidean=%.4f, mean_centroid_euclidean=%.4f, mean_intra_cosine=%.4f, mean_centroid_cosine=%.4f",
                                g.get('mean_intra_euclidean', 0.0), g.get('mean_centroid_euclidean', 0.0),
                                g.get('mean_intra_cosine', 0.0), g.get('mean_centroid_cosine', 0.0))
                except Exception as e:
                    logger.exception("Failed to compute/save embedding stats: %s", e)

                logger.info("Visualizations saved under %s", viz_dir)
        except Exception as e:
            logger.exception("trainer.predict for forced viz failed: %s", e)

    # optional test evaluation
    test_metrics = None
    if "test" in encoded_ds:
        test_metrics = evaluate_and_save_test(trainer, encoded_ds["test"], output_dir, "continuous", save_json)

    if test_metrics is not None:
        process_test_metrics(trainer, test_metrics, output_dir)

    # save artifacts and summarize
    timings = {"total": None, "train": train_elapsed}
    save_common_artifacts(
        trainer,
        args,
        output_dir,
        params,
        "continuous",
        save_json,
        summarize_run_variant,
        timings,
        test_metrics,
    )

    logger.info("Continuous w/ embb finished")

# End of file
