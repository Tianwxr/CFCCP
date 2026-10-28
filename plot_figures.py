"""Regenerate figures from pre-computed result arrays.

Usage:
    python plot_figures.py                    # all datasets, all K, all scores
    python plot_figures.py --dataset food101  # single dataset
    python plot_figures.py --K 10             # single K
    python plot_figures.py --score HPS        # single score function
    python plot_figures.py --grid-only        # only the multi-dataset grid
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

METHODS = ['Standard', 'Classwise', 'Clustered', 'CFCP', 'EnsStd', 'EnsCFCP']
METHOD_STYLES = {
    'Standard':  {'color': 'tab:blue',   'ls': '-',  'marker': 'o',  'label': 'Standard'},
    'Classwise': {'color': 'tab:red',    'ls': '--', 'marker': 's',  'label': 'Classwise'},
    'Clustered': {'color': 'tab:green',  'ls': '-.', 'marker': '^',  'label': 'Clustered'},
    'CFCP':      {'color': 'tab:orange', 'ls': '-',  'marker': 'D',  'label': 'CFCP'},
    'EnsStd':    {'color': 'tab:cyan',   'ls': '--', 'marker': 'v',  'label': 'EnsStd'},
    'EnsCFCP':   {'color': 'tab:purple', 'ls': '-.', 'marker': 'P',  'label': 'EnsCFCP'},
}
PLOT_ORDER = ['Standard', 'Classwise', 'Clustered', 'CFCP', 'EnsStd', 'EnsCFCP']

DATASETS = ['food101', 'food101_unbal', 'caltech256', 'sun397']
DATASET_LABELS = {
    'food101': 'Food-101',
    'food101_unbal': 'Food-101 (Unbalanced)',
    'caltech256': 'Caltech-256',
    'sun397': 'SUN-397',
}
K_VALUES = [3, 5, 10]
SCORES = ['HPS', 'APS', 'RAPS']

METRIC_NAMES = [
    'marginal', 'covgap', 'avgsize', 'c_nr', 'epsilon',
    'pred_acc', 'ens_acc', 'avg_cluster_acc',
    'pred_acc3', 'ens_acc3', 'avg_cluster_acc3',
    'main_time', 'avg_fold_time', 'cal_time', 'test_time',
]
MI = {name: i for i, name in enumerate(METRIC_NAMES)}

NAVGS = list(range(4, 42, 2))

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pkl(dataset, K, score, results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, dataset, f'{dataset}_{K}F_{score}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_covgap(data, method_idx):
    """Return (navgs, means, sems) for CovGap, converting percentage to fraction."""
    arr = data['data']
    covgap = arr[method_idx, :, :, MI['covgap']] / 100.0
    means = np.nanmean(covgap, axis=1)
    stds = np.nanstd(covgap, axis=1, ddof=1)
    n = np.sum(~np.isnan(covgap), axis=1).clip(min=1)
    sems = stds / np.sqrt(n)
    return NAVGS, means, sems


def extract_metric(data, method_idx, metric, scale=1.0):
    """Return (navgs, means, sems) for any metric."""
    arr = data['data']
    vals = arr[method_idx, :, :, MI[metric]] * scale
    means = np.nanmean(vals, axis=1)
    stds = np.nanstd(vals, axis=1, ddof=1)
    n = np.sum(~np.isnan(vals), axis=1).clip(min=1)
    sems = stds / np.sqrt(n)
    return NAVGS, means, sems


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_covgap_vs_navg(data, dataset, K, score, log_y=False, figures_dir=FIGURES_DIR):
    """CovGap vs calibration size for one dataset/K/score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for mi, method in enumerate(PLOT_ORDER):
        s = METHOD_STYLES[method]
        xs, ys, errs = extract_covgap(data, mi)
        ax.errorbar(xs, ys, yerr=errs,
                    color=s['color'], linestyle=s['ls'],
                    marker=s['marker'], markersize=6,
                    capsize=3, linewidth=1.5, label=s['label'])
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.grid(True, alpha=0.3, which='both')
    else:
        ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'Calibration samples per class ($n_{\mathrm{avg}}$)', fontsize=12)
    ax.set_ylabel('CovGap', fontsize=12)
    score_tag = f' ({score})' if score != 'HPS' else ''
    ax.set_title(f'CovGap vs. Calibration Size — {DATASET_LABELS[dataset]}{score_tag}, K={K}',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper right')

    suffix = '_log' if log_y else ''
    fname = f'{dataset}_ft_{score.lower()}_covgap_vs_navg_dense{suffix}_{K}f.png'
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out


def plot_avgsize_vs_navg(data, dataset, K, score, figures_dir=FIGURES_DIR):
    """Average prediction-set size vs calibration size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for mi, method in enumerate(PLOT_ORDER):
        s = METHOD_STYLES[method]
        xs, ys, errs = extract_metric(data, mi, 'avgsize')
        ax.errorbar(xs, ys, yerr=errs,
                    color=s['color'], linestyle=s['ls'],
                    marker=s['marker'], markersize=6,
                    capsize=3, linewidth=1.5, label=s['label'])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'Calibration samples per class ($n_{\mathrm{avg}}$)', fontsize=12)
    ax.set_ylabel('Avg. Prediction-Set Size', fontsize=12)
    score_tag = f' ({score})' if score != 'HPS' else ''
    ax.set_title(f'Avg. Set Size vs. Calibration Size — {DATASET_LABELS[dataset]}{score_tag}, K={K}',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper right')

    fname = f'{dataset}_ft_{score.lower()}_avgsize_vs_navg_{K}f.png'
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out


def plot_covgap_grid(K=10, score='HPS', results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR):
    """Multi-panel CovGap grid across all four datasets."""
    grid_datasets = ['food101', 'food101_unbal', 'caltech256', 'sun397']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for idx, dataset in enumerate(grid_datasets):
        ax = axes[idx]
        data = load_pkl(dataset, K, score, results_dir)
        if data is None:
            ax.set_title(f'{DATASET_LABELS[dataset]} — not found')
            continue
        for mi, method in enumerate(PLOT_ORDER):
            s = METHOD_STYLES[method]
            xs, ys, errs = extract_covgap(data, mi)
            ax.errorbar(xs, ys, yerr=errs,
                        color=s['color'], linestyle=s['ls'],
                        marker=s['marker'], markersize=5,
                        capsize=2, linewidth=1.5, label=s['label'])
        ax.grid(True, alpha=0.3)
        ax.set_title(DATASET_LABELS[dataset], fontsize=13)
        ax.set_ylabel('CovGap', fontsize=11)
        if idx >= 2:
            ax.set_xlabel(r'$n_{\mathrm{avg}}$', fontsize=11)

    axes[0].legend(fontsize=8, loc='upper right')
    score_tag = f' ({score})' if score != 'HPS' else ''
    fig.suptitle(f'CovGap vs. Calibration Size{score_tag}, K={K}', fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f'covgap_vs_navg_grid_{K}f.png'
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out


def plot_covgap_K_comparison(dataset='food101', score='HPS',
                             results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR):
    """Compare CovGap across K=3, 5, 10 for CFCP and EnsCFCP."""
    fig, ax = plt.subplots(figsize=(8, 5))
    k_colors = {3: 'tab:green', 5: 'tab:blue', 10: 'tab:red'}
    k_styles = {3: '--', 5: '-.', 10: '-'}

    for method, marker in [('CFCP', 'D'), ('EnsCFCP', 'P')]:
        mi = METHODS.index(method)
        for K in K_VALUES:
            data = load_pkl(dataset, K, score, results_dir)
            if data is None:
                continue
            xs, ys, errs = extract_covgap(data, mi)
            ax.errorbar(xs, ys, yerr=errs,
                        color=k_colors[K], linestyle=k_styles[K],
                        marker=marker, markersize=6,
                        capsize=3, linewidth=1.5,
                        label=f'{method} (K={K})')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'Calibration samples per class ($n_{\mathrm{avg}}$)', fontsize=12)
    ax.set_ylabel('CovGap', fontsize=12)
    ax.set_title(f'CovGap: K Comparison — {DATASET_LABELS[dataset]}', fontsize=13)
    ax.legend(fontsize=9)

    fname = f'{dataset}_covgap_K_comparison.png'
    out = os.path.join(figures_dir, fname)
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate figures from result arrays')
    parser.add_argument('--dataset', choices=DATASETS, help='Single dataset')
    parser.add_argument('--K', type=int, choices=K_VALUES, help='Single K value')
    parser.add_argument('--score', choices=SCORES, help='Single score function')
    parser.add_argument('--grid-only', action='store_true',
                        help='Only generate the multi-dataset grid figure')
    parser.add_argument('--results-dir', default=RESULTS_DIR)
    parser.add_argument('--figures-dir', default=FIGURES_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir
    figures_dir = args.figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    if args.grid_only:
        for K in ([args.K] if args.K else K_VALUES):
            for score in ([args.score] if args.score else ['HPS']):
                out = plot_covgap_grid(K=K, score=score,
                                       results_dir=results_dir, figures_dir=figures_dir)
                print(f'Saved: {out}')
        return

    datasets = [args.dataset] if args.dataset else DATASETS
    ks = [args.K] if args.K else K_VALUES
    scores = [args.score] if args.score else SCORES

    for dataset in datasets:
        for K in ks:
            for score in scores:
                data = load_pkl(dataset, K, score, results_dir)
                if data is None:
                    print(f'  Skip {dataset} K={K} {score} (not found)')
                    continue

                out = plot_covgap_vs_navg(data, dataset, K, score,
                                          figures_dir=figures_dir)
                print(f'Saved: {out}')

                out = plot_covgap_vs_navg(data, dataset, K, score, log_y=True,
                                          figures_dir=figures_dir)
                print(f'Saved: {out}')

                out = plot_avgsize_vs_navg(data, dataset, K, score,
                                           figures_dir=figures_dir)
                print(f'Saved: {out}')

    for K in ks:
        for score in ([args.score] if args.score else ['HPS']):
            out = plot_covgap_grid(K=K, score=score,
                                   results_dir=results_dir, figures_dir=figures_dir)
            print(f'Saved: {out}')

    if not args.dataset:
        for dataset in datasets:
            out = plot_covgap_K_comparison(dataset,
                                           results_dir=results_dir, figures_dir=figures_dir)
            print(f'Saved: {out}')


if __name__ == '__main__':
    main()
