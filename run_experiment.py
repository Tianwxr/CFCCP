#!/usr/bin/env python
"""
Run CFCP (Cross-Fitted Clustered Conformal Prediction) experiments.

Usage:
    python run_experiment.py --dataset food101 --K 10
    python run_experiment.py --dataset food101_unbal --K 3
    python run_experiment.py --dataset caltech256 --K 5
    python run_experiment.py --dataset sun397 --K 10

Trains ResNet-18 models, computes conformal prediction sets with six
methods (Standard, Classwise, Clustered, CFCP, EnsStd, EnsCFCP), and
saves results as 4D arrays in .pkl files (one per score function).
"""

import argparse
import gc
import os
import pickle
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from methods import (
    DEVICE,
    IMG_SIZE,
    aps_raps_all,
    aps_raps_true,
    build_index_imagefolder,
    cccp,
    classwise_cp,
    clustered_cp,
    epsilon_records,
    exponential_subsample,
    load_or_build_cache,
    load_or_score,
    score_data,
    scores_by_class,
    standard_cp,
)

# ════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    'food101': {
        'images_subdir': os.path.join('food-101', 'images'),
        'prefix': 'food101',
        'cache_prefix': 'food101',
        'imbalance_ratio': None,
    },
    'food101_unbal': {
        'images_subdir': os.path.join('food-101', 'images'),
        'prefix': 'food101_unbal',
        'cache_prefix': 'food101',
        'imbalance_ratio': 10,
    },
    'caltech256': {
        'images_subdir': '256_ObjectCategories',
        'prefix': 'caltech256',
        'cache_prefix': 'caltech256',
        'imbalance_ratio': None,
    },
    'sun397': {
        'images_subdir': 'SUN397',
        'prefix': 'sun397',
        'cache_prefix': 'sun397',
        'imbalance_ratio': None,
    },
}

NAVGS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
ALPHA = 0.1
CAL_RESAMPLES = 10
RAPS_LAM = 0.01
RAPS_KREG = 5

ALL_METHODS = ['Standard', 'CCP', 'Cluster3', 'KFCC3', 'KFCCEns3', 'EnsStd']
CLUSTER_METHODS = ['Cluster3', 'KFCC3', 'KFCCEns3']
FOLD_LABELS = [f'navg={n}' for n in NAVGS]

# Internal → display name mapping for output pkl
_METHODS_INTERNAL = ['Standard', 'CCP', 'Cluster3', 'KFCC3', 'EnsStd', 'KFCCEns3']
_METHOD_NAMES = ['Standard', 'Classwise', 'Clustered', 'CFCP', 'EnsStd', 'EnsCFCP']
_METRICS = ['marginal', 'covgap', 'avgsize', 'c_nr', 'epsilon',
            'pred_acc', 'ens_acc', 'avg_cluster_acc',
            'pred_acc3', 'ens_acc3', 'avg_cluster_acc3',
            'main_time', 'avg_fold_time', 'cal_time', 'test_time']


def _empty_method_dict():
    return {'marginal': [], 'avg_size': [], 'cov_gap': [],
            'cov_gap_clustered': [], 'cov_gap_rare': [],
            'class_cov': [], 'cal_time': [], 'test_time': []}


def _safe_nanmean(vals):
    v = [x for x in vals if not np.isnan(x)]
    return float(np.mean(v)) if v else float('nan')


# ════════════════════════════════════════════════════════════════════
# HPS experiment loop
# ════════════════════════════════════════════════════════════════════

def run_hps(all_paths, all_labels, C, N_TOTAL, K_CV, N_ITERS, TRAIN_SIZE,
            IMBALANCE_RATIO, cache_dir, score_dir, img_cache, path_to_row,
            batch_size, num_workers, lr, num_epochs, patience, use_amp):
    """Run the main HPS experiment: train models, cross-fit, sweep navg."""

    results = {fl: {m: _empty_method_dict() for m in ALL_METHODS} for fl in FOLD_LABELS}
    accs = {fl: [] for fl in FOLD_LABELS}
    accs3 = {fl: [] for fl in FOLD_LABELS}
    ens_accs = {fl: [] for fl in FOLD_LABELS}
    ens_accs3 = {fl: [] for fl in FOLD_LABELS}
    train_times = {fl: [] for fl in FOLD_LABELS}
    eps_rows, cluster_info = [], []
    sf_accs_all, sf_accs3_all = [], []
    main_times, fold_times = [], []

    train_kw = dict(batch_size=batch_size, num_workers=num_workers,
                    lr=lr, num_epochs=num_epochs, patience=patience, use_amp=use_amp)

    for it in range(N_ITERS):
        print(f'\n{"#"*60}\n#  ITERATION {it}\n{"#"*60}')

        # ── Train / rest split ─────────────────────────────────────
        if IMBALANCE_RATIO is not None:
            rng_sub = np.random.RandomState(42 + it)
            sub_idx, _ = exponential_subsample(all_labels, C, IMBALANCE_RATIO, rng_sub)
            sub_labels = all_labels[sub_idx]
            idx_train_local, idx_rest_local = train_test_split(
                np.arange(len(sub_idx)), train_size=TRAIN_SIZE,
                random_state=it, stratify=sub_labels)
            idx_train = sub_idx[np.sort(idx_train_local)]
            idx_rest = sub_idx[np.sort(idx_rest_local)]
            n_pool = len(sub_idx)
        else:
            idx_train, idx_rest = train_test_split(
                np.arange(N_TOTAL), train_size=TRAIN_SIZE,
                random_state=it, stratify=all_labels)
            idx_train, idx_rest = np.sort(idx_train), np.sort(idx_rest)
            n_pool = N_TOTAL

        train_labels = all_labels[idx_train]
        rest_labels = all_labels[idx_rest]
        r_train = TRAIN_SIZE / n_pool
        rtag = f'r{r_train}'

        max_cal = max(NAVGS) * C
        fixed_test_size = len(idx_rest) - max_cal
        cal_pool_idx, _ = train_test_split(
            np.arange(len(idx_rest)), train_size=max_cal,
            random_state=300 + it, stratify=rest_labels)
        cal_pool_abs = idx_rest[np.sort(cal_pool_idx)]

        # ── Main model ─────────────────────────────────────────────
        rest_sm, rest_lab, _, _tsm, tacc, _, ttime, tacc3, main_stime = \
            load_or_score(it, rtag, 0, idx_train, idx_rest,
                          all_paths, all_labels,
                          all_paths[idx_rest], all_labels[idx_rest],
                          C, cache_dir, score_dir,
                          img_cache, path_to_row,
                          tag=f'[it{it} main]', val_local=cal_pool_abs,
                          **train_kw)
        del _tsm
        rest_true = (1.0 - rest_sm[np.arange(len(rest_lab)), rest_lab]).astype(np.float32)
        print(f'  main acc = {tacc:.2f}%  (rest={len(idx_rest)})')

        # ── K-fold cross-fitted subfold models ─────────────────────
        sub_skf = StratifiedKFold(n_splits=K_CV, shuffle=True, random_state=200 + it)
        sub_fold_abs = [np.sort(idx_train[val_idx])
                        for _, val_idx in sub_skf.split(np.arange(TRAIN_SIZE), train_labels)]

        kfcc_rtag = f'r{r_train}_Kcv{K_CV}'
        sf_scores, sf_labels = [], []
        sf_iter_accs, sf_iter_accs3, sf_fold_times, sf_score_times = [], [], [], []

        for sj in range(K_CV):
            sf_cal = sub_fold_abs[sj]
            sf_train_idx = np.sort(np.concatenate(
                [sub_fold_abs[si] for si in range(K_CV) if si != sj]))
            sf_sm, sf_lab, _, _, sf_acc, _, sf_time, sf_acc3, sf_stime = load_or_score(
                it, kfcc_rtag, sj, sf_train_idx, sf_cal,
                all_paths, all_labels,
                all_paths[idx_rest], all_labels[idx_rest],
                C, cache_dir, score_dir,
                img_cache, path_to_row,
                tag=f'[it{it} sf{sj}]', **train_kw)
            sf_true = (1.0 - sf_sm[np.arange(len(sf_lab)), sf_lab]).astype(np.float32)
            sf_scores.append(sf_true)
            sf_labels.append(sf_lab)
            sf_iter_accs.append(float(sf_acc))
            sf_iter_accs3.append(float(sf_acc3) if sf_acc3 is not None else float('nan'))
            sf_fold_times.append(sf_time)
            sf_score_times.append(sf_stime)

        kfcc_clus_true = np.concatenate(sf_scores)
        kfcc_clus_lab = np.concatenate(sf_labels)
        sf_accs_all.append(np.mean(sf_iter_accs))
        sf_accs3_all.append(np.mean(sf_iter_accs3))
        main_times.append(ttime)
        fold_times.append(np.mean(sf_fold_times))

        # ── Ensemble softmax ───────────────────────────────────────
        ens_tag = f'r{r_train}_Kcv{K_CV}_ens'
        ens_spath = os.path.join(score_dir, f'scores_iter{it}_{ens_tag}.npz')
        if os.path.exists(ens_spath):
            d = np.load(ens_spath)
            ens_rest_sm = d['ens_rest_sm'].copy()
            ens_rest_acc = float(d['ens_rest_acc'])
            ens_rest_acc3 = float(d['ens_rest_acc3']) if 'ens_rest_acc3' in d else None
            d.close()
            if ens_rest_acc3 is None:
                ens_rest_acc3 = float((np.argsort(-ens_rest_sm, 1)[:, :3]
                                       == rest_labels[:, None]).any(1).mean() * 100)
                np.savez_compressed(ens_spath, ens_rest_sm=ens_rest_sm,
                                    ens_rest_acc=ens_rest_acc, ens_rest_acc3=ens_rest_acc3)
        else:
            ens_sms = []
            for sj in range(K_CV):
                d_sf = np.load(os.path.join(
                    score_dir, f'scores_iter{it}_{kfcc_rtag}_fold{sj}.npz'))
                ens_sms.append(d_sf['test_softmax'].copy())
                d_sf.close()
            ens_rest_sm = np.mean(ens_sms, axis=0).astype(np.float32)
            ens_rest_acc = float((ens_rest_sm.argmax(1) == rest_labels).mean() * 100)
            ens_rest_acc3 = float((np.argsort(-ens_rest_sm, 1)[:, :3]
                                   == rest_labels[:, None]).any(1).mean() * 100)
            np.savez_compressed(ens_spath, ens_rest_sm=ens_rest_sm,
                                ens_rest_acc=ens_rest_acc, ens_rest_acc3=ens_rest_acc3)

        # ── Sweep navg ─────────────────────────────────────────────
        for navg in NAVGS:
            flabel = f'navg={navg}'
            cal_size = navg * C

            accs[flabel].append(float(tacc))
            accs3[flabel].append(float(tacc3) if tacc3 is not None else float('nan'))
            if ttime > 0:
                train_times[flabel].append(ttime)
            ens_accs[flabel].append(ens_rest_acc)
            ens_accs3[flabel].append(ens_rest_acc3)

            ra = {m: {k: [] for k in ('marginal', 'avg_size', 'cov_gap',
                       'cov_gap_clustered', 'cov_gap_rare')} for m in ALL_METHODS}
            ra_cc = {m: [] for m in ALL_METHODS}
            ra_ci = {k: [] for k in ('Cluster3_nn', 'Cluster3_M', 'KFCC3_nn', 'KFCC3_M')}
            cal_ts = {m: [] for m in ALL_METHODS}
            test_ts = {m: [] for m in ALL_METHODS}

            for rs in range(CAL_RESAMPLES):
                _ci, _rem = train_test_split(
                    np.arange(len(idx_rest)), train_size=cal_size,
                    random_state=5000*it + 100*navg + rs, stratify=rest_lab)
                if len(_rem) > fixed_test_size:
                    _ti, _ = train_test_split(
                        _rem, train_size=fixed_test_size,
                        random_state=5000*it + 100*navg + rs + 50000,
                        stratify=rest_lab[_rem])
                else:
                    _ti = _rem
                _ci, _ti = np.sort(_ci), np.sort(_ti)

                ct, cl = rest_true[_ci], rest_lab[_ci]
                tn = (1.0 - rest_sm[_ti]).astype(np.float32)
                tl = rest_lab[_ti]
                ect = (1.0 - ens_rest_sm[_ci][np.arange(cal_size), cl]).astype(np.float32)
                etn = (1.0 - ens_rest_sm[_ti]).astype(np.float32)

                test_frac = len(_ti) / len(idx_rest)
                main_infer = main_stime * test_frac
                ens_infer = np.mean(sf_score_times) * K_CV * test_frac

                r_std = standard_cp(ct, cl, tn, tl, ALPHA, C)
                r_ccp = classwise_cp(ct, cl, tn, tl, ALPHA, C)
                r_cl = clustered_cp(ct, cl, tn, tl, ALPHA, C)
                r_kf = cccp(kfcc_clus_true, kfcc_clus_lab, ct, cl, tn, tl, ALPHA, C)
                r_ens = cccp(kfcc_clus_true, kfcc_clus_lab, ect, cl, etn, tl, ALPHA, C)
                r_estd = standard_cp(ect, cl, etn, tl, ALPHA, C)

                for nm, r in [('Standard', r_std), ('CCP', r_ccp),
                              ('Cluster3', r_cl), ('KFCC3', r_kf),
                              ('KFCCEns3', r_ens), ('EnsStd', r_estd)]:
                    cal_ts[nm].append(r.pop('_cal_time'))
                    bt = r.pop('_test_time')
                    test_ts[nm].append(bt + (ens_infer if nm in ('KFCCEns3', 'EnsStd') else main_infer))
                    for key in ('marginal', 'avg_size', 'cov_gap'):
                        ra[nm][key].append(float(r[key]))
                    ra_cc[nm].append(r['class_cov'].copy())
                    if nm in CLUSTER_METHODS:
                        ra[nm]['cov_gap_clustered'].append(float(r.get('cov_gap_clustered', float('nan'))))
                        ra[nm]['cov_gap_rare'].append(float(r.get('cov_gap_rare', float('nan'))))

                ra_ci['Cluster3_nn'].append(float(r_cl['n_clustered']))
                ra_ci['Cluster3_M'].append(float(r_cl['n_clusters']))
                ra_ci['KFCC3_nn'].append(float(r_kf['n_clustered']))
                ra_ci['KFCC3_M'].append(float(r_kf['n_clusters']))

            for name in ALL_METHODS:
                for key in ('marginal', 'avg_size', 'cov_gap'):
                    results[flabel][name][key].append(float(np.mean(ra[name][key])))
                results[flabel][name]['class_cov'].append(np.mean(ra_cc[name], axis=0))
                results[flabel][name]['cal_time'].append(float(np.mean(cal_ts[name])))
                results[flabel][name]['test_time'].append(float(np.mean(test_ts[name])))
                if name in CLUSTER_METHODS:
                    results[flabel][name]['cov_gap_clustered'].append(
                        _safe_nanmean(ra[name]['cov_gap_clustered']))
                    results[flabel][name]['cov_gap_rare'].append(
                        _safe_nanmean(ra[name]['cov_gap_rare']))

            ci_entry = {'navg': navg, 'iter': it}
            for k in ('Cluster3_nn', 'Cluster3_M', 'KFCC3_nn', 'KFCC3_M'):
                ci_entry[k] = float(np.mean(ra_ci[k]))
            cluster_info.append(ci_entry)

            sbc = scores_by_class(tn, tl, C)
            for rec in epsilon_records(r_cl['c2c'], r_cl['class_cov'], sbc, ALPHA, C):
                eps_rows.append({'navg': navg, 'iter': it, 'method': 'Cluster3', **rec})
            for rec in epsilon_records(r_kf['c2c'], r_kf['class_cov'], sbc, ALPHA, C):
                eps_rows.append({'navg': navg, 'iter': it, 'method': 'KFCC3', **rec})
            sbc_ens = scores_by_class(etn, tl, C)
            for rec in epsilon_records(r_ens['c2c'], r_ens['class_cov'], sbc_ens, ALPHA, C):
                eps_rows.append({'navg': navg, 'iter': it, 'method': 'KFCCEns3', **rec})

            print(f'  navg={navg} (cal={cal_size}): '
                  f'Std={np.mean(ra["Standard"]["cov_gap"]):.2f} '
                  f'CCP={np.mean(ra["CCP"]["cov_gap"]):.2f} '
                  f'Cl={np.mean(ra["Cluster3"]["cov_gap"]):.2f} '
                  f'CFCP={np.mean(ra["KFCC3"]["cov_gap"]):.2f} '
                  f'Ens={np.mean(ra["KFCCEns3"]["cov_gap"]):.2f}')

        del rest_sm, rest_true, rest_lab, ens_rest_sm, kfcc_clus_true, kfcc_clus_lab
        gc.collect()
        torch.cuda.empty_cache()

    return {
        'results': results, 'accs': accs, 'train_times': train_times,
        'NAVGS': NAVGS, 'FOLD_LABELS': FOLD_LABELS, 'K_CV': K_CV,
        'n_iters': N_ITERS, 'eps_rows': eps_rows,
        'cluster_info': cluster_info, 'ens_accs': ens_accs,
        'sf_accs': sf_accs_all, 'accs3': accs3, 'ens_accs3': ens_accs3,
        'sf_accs3': sf_accs3_all,
        'main_times_iter': main_times, 'fold_times_iter': fold_times,
    }


# ════════════════════════════════════════════════════════════════════
# APS + RAPS pass (reuses cached softmax from HPS)
# ════════════════════════════════════════════════════════════════════

def run_aps_raps(all_labels, C, N_TOTAL, K_CV, N_ITERS, TRAIN_SIZE,
                 IMBALANCE_RATIO, score_dir):
    """Compute APS and RAPS results from cached softmax scores."""

    aps_results = {fl: {m: {'cov_gap': [], 'marginal': [], 'avg_size': []}
                        for m in ALL_METHODS} for fl in FOLD_LABELS}
    raps_results = {fl: {m: {'cov_gap': [], 'marginal': [], 'avg_size': []}
                         for m in ALL_METHODS} for fl in FOLD_LABELS}
    aps_eps_lookup, raps_eps_lookup = {}, {}

    for it in range(N_ITERS):
        print(f'  APS+RAPS iter {it}')

        if IMBALANCE_RATIO is not None:
            rng_sub = np.random.RandomState(42 + it)
            sub_idx, _ = exponential_subsample(all_labels, C, IMBALANCE_RATIO, rng_sub)
            sub_labels = all_labels[sub_idx]
            idx_train_local, idx_rest_local = train_test_split(
                np.arange(len(sub_idx)), train_size=TRAIN_SIZE,
                random_state=it, stratify=sub_labels)
            idx_train = sub_idx[np.sort(idx_train_local)]
            n_pool = len(sub_idx)
        else:
            idx_train, _ = train_test_split(
                np.arange(N_TOTAL), train_size=TRAIN_SIZE,
                random_state=it, stratify=all_labels)
            n_pool = N_TOTAL

        train_labels = all_labels[idx_train]
        r_train = TRAIN_SIZE / n_pool
        rtag = f'r{r_train}'
        kfcc_rtag = f'r{r_train}_Kcv{K_CV}'

        d0 = np.load(os.path.join(score_dir, f'scores_iter{it}_{rtag}_fold0.npz'))
        rest_sm = d0['cal_softmax']
        rest_lab = d0['cal_lab']
        d0.close()

        n_rest = len(rest_lab)
        fixed_test_size = n_rest - max(NAVGS) * C

        rest_true_aps, rest_true_raps = aps_raps_true(rest_sm, rest_lab, RAPS_LAM, RAPS_KREG)
        rest_all_aps, rest_all_raps = aps_raps_all(rest_sm, RAPS_LAM, RAPS_KREG)

        sub_skf = StratifiedKFold(n_splits=K_CV, shuffle=True, random_state=200 + it)
        list(sub_skf.split(np.arange(TRAIN_SIZE), train_labels))
        sf_aps_list, sf_raps_list, sf_lab_list = [], [], []
        for sj in range(K_CV):
            sf_sp = os.path.join(score_dir, f'scores_iter{it}_{kfcc_rtag}_fold{sj}.npz')
            d_sf = np.load(sf_sp)
            sf_a, sf_r = aps_raps_true(d_sf['cal_softmax'], d_sf['cal_lab'], RAPS_LAM, RAPS_KREG)
            sf_aps_list.append(sf_a)
            sf_raps_list.append(sf_r)
            sf_lab_list.append(d_sf['cal_lab'])
            d_sf.close()
        kfcc_true_aps = np.concatenate(sf_aps_list)
        kfcc_true_raps = np.concatenate(sf_raps_list)
        kfcc_lab = np.concatenate(sf_lab_list)

        ens_tag = f'r{r_train}_Kcv{K_CV}_ens'
        ens_spath = os.path.join(score_dir, f'scores_iter{it}_{ens_tag}.npz')
        ens_rest_sm = None
        if os.path.exists(ens_spath):
            d_ens = np.load(ens_spath)
            ens_rest_sm = d_ens.get('ens_rest_sm', rest_sm.copy())
            d_ens.close()

        for navg in NAVGS:
            flabel = f'navg={navg}'
            cal_size = navg * C

            ra_a = {m: {k: [] for k in ('marginal', 'avg_size', 'cov_gap')} for m in ALL_METHODS}
            ra_r = {m: {k: [] for k in ('marginal', 'avg_size', 'cov_gap')} for m in ALL_METHODS}

            for rs in range(CAL_RESAMPLES):
                _ci, _rem = train_test_split(
                    np.arange(n_rest), train_size=cal_size,
                    random_state=5000*it + 100*navg + rs, stratify=rest_lab)
                if len(_rem) > fixed_test_size:
                    _ti, _ = train_test_split(
                        _rem, train_size=fixed_test_size,
                        random_state=5000*it + 100*navg + rs + 50000,
                        stratify=rest_lab[_rem])
                else:
                    _ti = _rem
                _ci, _ti = np.sort(_ci), np.sort(_ti)

                cal_lab_rs = rest_lab[_ci]
                tl = rest_lab[_ti]

                for score_nm, cal_true, test_nc, kf_true, ra_dict in [
                        ('APS', rest_true_aps, rest_all_aps, kfcc_true_aps, ra_a),
                        ('RAPS', rest_true_raps, rest_all_raps, kfcc_true_raps, ra_r)]:
                    ct, tn = cal_true[_ci], test_nc[_ti]
                    s_std = standard_cp(ct, cal_lab_rs, tn, tl, ALPHA, C)
                    s_ccp = classwise_cp(ct, cal_lab_rs, tn, tl, ALPHA, C)
                    s_cl = clustered_cp(ct, cal_lab_rs, tn, tl, ALPHA, C)
                    s_kf = cccp(kf_true, kfcc_lab, ct, cal_lab_rs, tn, tl, ALPHA, C)

                    for nm, r in [('Standard', s_std), ('CCP', s_ccp),
                                  ('Cluster3', s_cl), ('KFCC3', s_kf)]:
                        for key in ('marginal', 'avg_size', 'cov_gap'):
                            ra_dict[nm][key].append(float(r[key]))

                    if ens_rest_sm is not None:
                        ens_cal, ens_cal_r = aps_raps_true(ens_rest_sm[_ci], cal_lab_rs, RAPS_LAM, RAPS_KREG)
                        ens_test, ens_test_r = aps_raps_all(ens_rest_sm[_ti], RAPS_LAM, RAPS_KREG)
                        if score_nm == 'APS':
                            ec, et = ens_cal, ens_test
                        else:
                            ec, et = ens_cal_r, ens_test_r
                        s_ens = cccp(kf_true, kfcc_lab, ec, cal_lab_rs, et, tl, ALPHA, C)
                        s_estd = standard_cp(ec, cal_lab_rs, et, tl, ALPHA, C)
                        for key in ('marginal', 'avg_size', 'cov_gap'):
                            ra_dict['KFCCEns3'][key].append(float(s_ens[key]))
                            ra_dict['EnsStd'][key].append(float(s_estd[key]))

            for name in ALL_METHODS:
                for key in ('marginal', 'avg_size', 'cov_gap'):
                    if ra_a[name][key]:
                        aps_results[flabel][name][key].append(float(np.mean(ra_a[name][key])))
                    if ra_r[name][key]:
                        raps_results[flabel][name][key].append(float(np.mean(ra_r[name][key])))

            # Epsilon (last resample)
            for score_nm, ep_lookup, test_nc, kf_true, r_cl_obj, r_kf_obj in [
                    ('APS', aps_eps_lookup, rest_all_aps[_ti], kfcc_true_aps, s_cl, s_kf),
                    ('RAPS', raps_eps_lookup, rest_all_raps[_ti], kfcc_true_raps, s_cl, s_kf)]:
                sbc_s = scores_by_class(test_nc, tl, C)
                for em_nm, r_obj in [('Cluster3', r_cl_obj), ('KFCC3', r_kf_obj)]:
                    erecs = epsilon_records(r_obj['c2c'], r_obj['class_cov'], sbc_s, ALPHA, C)
                    if erecs:
                        ep_lookup[(navg, it, em_nm)] = np.mean([e['epsilon'] for e in erecs])

    return aps_results, raps_results, aps_eps_lookup, raps_eps_lookup


# ════════════════════════════════════════════════════════════════════
# Save 4D pkl arrays
# ════════════════════════════════════════════════════════════════════

def save_4d_pkls(hps_data, aps_results, raps_results,
                 aps_eps_lookup, raps_eps_lookup,
                 DATASET_PREFIX, K_CV, N_ITERS, out_dir):
    """Aggregate all scores into (6 methods × 19 navgs × N_iters × 15 metrics) arrays."""

    ci_lookup = {}
    for ci in hps_data['cluster_info']:
        ci_lookup[(ci['navg'], ci['iter'])] = ci

    hps_eps_lookup = {}
    for er in hps_data.get('eps_rows', []):
        ek = (er['navg'], er['iter'], er['method'])
        hps_eps_lookup.setdefault(ek, []).append(er['epsilon'])
    hps_eps_lookup = {k: np.mean(v) for k, v in hps_eps_lookup.items()}

    score_data_map = {
        'HPS': (hps_data['results'], hps_eps_lookup),
        'APS': (aps_results, aps_eps_lookup),
        'RAPS': (raps_results, raps_eps_lookup),
    }

    os.makedirs(out_dir, exist_ok=True)

    for score_name in ['HPS', 'APS', 'RAPS']:
        res_dict, ep_lookup = score_data_map[score_name]
        arr = np.full((len(_METHODS_INTERNAL), len(NAVGS), N_ITERS, len(_METRICS)), np.nan)

        for mi, method in enumerate(_METHODS_INTERNAL):
            for ni, navg in enumerate(NAVGS):
                fl = f'navg={navg}'
                r = res_dict[fl][method]
                n_done = len(r.get('marginal', r.get('cov_gap', [])))
                for it in range(min(n_done, N_ITERS)):
                    ci_e = ci_lookup.get((navg, it), {})
                    if method == 'Cluster3':
                        c_nr = ci_e.get('Cluster3_nn', np.nan)
                    elif method in ('KFCC3', 'KFCCEns3'):
                        c_nr = ci_e.get('KFCC3_nn', np.nan)
                    else:
                        c_nr = np.nan

                    arr[mi, ni, it, 0] = r['marginal'][it]
                    arr[mi, ni, it, 1] = r['cov_gap'][it]
                    arr[mi, ni, it, 2] = r['avg_size'][it]
                    arr[mi, ni, it, 3] = c_nr
                    arr[mi, ni, it, 4] = ep_lookup.get((navg, it, method), np.nan)
                    arr[mi, ni, it, 5] = hps_data['accs'][fl][it]
                    arr[mi, ni, it, 6] = hps_data['ens_accs'][fl][it]
                    arr[mi, ni, it, 7] = hps_data['sf_accs'][it]
                    arr[mi, ni, it, 8] = hps_data['accs3'][fl][it] if 'accs3' in hps_data else np.nan
                    arr[mi, ni, it, 9] = hps_data['ens_accs3'][fl][it] if 'ens_accs3' in hps_data else np.nan
                    arr[mi, ni, it, 10] = hps_data['sf_accs3'][it] if 'sf_accs3' in hps_data else np.nan
                    arr[mi, ni, it, 11] = hps_data['main_times_iter'][it] if 'main_times_iter' in hps_data else np.nan
                    arr[mi, ni, it, 12] = hps_data['fold_times_iter'][it] if 'fold_times_iter' in hps_data else np.nan
                    arr[mi, ni, it, 13] = r.get('cal_time', [np.nan]*N_ITERS)[it] if it < len(r.get('cal_time', [])) else np.nan
                    arr[mi, ni, it, 14] = r.get('test_time', [np.nan]*N_ITERS)[it] if it < len(r.get('test_time', [])) else np.nan

        out_pkl = os.path.join(out_dir, f'{DATASET_PREFIX}_{K_CV}F_{score_name}.pkl')
        with open(out_pkl, 'wb') as f:
            pickle.dump({
                'data': arr,
                'axes': ['method', 'navg', 'iter', 'metric'],
                'methods': _METHOD_NAMES,
                'navgs': NAVGS,
                'n_iters': N_ITERS,
                'metrics': _METRICS,
                'dataset': DATASET_PREFIX,
                'K': K_CV,
                'score': score_name,
                'raps_lam': RAPS_LAM if score_name == 'RAPS' else None,
                'raps_kreg': RAPS_KREG if score_name == 'RAPS' else None,
            }, f)
        print(f'  {out_pkl}  {arr.shape}')


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def run_experiment(args):
    cfg = DATASET_CONFIGS[args.dataset]
    DATASET_PREFIX = cfg['prefix']
    IMBALANCE_RATIO = cfg['imbalance_ratio']
    K_CV = args.K
    N_ITERS = args.n_iters
    TRAIN_SIZE = args.train_size

    print(f'Device: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Dataset: {args.dataset}  K={K_CV}  iters={N_ITERS}  train={TRAIN_SIZE}')

    # ── Load dataset ───────────────────────────────────────────────
    images_dir = os.path.join(args.data_dir, cfg['images_subdir'])
    assert os.path.isdir(images_dir), \
        f'Dataset not found: {images_dir}. Run prepare_data.py first.'

    all_paths, str_labels = build_index_imagefolder(images_dir)
    keep = np.array([str(l).strip().lower() not in ('', 'nan') for l in str_labels], dtype=bool)
    all_paths, str_labels = all_paths[keep], str_labels[keep]
    class_names = sorted(np.unique(str_labels).tolist())
    label_to_id = {c: i for i, c in enumerate(class_names)}
    all_labels = np.array([label_to_id[l] for l in str_labels], dtype=np.int64)
    C = len(class_names)
    N_TOTAL = len(all_labels)

    counts = np.bincount(all_labels, minlength=C)
    print(f'Classes: {C}  Images: {N_TOTAL}')
    print(f'Per-class: min={counts.min()}, median={int(np.median(counts))}, max={counts.max()}')

    # ── Image cache ────────────────────────────────────────────────
    cache_path = os.path.join(args.data_dir,
                              f'{cfg["cache_prefix"]}_img{IMG_SIZE}_uint8.npy')
    img_cache = load_or_build_cache(all_paths, cache_path)
    path_to_row = {p: i for i, p in enumerate(all_paths)}

    # ── Output directories ─────────────────────────────────────────
    suffix = f'{DATASET_PREFIX}_ft_kfcc'
    cache_dir = os.path.join(args.results_dir, f'{suffix}_cache')
    score_dir = os.path.join(args.results_dir, f'{suffix}_scores')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    # ── HPS pass ───────────────────────────────────────────────────
    hps_data = run_hps(
        all_paths, all_labels, C, N_TOTAL, K_CV, N_ITERS, TRAIN_SIZE,
        IMBALANCE_RATIO, cache_dir, score_dir, img_cache, path_to_row,
        batch_size=args.batch_size, num_workers=0,
        lr=args.lr, num_epochs=args.num_epochs, patience=args.patience,
        use_amp=True)

    hps_pkl = os.path.join(args.results_dir, f'{suffix}_combined_results.pkl')
    with open(hps_pkl, 'wb') as f:
        pickle.dump(hps_data, f)
    print(f'\nSaved HPS results to {hps_pkl}')

    # ── APS + RAPS pass ───────────────────────────────────────────
    print(f'\n{"="*60}\n  APS + RAPS pass\n{"="*60}')
    aps_results, raps_results, aps_eps, raps_eps = run_aps_raps(
        all_labels, C, N_TOTAL, K_CV, N_ITERS, TRAIN_SIZE,
        IMBALANCE_RATIO, score_dir)

    # ── Save 4D pkl arrays ─────────────────────────────────────────
    print(f'\n{"="*60}\n  Saving 4D pkl arrays\n{"="*60}')
    out_dir = os.path.join(args.results_dir, DATASET_PREFIX)
    save_4d_pkls(hps_data, aps_results, raps_results,
                 aps_eps, raps_eps,
                 DATASET_PREFIX, K_CV, N_ITERS, out_dir)

    print('\nDone.')


def main():
    parser = argparse.ArgumentParser(
        description='Run CFCP conformal prediction experiments.')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--K', type=int, required=True, choices=[3, 5, 10],
                        help='Number of cross-fitting folds')
    parser.add_argument('--data-dir', default='data',
                        help='Base directory for datasets (default: data/)')
    parser.add_argument('--results-dir', default='results',
                        help='Results directory (default: results/)')
    parser.add_argument('--n-iters', type=int, default=20,
                        help='Number of iterations (default: 20)')
    parser.add_argument('--train-size', type=int, default=10000,
                        help='Training set size (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='Max training epochs (default: 1000)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
