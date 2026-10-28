"""
CFCP: conformal prediction methods, model training, scoring, and data loading.

This module contains all reusable components for the CFCP experiments:
- Data loading and image caching
- ResNet-18 training and softmax scoring
- Conformal prediction methods (Standard, Classwise, Clustered, CFCP)
- APS and RAPS nonconformity scores
- Proposition 3 epsilon (per-class KS distance)
"""

import gc
import os
import time
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# ════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

_eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


# ════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════

def build_index_imagefolder(images_dir):
    """Index an imagefolder directory: sorted classes, sorted files per class."""
    paths, labels = [], []
    for cls in sorted(os.listdir(images_dir)):
        d = os.path.join(images_dir, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                paths.append(os.path.join(d, fn))
                labels.append(cls)
    return np.array(paths), np.array(labels)


class PathDataset(Dataset):
    """Load images from disk on the fly."""
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform or _eval_tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.transform(img), int(self.labels[i])


class CachedDataset(Dataset):
    """Load images from a uint8 memory-mapped numpy cache."""
    def __init__(self, rows, labels, img_cache):
        self.rows = rows
        self.labels = labels
        self.img_cache = img_cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        arr = np.array(self.img_cache[self.rows[i]])
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        return _normalize(t), int(self.labels[i])


def load_or_build_cache(all_paths, cache_path):
    """Load an existing uint8 image cache, or build one from scratch."""
    _resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    shape = (len(all_paths), IMG_SIZE, IMG_SIZE, 3)
    done_flag = cache_path + '.done'

    if os.path.exists(cache_path) and os.path.exists(done_flag):
        try:
            arr = np.load(cache_path, mmap_mode='r')
            if arr.shape == shape:
                print(f'Reusing image cache: {cache_path}  {shape}')
                return arr
        except Exception:
            pass

    gb = np.prod(shape) / 1e9
    print(f'Building image cache -> {cache_path} ({shape}, ~{gb:.1f} GB)...')
    if os.path.exists(done_flag):
        os.remove(done_flag)
    mm = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.uint8, shape=shape)
    for i, p in enumerate(all_paths):
        mm[i] = np.asarray(_resize(Image.open(p).convert('RGB')), dtype=np.uint8)
        if (i + 1) % 3000 == 0:
            print(f'  cached {i+1}/{len(all_paths)}', flush=True)
    mm.flush()
    del mm
    gc.collect()
    open(done_flag, 'w').close()
    return np.load(cache_path, mmap_mode='r')


def make_loader(paths, labels, shuffle, batch_size=32, num_workers=0,
                img_cache=None, path_to_row=None):
    """Create a DataLoader, using the image cache if available."""
    if img_cache is not None and path_to_row is not None:
        rows = np.fromiter((path_to_row[p] for p in paths),
                           dtype=np.int64, count=len(paths))
        ds = CachedDataset(rows, labels, img_cache)
    else:
        ds = PathDataset(paths, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=num_workers)


# ════════════════════════════════════════════════════════════════════
# Model training and scoring
# ════════════════════════════════════════════════════════════════════

def train_model(train_paths, train_labels, val_paths, val_labels,
                num_classes, img_cache=None, path_to_row=None,
                lr=0.0001, num_epochs=1000, patience=3,
                batch_size=32, num_workers=0, use_amp=True, tag=''):
    """Fine-tune a pretrained ResNet-18, return (model, best_acc, train_time)."""
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     img_cache=img_cache, path_to_row=path_to_row)
    train_loader = make_loader(train_paths, train_labels, True, **loader_kw)
    val_loader = make_loader(val_paths, val_labels, False, **loader_kw)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE).train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    use_amp = use_amp and DEVICE == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc, best_epoch, best_state = 0.0, 0, None
    wait = 0
    t0 = time.time()
    for ep in range(num_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(xb)
                correct += out.argmax(1).eq(yb).sum().item()
                total += yb.size(0)
        model.train()
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc, best_epoch = acc, ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    train_time = time.time() - t0
    print(f'    {tag} ValAcc={best_acc:.2f}% @ep{best_epoch} '
          f'({ep+1}/{num_epochs} epochs, {train_time:.1f}s)')
    model.load_state_dict(best_state)
    return model, best_acc, train_time


def score_data(model, paths, labels, img_cache=None, path_to_row=None,
               batch_size=32, num_workers=0):
    """Compute softmax scores, return (softmax, top1_acc, top3_acc)."""
    model.eval()
    loader = make_loader(paths, labels, False, batch_size=batch_size,
                         num_workers=num_workers,
                         img_cache=img_cache, path_to_row=path_to_row)
    all_logits = []
    with torch.no_grad():
        for xb, _ in loader:
            all_logits.append(model(xb.to(DEVICE, non_blocking=True)).cpu())
    sm = F.softmax(torch.cat(all_logits, 0), dim=1).numpy()
    acc = (sm.argmax(1) == labels).mean() * 100
    acc3 = (np.argsort(-sm, 1)[:, :3] == labels[:, None]).any(1).mean() * 100
    return sm.astype(np.float32), acc, acc3


def load_or_score(it, rtag, j, train_local, cal_local,
                  pool_paths, pool_labels, test_paths, test_labels,
                  num_classes, cache_dir, score_dir,
                  img_cache=None, path_to_row=None,
                  batch_size=32, num_workers=0, lr=0.0001,
                  num_epochs=1000, patience=3, use_amp=True,
                  tag='', val_local=None):
    """Load cached scores or train a model and compute them."""
    spath = os.path.join(score_dir, f'scores_iter{it}_{rtag}_fold{j}.npz')
    if os.path.exists(spath):
        try:
            d = np.load(spath)
            tsm = d['test_softmax'].copy()
            acc3 = float(d['test_acc3']) if 'test_acc3' in d else None
            ttime = float(d['train_time']) if 'train_time' in d else 0.0
            stime = float(d['score_time']) if 'score_time' in d else 0.0
            ret = (d['cal_softmax'].copy(), d['cal_lab'].copy(), d['cal_idx'].copy(),
                   tsm, float(d['test_acc']), True, ttime)
            d.close()
            if acc3 is None:
                acc3 = float((np.argsort(-tsm, 1)[:, :3] == test_labels[:, None]).any(1).mean() * 100)
            return ret + (acc3, stime)
        except Exception as e:
            print(f'    {tag} corrupt {spath}: {e} -> rescore')
            os.remove(spath)

    mpath = os.path.join(cache_dir, f'model_iter{it}_{rtag}_fold{j}.pt')
    train_time = 0.0
    loaded = False
    if os.path.exists(mpath):
        try:
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(mpath, map_location=DEVICE, weights_only=True))
            model = model.to(DEVICE)
            loaded = True
        except (EOFError, RuntimeError) as e:
            print(f'    {tag} corrupt {mpath}: {e} -> retrain')
            os.remove(mpath)
    if not loaded:
        vl = val_local if val_local is not None else cal_local
        model, _, train_time = train_model(
            pool_paths[train_local], pool_labels[train_local],
            pool_paths[vl], pool_labels[vl],
            num_classes, img_cache, path_to_row,
            lr=lr, num_epochs=num_epochs, patience=patience,
            batch_size=batch_size, num_workers=num_workers,
            use_amp=use_amp, tag=tag)
        torch.save(model.state_dict(), mpath)

    score_kw = dict(img_cache=img_cache, path_to_row=path_to_row,
                    batch_size=batch_size, num_workers=num_workers)
    cal_lab = pool_labels[cal_local]
    cal_sm, _, _ = score_data(model, pool_paths[cal_local], cal_lab, **score_kw)
    t0 = time.time()
    test_sm, test_acc, test_acc3 = score_data(model, test_paths, test_labels, **score_kw)
    score_time = time.time() - t0
    del model
    torch.cuda.empty_cache()
    np.savez_compressed(spath, cal_softmax=cal_sm, cal_lab=cal_lab,
                        cal_idx=cal_local, test_softmax=test_sm,
                        test_acc=test_acc, test_acc3=test_acc3,
                        train_time=train_time, score_time=score_time)
    return cal_sm, cal_lab, cal_local, test_sm, test_acc, False, train_time, test_acc3, score_time


# ════════════════════════════════════════════════════════════════════
# Conformal prediction methods
# ════════════════════════════════════════════════════════════════════

def get_quantile(scores, alpha, default=np.inf):
    """Conformal quantile with finite-sample correction."""
    n = len(scores)
    if n == 0:
        return default
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return default
    return np.quantile(scores, level, method='inverted_cdf')


def compute_metrics(in_set, test_labels, alpha, C):
    """Marginal coverage, average set size, and coverage gap."""
    n = len(test_labels)
    covered = in_set[np.arange(n), test_labels]
    size = in_set.sum(axis=1).mean()
    class_cov = np.full(C, np.nan)
    for y in range(C):
        m = test_labels == y
        if m.any():
            class_cov[y] = covered[m].mean()
    valid = ~np.isnan(class_cov)
    return {'marginal': float(covered.mean()),
            'avg_size': float(size),
            'cov_gap': float(np.mean(np.abs(class_cov[valid] - (1 - alpha))) * 100),
            'class_cov': class_cov}


def _predict(test_nc, test_labels, qhats, alpha, C):
    in_set = test_nc <= qhats[np.newaxis, :]
    return compute_metrics(in_set, test_labels, alpha, C)


def _add_split_cov_gap(r, c2c_arr, alpha):
    cc = r['class_cov']
    nn_mask = c2c_arr >= 0
    rare_mask = c2c_arr < 0
    valid_nn = nn_mask & ~np.isnan(cc)
    valid_rare = rare_mask & ~np.isnan(cc)
    r['cov_gap_clustered'] = (float(np.mean(np.abs(cc[valid_nn] - (1 - alpha))) * 100)
                              if valid_nn.any() else float('nan'))
    r['cov_gap_rare'] = (float(np.mean(np.abs(cc[valid_rare] - (1 - alpha))) * 100)
                         if valid_rare.any() else float('nan'))


def standard_cp(cal_true, cal_lab, test_nc, test_labels, alpha, C):
    """Standard conformal prediction: single global quantile."""
    t0 = time.time()
    q = get_quantile(cal_true, alpha)
    cal_t = time.time() - t0
    t0 = time.time()
    r = _predict(test_nc, test_labels, np.full(C, q), alpha, C)
    r['_cal_time'] = cal_t
    r['_test_time'] = time.time() - t0
    return r


def classwise_cp(cal_true, cal_lab, test_nc, test_labels, alpha, C):
    """Classwise conformal prediction: per-class quantile."""
    t0 = time.time()
    qhats = np.full(C, np.inf)
    for y in range(C):
        m = cal_lab == y
        if m.any():
            qhats[y] = get_quantile(cal_true[m], alpha)
    cal_t = time.time() - t0
    t0 = time.time()
    r = _predict(test_nc, test_labels, qhats, alpha, C)
    r['_cal_time'] = cal_t
    r['_test_time'] = time.time() - t0
    return r


def clustered_cp(cal_true, cal_lab, test_nc, test_labels, alpha, C):
    """Clustered CP: gamma-split, M = total_prop_nn/300, post-hoc merging."""
    t0 = time.time()
    n_alpha = int(np.ceil(1 / alpha) - 1)
    n_cal = len(cal_lab)
    counts = np.bincount(cal_lab, minlength=C)
    K_nn = int(np.sum(counts >= n_alpha))
    gamma = K_nn / (K_nn + 75)

    rng = np.random.RandomState(42)
    n_clust = int(np.floor(gamma * n_cal))
    perm = rng.permutation(n_cal)
    ci, pi = perm[:n_clust], perm[n_clust:]
    cs_t, cl = cal_true[ci], cal_lab[ci]
    ps_t, pl = cal_true[pi], cal_lab[pi]

    null_q = get_quantile(ps_t, alpha)
    emb, nullc = {}, set()
    for kk in range(C):
        mk = cl == kk
        if mk.sum() < n_alpha:
            nullc.add(kk)
            continue
        emb[kk] = np.quantile(cs_t[mk], QL)
    nn = sorted(set(range(C)) - nullc)
    c2c_arr = np.full(C, -1, dtype=int)

    if len(nn) < 2:
        cal_t = time.time() - t0
        t1 = time.time()
        qh = np.full(C, null_q)
        r = compute_metrics(test_nc <= qh[None, :], test_labels, alpha, C)
        r.update(c2c=c2c_arr, n_clustered=len(nn), n_clusters=0,
                 _cal_time=cal_t, _test_time=time.time() - t1)
        _add_split_cov_gap(r, c2c_arr, alpha)
        return r

    prop_cnt = np.bincount(pl, minlength=C)
    total_prop_nn = sum(prop_cnt[c] for c in nn)
    M = max(2, int(total_prop_nn / 300))
    if len(nn) <= M:
        M = max(2, len(nn) // 2)

    em = np.array([emb[kk] for kk in nn])
    wt = np.array([np.sqrt((cl == kk).sum()) for kk in nn])
    km = KMeans(n_clusters=M, random_state=0, n_init=10).fit(em, sample_weight=wt)
    c2c = {kk: km.labels_[i] for i, kk in enumerate(nn)}
    centroids = {m: km.cluster_centers_[m].copy() for m in range(M)}

    # Post-hoc merging: merge small clusters into nearest neighbor
    while len(set(c2c.values())) > 2:
        active = set(c2c.values())
        csz = {m: sum(prop_cnt[kk] for kk in nn if c2c[kk] == m) for m in active}
        small = [m for m, s in csz.items() if s < 150]
        if not small:
            break
        victim = min(small, key=lambda m: csz[m])
        remaining = sorted(active - {victim})
        for kk in [k for k in nn if c2c[k] == victim]:
            idx = nn.index(kk)
            c2c[kk] = min(remaining, key=lambda m: np.linalg.norm(em[idx] - centroids[m]))
        for m in remaining:
            mi = [nn.index(kk) for kk in nn if c2c[kk] == m]
            if mi:
                centroids[m] = em[mi].mean(axis=0)
    active = sorted(set(c2c.values()))
    remap = {old: new for new, old in enumerate(active)}
    c2c = {kk: remap[m] for kk, m in c2c.items()}
    M = len(active)

    cq = np.zeros(M)
    for m in range(M):
        cls = [kk for kk, c in c2c.items() if c == m]
        mm = np.isin(pl, cls)
        cq[m] = get_quantile(ps_t[mm], alpha) if mm.any() else null_q
    qh = np.full(C, null_q)
    for kk in nn:
        qh[kk] = cq[c2c[kk]]
        c2c_arr[kk] = c2c[kk]

    cal_t = time.time() - t0
    t1 = time.time()
    r = compute_metrics(test_nc <= qh[None, :], test_labels, alpha, C)
    r.update(c2c=c2c_arr, n_clustered=len(nn), n_clusters=M,
             _cal_time=cal_t, _test_time=time.time() - t1)
    _add_split_cov_gap(r, c2c_arr, alpha)
    return r


def cccp(clus_true, clus_lab, cal_true, cal_lab,
         test_nc, test_labels, alpha, C):
    """CFCP: cross-fitted clustering scores + full calibration set for thresholds."""
    t0 = time.time()
    n_alpha = int(np.ceil(1 / alpha) - 1)
    null_q = get_quantile(cal_true, alpha)
    cnt = np.bincount(clus_lab, minlength=C)

    emb, nullc = {}, set()
    for kk in range(C):
        mk = clus_lab == kk
        if mk.sum() < n_alpha:
            nullc.add(kk)
            continue
        emb[kk] = np.quantile(clus_true[mk], QL)
    nn = sorted(set(range(C)) - nullc)
    c2c_arr = np.full(C, -1, dtype=int)

    if len(nn) < 2:
        cal_t = time.time() - t0
        t1 = time.time()
        qh = np.full(C, null_q)
        r = compute_metrics(test_nc <= qh[None, :], test_labels, alpha, C)
        r.update(c2c=c2c_arr, n_clustered=len(nn), n_clusters=0,
                 _cal_time=cal_t, _test_time=time.time() - t1)
        _add_split_cov_gap(r, c2c_arr, alpha)
        return r

    cal_cnt = np.bincount(cal_lab, minlength=C)
    total_cal_nn = sum(cal_cnt[c] for c in nn)
    M = max(2, int(total_cal_nn / 300))
    if len(nn) <= M:
        M = max(2, len(nn) // 2)

    em = np.array([emb[kk] for kk in nn])
    wt = np.array([np.sqrt(cnt[kk]) for kk in nn])
    km = KMeans(n_clusters=M, random_state=0, n_init=10).fit(em, sample_weight=wt)
    c2c = {kk: km.labels_[i] for i, kk in enumerate(nn)}
    centroids = {m: km.cluster_centers_[m].copy() for m in range(M)}

    while len(set(c2c.values())) > 2:
        active = set(c2c.values())
        csz = {m: sum(cal_cnt[kk] for kk in nn if c2c[kk] == m) for m in active}
        small = [m for m, s in csz.items() if s < 150]
        if not small:
            break
        victim = min(small, key=lambda m: csz[m])
        remaining = sorted(active - {victim})
        for kk in [k for k in nn if c2c[k] == victim]:
            idx = nn.index(kk)
            c2c[kk] = min(remaining, key=lambda m: np.linalg.norm(em[idx] - centroids[m]))
        for m in remaining:
            mi = [nn.index(kk) for kk in nn if c2c[kk] == m]
            if mi:
                centroids[m] = em[mi].mean(axis=0)
    active = sorted(set(c2c.values()))
    remap = {old: new for new, old in enumerate(active)}
    c2c = {kk: remap[m] for kk, m in c2c.items()}
    M = len(active)

    cq = np.zeros(M)
    for m in range(M):
        cls = [kk for kk, c in c2c.items() if c == m]
        mm = np.isin(cal_lab, cls)
        cq[m] = get_quantile(cal_true[mm], alpha) if mm.any() else null_q
    qh = np.full(C, null_q)
    for kk in nn:
        qh[kk] = cq[c2c[kk]]
        c2c_arr[kk] = c2c[kk]

    cal_t = time.time() - t0
    t1 = time.time()
    r = compute_metrics(test_nc <= qh[None, :], test_labels, alpha, C)
    r.update(c2c=c2c_arr, n_clustered=len(nn), n_clusters=M,
             _cal_time=cal_t, _test_time=time.time() - t1)
    _add_split_cov_gap(r, c2c_arr, alpha)
    return r


# ════════════════════════════════════════════════════════════════════
# APS and RAPS nonconformity scores
# ════════════════════════════════════════════════════════════════════

def _sorted_ranks(softmax):
    n, C_s = softmax.shape
    sorted_idx = np.argsort(-softmax, axis=1)
    sorted_sm = np.take_along_axis(softmax, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_sm, axis=1)
    ranks = np.zeros_like(sorted_idx)
    ranks[np.arange(n)[:, None], sorted_idx] = np.arange(C_s)[None, :]
    return cumsum, ranks


def aps_raps_true(softmax, labels, lam=0.01, k_reg=5):
    """APS and RAPS scores for true labels. Returns (aps, raps)."""
    cumsum, ranks = _sorted_ranks(softmax)
    true_ranks = ranks[np.arange(len(labels)), labels]
    aps = cumsum[np.arange(len(labels)), true_ranks].astype(np.float32)
    penalty = lam * np.maximum(true_ranks + 1 - k_reg, 0).astype(np.float32)
    return aps, (aps + penalty).astype(np.float32)


def aps_raps_all(softmax, lam=0.01, k_reg=5):
    """APS and RAPS scores for all classes. Returns (aps, raps)."""
    cumsum, ranks = _sorted_ranks(softmax)
    aps = np.take_along_axis(cumsum, ranks, axis=1).astype(np.float32)
    penalty = lam * np.maximum(ranks + 1 - k_reg, 0).astype(np.float32)
    return aps, (aps + penalty).astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# Proposition 3: epsilon (per-class KS distance)
# ════════════════════════════════════════════════════════════════════

def scores_by_class(test_nc, test_labels, C):
    true_nc = test_nc[np.arange(len(test_labels)), test_labels]
    return [true_nc[test_labels == y] for y in range(C)]


def _cluster_ks_epsilons(score_list):
    k = len(score_list)
    sorted_list = [np.sort(s) for s in score_list]
    grid = np.unique(np.concatenate(sorted_list))
    F = np.stack([np.searchsorted(s, grid, side='right') / len(s) for s in sorted_list])
    eps = np.empty(k)
    for i in range(k):
        d = np.max(np.abs(F - F[i][None, :]), axis=1)
        eps[i] = d.sum() / (k - 1)
    return eps


def epsilon_records(c2c, class_cov, sbc, alpha, C):
    """Compute per-class epsilon for all clusters."""
    groups = defaultdict(list)
    for y in range(C):
        if c2c[y] >= 0:
            groups[c2c[y]].append(y)
    recs = []
    for gid, ys in groups.items():
        ys = [y for y in ys if len(sbc[y]) > 0]
        if len(ys) < 2:
            continue
        eps = _cluster_ks_epsilons([sbc[y] for y in ys])
        for y, e in zip(ys, eps):
            cov = class_cov[y]
            recs.append({'class': int(y), 'epsilon': float(e),
                         'cluster_size': len(ys),
                         'cov_dev': (float(abs(cov - (1 - alpha)) * 100)
                                     if not np.isnan(cov) else float('nan'))})
    return recs


# ════════════════════════════════════════════════════════════════════
# Imbalanced subsampling
# ════════════════════════════════════════════════════════════════════

def exponential_subsample(labels, num_classes, ratio, rng):
    """Exponential class-frequency imbalance for Food-101 (unbal)."""
    max_size = 1000
    min_size = int(max_size / ratio)
    lam = np.log(ratio) / (num_classes - 1)
    perm = rng.permutation(num_classes)
    target = {perm[i]: max(min_size, int(max_size * np.exp(-lam * i)))
              for i in range(num_classes)}
    idx = []
    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        n_take = min(target[c], len(c_idx))
        chosen = rng.choice(c_idx, n_take, replace=False)
        idx.extend(chosen)
    return np.sort(np.array(idx)), target
