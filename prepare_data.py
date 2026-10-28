#!/usr/bin/env python
"""
Download datasets and build image caches for CFCP experiments.

Usage:
    python prepare_data.py --dataset food101
    python prepare_data.py --dataset caltech256
    python prepare_data.py --dataset sun397
    python prepare_data.py --all

Options:
    --data-dir DIR    Base directory for datasets (default: data/)
    --cache-only      Skip download, only build image cache
    --no-cache        Skip cache building, only download

Downloads each dataset into an imagefolder layout under data/, then
builds a 224x224 uint8 memory-mapped .npy cache for fast loading.
"""

import argparse
import gc
import io
import os
import shutil
import sys
import tarfile
from urllib.request import urlretrieve

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 224

DATASETS = {
    'food101': {
        'url': 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',
        'images_subdir': os.path.join('food-101', 'images'),
        'cache_name': 'food101',
        'min_classes': 101,
        'description': 'Food-101 (101 classes, ~101K images, ~5 GB download)',
    },
    'caltech256': {
        'url': 'https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar',
        'images_subdir': '256_ObjectCategories',
        'cache_name': 'caltech256',
        'min_classes': 257,
        'description': 'Caltech-256 (257 classes, ~30K images, ~1.2 GB download)',
    },
    'sun397': {
        'url': None,  # HuggingFace parquet
        'images_subdir': 'SUN397',
        'cache_name': 'sun397',
        'min_classes': 397,
        'description': 'SUN-397 (397 classes, ~108K images, ~40 GB download)',
    },
}


def _progress_hook(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size * 100 / total_size, 100)
        mb = count * block_size / 1e6
        total_mb = total_size / 1e6
        print(f'\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)', end='', flush=True)


def _count_classes(images_dir):
    if not os.path.isdir(images_dir):
        return 0
    return sum(1 for d in os.listdir(images_dir)
               if os.path.isdir(os.path.join(images_dir, d)))


# ── Food-101 and Caltech-256: direct tar download ──────────────────

def download_tar(data_dir, dataset_name):
    cfg = DATASETS[dataset_name]
    images_dir = os.path.join(data_dir, cfg['images_subdir'])

    if _count_classes(images_dir) >= cfg['min_classes']:
        print(f'{dataset_name}: already exists at {images_dir}')
        return images_dir

    url = cfg['url']
    ext = '.tar.gz' if url.endswith('.tar.gz') else '.tar'
    tar_path = os.path.join(data_dir, f'{dataset_name}{ext}')

    os.makedirs(data_dir, exist_ok=True)
    print(f'Downloading {cfg["description"]}...')
    urlretrieve(url, tar_path, _progress_hook)
    print()

    print('Extracting...')
    mode = 'r:gz' if ext == '.tar.gz' else 'r'
    with tarfile.open(tar_path, mode) as tf:
        tf.extractall(data_dir)
    os.remove(tar_path)

    n = _count_classes(images_dir)
    print(f'{dataset_name}: {n} classes at {images_dir}')
    return images_dir


# ── SUN-397: HuggingFace parquet download ──────────────────────────

def download_sun397(data_dir):
    images_dir = os.path.join(data_dir, 'SUN397')

    if _count_classes(images_dir) >= 397:
        print(f'sun397: already exists at {images_dir}')
        return images_dir

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print('ERROR: pyarrow required for SUN-397. Install: pip install pyarrow')
        sys.exit(1)

    os.makedirs(images_dir, exist_ok=True)
    parquet_tmp = os.path.join(data_dir, '_sun397_parquet_tmp')
    os.makedirs(parquet_tmp, exist_ok=True)

    n_shards = 93
    hf_base = 'https://huggingface.co/datasets/1aurent/SUN397/resolve/main/data'
    class_counters = {}
    total_saved = 0

    for si in range(n_shards):
        fname = f'train-{si:05d}-of-{n_shards:05d}.parquet'
        url = f'{hf_base}/{fname}'
        local = os.path.join(parquet_tmp, fname)

        if not os.path.exists(local):
            print(f'  [{si+1}/{n_shards}] Downloading {fname}...', flush=True)
            urlretrieve(url, local)
        else:
            print(f'  [{si+1}/{n_shards}] {fname} cached', flush=True)

        table = pq.read_table(local)
        labels = table.column('label').to_pylist()
        images = table.column('image')

        for row_idx in range(len(labels)):
            label = labels[row_idx]
            class_dir = os.path.join(images_dir, f'class_{label:04d}')
            os.makedirs(class_dir, exist_ok=True)
            count = class_counters.get(label, 0) + 1
            class_counters[label] = count

            img_struct = images[row_idx].as_py()
            if isinstance(img_struct, dict) and 'bytes' in img_struct:
                img_bytes = img_struct['bytes']
            elif isinstance(img_struct, bytes):
                img_bytes = img_struct
            else:
                continue

            out_path = os.path.join(class_dir, f'{count:05d}.jpg')
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img.save(out_path, 'JPEG', quality=95)
            total_saved += 1

        del table, labels, images
        os.remove(local)

    shutil.rmtree(parquet_tmp, ignore_errors=True)
    print(f'sun397: {total_saved} images, {len(class_counters)} classes at {images_dir}')
    return images_dir


# ── Image cache builder ────────────────────────────────────────────

def build_index(images_dir):
    paths = []
    for cls in sorted(os.listdir(images_dir)):
        d = os.path.join(images_dir, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                paths.append(os.path.join(d, fn))
    return paths


def build_image_cache(images_dir, cache_path):
    done_flag = cache_path + '.done'
    paths = build_index(images_dir)
    n = len(paths)
    shape = (n, IMG_SIZE, IMG_SIZE, 3)

    if os.path.exists(cache_path) and os.path.exists(done_flag):
        try:
            arr = np.load(cache_path, mmap_mode='r')
            if arr.shape == shape:
                print(f'  Cache valid: {cache_path}  {shape}')
                return
        except Exception:
            pass

    gb = np.prod(shape) / 1e9
    print(f'  Building cache: {n} images -> {cache_path} (~{gb:.1f} GB)')

    if os.path.exists(done_flag):
        os.remove(done_flag)

    mm = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.uint8, shape=shape)
    for i, p in enumerate(paths):
        img = Image.open(p).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mm[i] = np.asarray(img, dtype=np.uint8)
        if (i + 1) % 5000 == 0:
            print(f'    {i+1}/{n}', flush=True)
    mm.flush()
    del mm
    gc.collect()
    open(done_flag, 'w').close()
    print(f'  Cache done: {cache_path}')


# ── Main ───────────────────────────────────────────────────────────

def prepare_one(dataset_name, data_dir, download=True, cache=True):
    cfg = DATASETS[dataset_name]
    print(f'\n{"="*60}')
    print(f'  {dataset_name.upper()}: {cfg["description"]}')
    print(f'{"="*60}')

    images_dir = os.path.join(data_dir, cfg['images_subdir'])

    if download:
        if dataset_name == 'sun397':
            images_dir = download_sun397(data_dir)
        else:
            images_dir = download_tar(data_dir, dataset_name)

    if cache:
        cache_path = os.path.join(data_dir, f'{cfg["cache_name"]}_img{IMG_SIZE}_uint8.npy')
        if not os.path.isdir(images_dir):
            print(f'  ERROR: {images_dir} not found. Run without --cache-only first.')
            return
        build_image_cache(images_dir, cache_path)


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets and build image caches for CFCP experiments.')
    parser.add_argument('--dataset', choices=list(DATASETS.keys()),
                        help='Dataset to prepare')
    parser.add_argument('--all', action='store_true',
                        help='Prepare all datasets')
    parser.add_argument('--data-dir', default='data',
                        help='Base directory for datasets (default: data/)')
    parser.add_argument('--cache-only', action='store_true',
                        help='Skip download, only build image cache')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip cache building, only download')

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error('Specify --dataset NAME or --all')

    datasets = list(DATASETS.keys()) if args.all else [args.dataset]

    for ds in datasets:
        prepare_one(ds, args.data_dir,
                    download=not args.cache_only,
                    cache=not args.no_cache)

    print('\nDone.')


if __name__ == '__main__':
    main()
