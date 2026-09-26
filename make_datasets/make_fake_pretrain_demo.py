"""
Generate a small fake HDF5 dataset for smoke-testing the FEMBA pretraining
pipeline (Hydra config + Lightning training loop + real mamba_ssm CUDA kernels)
without needing real TUEG data.

Output layout matches what datasets.hdf5_dataset.HDF5Loader expects for
pretraining (finetune=False): one HDF5 file with one or more top-level
groups, each containing a dataset "X" of shape (N, num_channels, seq_length).

Usage (run on the GPU machine, no GPU required for this script itself):
    python make_datasets/make_fake_pretrain_demo.py \
        --out ${DATA_PATH}/demo/pretrain_fake.h5 \
        --num-samples 300 --num-channels 22 --seq-length 1280
"""

import argparse
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output .h5 path")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--num-channels", type=int, default=22)
    parser.add_argument("--seq-length", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    # Rough EEG-like amplitude (tens of microvolts), just needs to be a
    # non-degenerate signal for the masked-reconstruction loss to be meaningful.
    X = rng.normal(loc=0.0, scale=20.0, size=(args.num_samples, args.num_channels, args.seq_length)).astype(np.float32)

    with h5py.File(args.out, "w") as f:
        grp = f.create_group("demo")
        grp.create_dataset("X", data=X)

    print(f"Wrote {args.num_samples} fake samples ({args.num_channels}x{args.seq_length}) to {args.out}")


if __name__ == "__main__":
    main()
