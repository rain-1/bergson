#!/usr/bin/env python3
"""Example: score pile-10k against mean WMDP bio gradients using trackstar.

Computes adafactor normalizers and preconditioners on both query and value
datasets (up to 10k samples each). Preconditioners are not recomputed during
reduce or score.
"""

import subprocess
import sys

cmd = [
    sys.executable,
    "-m",
    "bergson",
    "trackstar",
    "runs/trackstar_wmdp",
    "--model",
    "EleutherAI/pythia-160m",
    "--normalizer",
    "adafactor",
    "--stats_sample_size",
    "10000",
    # Value dataset
    "--data.dataset",
    "NeelNanda/pile-10k",
    "--data.split",
    "train",
    "--data.truncation",
    # Query dataset
    "--query.dataset",
    "cais/wmdp",
    "--query.split",
    "test",
    "--query.subset",
    "wmdp-bio",
    "--query.prompt_column",
    "question",
    # Reduce and score methods
    "--method",
    "mean",
    "--score",
    "mean",
    "--overwrite",
]

print(" ".join(cmd))
subprocess.run(cmd, check=True)
