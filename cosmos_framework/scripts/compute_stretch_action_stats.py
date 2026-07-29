# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compute action-normalization stats for ``StretchLeRobotDataset``.

Walks every valid chunk of the Stretch LeRobot dataset (the same anchor-relative
10D ``[pos_delta(3), rot6d_delta(6), gripper(1)]`` action ``_build_action``
produces, but without decoding video — this only touches the parquet-derived
pose arrays) and writes per-dimension ``q01``/``q99``/``mean``/``std``/``min``/``max``
to a JSON file consumable by ``StretchLeRobotDataset``'s inherited
``ActionBaseDataset._load_norm_stats`` (``action_normalization.load_action_stats``
reads exactly those flat keys — see ``cosmos_framework/data/generator/action/action_normalization.py``).

Usage::

    PYTHONPATH=. python -m cosmos_framework.scripts.compute_stretch_action_stats \\
        --root /path/to/stretch_dataset_final_lerobot
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

from cosmos_framework.data.generator.action.datasets.stretch_lerobot_dataset import StretchLeRobotDataset
from cosmos_framework.utils import log

_DEFAULT_OUT_PATH = (
    Path(__file__).parent.parent / "data" / "generator" / "action" / "normalizer_stats" / "stretch_lerobot_ee_pose_rot6d.json"
)


def _collect_actions(dataset: StretchLeRobotDataset) -> np.ndarray:
    """Build every valid chunk's anchor-relative action WITHOUT decoding video."""
    rows = []
    for idx in range(len(dataset)):
        ep = int(np.searchsorted(dataset._valid_cum, idx, side="right"))
        prev = int(dataset._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(dataset._ep_starts[ep]) + (idx - prev)

        anchor_state = dataset._row_state[start]
        action_state_window = dataset._row_action_state[start : start + dataset._chunk_length]
        action_gripper_window = dataset._row_action_gripper[start : start + dataset._chunk_length]
        action = dataset._build_action(anchor_state, action_state_window, action_gripper_window)
        rows.append(action.numpy())
    return np.concatenate(rows, axis=0)  # [N * chunk_length, 10]


def main(
    root: Annotated[Path, tyro.conf.arg(help="Stretch-LeRobot dataset root (produced by convert_stretch_to_lerobot).")],
    split: Annotated[str, tyro.conf.arg(help="Which episodes to include: train/val/full.")] = "train",
    val_ratio: float = 0.02,
    seed: int = 0,
    out_path: Annotated[Path, tyro.conf.arg(help="Output stats JSON path.")] = _DEFAULT_OUT_PATH,
) -> None:
    """Compute and write Stretch ``ee_pose`` action normalization stats."""
    dataset = StretchLeRobotDataset(
        root=str(root),
        split=split,
        val_ratio=val_ratio,
        seed=seed,
        action_normalization=None,  # raw actions in, so stats reflect the un-normalized distribution
    )
    log.info(f"Computing action stats over {len(dataset)} chunks ({dataset._chunk_length} steps each)...")
    actions = _collect_actions(dataset)  # [N, 10]

    stats = {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).tolist(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Wrote action stats (from {actions.shape[0]} action rows) to {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
