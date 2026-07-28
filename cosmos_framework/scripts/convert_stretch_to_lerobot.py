# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert raw Stretch teleop episodes into a LeRobotDataset v3.0 directory.

Source layout (one directory per task under ``--raw-root``, one subdirectory per
episode, timestamp-named e.g. ``2026-03-26--20-53-22``):

    <raw_root>/<task>/<episode>/
        head_rgb/{frame:06d}.png        gripper_rgb/{frame:06d}.png
        extr_cam0cam/{frame:06d}.npz    {"T_base_eecam": (4,4), "T_base_headcam": (4,4)}
        dex_traj/{frame:06d}.npz        {"trajectory": (30,17), ...}; row 0's last
                                         column is that frame's gripper closure.
        success.txt                     "Success" | "Failure" (not always present)

Per episode, per frame ``i`` this writes:
  * ``observation.images.head_rgb`` / ``observation.images.gripper_rgb`` — raw
    PNGs, encoded to per-episode mp4 by the ``lerobot`` package on save.
  * ``observation.state.cartesian_position`` — ``[x, y, z, roll, pitch, yaw]``
    decomposed from ``T_base_eecam`` (absolute end-effector pose in the robot's
    base/world frame; euler ``xyz``, radians).
  * ``observation.state.gripper_position`` — 1-D gripper closure.

The **absolute** pose is stored, not a pre-computed delta — matching how the
DROID LeRobot conversion stores ``observation.state.cartesian_position`` and lets
the dataset loader (``StretchLeRobotDataset``) derive the relative ``ee_pose``
action at read time via
``cosmos_framework.data.generator.action.pose_utils.pose_abs_to_rel``.

Episode selection: each raw episode is joined by name against a
``stretchrobot_<task>_all_valid_frame_ranges.csv`` split file (see
``--splits-root``) to resolve ``success``, ``is_teleop``, and the validated
``valid_frame_range``. Only rows with ``success == 1 and is_teleop == 1`` are
converted, using the CSV's ``round == 0`` row when an episode has several. If an
episode's name isn't present in that CSV (this happens for ``wipe_table``, whose
CSV uses sequential ids that don't match the raw timestamped directory names),
this falls back to the episode's own ``success.txt`` (`"Success"` -> keep,
``is_teleop`` assumed true) over the full on-disk frame range; episodes with
neither signal are skipped.

Usage
-----
    PYTHONPATH=. python -m cosmos_framework.scripts.convert_stretch_to_lerobot \\
        --out-root examples/data/Stretch-LeRobot

    # Quick smoke run on a couple of episodes per task:
    PYTHONPATH=. python -m cosmos_framework.scripts.convert_stretch_to_lerobot \\
        --out-root /tmp/stretch_lerobot_smoke --max-episodes-per-task 2
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro
from PIL import Image
from scipy.spatial.transform import Rotation

from cosmos_framework.utils import log

_TASK_INSTRUCTIONS: dict[str, str] = {
    "pnp_basketball": "pick up the ball and place it in the basket",
    "pnp_microwave": "open the microwave and take the drink",
    "pnp_ricecooker": "open the ricecooker and put food inside",
    "pnp_socks": "put the socks in the drawer and close the drawer",
    "wipe_table": "take the sponge and wipe the table",
}

_TASK_SPLIT_CSV: dict[str, str] = {
    "pnp_basketball": "stretchrobot_pnp-basketball_all_valid_frame_ranges.csv",
    "pnp_microwave": "stretchrobot_pnp-microwave_all_valid_frame_ranges.csv",
    "pnp_ricecooker": "stretchrobot_pnp-ricecooker_all_valid_frame_ranges.csv",
    "pnp_socks": "stretchrobot_pnp-socks_all_valid_frame_ranges.csv",
    "wipe_table": "stretchrobot_wipe-table_all_valid_frame_ranges.csv",
}

_DEFAULT_RAW_ROOT = Path("/home/wiss/chenh/storage/group/srl/stretch_dataset_final")
_DEFAULT_SPLITS_ROOT = Path("/home/wiss/chenh/mobile_manip/egoasis3D/datasets/splits_robot")

_HEAD_IMAGE_SHAPE = (320, 240, 3)  # (H, W, C), portrait
_GRIPPER_IMAGE_SHAPE = (240, 320, 3)  # (H, W, C), landscape


@dataclass
class _EpisodeMeta:
    name: str
    start: int
    end: int  # inclusive
    success: bool
    is_teleop: bool
    source: str  # "csv" | "success_txt"


def _load_split_csv(path: Path) -> dict[str, _EpisodeMeta]:
    """Parse one ``stretchrobot_*_all_valid_frame_ranges.csv`` into ``{episode_name: meta}``.

    Prefers the ``round == 0`` row when an episode has multiple rows.
    """
    import csv

    by_episode: dict[str, list[dict]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            episode_name = row["sample"].split("/")[0]
            by_episode.setdefault(episode_name, []).append(row)

    out: dict[str, _EpisodeMeta] = {}
    for episode_name, rows in by_episode.items():
        row = next((r for r in rows if r.get("round") == "0"), rows[0])
        start, end = json.loads(row["valid_frame_range"])
        out[episode_name] = _EpisodeMeta(
            name=episode_name,
            start=int(start),
            end=int(end),
            success=row["success"] == "1",
            is_teleop=row["is_teleop"] == "1",
            source="csv",
        )
    return out


def _resolve_episode(task_dir: Path, episode_name: str, csv_meta: dict[str, _EpisodeMeta]) -> _EpisodeMeta | None:
    if episode_name in csv_meta:
        return csv_meta[episode_name]

    # Fallback: raw success.txt over the full on-disk frame range (e.g. wipe_table,
    # whose CSV uses sequential ids that don't match these timestamped dir names).
    ep_dir = task_dir / episode_name
    n_frames = len(list((ep_dir / "head_rgb").glob("*.png")))
    if n_frames == 0:
        return None
    success_txt = ep_dir / "success.txt"
    if not success_txt.exists():
        return None  # no success signal available at all; can't honor success-only filtering
    success = success_txt.read_text().strip().lower() == "success"
    return _EpisodeMeta(name=episode_name, start=0, end=n_frames - 1, success=success, is_teleop=True, source="success_txt")


def _episode_files_present(ep_dir: Path, start: int, end: int) -> bool:
    for i in range(start, end + 1):
        if not (
            (ep_dir / "head_rgb" / f"{i:06d}.png").exists()
            and (ep_dir / "gripper_rgb" / f"{i:06d}.png").exists()
            and (ep_dir / "extr_cam0cam" / f"{i:06d}.npz").exists()
            and (ep_dir / "dex_traj" / f"{i:06d}.npz").exists()
        ):
            return False
    return True


def _pose_to_state(pose: np.ndarray) -> np.ndarray:
    xyz = pose[:3, 3]
    euler = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)
    return np.concatenate([xyz, euler]).astype(np.float32)


def _convert_episode(dataset, ep_dir: Path, meta: _EpisodeMeta, task_text: str) -> int:
    n_frames = 0
    for i in range(meta.start, meta.end + 1):
        head = np.array(Image.open(ep_dir / "head_rgb" / f"{i:06d}.png").convert("RGB"))
        gripper_img = np.array(Image.open(ep_dir / "gripper_rgb" / f"{i:06d}.png").convert("RGB"))
        pose = np.load(ep_dir / "extr_cam0cam" / f"{i:06d}.npz")["T_base_eecam"]
        gripper_val = float(np.load(ep_dir / "dex_traj" / f"{i:06d}.npz")["trajectory"][0, 16])

        dataset.add_frame(
            {
                "observation.images.head_rgb": head,
                "observation.images.gripper_rgb": gripper_img,
                "observation.state.cartesian_position": _pose_to_state(pose),
                "observation.state.gripper_position": np.array([gripper_val], dtype=np.float32),
                "task": task_text,
            }
        )
        n_frames += 1
    # parallel_encoding uses a multiprocessing ProcessPoolExecutor, whose semaphores are
    # backed by /dev/shm; on shared machines where /dev/shm is exhausted by unrelated
    # processes this raises OSError(28). Serial encoding avoids /dev/shm entirely and is
    # fine for a 2-camera dataset.
    dataset.save_episode(parallel_encoding=False)
    return n_frames


def main(
    out_root: Annotated[Path, tyro.conf.arg(help="Output LeRobotDataset v3.0 root directory (must not exist).")],
    raw_root: Annotated[Path, tyro.conf.arg(help="Root of stretch_dataset_final (one subdir per task).")] = (
        _DEFAULT_RAW_ROOT
    ),
    splits_root: Annotated[
        Path, tyro.conf.arg(help="Directory of stretchrobot_*_all_valid_frame_ranges.csv split files.")
    ] = _DEFAULT_SPLITS_ROOT,
    fps: float = 15.0,
    max_episodes_per_task: Annotated[
        int, tyro.conf.arg(help="Cap converted episodes per task, for a quick smoke run. -1 converts all.")
    ] = -1,
    tasks: Annotated[tuple[str, ...], tyro.conf.arg(help="Task subset to convert.")] = tuple(_TASK_INSTRUCTIONS),
) -> None:
    """Convert Stretch teleop episodes into a single LeRobotDataset v3.0 root."""
    # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    unknown_tasks = set(tasks) - set(_TASK_INSTRUCTIONS)
    if unknown_tasks:
        raise ValueError(f"Unknown task(s): {sorted(unknown_tasks)}. Known: {sorted(_TASK_INSTRUCTIONS)}")

    features = {
        "observation.images.head_rgb": {"dtype": "video", "shape": _HEAD_IMAGE_SHAPE, "names": ["height", "width", "channel"]},
        "observation.images.gripper_rgb": {
            "dtype": "video",
            "shape": _GRIPPER_IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "observation.state.cartesian_position": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "observation.state.gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/stretch",
        fps=int(fps),
        features=features,
        root=out_root,
        robot_type="stretch",
        use_videos=True,
    )

    stats: Counter[str] = Counter()
    for task in tasks:
        task_dir = raw_root / task
        csv_path = splits_root / _TASK_SPLIT_CSV[task]
        csv_meta = _load_split_csv(csv_path) if csv_path.exists() else {}
        if not csv_meta:
            log.warning(f"{task}: no split CSV found at {csv_path}; every episode falls back to success.txt")

        episode_names = sorted(p.name for p in task_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
        n_converted = 0
        for name in episode_names:
            if max_episodes_per_task >= 0 and n_converted >= max_episodes_per_task:
                break

            meta = _resolve_episode(task_dir, name, csv_meta)
            if meta is None:
                stats["unresolved"] += 1
                log.warning(f"{task}/{name}: no success/teleop signal found; skipping")
                continue
            if not (meta.success and meta.is_teleop):
                stats["filtered_success_or_teleop"] += 1
                continue

            ep_dir = task_dir / name
            if not _episode_files_present(ep_dir, meta.start, meta.end):
                stats["missing_files"] += 1
                log.warning(f"{task}/{name}: missing frame file(s) in range [{meta.start}, {meta.end}]; skipping")
                continue

            n_frames = _convert_episode(dataset, ep_dir, meta, _TASK_INSTRUCTIONS[task])
            stats[f"converted_via_{meta.source}"] += 1
            stats["converted_frames"] += n_frames
            n_converted += 1

        log.info(f"{task}: converted {n_converted}/{len(episode_names)} episodes")

    log.info(f"Done. Wrote {out_root} — summary: {dict(stats)}")


if __name__ == "__main__":
    tyro.cli(main)
