# Action FD Stretch Post-Training

This document describes how to run the `action_fd_stretch_posttrain` experiment.
It trains Cosmos3 forward dynamics on a Stretch mobile-manipulator teleop
dataset in `cosmos_framework`.

## Overview

| Piece                     | Value                                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------------------|
| Experiment                | `action_fd_stretch_posttrain`                                                                    |
| TOML                      | `examples/toml/sft_config/action_fd_stretch_posttrain.toml`                                      |
| Launch shell              | `examples/launch_sft_action_fd_stretch_posttrain.sh`                                             |
| Config module             | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_stretch_posttrain.py`|
| Dataset wrapper           | `cosmos_framework/data/generator/action/datasets/stretch_lerobot_dataset.py`                     |
| Converter (raw -> LeRobot)| `cosmos_framework/scripts/convert_stretch_to_lerobot.py`                                         |
| Dataset root              | LeRobotDataset v3.0 root produced by the converter above                                         |
| Task mode                 | `forward_dynamics`                                                                                |
| Action space               | `ee_pose`: 10-D `[pos_delta, rot6d_delta, gripper]`, derived from absolute `observation.state.cartesian_position` |
| Chunk length / resolution | `16` frames at `480`                                                                              |

Unlike DROID/LIBERO, there is no upstream pre-converted Stretch LeRobot
dataset — you convert your own raw teleop capture first.

## Prerequisites

- Install the training environment as described in [`docs/setup.md`](./setup.md).
- Run commands from the repository root.
- In NGC / PyTorch containers, set `LD_LIBRARY_PATH=''` before Python commands.

## Step 1: Convert raw Stretch episodes to LeRobot format

Raw episodes are per-episode directories of PNG frames + per-frame `.npz`
(head/gripper RGB, `dex_traj/*.npz` gripper closure, `extr_cam0cam/*.npz`
absolute end-effector pose), one task subdirectory per manipulation task. Convert
them into a single LeRobotDataset v3.0 root:

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.convert_stretch_to_lerobot \
  --out-root /path/to/Stretch-LeRobot
```

By default this reads raw episodes from `/home/wiss/chenh/storage/group/srl/stretch_dataset_final`
and resolves each episode's success/teleop status and language instruction from
the `stretchrobot_*_all_valid_frame_ranges.csv` split files under
`/home/wiss/chenh/mobile_manip/egoasis3D/datasets/splits_robot` (override both
with `--raw-root` / `--splits-root`). Only episodes with `success == 1` and
`is_teleop == 1` are converted; see the script's module docstring for the exact
join/fallback logic (one task, `wipe_table`, falls back to each episode's own
`success.txt` since its split CSV uses non-matching episode ids).

For a quick smoke run on a couple of episodes per task:

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.convert_stretch_to_lerobot \
  --out-root /tmp/stretch_lerobot_smoke --max-episodes-per-task 2
```

## Data Layout

Set `DATASET_PATH` to the converted Stretch-LeRobot root:

```shell
export DATASET_PATH=/path/to/Stretch-LeRobot
```

```text
$DATASET_PATH/
├── meta/info.json
├── data/chunk-*/file-*.parquet
└── videos/observation.images.{head_rgb,gripper_rgb}/chunk-*/file-*.mp4
```

## Full Reproduction

```shell
# Step 1: convert raw Stretch episodes -> $DATASET_PATH (see above).
export DATASET_PATH=examples/data/Stretch-LeRobot

# Step 2: point to the base DCP checkpoint and Wan2.2 VAE.
export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth

# Step 3: choose the output root and launch.
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root
export LD_LIBRARY_PATH=''

bash examples/launch_sft_action_fd_stretch_posttrain.sh
```

## Validate The Config

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_fd_stretch_posttrain.toml \
  --dryrun
```

## Run Training

Recommended paired launch shell:

```shell
bash examples/launch_sft_action_fd_stretch_posttrain.sh
```

Pass short smoke-run overrides through `EXTRA_TAIL_OVERRIDES`:

```shell
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
bash examples/launch_sft_action_fd_stretch_posttrain.sh
```

Single-node, 8 GPU:

```shell
PYTHONPATH=. torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_fd_stretch_posttrain.toml
```

Multi-node HSDP:

```shell
PYTHONPATH=. torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=8 \
  -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_fd_stretch_posttrain.toml \
  -- model.parallelism.data_parallel_replicate_degree=$NNODES
```

## Outputs

```text
$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_fd/action_sft/<job.name>/
```

DCP checkpoints under `$RUN_DIR/checkpoints/iter_<N>/`.

## Notes

- This recipe uses `mode="forward_dynamics"`, so actions are conditioning and
  the model trains video prediction from the first frame plus action sequence.
- `StretchLeRobotDataset` stores **absolute** end-effector pose per frame
  (`observation.state.cartesian_position`) and derives the 10-D relative
  `ee_pose` action at read time via `pose_utils.pose_abs_to_rel`
  (`world_framewise`, `rot6d`): rotation delta is `R_i^T @ R_{i+1}`, but the
  translation delta stays in **world** axes (`p_{i+1} - p_i`, not rotated
  into the current end-effector frame) — matching the convention validated in
  ego-moma's `RobotDataset` (`transform_hand_trajectory_absolute_to_relative`).
  This differs from DROID/Bridge/RoboMIND, which use body-frame
  `backward_framewise` deltas.
- `pnp_socks_human` (ARIA human hand-pose recordings) is out of scope for this
  recipe; only Stretch teleop episodes are converted.
