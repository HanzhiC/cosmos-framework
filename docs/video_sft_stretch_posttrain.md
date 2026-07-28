# Video SFT Stretch Post-Training (Pure Video Generation, No Actions)

This trains **pure video generation — no action supervision at all** — on
your converted Stretch teleop LeRobot dataset. It reuses the exact same
dataset/model shape as
[`docs/action_fd_stretch_posttrain.md`](./action_fd_stretch_posttrain.md)
(`StretchLeRobotDataset`, concat_view head+gripper camera, chunk_length=16 at
720p vision tokens), but the dataset factory is called with
`mask_action=True`: the action tensor fed to the model is zeroed, and only
the generation heads (`moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`) are
trained — `action2llm`/`llm2action`/`action_modality_embed` stay loaded from
the base checkpoint but frozen (never in `optimizer.keys_to_select`, and
never see a nonzero action to learn from).

Experiment `video_sft_stretch_posttrain`, launcher
`launch_sft_video_sft_stretch_posttrain.sh`.

| Piece            | Value                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------- |
| Experiment        | `video_sft_stretch_posttrain`                                                                |
| TOML              | `examples/toml/sft_config/video_sft_stretch_posttrain.toml`                                  |
| Launch shell      | `examples/launch_sft_video_sft_stretch_posttrain.sh`                                         |
| Config module     | `cosmos_framework/configs/base/experiment/action/posttrain_config/video_sft_stretch_posttrain.py` |
| Dataset wrapper   | `cosmos_framework/data/generator/action/datasets/stretch_lerobot_dataset.py` (`mask_action=True`) |
| Task mode         | `"forward_dynamics"` dataset mode (unchanged), but action is always zeroed                    |
| Chunk length / resolution | `16` frames at `480` (data decoded/trained at 720p vision tokens; same as FD)          |

## Important caveat: this is I2V-style, not multi-frame V2V prefix continuation

Every mode in this framework's `SequencePlan` machinery
(`forward_dynamics`/`inverse_dynamics`/`wam`) conditions on exactly the
**first** video frame (`condition_frame_indexes_vision=[0]`) — there is no
mode that conditions on a multi-frame video prefix. With the action zeroed,
training reduces to **first-frame-conditioned (I2V-style) generation with a
null action signal**, not a true multi-frame video-to-video continuation.

If you need genuine multi-frame V2V prefix conditioning, that's what the
separate JSONL-based `vision_sft_*` recipes support natively (see
[`docs/vision_sft_posttrain.md`](./vision_sft_posttrain.md)) — but nothing
converts the Stretch LeRobot dataset into that JSONL format today; it would
require a new converter script. This recipe was built as the lower-risk
alternative: reuse the existing, validated Stretch LeRobot pipeline and mask
out the action, rather than build a new converter.

## Prerequisites

- Install the training environment as described in [`docs/setup.md`](./setup.md).
- Run commands from the repository root.
- Clear `LD_LIBRARY_PATH` before Python commands, or `module load cuda/<version matching your torch build>` (e.g. `cuda/12.8.1` for `torch+cu128`) if you hit a `cudart shared object not found` error from `transformer_engine`.

## Step 1: Convert raw Stretch episodes to LeRobot format

Same conversion as the FD/WAM recipes — see
[`docs/action_fd_stretch_posttrain.md` → Step 1](./action_fd_stretch_posttrain.md#step-1-convert-raw-stretch-episodes-to-lerobot-format).
Reuse the same `DATASET_PATH` if you've already converted your data for one
of the action recipes — the underlying LeRobot dataset is identical; only
this recipe's `mask_action=True` flag differs.

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.convert_stretch_to_lerobot \
  --out-root /home/wiss/chenh/storage/group/srl/stretch_dataset_final_lerobot
```

## Data Layout

```shell
export DATASET_PATH=/home/wiss/chenh/storage/group/srl/stretch_dataset_final_lerobot
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
export DATASET_PATH=/home/wiss/chenh/storage/group/srl/stretch_dataset_final_lerobot

# Step 2: point to the base DCP checkpoint and Wan2.2 VAE
# (both produced by docs/training.md Step 1/2 — these are the launcher's defaults).
export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth

# Step 3: choose the output root and launch.
export IMAGINAIRE_OUTPUT_ROOT=outputs/train
export LD_LIBRARY_PATH=''

bash examples/launch_sft_video_sft_stretch_posttrain.sh
```

## Validate The Config

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/video_sft_stretch_posttrain.toml \
  --dryrun
```

## Run Training

Recommended paired launch shell:

```shell
bash examples/launch_sft_video_sft_stretch_posttrain.sh
```

Pass short smoke-run overrides through `EXTRA_TAIL_OVERRIDES`:

```shell
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
bash examples/launch_sft_video_sft_stretch_posttrain.sh
```

Single-node, 8 GPU:

```shell
PYTHONPATH=. torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/video_sft_stretch_posttrain.toml
```

Multi-node HSDP:

```shell
PYTHONPATH=. torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=8 \
  -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/video_sft_stretch_posttrain.toml \
  -- model.parallelism.data_parallel_replicate_degree=$NNODES
```

## Outputs

```text
$IMAGINAIRE_OUTPUT_ROOT/cosmos3_video_sft/sft/<job.name>/
```

DCP checkpoints under `$RUN_DIR/checkpoints/iter_<N>/`.

## Notes

- **Idle-frame captioning stays truthful despite masking.** `mask_action=True`
  zeros the `action` tensor the model actually conditions on, but
  `StretchLeRobotDataset` computes the idle-frame count (embedded in the
  structured-JSON caption) from the **real, unmasked** action first — so
  captions don't wrongly claim every clip is "fully idle."
- **Action heads stay loaded but inert.** `optimizer.keys_to_select` excludes
  `action2llm`/`llm2action`/`action_modality_embed` — they're loaded from the
  base checkpoint (`checkpoint.keys_to_skip_loading` only skips `net_ema.`)
  but never updated and never see a nonzero action.
- Hyperparameters (chunk_length, resolution, `lr=1e-4`, token-packing) are
  inherited unchanged from the validated `action_fd_stretch_posttrain`
  recipe — not independently tuned for the video-only objective.
- Both this recipe and `action_fd_stretch_posttrain`/`action_wam_stretch_posttrain`
  read the same `DATASET_PATH` — no separate conversion needed if you've
  already run Step 1 for one of them.
