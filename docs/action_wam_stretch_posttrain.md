# Action WAM Stretch Post-Training (World-Action-Model / Policy)

This trains a **world-action-model (WAM)** — colloquially "action policy" —
on your converted Stretch teleop LeRobot dataset: given only the FIRST video
frame (no clean actions), the model jointly denoises the future video AND the
matching future 10D `ee_pose` actions. At inference, only the generated
action stream is used to drive the robot; the jointly-generated video is an
auxiliary training signal that grounds the action prediction in visual
dynamics.

Experiment `action_wam_stretch_posttrain`, launcher
`launch_sft_action_wam_stretch_posttrain.sh`. This is the counterpart to
[`docs/action_fd_stretch_posttrain.md`](./action_fd_stretch_posttrain.md)
(forward dynamics: action → video) — same dataset, model shape, and
dataloading strategy, with only the `mode` swapped from
`"forward_dynamics"` to `"wam"`.

| Piece            | Value                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------- |
| Experiment        | `action_wam_stretch_posttrain`                                                              |
| TOML              | `examples/toml/sft_config/action_wam_stretch_posttrain.toml`                                |
| Launch shell      | `examples/launch_sft_action_wam_stretch_posttrain.sh`                                       |
| Config module     | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_wam_stretch_posttrain.py` |
| Dataset wrapper   | `cosmos_framework/data/generator/action/datasets/stretch_lerobot_dataset.py` (`mode="wam"`) |
| Task mode         | `"wam"` (world-action-model) — vs. `"forward_dynamics"` (FD doc) / `"inverse_dynamics"`      |
| Action space      | `ee_pose`: 10-D `[pos_delta, rot6d_delta, gripper]`, derived from absolute `observation.state.cartesian_position` |
| Chunk length / resolution | `16` frames at `480` (data decoded/trained at 720p vision tokens; same as FD)         |

## What `mode="wam"` actually does

From `cosmos_framework/data/generator/action/transforms.py::build_sequence_plan_from_mode`:

- `condition_frame_indexes_vision = [0]` — only the first video frame is clean/conditioning (same as `forward_dynamics`).
- No action steps are conditioning — **all** actions are supervised targets (unlike `forward_dynamics`, where all actions are given as conditioning).
- The model therefore denoises the remaining video frames and the entire action sequence jointly, from a single starting frame.

This differs from `"inverse_dynamics"` (which observes the **full** video and only predicts actions) — WAM only ever sees frame 0.

## Prerequisites

- Install the training environment as described in [`docs/setup.md`](./setup.md).
- Run commands from the repository root.
- Clear `LD_LIBRARY_PATH` before Python commands, or `module load cuda/<version matching your torch build>` (e.g. `cuda/12.8.1` for `torch+cu128`) if you hit a `cudart shared object not found` error from `transformer_engine`.

## Step 1: Convert raw Stretch episodes to LeRobot format

Same conversion as the FD recipe — see
[`docs/action_fd_stretch_posttrain.md` → Step 1](./action_fd_stretch_posttrain.md#step-1-convert-raw-stretch-episodes-to-lerobot-format).
If you've already converted your data for the FD recipe, reuse the same
`DATASET_PATH` here — both recipes read the identical LeRobot dataset, only
the training `mode` differs.

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

bash examples/launch_sft_action_wam_stretch_posttrain.sh
```

## Validate The Config

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_wam_stretch_posttrain.toml \
  --dryrun
```

## Run Training

Recommended paired launch shell:

```shell
bash examples/launch_sft_action_wam_stretch_posttrain.sh
```

Pass short smoke-run overrides through `EXTRA_TAIL_OVERRIDES`:

```shell
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
bash examples/launch_sft_action_wam_stretch_posttrain.sh
```

Single-node, 8 GPU:

```shell
PYTHONPATH=. torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_wam_stretch_posttrain.toml
```

Multi-node HSDP:

```shell
PYTHONPATH=. torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=8 \
  -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_wam_stretch_posttrain.toml \
  -- model.parallelism.data_parallel_replicate_degree=$NNODES
```

## Outputs

```text
$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_wam/action_sft/<job.name>/
```

DCP checkpoints under `$RUN_DIR/checkpoints/iter_<N>/`.

## Notes

- **Hyperparameters are inherited from the FD Stretch recipe, not independently
  tuned for WAM.** Same dataset/model shape (chunk_length=16, 720p vision
  tokens, token-packed `PackingDataLoader`, `lr=1e-4`) — only `mode="wam"`
  changed. Watch the action loss on your first smoke runs (`EXTRA_TAIL_OVERRIDES="trainer.max_iter=10"`)
  and adjust `optimizer.lr` in the TOML if it doesn't converge — DROID's
  analogous policy recipe (`docs/action_policy_droid_posttrain.md`) uses
  `2e-4`, but that recipe uses a fundamentally different dataloading strategy
  (count-based batching at 480p, chunk 32) so its tuned value doesn't
  transfer directly.
- Same action convention as the FD recipe: **world-frame** translation delta
  (`world_framewise`, rotation delta `R_i^T @ R_{i+1}`, translation delta
  `p_{i+1}-p_i` in world axes) — see
  [`docs/action_fd_stretch_posttrain.md` → Notes](./action_fd_stretch_posttrain.md#notes)
  for the full rationale.
- Both `action_fd_stretch_posttrain` and `action_wam_stretch_posttrain` read
  the exact same `DATASET_PATH` — no separate conversion needed if you've
  already run Step 1 for one of them.
- To reproduce inverse-dynamics-style pure policy inference (given the full
  video, predict actions) rather than WAM's joint video+action generation
  from frame 0, a third `mode="inverse_dynamics"` experiment would need to be
  registered the same way — not built here since the WAM framing is what was
  requested.

## Cosmos3-Edge + LoRA variant (single ~48GB GPU)

`action_wam_stretch_edge_lora_posttrain` is a lighter sibling recipe aimed at
fitting a single ~48GB GPU, since the recipe above (Nano backbone, full
fine-tune) is only validated on multi-GPU FSDP. It uses `EDGE_MODEL_CONFIG`
(Nemotron-2B-Dense-VL, ~4x smaller than Nano's Qwen3-VL-8B) with a **hybrid**
training scheme:

- **LoRA** on the generation backbone (`lora_enabled=True`, rank 16, targeting
  `q/k/v/o_proj_moe_gen` — same knobs as `vision_sft_super`'s LoRA recipe).
- **Full fine-tuning** of the action heads (`action2llm`/`llm2action`/
  `action_modality_embed`). These are tiny standalone modules (`cosmos3_vfm_network.py:157-160`:
  a couple of small linear projections + one embedding vector), architecturally
  separate from the LoRA-targeted attention projections — LoRA alone would give
  the model no capacity to actually learn the action-prediction task WAM needs,
  so they're kept fully trainable. Being small, this adds negligible extra
  optimizer-state memory on top of the LoRA adapters.

`optimizer.keys_to_select=["lora_", "action2llm", "llm2action", "action_modality_embed"]`
— everything else (the frozen Edge backbone) is excluded.

| Piece          | Value                                                                                                    |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| Experiment      | `action_wam_stretch_edge_lora_posttrain`                                                                    |
| TOML            | `examples/toml/sft_config/action_wam_stretch_edge_lora_posttrain.toml`                                      |
| Launch shell    | `examples/launch_sft_action_wam_stretch_edge_lora_posttrain.sh`                                             |
| Config module   | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_wam_stretch_edge_lora_posttrain.py` |
| Base checkpoint | `Cosmos3-Edge` (not `Cosmos3-Nano`) — convert separately: `BASE_CHECKPOINT_NAME=Cosmos3-Edge`               |
| Resolution / tokens | `480` / `max_num_tokens_after_packing=45056` — Edge's own lighter defaults, NOT the 720p/74000-token settings the Nano-based recipe above uses |
| Activation checkpointing | `"full"` (max memory savings) — not `"selective"`                                                  |
| Parallelism     | `data_parallel_shard_degree=1`, single GPU (`NPROC_PER_NODE=1`)                                             |

```shell
# Step 1: same DATASET_PATH as the other Stretch recipes.
export DATASET_PATH=/home/wiss/chenh/storage/group/srl/stretch_dataset_final_lerobot

# Step 2: convert Cosmos3-Edge (not Cosmos3-Nano) to DCP.
BASE_CHECKPOINT_NAME=Cosmos3-Edge
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/$BASE_CHECKPOINT_NAME \
  --checkpoint-path $BASE_CHECKPOINT_NAME
export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth

# Step 3: single-GPU smoke run.
export IMAGINAIRE_OUTPUT_ROOT=outputs/train
export LD_LIBRARY_PATH=''
export NPROC_PER_NODE=1
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
bash examples/launch_sft_action_wam_stretch_edge_lora_posttrain.sh
```

**Unvalidated — treat as a starting point for an empirical smoke test, not a
known-good config.** Nobody has measured actual peak GPU memory for this
recipe (or for this LoRA+full-tune hybrid at all) in this repo. Watch the
action loss especially — the `lr_multipliers` on the action heads (5x the base
LoRA `lr=5e-4`, so an effective ~2.5e-3) are inherited from the Nano WAM
recipe's full-fine-tune tuning, not re-tuned for this hybrid scheme. If it
OOMs on your 48GB card, try in order:
1. Lower `dataloader_train.max_sequence_length` below 45056 via
   `EXTRA_TAIL_OVERRIDES="dataloader_train.max_sequence_length=<N>"`.
2. Reduce `chunk_length` (currently 16) — requires editing the experiment
   Python's dataset factory call, not just a Hydra override.
3. Disable the `compile_tokenizer` callback
   (`trainer.callbacks.compile_tokenizer.enabled=false`) — `torch.compile`
   warmup can spike memory transiently.
