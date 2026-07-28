# Vision SFT Post-Training (Video Generation Only)

Fine-tune Cosmos3 for pure T2V / I2V / V2V video generation — no actions,
no reasoner/VLM component. This is the `vision_sft_*` family of recipes
from [`docs/training.md`](./training.md), pulled into its own page for a
focused walkthrough.

## Overview

**Cosmos3-Edge (`vision_sft_edge`) is the recommended default here** — only
2B params, fits a 4-GPU node, and is the fastest to iterate on. Nano/Super
are documented too, for reference.

| Piece            | `vision_sft_edge` (recommended)                        | `vision_sft_nano`                                    | `vision_sft_super` (LoRA)                             |
| ----------------- | --------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| Backbone          | Nemotron-2B-Dense-VL (Cosmos3-Edge)                       | Qwen3-VL-8B (Cosmos3-Nano)                            | Qwen3-VL-32B-Instruct MoT (Cosmos3-Super)               |
| TOML              | `examples/toml/sft_config/vision_sft_edge.toml`           | `examples/toml/sft_config/vision_sft_nano.toml`       | `examples/toml/sft_config/vision_sft_super.toml`        |
| Launch shell      | `examples/launch_sft_vision_edge.sh`                       | `examples/launch_sft_vision_nano.sh`                  | `examples/launch_sft_vision_super.sh`                   |
| Config module     | `cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py` | `cosmos_framework/configs/base/experiment/sft/vision_sft_nano.py` | `cosmos_framework/configs/base/experiment/sft/vision_sft_super.py` |
| Fine-tune mode    | Full                                                       | Full                                                  | LoRA (`lora_enabled=true`)                              |
| Topology          | 8-GPU FSDP (also fits 4-GPU, e.g. `NPROC_PER_NODE=4`)      | 8-GPU FSDP                                            | 8-GPU FSDP, CP=2 / DP=4                                 |
| `[job].task`      | `"vfm"`                                                    | `"vfm"`                                               | `"vfm"`                                                  |

All three train on the same dataset and share `[job].task = "vfm"` — there
is no action-conditioning and no VLM/reasoner head involved, unlike the
DROID/LIBERO/Stretch forward-dynamics recipes or the Reasoner alignment
recipes documented elsewhere in `docs/training.md`.

Dataset: [nvidia/BridgeData2-Subset-Synthetic-Captions](https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions/tree/main).
Each clip carries a structured-JSON caption (`caption_json`) — the model's
native prompt format — which the SFT loader trains on by default, keeping
training aligned with [Inference](./dataset_jsonl.md#inference); see
[JSONL Dataset → Format](./dataset_jsonl.md#format).

## Prerequisites

- Install the training environment as described in [`docs/setup.md`](./setup.md)
  (`cu130-train` / `cu128-train` group).
- Run commands from the repository root.
- Clear `LD_LIBRARY_PATH` before Python commands (`export LD_LIBRARY_PATH=''`)
  to avoid host CUDA/NCCL libs bleeding into the venv's torch import; on a
  cluster with environment modules, `module load cuda/<version matching your
  torch build>` (e.g. `cuda/12.8.1` for `torch+cu128`) is the cleaner fix if
  you still hit a `cudart shared object not found` error.

## Step 1: Download data + Wan2.2 VAE

```shell
uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions \
    --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 \
    --local-dir examples/data/BridgeData2-Subset-Synthetic-Captions --quiet

uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

`$DATASET_PATH` should point at the directory containing
`train/video_dataset_file.jsonl` — that's the `sft_dataset_bridge`
subdirectory inside the downloaded repo:
`examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge`
(this is the launch shells' default).

## Step 2: Convert the base checkpoint to DCP

Pick the tier matching the recipe you're running (`Cosmos3-Edge` /
`Cosmos3-Nano` / `Cosmos3-Super`):

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Edge   # or Cosmos3-Nano / Cosmos3-Super

python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/$BASE_CHECKPOINT_NAME \
  --checkpoint-path $BASE_CHECKPOINT_NAME
```

This downloads the matching repo from the HF Hub and writes a DCP
checkpoint to `examples/checkpoints/$BASE_CHECKPOINT_NAME`.

## Step 3: Run training

### Option A (recommended): paired launch shell

The launcher auto-resolves `DATASET_PATH` / `BASE_CHECKPOINT_PATH` /
`WAN_VAE_PATH` from the `examples/` locations populated by Steps 1+2 — no
env vars required if you kept the defaults:

```shell
bash examples/launch_sft_vision_edge.sh   # or launch_sft_vision_nano.sh / launch_sft_vision_super.sh
```

Only 2B — on a 4-GPU allocation (e.g. GB200x4) set
`NPROC_PER_NODE=4 bash examples/launch_sft_vision_edge.sh`.

To override any default (e.g. data on a different filesystem):

```shell
export DATASET_PATH=/scratch/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
export BASE_CHECKPOINT_PATH=/nfs/ckpts/Cosmos3-Edge
export WAN_VAE_PATH=/nfs/ckpts/wan22_vae/Wan2.2_VAE.pth
bash examples/launch_sft_vision_edge.sh
```

Each var falls back to its launcher default if unset — only export the
ones you're moving.

### Validate the config

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/vision_sft_edge.toml \
  --dryrun
```

### Option B: raw `torchrun`

Unlike Option A, raw `torchrun` does **not** auto-resolve the env-var trio
from `examples/` — export them yourself first:

```shell
TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"

export DATASET_PATH="$PWD/examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge"
export BASE_CHECKPOINT_PATH="$PWD/examples/checkpoints/Cosmos3-Edge"
export WAN_VAE_PATH="$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"

IMAGINAIRE_OUTPUT_ROOT=outputs/train PYTHONPATH=. \
torchrun --nproc_per_node=4 -m cosmos_framework.scripts.train \
    --sft-toml=$TOML_FILE
```

For `vision_sft_nano`/`vision_sft_super`, swap `TOML_FILE` and
`BASE_CHECKPOINT_PATH` (`Cosmos3-Edge` → `Cosmos3-Nano` / `Cosmos3-Super`)
and use `--nproc_per_node=8` accordingly.

## Outputs

`$RUN_DIR = $IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>` — e.g. with
`IMAGINAIRE_OUTPUT_ROOT=outputs/train` and the `vision_sft_edge` recipe,
`$RUN_DIR` is `outputs/train/cosmos3/sft/vision_sft_edge`. DCP checkpoints
land under `$RUN_DIR/checkpoints/iter_<N>/`; the latest iter name is in
`$RUN_DIR/checkpoints/latest_checkpoint.txt`.

## Inference

Point `cosmos_framework.scripts.inference` at
`$RUN_DIR/checkpoints/iter_<N>` together with `--config-file
$RUN_DIR/config.yaml` — see the `cosmos3-inference` skill / `docs/inference.md`
for parameter and input-format details.

## Export checkpoint to Hugging Face safetensors (optional)

```shell
RUN_DIR=$IMAGINAIRE_OUTPUT_ROOT/cosmos3/sft/vision_sft_edge
CHECKPOINT_ITER=$(cat $RUN_DIR/checkpoints/latest_checkpoint.txt)

python -m cosmos_framework.scripts.export_model \
  --checkpoint-path $RUN_DIR/checkpoints/$CHECKPOINT_ITER \
  --config-file $RUN_DIR/config.yaml \
  --no-vit \
  -o $RUN_DIR/model
```

`--no-vit` is the right choice for these recipes since there's no
reasoner/VLM head to preserve — it produces a ~1 GB smaller,
generation-only export (T2V/I2V/V2V/T2I). Drop the flag if you plan to
reuse the exported checkpoint's vision tower elsewhere.

## Notes

- **`vision_sft_super` is LoRA, not a full fine-tune** — `[model].lora_enabled`
  is `true` with `lora_rank=16` / `lora_alpha=32` on
  `q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen`. Its launch
  shell also clears `LD_LIBRARY_PATH` and sets
  `PYTORCH_ALLOC_CONF=expandable_segments:True` so the 32B backbone fits
  without OOM during compile/decode.
- **`vision_sft_edge` is small** — Cosmos3-Edge is only 2B and fits a 4-GPU
  node (e.g. `NPROC_PER_NODE=4 bash examples/launch_sft_vision_edge.sh`,
  tested on 4×GB200). No `HF_TOKEN` is required — `nvidia/Cosmos3-Edge` is
  ungated.
- **Alternative CosmosDataLoader variant**: `vision_sft_nano_mapstyle_dataloader.toml`
  + `examples/launch_sft_vision_nano_cosmosdataloader.sh` is a dataflow-loader
  mirror of `vision_sft_nano` (CosmosDataLoader + RankPartitionedDistributor +
  SequentialPackingBatcher) used for loss-curve regression testing against the
  baseline `PackingDataLoader` path — not the recommended default, but useful
  if you're debugging dataloader-specific behavior.
- `.gitignore` excludes `examples/data/` and `examples/checkpoints/`, so the
  multi-GB downloads from Steps 1+2 aren't tracked if you keep the defaults.
