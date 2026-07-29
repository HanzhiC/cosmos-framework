#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for action_wam_stretch_edge_lora_posttrain.
#
# Single-GPU (~48GB), Cosmos3-Edge + LoRA variant of action_wam_stretch_posttrain
# (world-action-model / policy) on a Stretch teleop LeRobot dataset produced by
# `python -m cosmos_framework.scripts.convert_stretch_to_lerobot`. LoRA trains
# the generation backbone; the small action heads (action2llm/llm2action/
# action_modality_embed) are fully fine-tuned. See
# docs/action_wam_stretch_posttrain.md (Edge + LoRA section).
#
# UNVALIDATED: nobody has measured actual peak GPU memory for this recipe here.
# Treat this as a starting point for an empirical smoke test on your 48GB card,
# not a known-good config.
#
# Env vars (override for your filesystem):
#   DATASET_PATH                 Stretch-LeRobot root (single flat LeRobot v3 dir)
#   BASE_CHECKPOINT_PATH         Cosmos3-Edge DCP checkpoint (NOT Cosmos3-Nano)
#   WAN_VAE_PATH                 Wan2.2 VAE .pth
#   WANDB_API_KEY                for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE               torchrun --nproc_per_node (default 1 here — single GPU)
#   EXTRA_TAIL_OVERRIDES         space-separated Hydra overrides
#
# Single-GPU smoke run:
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
#   bash examples/launch_sft_action_wam_stretch_edge_lora_posttrain.sh
#
# If this OOMs on your 48GB GPU, try (in order): lowering
# dataloader_train.max_sequence_length below 45056 via EXTRA_TAIL_OVERRIDES,
# reducing chunk_length (edit the TOML/experiment), or disabling the
# compile_tokenizer callback (trainer.callbacks.compile_tokenizer.enabled=false).
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_wam_stretch_edge_lora_posttrain.toml"
: "${DATASET_PATH:=examples/data/Stretch-LeRobot}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${NPROC_PER_NODE:=1}"

WAN_VAE_PATH="${WAN_VAE_PATH:-examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

EXTRA_DATASET_CHECK='[[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$WORKDIR/$WAN_VAE_PATH"; export WAN_VAE_PATH; [[ -f "$DATASET_PATH/meta/info.json" ]] || { echo "ERROR: missing Stretch-LeRobot dataset under $DATASET_PATH (expected meta/info.json; run cosmos_framework.scripts.convert_stretch_to_lerobot first, see docs/action_wam_stretch_posttrain.md)" >&2; exit 1; }; [[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
