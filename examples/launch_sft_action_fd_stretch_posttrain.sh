#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for action_fd_stretch_posttrain.
#
# This trains forward dynamics on a Stretch teleop LeRobot dataset produced by
# `python -m cosmos_framework.scripts.convert_stretch_to_lerobot`. See
# docs/action_fd_stretch_posttrain.md.
#
# Env vars (override for your filesystem):
#   DATASET_PATH                 Stretch-LeRobot root (single flat LeRobot v3 dir)
#   BASE_CHECKPOINT_PATH         Base DCP checkpoint
#   WAN_VAE_PATH                 Wan2.2 VAE .pth
#   WANDB_API_KEY                for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE               torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES         space-separated Hydra overrides
#
# Single-node smoke:
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
#   bash examples/launch_sft_action_fd_stretch_posttrain.sh
#
# Multi-node: launch on every worker. For HSDP set
# model.parallelism.data_parallel_replicate_degree = <num_nodes> (shard stays 8).
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_fd_stretch_posttrain.toml"
: "${DATASET_PATH:=examples/data/Stretch-LeRobot}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

WAN_VAE_PATH="${WAN_VAE_PATH:-examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

EXTRA_DATASET_CHECK='[[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$WORKDIR/$WAN_VAE_PATH"; export WAN_VAE_PATH; [[ -f "$DATASET_PATH/meta/info.json" ]] || { echo "ERROR: missing Stretch-LeRobot dataset under $DATASET_PATH (expected meta/info.json; run cosmos_framework.scripts.convert_stretch_to_lerobot first, see docs/action_fd_stretch_posttrain.md)" >&2; exit 1; }; [[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
