# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_wam_stretch_edge_lora_posttrain`` — Cosmos3-Edge + LoRA world-action-model
Stretch post-training.

Single-GPU-friendly variant of ``action_wam_stretch_posttrain``, aimed at fitting a
single ~48GB GPU: swaps ``NANO_MODEL_CONFIG`` for ``EDGE_MODEL_CONFIG``
(Nemotron-2B-Dense-VL, ~4x smaller than Nano's Qwen3-VL-8B) and applies a HYBRID
training scheme:

  * LoRA (``lora_enabled=True``, rank 16, targeting the generation backbone's
    ``q/k/v/o_proj_moe_gen``) — same knobs as ``vision_sft_super``'s LoRA recipe.
  * FULL fine-tuning of the action heads (``action2llm``/``llm2action``/
    ``action_modality_embed``) — these are tiny standalone modules (a couple of
    small linear projections + one embedding vector,
    ``cosmos3_vfm_network.py``), NOT part of the LoRA-targeted attention
    projections, so LoRA alone gives the model no capacity to actually learn the
    action-prediction task WAM needs. Full-tuning them is architecturally simple
    (they're unrelated to the LoRA-injected modules) and cheap (negligible
    additional optimizer state given their size).

``optimizer.keys_to_select=["lora_", "action2llm", "llm2action",
"action_modality_embed"]`` — everything else (the frozen Edge backbone) stays
untouched, so FusedAdam's fp32 master-weight/momentum/variance state is only
paid for the LoRA adapters + these small heads, not the whole model.

Same as ``video_sft_stretch_edge_lora_posttrain``: keeps Edge's OWN lighter
defaults (480p, 45056 tokens, ``activation_checkpointing.mode="full"``) rather
than the Nano/FD-inherited heavier settings. ``mask_action`` stays ``False`` here
(unlike the video-only recipe) — WAM needs the real action signal to supervise
against.

For single-GPU use, set ``model.parallelism.data_parallel_shard_degree=1`` and
``NPROC_PER_NODE=1`` at launch (see the TOML / launch shell).

Unvalidated in this repo: neither Cosmos3-Edge, LoRA, nor this LoRA+full-tune
hybrid has been run on Stretch WAM training before, and no single-GPU memory
footprint has been measured here for any recipe. Treat this as a starting point
for an empirical 48GB smoke test, not a known-good config — watch the action
loss especially, since the `lr_multipliers` on the action heads (5x the base
LoRA lr) are inherited from the Nano WAM recipe's full-fine-tune tuning, not
re-tuned for this hybrid scheme.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_stretch_fd_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _make_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)

    # Dataset window is always chunk_length(16)+1=17 frames regardless of resolution.
    cfg["tokenizer"]["encode_exact_durations"] = [17]
    # Load fully from the converted Cosmos3-Edge DCP checkpoint (checkpoint.load_path),
    # not a separate HF-pretrained diffusion-expert fetch.
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False

    # LoRA: same knobs as vision_sft_super's LoRA recipe.
    cfg["lora_enabled"] = True
    cfg["lora_rank"] = 16
    cfg["lora_alpha"] = 32
    cfg["lora_target_modules"] = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"

    return cfg


action_wam_stretch_edge_lora_posttrain = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /checkpoint": "gcp"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "training_stats"]},
            {"override /ema": "power"},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_wam",
            group="action_sft",
            name="${now:%Y-%m-%d_%H-%M-%S}_action_wam_stretch_edge_lora_posttrain",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_make_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # LoRA adapters on the generation backbone + full fine-tune of the
            # (tiny) action heads. Everything else (frozen Edge backbone) is
            # excluded.
            keys_to_select=[
                "lora_",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=5.0e-04,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            cycle_lengths=[20000],
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            lr_scheduler_type="LambdaLinear",
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=50,
            max_iter=20000,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=100, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            straggler_detection=dict(enabled=False, report_freq=50),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(every_n=200, log_memory_detail=True, save_s3=False, step_size=1),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=50, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=200, gc_level=1, warm_up=1),
                norm_monitor=dict(every_n=100),
                param_count=dict(save_s3=False),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                compile_tokenizer=dict(enabled=True, warmup_resolutions=["480"]),
            ),
        ),
        checkpoint=dict(
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema. (EMA warm-starts from net, see dcp.py) and lora_ (LoRA
            # tensors don't exist in the base checkpoint; init fresh). Action heads
            # ARE loaded from base (warm-start), matching action_wam_stretch_posttrain.
            keys_to_skip_loading=[
                "net_ema.",
                "lora_",
            ],
            load_ema_to_reg=False,
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
            load_path="???",  # Cosmos3-Edge DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=500,
            strict_resume=True,
            verbose=True,
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_stretch_wam_edge_lora",
            max_samples_per_batch=None,
            max_sequence_length="${model.config.max_num_tokens_after_packing}",
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            sound_latent_fps="${model.config.sound_latent_fps}",
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=3,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=2,
                sampler=None,
                datasets=dict(
                    stretch=dict(
                        ratio=1,
                        dataset=L(get_action_stretch_fd_sft_dataset)(
                            root="${oc.env:DATASET_PATH}",
                            fps=15.0,
                            chunk_length=16,
                            mode="wam",
                            camera_mode="concat_view",
                            split="train",
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            action_normalization=None,
                            resolution="480",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            append_idle_frames=True,
                            idle_frames_dropout=0.05,
                            format_prompt_as_json=True,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


cs.store(
    group="experiment",
    package="_global_",
    name="action_wam_stretch_edge_lora_posttrain",
    node=action_wam_stretch_edge_lora_posttrain,
)
