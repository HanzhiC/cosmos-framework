# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stretch mobile-manipulator LeRobot dataset (forward-dynamics ``ee_pose`` action).

Mirrors ``LIBEROLeRobotDataset``: reads the LeRobot parquet directly, windows by
frame index, and decodes video at each frame's REAL timestamp (FPS-agnostic).
Unlike LIBERO, the stored per-frame quantities are **absolute** end-effector
poses — not a pre-computed delta — matching how ``DROIDLeRobotDataset``'s
``ee_pose``/``midtrain`` action space works, but with two poses per frame:
``observation.state.cartesian_position`` (the frame's *current* pose,
``dex_traj``'s ``history_trajectory[-1]``) and ``action.cartesian_position``
(the *one-step teleop-commanded target* pose, ``dex_traj``'s ``trajectory[0]``)
— plus matching ``*.gripper_position`` scalars for each. The relative
``[pos_delta(3), rot6d_delta(6), gripper(1)]`` action (10D) for step ``i`` in
a chunk is derived at read time as the delta from a single **anchor** state —
``self._row_state[start]``, the chunk's first frame — to that step's own
commanded target (``action_state_window[i]``), NOT a per-step framewise delta
(i.e. not ``state[i] -> target[i]`` re-anchored every step, and not
``state[i] -> state[i+1]``) via ``pose_utils.build_abs_pose_from_components`` +
``pose_utils.pose_abs_to_rel`` (``world_framewise``: rotation delta is
``R_anchor^T @ R_target[i]``, but translation delta stays in **world** axes,
i.e. ``p_target[i] - p_anchor``, not rotated into the anchor's frame). This
matches the convention validated in ego-moma's ``RobotDataset``
(``transform_hand_trajectory_absolute_to_relative``) — unlike DROID/Bridge/
RoboMIND, which stay on body-frame ``backward_framewise`` deltas. Using the
teleop's own commanded target (rather than the next *realized* state) avoids
baking control-loop settling lag into the action.

Each sample also carries ``history_action``: up to ``history_length`` past
``observation.state.*`` (proprioceptive, NOT ``action.*``) frames, re-expressed
in the same anchor-relative 10D format as ``action`` (i.e. each past state's
delta from the same ``self._row_state[start]`` anchor, plus its own raw
gripper reading) rather than as raw absolute poses — with the chunk's own
current state as the LAST row (whose delta is therefore ~0, since it *is* the
anchor). Fewer rows are returned near an episode's start (clamped, not
padded). This reuses ``DROIDLeRobotDataset``'s ``history_action``/``use_state``
key verbatim — ``transforms.py`` already pops ``history_action`` and prepends
it onto ``action`` as conditioning frames for any action dataset, so no model
or transform-pipeline changes are needed to make Stretch's proprio history
reach the model; only the *source* differs (past states here vs. past actions
for DROID).

The data is produced by ``cosmos_framework.scripts.convert_stretch_to_lerobot``,
which converts raw Stretch teleop episodes (per-frame PNG + npz) into this
LeRobotDataset v3.0 layout — there is no upstream pre-converted HF dataset for
Stretch, unlike DROID/LIBERO.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.pose_utils import build_abs_pose_from_components, pose_abs_to_rel
from cosmos_framework.utils import log

CameraMode = Literal["head_rgb", "gripper_rgb", "concat_view"]

_STATE_FEATURE = "observation.state.cartesian_position"
_GRIPPER_FEATURE = "observation.state.gripper_position"
_ACTION_STATE_FEATURE = "action.cartesian_position"
_ACTION_GRIPPER_FEATURE = "action.gripper_position"
_HEAD_CAMERA = "observation.images.head_rgb"
_GRIPPER_CAMERA = "observation.images.gripper_rgb"
_NORMALIZERS_DIR = Path(__file__).parent.parent / "normalizer_stats"

_VIEWPOINT_BY_CAMERA = {
    "head_rgb": "third_person_view",
    "gripper_rgb": "wrist_view",
    "concat_view": "concat_view",
}


class StretchLeRobotDataset(ActionBaseDataset):
    """Stretch action-forward-dynamics dataset: 10D absolute-state-derived ``ee_pose`` action.

    ``concat_view`` (head + gripper camera, horizontally concatenated) video,
    ``action_normalization=None`` by default (raw actions, matching the DROID FD
    recipe) since no normalizer stats have been computed for Stretch yet.
    """

    def __init__(
        self,
        root: str,
        fps: float = 15.0,
        chunk_length: int = 16,
        mode: str = "forward_dynamics",
        tolerance_s: float = 1e-4,
        camera_mode: CameraMode = "concat_view",
        image_size: int = 256,
        embodiment_type: str = "stretch_lerobot",
        action_normalization: str | None = None,
        split: str = "train",
        val_ratio: float = 0.02,
        seed: int = 0,
        sample_stride: int = 1,
        mask_action: bool = False,
        center_crop: bool = False,
        history_length: int = 15,
    ) -> None:
        if camera_mode not in _VIEWPOINT_BY_CAMERA:
            raise ValueError(f"Unsupported camera_mode={camera_mode!r}. Use head_rgb/gripper_rgb/concat_view.")
        split = split.lower().strip()
        if split not in {"train", "val", "valid", "validation", "eval", "test", "full"}:
            raise ValueError(f"Unsupported split={split!r}. Use train/val/full.")

        super().__init__(
            root=root,
            domain_name=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="world_framewise",
            tolerance_s=tolerance_s,
            viewpoint=_VIEWPOINT_BY_CAMERA[camera_mode],
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        info_fps = self._info.get("fps")
        if info_fps and int(info_fps) != int(fps):
            log.info(f"Using dataset native fps={info_fps} for conditioning (requested {fps}).")
            self._fps = float(info_fps)
            self._dt = 1.0 / self._fps
        self._camera_mode = camera_mode
        self._image_size = int(image_size)
        self._embodiment_type = embodiment_type
        self._mask_action = bool(mask_action)
        self._center_crop = bool(center_crop)
        if history_length < 1:
            raise ValueError(f"history_length must be >= 1, got {history_length}.")
        self._history_length = int(history_length)

        if self._camera_mode == "head_rgb":
            self._video_keys = [_HEAD_CAMERA]
        elif self._camera_mode == "gripper_rgb":
            self._video_keys = [_GRIPPER_CAMERA]
        else:
            self._video_keys = [_HEAD_CAMERA, _GRIPPER_CAMERA]

        # Compact, lazy frame index (mirrors LIBEROLeRobotDataset): read only the
        # columns the sample builder needs into contiguous arrays, ordered by global
        # frame index, so DataLoader worker forks share them copy-on-write.
        index_parts, episode_parts, task_parts, ts_parts = [], [], [], []
        state_parts, gripper_parts, action_state_parts, action_gripper_parts = [], [], [], []
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(
                path,
                columns=[
                    "index",
                    "episode_index",
                    "task_index",
                    "timestamp",
                    _STATE_FEATURE,
                    _GRIPPER_FEATURE,
                    _ACTION_STATE_FEATURE,
                    _ACTION_GRIPPER_FEATURE,
                ],
            )
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            state_parts.append(np.asarray(table[_STATE_FEATURE].to_pylist(), dtype=np.float32))
            action_state_parts.append(np.asarray(table[_ACTION_STATE_FEATURE].to_pylist(), dtype=np.float32))
            # LeRobot flattens 1-element feature vectors to bare scalars in parquet
            # (shape (1,) -> plain float column); restore the (N, 1) shape here.
            gripper_parts.append(np.asarray(table[_GRIPPER_FEATURE].to_pylist(), dtype=np.float32).reshape(-1, 1))
            action_gripper_parts.append(
                np.asarray(table[_ACTION_GRIPPER_FEATURE].to_pylist(), dtype=np.float32).reshape(-1, 1)
            )
        if not index_parts:
            raise FileNotFoundError(f"No data parquet found under {self._root / 'data'}.")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._row_state = np.concatenate(state_parts, axis=0).astype(np.float32)[order]  # [N,6] xyz+euler_xyz
        self._row_gripper = np.concatenate(gripper_parts, axis=0).astype(np.float32)[order]  # [N,1]
        self._row_action_state = np.concatenate(action_state_parts, axis=0).astype(np.float32)[order]  # [N,6]
        self._row_action_gripper = np.concatenate(action_gripper_parts, axis=0).astype(np.float32)[order]  # [N,1]

        assert np.all(np.diff(self._row_episode) >= 0), "episode_index not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)

        # Deterministic per-episode train/val split (seeded; same on every rank).
        keep = self._split_episode_ids(ep_vals.tolist(), split, val_ratio, seed)
        kept = np.array([int(v) in keep for v in ep_vals], dtype=bool)
        self._ep_vals = ep_vals.astype(np.int64)[kept]
        self._ep_starts = ep_starts.astype(np.int64)[kept]
        kept_counts = ep_counts.astype(np.int64)[kept]
        # Each sample needs chunk_length+1 states -> chunk_length valid windows per episode.
        self._valid_cum = np.cumsum(np.maximum(0, kept_counts - self._chunk_length)).astype(np.int64)

        log.info(
            f"Loaded Stretch dataset root={self._root} split={split!r} camera_mode={camera_mode!r} "
            f"fps={self._fps} kept_episodes={len(self._ep_vals)}/{len(ep_vals)} "
            f"valid_indices={int(self._valid_cum[-1]) if self._valid_cum.size else 0}"
        )

    # ---- spec / dims -------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 10  # [pos(3), rot6d(6), gripper(1)]

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZERS_DIR / "stretch_lerobot_ee_pose_rot6d.json"

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self.action_normalization is None:
            raise RuntimeError("action_normalization is None; normalizer stats were not requested.")
        return super()._load_norm_stats()

    # ---- index helpers -----------------------------------------------------

    @staticmethod
    def _split_episode_ids(ep_ids: list[int], split: str, val_ratio: float, seed: int) -> set[int]:
        if split == "full":
            return set(int(v) for v in ep_ids)
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")
        n_val = max(1, int(round(len(ep_ids) * val_ratio)))
        rng = random.Random(seed)  # identical selection on every rank
        val = set(int(v) for v in rng.sample(list(ep_ids), n_val))
        if split == "train":
            return set(int(v) for v in ep_ids) - val
        return val  # val/valid/validation/eval/test

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode ``(start, length)`` flat-index blocks for
        ``ActionIterableShuffleDataset`` (shuffle block ORDER + shard across
        ranks, sequential within a block)."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(self._valid_cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

    # ---- sample build ------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Resample a different valid window if a frame fails to decode (bounded retries).
        n = len(self)
        last_err: Exception | None = None
        for _attempt in range(8):
            try:
                return self._build_item(idx)
            except Exception as e:  # noqa: BLE001 — skip past undecodable frames
                last_err = e
                log.warning(f"Stretch: sample idx={idx} failed to load ({type(e).__name__}: {e}); resampling")
                if n > 0:
                    idx = random.randint(0, n - 1)
        raise RuntimeError(f"Stretch: failed to load a sample after 8 resamples; last error: {last_err}")

    def _build_item(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]

        stop = start + self._chunk_length + 1  # chunk_length+1 frames of video, chunk_length actions
        timestamps = [float(self._row_timestamp[j]) for j in range(start, stop)]
        video = self._load_video(episode, timestamps)

        # Every action step in the chunk is relative to the SAME anchor: the
        # current state at the start of the chunk (not each step's own current
        # state) — an "anchored" encoding rather than a per-step framewise one.
        anchor_state = self._row_state[start]  # [6] xyz+euler_xyz
        action_state_window = self._row_action_state[start : start + self._chunk_length]  # [chunk, 6]
        action_gripper_window = self._row_action_gripper[start : start + self._chunk_length]  # [chunk, 1]
        action = self._build_action(anchor_state, action_state_window, action_gripper_window)

        # Proprioceptive history, re-expressed in the SAME anchor-relative 10D
        # format as `action` (not raw absolute poses) and emitted under the
        # `history_action` key: `transforms.py` already pops that key and
        # prepends it onto `action` as conditioning frames for every action
        # dataset (see DROIDLeRobotDataset's `history_action`/`use_state`), so
        # this needs no changes outside this file to reach the model.
        ep_start_row = int(self._ep_starts[ep])
        num_past = min(self._history_length - 1, start - ep_start_row)
        hist_start = start - num_past
        history_state_window = self._row_state[hist_start : start + 1]  # [<=history_length, 6], last row = current
        history_gripper_window = self._row_gripper[hist_start : start + 1]  # [<=history_length, 1]
        history_action = self._build_action(anchor_state, history_state_window, history_gripper_window)

        task = self._tasks[int(self._row_task[start])]
        ai_caption = random.choice([p.strip() for p in task.split(" | ") if p.strip()] or [task])

        extras: dict[str, Any] = {"history_action": history_action}
        if self._camera_mode == "concat_view":
            extras["additional_view_description"] = (
                "The left half shows the head camera view; the right half shows the gripper-mounted camera."
            )
        if self._mask_action:
            # Video-only SFT: zero the action AND history fed to the model (inert
            # conditioning), but keep idle-frame captioning truthful by computing
            # it from the real action.
            extras["history_action"] = torch.zeros_like(extras["history_action"])
            return self._build_result(
                mode=mode,
                video=video,
                action=torch.zeros_like(action),
                idle_frames_action=action,
                ai_caption=ai_caption,
                **extras,
            )
        return self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption, **extras)

    def _build_action(
        self,
        anchor_state: np.ndarray,
        action_state_window: np.ndarray,
        action_gripper_window: np.ndarray,
    ) -> torch.Tensor:
        anchor_pose = build_abs_pose_from_components(anchor_state[None, 0:3], anchor_state[None, 3:6], "euler_xyz")[0]
        poses_target = build_abs_pose_from_components(
            action_state_window[:, 0:3], action_state_window[:, 3:6], "euler_xyz"
        )
        chunk_length = len(poses_target)
        poses_anchor = np.repeat(anchor_pose[None], chunk_length, axis=0)
        # pose_abs_to_rel only encodes deltas between CONSECUTIVE entries of one
        # trajectory, so interleave [anchor, target_0, anchor, target_1, ...] and
        # keep every other output — the anchor -> target[i] deltas — discarding
        # the target[i] -> anchor deltas we don't want.
        interleaved = np.empty((2 * chunk_length, 4, 4), dtype=poses_anchor.dtype)
        interleaved[0::2] = poses_anchor
        interleaved[1::2] = poses_target
        poses_rel_all = pose_abs_to_rel(interleaved, rotation_format="rot6d", pose_convention="world_framewise")
        poses_rel = poses_rel_all[0::2]  # [chunk_length, 9]: anchor -> target[i]
        action = np.concatenate([poses_rel, action_gripper_window], axis=-1)  # [chunk_length, 10]
        return torch.from_numpy(np.ascontiguousarray(action)).float()

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        frames_by_view = {}
        for key in self._video_keys:
            from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
            frames = decode_video_frames(
                self._video_path(episode, key),
                [from_ts + ts for ts in timestamps],
                self._tolerance_s,
            )  # [T, C, H, W] in [0, 1]
            if self._center_crop:
                frames = self._crop_to_square(frames)
            frames = self._resize(frames)
            frames_by_view[key] = frames
        if self._camera_mode == "concat_view":
            # head (left) + gripper (right), horizontally concatenated -> [T, C, H, 2W]
            return torch.cat([frames_by_view[_HEAD_CAMERA], frames_by_view[_GRIPPER_CAMERA]], dim=-1)
        return frames_by_view[self._video_keys[0]]

    @staticmethod
    def _crop_to_square(frames: torch.Tensor) -> torch.Tensor:
        """Center-crop [T, C, H, W] frames to a square on the shorter side.

        Stretch's raw ``head_rgb`` (320x240, portrait) and ``gripper_rgb``
        (240x320, landscape) frames are non-square, so the plain
        ``F.interpolate``-to-square in ``_resize`` would squash/distort them.
        Cropping first (like ``imaginaire.webdataset.augmentors.image.cropping.CenterCrop``)
        preserves aspect ratio at the cost of trimming the long side.
        """
        h, w = frames.shape[-2], frames.shape[-1]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        return frames[..., top : top + side, left : left + side]

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-1] == self._image_size and frames.shape[-2] == self._image_size:
            return frames
        return F.interpolate(frames, size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)
