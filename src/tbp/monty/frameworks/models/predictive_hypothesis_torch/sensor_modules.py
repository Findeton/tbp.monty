from __future__ import annotations

from typing import Any

import numpy as np
from skimage.color import rgb2hsv

from tbp.monty.frameworks.models.change_detecting_sm import ChangeDetectingSM
from tbp.monty.frameworks.models.sensor_modules import CameraSM


_DEFAULT_DETAIL_GRID_SHAPE = (5, 5)
_MICRO_PATCH_SHAPE = (4, 4)


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    max_value = float(np.max(rgb)) if rgb.size else 0.0
    if max_value > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _normalize_grid_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    return (
        max(int(grid_shape[0]), 1),
        max(int(grid_shape[1]), 1),
    )


def _pool_bounds(start: float, end: float, limit: int) -> tuple[int, int]:
    start_idx = int(np.floor(start))
    end_idx = int(np.floor(end))
    start_idx = min(max(start_idx, 0), max(limit - 1, 0))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, limit)
    else:
        end_idx = min(max(end_idx, start_idx + 1), limit)
    return start_idx, end_idx


def _resample_patch(patch: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(patch, dtype=np.float32)
    squeeze_output = False
    if array.ndim == 2:
        array = array[..., None]
        squeeze_output = True

    output_rows, output_cols = max(int(output_shape[0]), 1), max(int(output_shape[1]), 1)
    channels = int(array.shape[2]) if array.ndim == 3 else 1
    result = np.zeros((output_rows, output_cols, channels), dtype=np.float32)

    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        return result[..., 0] if squeeze_output else result

    row_edges = np.linspace(0, array.shape[0], num=output_rows + 1)
    col_edges = np.linspace(0, array.shape[1], num=output_cols + 1)
    for out_row in range(output_rows):
        row_start, row_end = _pool_bounds(
            row_edges[out_row],
            row_edges[out_row + 1],
            array.shape[0],
        )
        for out_col in range(output_cols):
            col_start, col_end = _pool_bounds(
                col_edges[out_col],
                col_edges[out_col + 1],
                array.shape[1],
            )
            window = array[row_start:row_end, col_start:col_end]
            result[out_row, out_col] = window.mean(axis=(0, 1))

    return result[..., 0] if squeeze_output else result


def _reshape_patch_map(
    data: Any,
    height: int,
    width: int,
    channels: int,
) -> np.ndarray | None:
    array = np.asarray(data)
    expected_size = height * width
    if array.size != expected_size * channels:
        return None
    return np.asarray(array, dtype=np.float32).reshape(height, width, channels)


def _reshape_scalar_map(data: Any, height: int, width: int) -> np.ndarray | None:
    array = np.asarray(data)
    expected_size = height * width
    if array.size != expected_size:
        return None
    return np.asarray(array, dtype=np.float32).reshape(height, width)


def _build_state_anchor(
    state,
    *,
    include_world_pose: bool = True,
) -> dict[str, Any]:
    if state is None:
        return {}

    morph = getattr(state, "morphological_features", None) or {}
    anchor = {
        "pose_fully_defined": bool(morph.get("pose_fully_defined", True)),
        "confidence": float(getattr(state, "confidence", 0.0)),
    }
    if include_world_pose:
        anchor["location"] = np.asarray(
            getattr(state, "location", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        )
        anchor["pose_vectors"] = np.asarray(
            morph.get("pose_vectors", np.eye(3, dtype=np.float32)),
            dtype=np.float32,
        )
    if "on_object" in morph:
        anchor["on_object"] = float(morph.get("on_object", 0.0))
    return anchor


def _build_sensor_context(
    observation,
    *,
    resolution: tuple[int, int],
    include_world_pose: bool = True,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "resolution": np.asarray(resolution, dtype=np.float32),
    }
    world_camera = observation.get("world_camera")
    if include_world_pose and world_camera is not None:
        context["camera_pose_world"] = np.asarray(world_camera, dtype=np.float32)
    return context


def _build_packet_cells(
    observation,
    *,
    grid_shape: tuple[int, int],
    micro_patch_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rgba = observation.get("rgba")
    if not isinstance(rgba, np.ndarray) or rgba.ndim < 3:
        return [], {}

    height, width = rgba.shape[:2]
    grid_rows, grid_cols = _normalize_grid_shape(grid_shape)

    depth_map = _reshape_scalar_map(observation.get("depth"), height, width)
    sensor_frame_map = _reshape_patch_map(
        observation.get("sensor_frame_data"),
        height,
        width,
        4,
    )
    semantic_map = _reshape_patch_map(
        observation.get("semantic_3d"),
        height,
        width,
        4,
    )

    support_map = None
    if sensor_frame_map is not None:
        support_map = (sensor_frame_map[..., 3] > 0).astype(np.float32)
    elif semantic_map is not None:
        support_map = (semantic_map[..., 3] > 0).astype(np.float32)

    sensor_points = None
    if sensor_frame_map is not None:
        sensor_points = np.asarray(sensor_frame_map[..., :3], dtype=np.float32)

    row_edges = np.linspace(0, height, num=grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, num=grid_cols + 1, dtype=int)
    cells: list[dict[str, Any]] = []

    for row_idx in range(grid_rows):
        for col_idx in range(grid_cols):
            row_start, row_end = row_edges[row_idx], row_edges[row_idx + 1]
            col_start, col_end = col_edges[col_idx], col_edges[col_idx + 1]
            patch_rgb = rgba[row_start:row_end, col_start:col_end, :3]
            if patch_rgb.size == 0:
                continue

            cell: dict[str, Any] = {
                "uv_center": np.array(
                    [
                        (row_idx + 0.5) / grid_rows,
                        (col_idx + 0.5) / grid_cols,
                    ],
                    dtype=np.float32,
                ),
                "image_bounds": np.array(
                    [row_start, row_end, col_start, col_end],
                    dtype=np.float32,
                ),
                "rgb_patch": _resample_patch(
                    _normalize_rgb(patch_rgb),
                    micro_patch_shape,
                ),
            }

            if depth_map is not None:
                depth_patch = depth_map[row_start:row_end, col_start:col_end]
                cell["depth_patch"] = _resample_patch(depth_patch, micro_patch_shape)

            if support_map is not None:
                support_patch = support_map[row_start:row_end, col_start:col_end]
                cell["support_patch"] = np.clip(
                    _resample_patch(support_patch, micro_patch_shape),
                    0.0,
                    1.0,
                )
                cell["support_mean"] = float(np.mean(support_patch))
                cell["valid_fraction"] = float(np.mean(support_patch > 0.0))

            if sensor_points is not None:
                sensor_patch = sensor_points[row_start:row_end, col_start:col_end]
                cell["sensor_frame_patch"] = _resample_patch(
                    sensor_patch,
                    micro_patch_shape,
                )

            cells.append(cell)

    metadata = {
        "resolution": (height, width),
        "support_mean": float(np.mean(support_map)) if support_map is not None else 0.0,
    }
    return cells, metadata


def build_visual_detail_packet(
    observation,
    sender_id: str,
    grid_shape: tuple[int, int] = _DEFAULT_DETAIL_GRID_SHAPE,
    state=None,
    micro_patch_shape: tuple[int, int] = _MICRO_PATCH_SHAPE,
    include_world_pose: bool = True,
) -> dict[str, Any]:
    grid_rows, grid_cols = _normalize_grid_shape(grid_shape)
    micro_rows, micro_cols = _normalize_grid_shape(micro_patch_shape)
    cells, metadata = _build_packet_cells(
        observation,
        grid_shape=(grid_rows, grid_cols),
        micro_patch_shape=(micro_rows, micro_cols),
    )
    return {
        "packet_type": "visual_observation_packet_v2",
        "packet_version": 2,
        "sender_id": sender_id,
        "grid_shape": [grid_rows, grid_cols],
        "micro_patch_shape": [micro_rows, micro_cols],
        "cell_count": len(cells),
        "shard_count": len(cells),
        "sensor_context": _build_sensor_context(
            observation,
            resolution=tuple(metadata.get("resolution", (0, 0))),
            include_world_pose=include_world_pose,
        ),
        "state_anchor": _build_state_anchor(
            state,
            include_world_pose=include_world_pose,
        ),
        "frame_support_mean": float(metadata.get("support_mean", 0.0)),
        "cells": cells,
    }


def build_change_detail_packet(
    observation,
    state,
    sender_id: str,
    grid_shape: tuple[int, int] = _DEFAULT_DETAIL_GRID_SHAPE,
    micro_patch_shape: tuple[int, int] = _MICRO_PATCH_SHAPE,
    include_world_pose: bool = True,
) -> dict[str, Any]:
    packet = build_visual_detail_packet(
        observation,
        sender_id=sender_id,
        grid_shape=grid_shape,
        state=state,
        micro_patch_shape=micro_patch_shape,
        include_world_pose=include_world_pose,
    )
    non_morph = getattr(state, "non_morphological_features", None) or {}
    packet["packet_type"] = "change_observation_packet_v2"

    flow_magnitude = non_morph.get("flow_magnitude", 0.0)
    if isinstance(flow_magnitude, np.ndarray):
        flow_magnitude = float(flow_magnitude.reshape(-1)[0]) if flow_magnitude.size else 0.0

    feature_deltas: dict[str, Any] = {}
    raw_feature_deltas = non_morph.get("feature_deltas", {}) or {}
    if isinstance(raw_feature_deltas, dict):
        feature_deltas.update(raw_feature_deltas)
    for key, value in non_morph.items():
        if str(key).startswith("delta_"):
            feature_deltas[str(key)[6:]] = value

    temporal_context = {
        "flow_direction": np.asarray(
            non_morph.get("flow_direction", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ),
        "flow_magnitude": float(flow_magnitude),
    }
    if feature_deltas:
        temporal_context["feature_deltas"] = {
            str(key): np.asarray(value, dtype=np.float32)
            for key, value in sorted(feature_deltas.items())
        }
    packet["temporal_context"] = temporal_context
    return packet


class DetailAwareCameraSM(CameraSM):
    def __init__(
        self,
        *args,
        detail_grid_shape: tuple[int, int] = _DEFAULT_DETAIL_GRID_SHAPE,
        include_world_pose: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._detail_grid_shape = detail_grid_shape
        self._include_world_pose = bool(include_world_pose)

    def step(self, ctx, observation, motor_only_step: bool = False):
        state = super().step(ctx, observation, motor_only_step=motor_only_step)
        if state is None:
            return None
        non_morph = dict(getattr(state, "non_morphological_features", None) or {})
        packet = build_visual_detail_packet(
            observation,
            sender_id=self.sensor_module_id,
            grid_shape=self._detail_grid_shape,
            state=state,
            include_world_pose=self._include_world_pose,
        )
        non_morph["observation_packet_v2"] = packet
        non_morph["detail_packet"] = packet
        state.non_morphological_features = non_morph
        return state


class DetailAwareChangeDetectingSM(ChangeDetectingSM):
    def __init__(
        self,
        *args,
        detail_grid_shape: tuple[int, int] = _DEFAULT_DETAIL_GRID_SHAPE,
        include_world_pose: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._detail_grid_shape = detail_grid_shape
        self._include_world_pose = bool(include_world_pose)

    def step(self, ctx, observation, motor_only_step: bool = False):
        state = super().step(ctx, observation, motor_only_step=motor_only_step)
        if state is None:
            return None
        non_morph = dict(getattr(state, "non_morphological_features", None) or {})
        packet = build_change_detail_packet(
            observation,
            state,
            sender_id=self.sensor_module_id,
            grid_shape=self._detail_grid_shape,
            include_world_pose=self._include_world_pose,
        )
        non_morph["observation_packet_v2"] = packet
        non_morph["detail_packet"] = packet
        state.non_morphological_features = non_morph
        return state