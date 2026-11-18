#!/usr/bin/env python3
"""Collect a configurable SAPIEN expert demonstration.

This is an adapter template for the SAPIEN scene used by this repository.  It
keeps the experiment-specific scene builder and robot controller private while
making the important data contract explicit.  ``--print-plan`` works without
SAPIEN and is useful for checking a new mug pose before connecting the adapter.

To run a real collection, provide an importable factory with ``--adapter``::

    python examples/sapien_custom_demonstration.py \
        --adapter my_sapien_adapter:create_adapter \
        --mug-x 0.45 --mug-y -0.04 --mug-yaw-deg 15 --seed 12 \
        --output runs/my_demo_12

The factory must return an object implementing ``SapienAdapter`` below.  The
adapter is where the local SAPIEN actor meshes, camera renderer, and MPlib (or
another collision-aware planner) are connected.  The collector records every
control-step observation and action, rather than only storing a final pose.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convert numpy values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def euler_xyz_deg_to_quat_xyzw(euler_deg: Sequence[float]) -> np.ndarray:
    """Return an XYZW quaternion for intrinsic XYZ Euler angles in degrees."""
    rx, ry, rz = (math.radians(float(v)) * 0.5 for v in euler_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # This convention matches the quaternion field expected by the adapter.
    return np.asarray(
        [
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ],
        dtype=np.float32,
    )


@dataclass
class SceneConfig:
    """Parameters that deliberately change the SAPIEN scene between episodes."""

    seed: int = 0
    mug_center: tuple[float, float, float] = (0.45, -0.03, 0.055)
    mug_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mug_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rack_center: tuple[float, float, float] = (0.694, -0.075, 0.145)
    table_height: float = 0.0
    distractors: int = 0
    camera_name: str = "six_camera_live_actor_mesh"


@dataclass
class Waypoint:
    """One Cartesian target sent to the collision-aware robot planner."""

    name: str
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    gripper_open: float
    max_speed_m_s: float = 0.25


class SapienAdapter(Protocol):
    """Interface implemented by a project-specific SAPIEN/MPlib adapter.

    ``execute_waypoint`` must yield one mapping per simulated control step.  A
    mapping contains ``observation`` with ``xyz`` and ``rgb`` arrays and may
    contain ``state`` and ``frame_rgb``.  The action is reconstructed from the
    waypoint and recorded at the same index as that observation.
    """

    def reset(self, scene: SceneConfig) -> None: ...

    def execute_waypoint(self, waypoint: Waypoint) -> Iterable[Mapping[str, Any]]: ...

    def success(self) -> bool: ...


def build_plan(scene: SceneConfig) -> list[Waypoint]:
    """Build a segmented grasp, transport, insertion, and retreat plan.

    Change these offsets when testing a different grasp or rack approach.  The
    mug pose is read from ``SceneConfig``; no target pose is used to construct
    the point-cloud observation.
    """
    mug = np.asarray(scene.mug_center, dtype=np.float32)
    rack = np.asarray(scene.rack_center, dtype=np.float32)
    grasp_rotation = euler_xyz_deg_to_quat_xyzw(scene.mug_euler_deg)
    def point(base: np.ndarray, offset: Sequence[float]) -> tuple[float, float, float]:
        return tuple((base + np.asarray(offset, dtype=np.float32)).tolist())

    return [
        Waypoint("approach", point(mug, (0.0, 0.0, 0.16)), tuple(grasp_rotation), 1.0),
        Waypoint("pre_grasp", point(mug, (0.0, 0.0, 0.055)), tuple(grasp_rotation), 1.0),
        Waypoint("grasp", point(mug, (0.0, 0.0, 0.012)), tuple(grasp_rotation), 0.0, 0.08),
        Waypoint("lift", point(mug, (0.0, 0.0, 0.18)), tuple(grasp_rotation), 0.0),
        Waypoint("safe_transport", point(rack, (-0.14, 0.0, 0.18)), tuple(grasp_rotation), 0.0),
        Waypoint("pre_insert", point(rack, (-0.05, 0.0, 0.10)), tuple(grasp_rotation), 0.0, 0.12),
        Waypoint("insert", point(rack, (0.0, 0.0, 0.045)), tuple(grasp_rotation), 0.0, 0.06),
        Waypoint("release", point(rack, (0.0, 0.0, 0.045)), tuple(grasp_rotation), 1.0, 0.04),
        Waypoint("retreat", point(rack, (-0.12, 0.0, 0.18)), tuple(grasp_rotation), 1.0),
    ]


def _action_for(waypoint: Waypoint) -> dict[str, Any]:
    return {
        "target_position": list(waypoint.position),
        "target_rotation_xyzw": list(waypoint.rotation_xyzw),
        "gripper_open": waypoint.gripper_open,
        "max_speed_m_s": waypoint.max_speed_m_s,
    }


def _write_gif(frames: list[np.ndarray], path: Path, fps: int = 12) -> None:
    """Write optional RGB frames as a playable GIF without requiring ffmpeg."""
    if not frames:
        return
    from PIL import Image

    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB") for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=max(1, int(1000 / fps)), loop=0)


def collect_episode(adapter: SapienAdapter, scene: SceneConfig, output: Path) -> dict[str, Any]:
    """Reset, execute, and persist one complete expert episode."""
    output.mkdir(parents=True, exist_ok=False)
    adapter.reset(scene)
    plan = build_plan(scene)
    records: list[dict[str, Any]] = []
    raw_steps: list[Mapping[str, Any]] = []
    frames: list[np.ndarray] = []
    for waypoint in plan:
        for step in adapter.execute_waypoint(waypoint):
            if "observation" not in step:
                raise ValueError("adapter step is missing observation")
            observation = step["observation"]
            if "xyz" not in observation or "rgb" not in observation:
                raise ValueError("each observation must contain xyz and rgb")
            raw_steps.append(step)
            records.append(
                {
                    "step_index": len(records),
                    "stage": waypoint.name,
                    "action": _action_for(waypoint),
                    "state": _jsonable(step.get("state", {})),
                    "timestamp_s": float(step.get("timestamp_s", len(records))),
                    "observation": {
                        "xyz_shape": list(np.asarray(observation["xyz"]).shape),
                        "rgb_shape": list(np.asarray(observation["rgb"]).shape),
                    },
                }
            )
            if step.get("frame_rgb") is not None:
                frames.append(np.asarray(step["frame_rgb"], dtype=np.uint8))

    if not records:
        raise RuntimeError("adapter returned no control steps")
    # The adapter keeps arrays in the step mappings so the collector can write
    # one aligned NPZ.  Storing every frame is required for sequence training.
    xyz = np.stack([np.asarray(step["observation"]["xyz"], dtype=np.float32) for step in raw_steps], axis=0)
    rgb = np.stack([np.asarray(step["observation"]["rgb"], dtype=np.float32) for step in raw_steps], axis=0)
    if xyz.shape[0] != len(records) or rgb.shape[0] != len(records):
        raise RuntimeError("observation/action length mismatch")
    np.savez_compressed(output / "observations.npz", xyz=xyz, rgb=rgb)
    (output / "scene_spec.json").write_text(json.dumps(_jsonable(asdict(scene)), indent=2) + "\n")
    trajectory = {
        "scene": _jsonable(asdict(scene)),
        "waypoints": [_jsonable(asdict(item)) for item in plan],
        "steps": records,
        "success": bool(adapter.success()),
        "observation_contract": "live SAPIEN actor-mesh XYZ/RGB captured before the corresponding action",
    }
    (output / "trajectory.json").write_text(json.dumps(_jsonable(trajectory), indent=2) + "\n")
    _write_gif(frames, output / "rollout.gif")
    metadata = {
        "episode_id": output.name,
        "frame_count": len(records),
        "success": bool(adapter.success()),
        "artifacts": {name: _sha256(output / name) for name in ("scene_spec.json", "trajectory.json", "observations.npz")},
    }
    if (output / "rollout.gif").exists():
        metadata["artifacts"]["rollout.gif"] = _sha256(output / "rollout.gif")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_factory(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--adapter must have the form module_name:factory")
    return getattr(importlib.import_module(module_name), attribute)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="Python factory module:attribute for a real SAPIEN adapter")
    parser.add_argument("--output", type=Path, default=Path("runs/custom_sapien_demo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mug-x", type=float, default=0.45)
    parser.add_argument("--mug-y", type=float, default=-0.03)
    parser.add_argument("--mug-z", type=float, default=0.055)
    parser.add_argument("--mug-roll-deg", type=float, default=0.0)
    parser.add_argument("--mug-pitch-deg", type=float, default=0.0)
    parser.add_argument("--mug-yaw-deg", type=float, default=0.0)
    parser.add_argument("--mug-scale", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("SX", "SY", "SZ"))
    parser.add_argument("--rack-x", type=float, default=0.694)
    parser.add_argument("--rack-y", type=float, default=-0.075)
    parser.add_argument("--rack-z", type=float, default=0.145)
    parser.add_argument("--table-height", type=float, default=0.0)
    parser.add_argument("--distractors", type=int, default=0)
    parser.add_argument("--print-plan", action="store_true", help="print the computed waypoints and exit")
    args = parser.parse_args()
    scene = SceneConfig(
        seed=args.seed,
        mug_center=(args.mug_x, args.mug_y, args.mug_z),
        mug_euler_deg=(args.mug_roll_deg, args.mug_pitch_deg, args.mug_yaw_deg),
        mug_scale=tuple(args.mug_scale),
        rack_center=(args.rack_x, args.rack_y, args.rack_z),
        table_height=args.table_height,
        distractors=args.distractors,
    )
    plan = build_plan(scene)
    if args.print_plan:
        print(json.dumps({"scene": _jsonable(asdict(scene)), "waypoints": [_jsonable(asdict(item)) for item in plan]}, indent=2))
        return
    if not args.adapter:
        parser.error("--adapter is required unless --print-plan is used")
    factory = load_factory(args.adapter)
    metadata = collect_episode(factory(), scene, args.output.resolve())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
