# Custom SAPIEN demonstration workflow

## What the existing episodes represent

The accepted ten-episode SAPIEN set is a calibrated T-condition collection.
Its mug center and orientation move slightly with the scene seed, while the
mug scale, rack, table, and distractor count stay fixed. This isolates the
observation/action convention before testing new-instance, new-pose, or
distractor generalization. A collection that intentionally changes only the
robot path for an identical observation is ambiguous for a deterministic
policy, because the input does not identify which path should be selected.

For a generalization collection, make these changes explicit in the frozen
scene manifest:

| Factor | What to change | What to keep paired |
|---|---|---|
| Mug pose | position and roll/pitch/yaw | camera model, robot reset, rack |
| Mug instance | mesh dimensions or scale | task semantics and action units |
| Distractors | count, mesh, colour, placement | target mug identity and camera calibration |
| Robot path | planner waypoints or speed limits | the scene observation and success predicate |

Keep the calibrated T episodes and the generalization episodes in separate
directories and report them separately.

## Connect the template to the simulator

[`examples/sapien_custom_demonstration.py`](../examples/sapien_custom_demonstration.py)
defines `SceneConfig`, `Waypoint`, and the `SapienAdapter` protocol. Implement
the adapter in a local module:

1. `reset(scene)` creates the table, rack, mug, robot, cameras, and optional
   distractors with the supplied seed and mug parameters. Save the resulting
   actor poses and a hash of the scene specification.
2. `execute_waypoint(waypoint)` calls the collision-aware planner, follows the
   returned joint trajectory at the fixed simulation period, and yields one
   mapping per control step. Each mapping must contain `observation.xyz` and
   `observation.rgb` captured from the live actor meshes, plus optional robot
   state and an RGB frame.
3. `success()` evaluates the same physical pick/place predicate used by the
   test harness. Do not use a target pose to construct the point-cloud input.

The template then writes an aligned `observations.npz`, a complete
`trajectory.json`, `scene_spec.json`, SHA-256 metadata, and an optional
playable `rollout.gif`. It keeps failed episodes for diagnosis.

## Change the environment and plan

Print a plan without launching SAPIEN:

```bash
python examples/sapien_custom_demonstration.py --print-plan \
  --seed 12 --mug-x 0.42 --mug-y -0.08 --mug-yaw-deg 15 \
  --mug-scale 1.0 1.0 1.05 --distractors 2
```

The named waypoints are `approach`, `pre_grasp`, `grasp`, `lift`,
`safe_transport`, `pre_insert`, `insert`, `release`, and `retreat`. Edit
`build_plan` when a new grasp frame or rack approach is required. The planner
must reject an unreachable or colliding waypoint; do not silently replace a
failed plan with a straight-line joint interpolation. Record planner status,
collision checks, stage names, and timestamps for every step.

Run a real adapter after binding the local SAPIEN builder and planner:

```bash
python examples/sapien_custom_demonstration.py \
  --adapter my_sapien_adapter:create_adapter \
  --seed 12 --mug-x 0.42 --mug-y -0.08 --mug-yaw-deg 15 \
  --output runs/custom_sapien_demo_12
```

Before training, audit finite values, timestamp order, XYZ/RGB shape equality,
quaternion normalization, gripper transitions, planner status, scene hashes,
and the physical success predicate. Split train and validation by episode.

## Use the trajectory for training

The current RiEMann target-pose adapter can select pre-grasp observations and a
future expert target, as documented in the RLBench workflow. A full trajectory
policy needs an explicit horizon `H` and aligned windows:

```text
observations: [episode, time, points, 3]
actions:      [episode, time, H, 8]  # xyz + XYZW + gripper
valid_mask:   [episode, time, H]
```

The action at time `t` is the action executed immediately after the observation
at time `t`. Keep terminal padding masked, train and validate by episode, and
replay predictions through the same planner and physical success predicate.
