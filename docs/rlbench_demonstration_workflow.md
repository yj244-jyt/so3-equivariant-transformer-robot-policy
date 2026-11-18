# RLBench demonstrations and training workflow

## Frozen demonstrations used by the current adapter

The current RLBench `PickUpCup` training line uses ten successful expert
episodes from the frozen `expert10_pregrasp` run. Each episode stores a full
trajectory before the dataset converter samples training frames. The public
repository contains low-quality playback GIFs for all ten episodes under
[`assets/rlbench/expert_demos_low_quality/`](../assets/rlbench/expert_demos_low_quality/).

The source episode file contains these arrays:

```text
xyz            [T, 8192, 3]  world-frame point cloud
rgb            [T, 8192, 3]  point colour aligned with xyz
gripper_pose  [T, 7]        Panda tip position + XYZW quaternion
gripper_open  [T]           gripper opening state
```

The episode record also stores the task variation, reset seed, initial object
poses, the first-close frame, the future target frame, planner events, native
task success, a video hash, and the source trajectory hash. The policy input
contains only the current XYZRGB observation. The future target pose is used
for supervision and evaluation after the observation has been captured.

## Writing a new complete demonstration

1. Pin the RLBench, PyRep, task, robot, action-mode, camera, and renderer
   versions. Record their source hashes in a run gate. Fix the task variation,
   reset seed, camera calibration, point-cloud coordinate frame, point count,
   and control period before collecting data.
2. Reset a fresh episode and save the initial object poses. At every control
   step, capture the point cloud and RGB values from the same observation, the
   Panda tip pose in one quaternion convention, the gripper state, joint state,
   timestamp, and the action that is executed next. Save all frames; do not
   keep only the first frame or only the final target.
3. Mark semantic events explicitly: approach, pre-grasp, close, lift, move,
   place, release, retreat, and terminal success. For `PickUpCup`, the current
   target contract uses the last open frame before the first close as the
   future grasp target. A different task must define its own event and target
   contract instead of reusing this rule.
4. Save a video synchronized with the trajectory and a JSON record containing
   `episode_id`, variation, seed, frame count, target frame, action stages,
   planner status, native task success, and SHA-256 hashes for every artifact.
   Failed episodes stay in the audit directory and are excluded from the
   training freeze only after their failure stage is recorded.
5. Audit each episode before training: finite values, strictly increasing
   timestamps, matching observation/action lengths, normalized quaternions,
   valid gripper transitions, collision-free planner results, no unexplained
   jumps, and a native task success check. Split by episode, never by adjacent
   frames from the same episode.

A minimal per-episode writer has the following shape. The exact RLBench
`Observation` fields depend on the camera configuration, so the adapter must
make the coordinate and colour conventions explicit.

```python
np.savez_compressed(
    episode_path,
    xyz=np.asarray(xyz_frames, dtype=np.float32),
    rgb=np.asarray(rgb_frames, dtype=np.float32),
    gripper_pose=np.asarray(tip_pose_xyzw, dtype=np.float32),
    gripper_open=np.asarray(gripper_open, dtype=np.float32),
    # Keep reset metadata in JSON, not inside the policy input.
)
```

## Converting complete trajectories to the current RiEMann dataset

The frozen target-pose converter performs these checks and transformations:

- choose 20 unique frames from the start through the last open frame before
  the first closure;
- stack `xyz` and `rgb` as `[episodes, 20, 8192, 3]`;
- repeat the future expert Panda-tip target position as `seg_center` with
  shape `[episodes, 20, 3]`;
- repeat the target world rotation as `axes` with shape `[episodes, 20, 9]`,
  stored as `R.T.reshape(9)`;
- freeze the train/validation split by episode (current split: episodes 0–7
  train and 8–9 validation), then record all source and dataset hashes.

The scripts implementing this contract are maintained in the experiment
workspace as `freeze_rlbench_expert_targets.py` and
`build_rlbench_target_pose_dataset.py`. The public portfolio stores the
resulting media and manifests rather than the private simulator code. For a
new task, copy the contract into a versioned converter and change the
event, target, and action semantics together; changing only the array names
will silently create incorrect labels.

## Training the current two-stage models

The current trainer consumes `pick_dataset.npz` and trains each stage
separately. It uses XYZRGB as input, `seg_center` for the position target, and
`axes` for the orientation target. The `phi` stage predicts the position; the
`psi` stage receives the target ROI during training and predicts position plus
orientation. This is a target-pose adapter, not a sequence policy.

Run a finite-gradient preflight before formal training:

```bash
python riemann_sapien_benchmark/scripts/train_rlbench_pickup_riemann.py \
  --dataset /path/to/pick_dataset.npz \
  --output /path/to/runs/B0/phi_preflight \
  --mode preflight --model B0 --stage phi --device cuda:0 --steps 8
```

After all preflights pass, run the formal stage with a separate output
directory. The current development budget is 200 epochs, seed 0, Adam with
learning rate `1e-4`, and an effective batch size of 4. Repeat the run for
`B0`, `M2`, and `M2wide`, and for both `phi` and `psi`; retain the best
validation-loss checkpoint and its hash.

```bash
python riemann_sapien_benchmark/scripts/train_rlbench_pickup_riemann.py \
  --dataset /path/to/pick_dataset.npz \
  --output /path/to/runs/B0/phi \
  --mode train --model B0 --stage phi --epochs 200 \
  --batch-size 4 --seed 0 --device cuda:0
```

Before a learned rollout, verify checkpoint reload, input hashes, the frozen
scene manifest, planner/controller identity, and the native RLBench success
predicate. Save predictions, planner events, failure stages, and a decoded
video for every episode.

## If the goal is a full trajectory policy

The current `phi -> psi` trainer does not learn every action in a complete
demonstration. To learn a full trajectory, define an action horizon `H` and
export aligned sequence windows instead:

```text
observations: [episode, time, points, 3]   # XYZRGB or separate XYZ/RGB
actions:      [episode, time, H, 8]         # xyz + XYZW + gripper
valid_mask:   [episode, time, H]
```

The action at time `t` must be the action executed after the observation at
time `t`; never pair an observation with a future frame using an undocumented
offset. Pad only at the terminal boundary and mask padded actions in the loss.
Use a position loss, an SO(3) geodesic or quaternion loss, and a gripper loss;
keep all action units and quaternion conventions fixed between training and
rollout. Train and validate by episode, then replay the predicted sequence
through the same native planner and task success predicate. ET-SEED or another
trajectory-level diffusion policy can consume this sequence format, while the
existing RiEMann phi/psi checkpoints cannot be interpreted as full-trajectory
policies without an explicit sequence adapter.
