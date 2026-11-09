# Stage 2 causal teacher v2 — results

## What changed

Version 1 only zeroed forward/lateral commands when it predicted a close encounter.
In failed diagonals the pedestrian could still move into the stopped robot, or the
robot waited long enough to lose the client. Three matched development diagonals
had two geometric intrusions and one client loss.

`diagnostics/stage2_causal_teacher.py` now evaluates 49 planar velocity-grid
candidates plus the desired formation velocity. It assumes constant pedestrian
velocity from the current causal observation over four seconds, retaining measured
robot motion for an assumed 0.4-second response lag. Its score is:

```
200 * max(0.85 - minimum_predicted_centre_distance, 0)^2
+ ||candidate_velocity - desired_formation_velocity||^2
+ 0.1 * ||candidate_velocity - previous_selected_velocity||^2
```

Desired velocity is current client velocity plus 0.65 times the current body-frame
formation target error, clipped per axis to +/-0.35 m/s. The grid is -0.3 through
+0.3 m/s in 0.1 increments. Velocity-to-command conversion uses the existing
follower's dead-zone compensation; commands are still clipped by the original
bounds. Yaw follows the original follower. Without a pedestrian, output is exactly
the original follower. The same teacher handles every family without a family input.
It can sidestep, retreat, brake and recover rather than stopping indefinitely.

This search is only high-level command selection. Every executed G1 step uses
unchanged frozen AGILE and full MuJoCo dynamics. The constant-velocity/response-lag
predictor is an approximation, not access to a future scripted trajectory and not
a replacement physics model. It gives no formal collision-avoidance guarantee.

## Executed full-G1 results

1. Three matched formerly failed development diagonals, seeds 853000/853004/853008:
   **3/3 success**, versus 0/3 for stop-only v1. Minimum geometric surface clearances
   were 0.069, 0.177 and 0.120 m. Evidence and an inspected XY/clearance comparison:
   `runs/stage2_teacher_v2_dev_20261108/old_vs_new_diagonal.png`.
2. A separate training-partition readiness batch, seed start 854000:
   **60/60 success**, ten each straight/turns/stops/crossing/head-on/diagonal.
   It produced **10,878** finite float32 40-observation/3-action demonstrations.
   Cases are paired across families using the seed range; there are 60 distinct
   family/seed cases, not 60 distinct numeric seeds. Failed outcomes would have
   remained in reports and been excluded from expert labels; none failed here.
   Data: `runs/stage2_teacher_v2_readiness_20261108/`. Dataset SHA-256:
   `f0f426e5f8652e97f9e35729c6b819208abb26ac39d4d73ddb7896e4bf681d26`.
3. Original fixed pedestrian validation, separate from fitting data:
   **60/60 success**, crossing 20/20, head-on 20/20, diagonal 20/20; zero geometric
   intrusions or recorded fall/supervisor/physics failures. Mean episode formation
   error 0.11928 m; minimum surface clearance 0.06089 m. Wall time 370.82 s.
   Evidence: `runs/stage2_teacher_v2_validation_20261108/summary.json` and traces.
   No fitting dataset is exported from validation. Final-test cases were untouched.

Minimum clearance is positive but sometimes narrow. These are noncolliding scripted
pedestrian proxies; success does not demonstrate real-person safety, human responses
or physical person contacts. This is one set of simulator cases, not a robustness
confidence bound. No reward, caps, scenario generator, observation schema, robot
physics, safety controller or gait checkpoint was changed.

## Code and tests

- New `diagnostics/stage2_causal_teacher.py`: causal planner and fixed validation CLI.
- Updated `diagnostics/stage2_teacher_bootstrap.py`: protocol v2, one teacher for all
  families, new teacher/config hash in collection provenance. Old v1 failed datasets
  are not silently accepted under the new protocol.
- New `tests/test_stage2_causal_teacher.py`: no-pedestrian exact preservation,
  reaction to approaching pedestrian, bounded finite commands, deterministic and
  independent state, far-pedestrian progress.
- Updated `STAGE2_TEACHER_BOOTSTRAP.md` and AGENTS.md with current status.
- Full existing environment/scenario regression plus new tests: **139 passed**,
  zero failures/skips, 14 third-party deprecation warnings, 99.38 seconds.

No neural model weights were created or modified. The model registry therefore
needs no new weight entry. Runtime/controller source hashes remain unchanged.
Teacher source snapshots matching recorded hashes accompany the readiness and
validation evidence; adding the validation CLI did not change the Teacher class.

## Next step

Use the already collected readiness dataset for a bounded actor-only imitation fit
from the selected Stage 1-derived transition archive. There is no need to collect
it again. Keep 50% Stage 1 replay, the original reward and unchanged critic/variance;
run both fixed validation suites as implemented. Save a resumable candidate only
if it passes those gates. Teacher success does not guarantee that a feed-forward
student will imitate it successfully. No MLP fitting or PPO was launched during
this teacher-improvement milestone.

See `STAGE2_TEACHER_BOOTSTRAP.md` for the exact fitting command. After fitting,
review imitation and validation results before any PPO; the changed actor may
require current-policy critic calibration. Teacher returns must not be used as
critic labels for a different stochastic policy.
