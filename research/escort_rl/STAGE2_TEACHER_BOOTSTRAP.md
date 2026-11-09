# Stage 2 causal teacher bootstrap v2

This extends the selected Stage 1 actor rather than replacing it with a fresh MLP.
Default parent is the provenance-checked Stage 2 transition archive
`runs/stage2_readiness_20261107/transition_ready/initial.zip`: selected Stage 1
weights with only the newly active pedestrian input columns initialized to zero.
The inherited counter is 5,000 PPO transitions, zero Stage 2 updates.

Implementation: `diagnostics/stage2_teacher_bootstrap.py`. It reuses the existing
causal yielding/following controller and original Stage 2 environment. No changes
to rewards, observations, gait, physics, caps or runtime source hashes. No PPO
`learn()` is called. No ROS/DDS authority exists.

## Important limitation

Version 1 stopped in place and failed diagonal encounters. Version 2 uses
current-observation velocity candidates to avoid the predicted encounter and recover
formation. Collection and fitting remain separate; readiness depends on measured outcomes. Collection retains
all outcomes and traces. Fitting refuses a dataset unless every one of the six
families reaches at least 80% success and there are no critical failures. Failed
teacher trajectories are never used as imitation targets. Do not repeatedly cherry
pick seeds or lower the gate to make a weak diagonal teacher appear reliable.

The SAME controller is used for every family; it is never given the family label.
Without a pedestrian it returns the original Stage 1 follower exactly. With a
pedestrian it scores 49 grid velocities plus the nominal formation-tracking velocity
over a four-second constant-velocity horizon with an assumed 0.4-second response
lag and a preferred 0.85 m centre distance. The objective combines predicted
clearance deficit, deviation from desired formation velocity, and change from the
previous selected velocity. All control uses the current 40-vector and causal
controller state, never resolved future trajectories. This predictor is a controller
approximation; ALL executed robot motion still uses full MuJoCo and AGILE.
The velocity-to-command compensation reuses the existing follower calibration.
There is no formal safety guarantee from this finite candidate search.
This is a diagnostic training teacher, not the final Reactive Baseline A.

## Run teacher collection first

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/stage2_teacher_XXXXXX")
set -o pipefail
"$PY" research/escort_rl/diagnostics/stage2_teacher_bootstrap.py collect \
  --episodes-per-family 10 --seed 852000 \
  --output "$OUT/data" 2>&1 | tee "$OUT/collection.log"
printf '\nEvidence: %s\n' "$OUT"
```

This is 60 full-G1 episodes, not PPO training. These are training-partition seeds
(divisible by four); held-out route combinations are excluded. Fixed validation and
final-test cases are not demonstrations. Stage 1 replay is collected with the same
causal following controller. Scenario metadata/family labels are only used for
reporting and balanced supervised batching, never policy inputs.

Inspect `data/summary.json`: `teacher_ready` must be true before fitting. On a
failed gate, stop and improve/test the teacher; do not launch a PPO continuation.
A failed teacher audit is useful evidence and should be preserved.

## Fit only after teacher readiness

Use the same shell's `OUT`, or set it to the exact previous output directory:

```bash
"$PY" research/escort_rl/diagnostics/stage2_teacher_bootstrap.py fit \
  --data "$OUT/data" --updates 3000 \
  --output "$OUT/bootstrap" 2>&1 | tee "$OUT/fit.log"
```

The fit uses equal-sized batches from each family, hence 50% Stage 1 replay and
50% pedestrian encounters. It updates existing actor MLP/action-head weights using
mean squared error to normalized teacher commands. It keeps the critic, log sigma,
PPO counters and non-actor optimizer state unchanged. Stale actor Adam moments are
explicitly cleared at this supervised-fine-tuning boundary. Preparation uses a
separate fixed fitting RNG; the parent's worker/Python/NumPy/Torch continuation
states are restored before saving. Offline supervised updates are recorded
separately from PPO transitions.

Whole episodes (at least three successful episodes per family) are split into
fitting and holdout sets. Best balanced holdout action MSE selects fitted weights.
This holdout is used for selection, not an untouched final test. Checkpoint lineage,
dataset hashes, fitting history and source hash are recorded. All old archives
remain unchanged.

After fitting, both original fixed validation suites run automatically. Only a
candidate passing the existing Stage 2 and Stage 1 gates gets `prepared.zip` plus
manifest/RNG sidecars and an automatic model registry entry. One passing round
still does not authorize promotion. Failed candidates retain reports and traces
without producing a resumable checkpoint. Review results before running PPO; no
PPO training starts automatically.

The supervised actor change invalidates the old critic's policy-return calibration.
Before continuing PPO, review whether current-policy critic preparation is needed;
never fit its values to teacher returns from a different policy. No 100k command
is recommended here.

## Implementation verification (2026-11-08)

- Full environment/scenario regression: **135 passed**, zero failed/skipped; 14
  third-party deprecation warnings, 96.34 seconds.
- Four new tests cover readiness rejection, whole-episode/family holdout,
  actor fitting with critic/variance/counter preservation, and seed-partition checks.
- Strict default-parent checkpoint loading verified: 5,000 inherited Stage 1 PPO
  transitions, zero Stage 2 updates.
- Eighteen full-G1 development episodes, three per family, seed start 853000:
  straight/turns/stops/crossing/head-on each 3/3 success; diagonal 0/3 (two geometric
  intrusions, one client loss). Evidence: `runs/stage2_teacher_smoke_20261108/`.
  This small sample tests integration, not teacher generalization. It established
  why the version 1 teacher needed improvement before fitting.
- No neural checkpoint was created or teacher imitation/PPO research run started.
  The fitting unit test updates only a temporary synthetic test model.

Do not run fitting on the failed version 1 dataset. Version 2 deliberately records
a new protocol and teacher hash; old datasets are not silently relabeled.
Read the version 2 result report before collecting or fitting new demonstrations.

## Version 2 verified result and next command

The improved causal teacher passed 60/60 training-partition readiness cases and
60/60 separate fixed pedestrian validation cases, including 20/20 diagonals.
Full regression: **139 tests passed**. See
[teacher v2 results](STAGE2_TEACHER_V2_RESULTS_2026-11-08.md).
The minimum validation surface clearance was 0.06089 m: positive but narrow,
not a formal safety guarantee. No learned policy was fitted in this milestone.

The readiness dataset is already collected; do not repeat collection just to fit:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/stage2_teacher_fit_XXXXXX")
set -o pipefail
"$PY" research/escort_rl/diagnostics/stage2_teacher_bootstrap.py fit \
  --checkpoint research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip \
  --data research/escort_rl/runs/stage2_teacher_v2_readiness_20261108 \
  --updates 3000 --output "$OUT/bootstrap" 2>&1 | tee "$OUT/run.log"
printf '\nResults: %s\n' "$OUT"
```

This performs supervised actor fitting and both validation suites, not PPO. If a
student fails validation, inspect the saved results without bypassing the gates.
