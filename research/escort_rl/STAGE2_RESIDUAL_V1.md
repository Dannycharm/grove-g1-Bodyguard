# Frozen Stage 1 residual avoidance v1

Implemented November 8, 2026. This is a separately versioned architecture and
supervised diagnostic, not a silent resume or a PPO implementation. Original
weights, runtime modules, observations, rewards, scenarios, gait and physics remain
unchanged. See `STAGE2_RESIDUAL_RESULTS_2026-11-08.md` for actual results.

## Architecture and exact preservation

The selected successful Stage 1 archive is loaded directly with its pinned SHA-256,
matching source/config/dependency/gait provenance, and unchanged manifest/RNG sidecars.
ALL parameters in its saved policy, including actor, critic and log sigma, have
`requires_grad=False`. The residual optimizer receives ONLY residual parameters.
Every fitting iteration checks original base tensors and the original archive hash.

The base ALWAYS sees zero in pedestrian slots 22:40, including when a pedestrian
is present. Those columns were inactive during Stage 1; feeding live pedestrian
values into its unused columns would alter its behavior despite frozen parameters.

```
a_base = Stage1.predict(observation with pedestrian slots zeroed, deterministic=True)
g = any valid pedestrian mask
Delta = 2 * tanh(residual_MLP(internal_features))
a_final = clip(a_base + g * Delta, -1, 1)
```

Inactive rows bypass residual inference and addition entirely. Thus effective
residual is EXACTLY zero and final action bitwise equals the selected Stage 1 action
when no pedestrian is valid, even if residual parameters are arbitrary. Version 1
executes only the Stage 2 single-pedestrian slot; other valid slots are rejected
rather than pretending later stages are supported. Output commands retain the
original limits. Base means are clipped exactly as SB3 deterministic applied actions
before residual addition, so teacher-minus-base targets are feasible within +/-2.

The external observation remains the same causal float32 40-vector. Internal inputs
are 51 values: the current 40 values with fixed rescaling for the NEW branch only,
eight analytic risk features, and the three detached base actions. The base's own
normalization/inputs are unchanged. Risk features are current distance, closing
speed, bounded time/minimum separation at closest approach under current relative
velocity, time/minimum separation under desired nominal formation velocity,
nominal clearance deficit and current client speed. They use a four-second
constant-velocity assumption, never scenario futures or privileged evaluator data.

Residual architecture: `51 → 128 Tanh → 128 Tanh → 3`, followed by `2*tanh`;
23,555 trainable parameters. Its last layer starts at zero. The frozen saved base
contains 44,039 actor/critic/variance parameters, none optimized here. No recurrence,
running observation normalizer or gait changes are introduced.

## Teacher audit and observable anchoring

The existing successful velocity-candidate teacher penalizes changes relative to
an internally remembered selected velocity. At identical observations, adversarial
changes of that memory changed actions in 59.58% of 5,394 pedestrian-state probes;
maximum normalized action L2 difference was 1.898. These are stress probes, not
proof that 59.58% of recorded labels conflict. Exact recorded-sequence replay matched
all labels, and no conflicting EXACT duplicate observation was found.

The previous applied action is already in indices 19:22. The versioned teacher mode
`observable_previous_command_v1` reconstructs the previous planar target velocity
from that action using the existing follower's command compensation, before invoking
the unchanged teacher. This removes dependence on unobserved/counterfactual teacher
memory during student-only DAgger. It changed only ONE of the 5,394 original labels
by more than 0.05 normalized L2, but was still explicitly revalidated, not silently
assumed equivalent. The original teacher/source datasets remain intact. The new
observable teacher passed all 60 fixed pedestrian cases with zero intrusions.

## Supervision, clipping and sample priorities

Each label is the observable teacher's normalized command. The residual target is
`teacher_action - frozen_base_action`, computed at the SAME current observation.
Loss combines final clipped action MSE with 0.25 times raw residual-target MSE and
0.01 times excess saturation cost. Raw residual supervision supplies a gradient
when the clipped final action is in a flat saturation region. Labels and final
commands retain existing action bounds.

Sampling weights are YAML-controlled: ordinary 1, causal anticipation 4, avoidance
8, recovery 4, stopping 4. Anticipation is classified from CURRENT predicted risk,
not by looking ahead through logged future labels. The teacher/base command gap
marks avoidance; current client velocity and formation error mark stopping/recovery.
These phase labels affect sampling only, never actor inputs. Actual high-level
states/actions and teacher execution flags are logged.

## Implemented bounded diagnostic

`diagnostics/stage2_residual.py` reuses Stage2Env, the original scenario generator,
existing teacher, fixed-suite evaluator and unchanged gates. Experiment settings
are `diagnostics/stage2_residual_v1.yaml`.

The diagnostic audits the teacher before training, checks training-partition cases
and excluded route combinations, validates the teacher on one crossing (904000) and
one diagonal (904004), and measures the zero-residual baseline. It then runs:

1. 4,000 residual-only BC updates on teacher trajectories; closed-loop two-case test.
2. Student-only DAgger on those development cases, queried by the teacher; another
   4,000 updates on aggregate labels; closed-loop two-case test.
3. A second student-only DAgger batch, another 4,000 updates and closed-loop test.

Optimizer and local batch RNG continue across iterations. There is no restarting
from the original parent each round and no base update. This is deliberate same-case
overfit diagnostics, not independent generalization or promotion. Final fixed-suite
validation is separate; no validation trajectory enters fitting. Total BC updates
are 12,000, not PPO timesteps. Final-test data is untouched.

## Artifacts and reproducibility

Each `.pt` stores residual state, optimizer state, Torch/local NumPy RNG states,
update count and resolved configuration. Its `.manifest.json` pins base archive,
base manifest/RNG sidecars, residual and teacher source hashes, full runtime contract,
feature schema and dataset hashes. Loading verifies hashes and original Stage 1
provenance; do not bypass a mismatch. Exact implementation/teacher/config snapshots
accompany the executed run. Original model archives are never overwritten.

The loader supports deterministic evaluation. Training CLI resume and partial
physics/episode restoration are NOT implemented; saved optimizer/RNG state is for
provenance and a future explicit tested continuation path, not a promise that rerunning
`diagnostic` resumes. Existing output paths are rejected. These files are NOT SB3
PPO archives; guarded residual PPO requires a separately scoped adapter and validation.
Stochastic prediction is explicitly rejected. No PPO updates ran here.

## Reproduction commands

Do not rerun completed experiments just to inspect results. The executed run is
`runs/residual_overfit_v1b_20261108/`; final checkpoint is `residual_2.pt`.
For a NEW reproducibility run:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/residual_reproduction_XXXXXX")
set -o pipefail
"$PY" research/escort_rl/diagnostics/stage2_residual.py diagnostic \
  --output "$OUT/diagnostic" 2>&1 | tee "$OUT/run.log"
```

Run fixed validation in fresh paths (the completed run already has these results):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
  research/escort_rl/diagnostics/stage2_residual.py evaluate \
  --checkpoint "$OUT/diagnostic/residual_2.pt" \
  --suite pedestrian --output "$OUT/validation_pedestrian"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
  research/escort_rl/diagnostics/stage2_residual.py evaluate \
  --checkpoint "$OUT/diagnostic/residual_2.pt" \
  --suite stage1 --output "$OUT/validation_stage1"
```

Teacher variant verification and read-only plotting:

```bash
"$PY" research/escort_rl/diagnostics/evaluate_observable_teacher.py \
  --output "$OUT/observable_teacher_validation"
"$PY" research/escort_rl/diagnostics/plot_residual_diagnostic.py \
  --run "$OUT/diagnostic"
```

Bottom-row plot base commands are the frozen base QUERIED on residual-visited
states, not commands executed in the separate base-only trajectory.

Tests:

```bash
PYTHONPATH=research/escort_rl:research/escort_scenarios \
research/escort_rl/.venv/bin/python -m pytest \
  research/escort_rl/tests research/escort_scenarios/tests -q
```

Final regression: 159 passed, zero failed/skipped; 14 third-party deprecation
warnings, 100.83 seconds. Nine new tests verify real-checkpoint exact preservation,
base input isolation, residual-only gradients, saturation gradient handling, finite
causal risk geometry, observable teacher consistency, checkpoint integrity/RNG state,
partition/unsupported-mode rejection and relative-path registry output. These tests
plus measured closed-loop experiments are different kinds of evidence; neither
establishes unmeasured live Grove numerical parity or real-person safety.
