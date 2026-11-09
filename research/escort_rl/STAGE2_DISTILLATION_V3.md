# Intervention-balanced imitation and student-state labeling

This is a separately versioned supervised experiment, not PPO. It responds to the
failed v2 student: the teacher passed 60/60 pedestrian validation, but the student
passed only 20/60 and its loss was dominated by ordinary following. Original v2
code, dataset, experiment outputs, rewards, model architecture, source scenarios,
observation schema, full-G1 dynamics and AGILE remain unchanged.

## Changes

- Tool: `diagnostics/stage2_distillation.py`.
- Versioned experiment settings: `diagnostics/stage2_distillation_v3.yaml`.
- Same selected Stage 1-derived parent, two 128-unit MLP layers and 40 causal inputs.
- Actor-only fitting; critic, action variance and non-actor optimizer moments stay
  unchanged. Stale actor Adam moments are explicitly cleared after supervised fit.
- Half of every 192-row batch is Stage 1 replay. The other half has equal crossing,
  head-on and diagonal representation. Each encounter family's 32 rows comprise
  16 substantial interventions and 16 ordinary-following labels, sampled with
  replacement. No change to the PPO environment's scenario probabilities.
- An intervention is a teacher action whose normalized 3-action L2 distance from
  the original follower exceeds 0.15. This is a label-balance definition, not a
  claim that all smaller differences are irrelevant. Intervention presence and
  bounds are verified; missing fitting strata abort explicitly.
- Whole episodes stay disjoint between fitting and holdout. Each source dataset
  keeps its own split when aggregating, so adding DAgger does not move the original
  holdout into fitting. Duplicate family/seed cases and final-test combinations are
  rejected. Family/intervention labels are only used for batching/reporting, never
  passed to the actor or critic.
- Holdout reports show MSE and physical command MAE separately for each family and
  its intervention/ordinary subsets. Selection equally weights families and, within
  encounters, the two subsets. Curves are saved as `fitting_curves.png`.
- The existing training-only dataset is reused with recorded hashes verified.
  Its teacher source predates the appended audit CLI: the compatibility path
  verifies the exact archived source hash, identical Teacher AST and identical
  parameters. It does not ignore a hash mismatch or change runtime provenance.

## Run this first: one 3,000-update weighted fit

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/stage2_distill_v3_XXXXXX")
set -o pipefail
"$PY" research/escort_rl/diagnostics/stage2_distillation.py fit \
  --experiment research/escort_rl/diagnostics/stage2_distillation_v3.yaml \
  --checkpoint research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip \
  --data research/escort_rl/runs/stage2_teacher_v2_readiness_20261108 \
  --updates 3000 --output "$OUT/weighted" 2>&1 | tee "$OUT/run.log"
printf '\nResults: %s\n' "$OUT"
```

This fits then evaluates the exact original 60-case pedestrian suite and 60-case
Stage 1 suite. Stop and review these results before PPO or DAgger. No expectation
of success is assumed. Useful files: `result.json`, `fit_report.json`, the curve,
and both validation directories. PPO counters remain 5,000 inherited Stage 1
transitions; supervised updates are separately logged.

## Diagnostic weights versus a continuation checkpoint

`candidate_actor.npz` and `candidate.json` are retained even if validation fails.
They contain ONLY actor MLP/action-head parameters and checked parent/config/data
lineage. They cannot be supplied to a PPO resume command. Only this tool's
`load_student()` composes them onto their verified original parent for deterministic
student-state collection. Original archives remain untouched.

If both validation gates pass, the tool also writes `prepared.zip` with manifest and
RNG sidecars, restoring the original parent's worker/Python/NumPy/Torch RNG states
before saving. The model registry records diagnostic actors and any accepted archive.
One passing round is not promotion. Changing the actor can invalidate its old
critic's policy-return predictions; review current-policy critic calibration before
PPO. Never fit that critic to a different teacher policy's returns.

## Optional next step if closed-loop avoidance is still weak

This is DAgger-style dataset aggregation: execute the current student in isolated
full-G1 dynamics and query the causal teacher on the states the student actually
visits. It adds command labels, not successful teacher returns or future trajectory
information. It is a versioned follow-up to the weighted-only control, not silently
part of it.

Using the previous `OUT` (or set it to that exact saved experiment directory):

```bash
"$PY" research/escort_rl/diagnostics/stage2_distillation.py dagger \
  --candidate "$OUT/weighted" \
  --episodes-per-family 3 --teacher-probability 0.0 \
  --output "$OUT/dagger" 2>&1 | tee "$OUT/dagger.log"
```

This runs 18 development/training-partition episodes, three per family, using a
fixed seed start 880000 and independent mixture RNG. With probability 0.0 the
student executes EVERY action; there are no hidden teacher overrides. Each trace
logs student command, teacher label, executed command, execution-owner flag and
actual state. Proxy intrusion terminates normally and labels from failed student
rollouts are retained, explicitly marked as student-state labels. Critical
fall/supervisor/physics failure aborts; incomplete output cannot be fitted.
Actual active physics/gait counters are recorded. No ROS/DDS is initialized.

An explicitly chosen nonzero `--teacher-probability` executes a teacher/student
mixture and is a changed experimental condition. Its probability and actual counts
are recorded. Such results are not autonomous student success.

Then, only after reviewing collection:

```bash
"$PY" research/escort_rl/diagnostics/stage2_distillation.py fit \
  --data research/escort_rl/runs/stage2_teacher_v2_readiness_20261108 \
  --data "$OUT/dagger" --updates 3000 \
  --output "$OUT/aggregated" 2>&1 | tee "$OUT/aggregated.log"
```

The aggregated fit starts again from the same verified Stage 1-derived parent,
not from a guessed optimizer state for a parameter-only candidate. All supervised
updates/collections are reported separately. Further rounds require new explicit
seed configuration and fresh paths; repeated family/seed cases are rejected.
Fixed validation/final-test trajectories must never become fitting data.

## Verification and limits

New tests cover rare-event batch composition, conditional error reporting, preservation
of trained critic/variance/optimizer state, exact source compatibility, per-source
holdout stability, duplicate/tamper rejection, actor-only reload, failed/passed gate
checkpoint handling, preserved RNG/counters, and beta=0/beta=1 DAgger execution logs.
DAgger schedule tests use a mocked environment and genuine SB3 actor, not evidence
of completed full-G1 DAgger rollouts. Fitting tests use temporary synthetic models.
No research fit or DAgger collection was launched during implementation. Existing
full-G1 regression tests are rerun separately. Live Grove dynamics parity remains
unverified. Balanced imitation and aggregation are hypotheses to test, not claimed
solutions or Stage 2 promotion.

Final implementation verification (2026-11-08): **150 passed, zero failed/skipped**,
14 third-party deprecation warnings, 97.96 seconds. Command:

```bash
PYTHONPATH=research/escort_rl:research/escort_scenarios \
research/escort_rl/.venv/bin/python -m pytest \
  research/escort_rl/tests research/escort_scenarios/tests -q
```

The 11 new tests passed separately. Development tests caught a missing environment
import, which was fixed. A suite started before the JSON-to-YAML configuration
rename retained the old path and failed; the final complete run above used the
final YAML configuration and passed. There are no outstanding test failures.
