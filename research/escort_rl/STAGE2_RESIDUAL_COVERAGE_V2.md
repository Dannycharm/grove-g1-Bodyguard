# Residual coverage v2: frozen-base continuation

This extends the completed v1 residual diagnostic without modifying its source,
checkpoint, frozen base, feature schema, teacher, loss, gait, physics or observations.
It is supervised BC/DAgger only; no PPO or Stage 3 implementation.

## Versioned continuation

- Driver: `diagnostics/stage2_residual_coverage.py`.
- YAML: `diagnostics/stage2_residual_coverage_v2.yaml`.
- Parent: `runs/residual_overfit_v1b_20261108/residual_2.pt`, SHA-256
  `2ff02f723de84308674a5b4d9ff656e4b7bac5331b1cf530e00a55f13f08d56c`.
- Existing strict v1 loader verifies inference/runtime/base/teacher hashes. It is
  reused unchanged; no source-hash check is disabled.
- Parent residual weights, Adam moments, Torch RNG and saved local NumPy batch RNG
  are restored. Missing optimizer/RNG fails explicitly rather than initializing fresh.
  Parent label hashes are verified and its three source datasets are retained.
- The new training protocol, resolved YAML, parent/source/config/data hashes and
  source snapshots are recorded separately in continuation provenance. Serialization
  remains the unchanged v1 inference format; new training lineage lives in `extra`.
- Base Stage 1 policy remains frozen; inherited PPO counter stays 5,000. Residual
  training proceeds from 12k to 16k to 20k SUPERVISED updates, not PPO transitions.
  The actor has not been refitted from scratch or initialized from the failed whole-MLP
  branches. Physical episodes restart at dataset/evaluation boundaries, not midway.

## Training-only case expansion

The case selector deterministically samples the existing generator, rejects reserved
final-test route combinations and uses only seeds divisible by four. It chooses
one NEW case per family per generator-side (-1/+1), before observing teacher/student
outcomes. Side denotes generator approach/lane choice, not an actor input. It retains
the two original development anchors. No cases are selected from validation failures.

| Family | Seed | Generator side | Role |
|---|---:|---:|---|
| Crossing | 904000 | -1 | Retained anchor |
| Diagonal | 904004 | +1 | Retained anchor |
| Crossing | 920000 | -1 | New |
| Crossing | 920016 | +1 | New |
| Head-on | 930004 | -1 | New |
| Head-on | 930020 | +1 | New |
| Diagonal | 940000 | +1 | New |
| Diagonal | 940004 | -1 | New |

All cases have distinct recorded family/seed signatures; repeated visits within
DAgger are intentional. Family/side/seed metadata is used only for selection,
batching and evaluation, never as neural input. Human parameters and robot physics
ranges are the existing Stage 2 configuration, not enlarged to force easier results.

## Sampling and bounded sequence

Each family receives 1/3 of fitting probability and each of its two sides receives
1/6 overall. Cases within a family/side have equal mass despite different trajectory
lengths or repeated labels. The existing causal phase weights prioritize anticipation,
avoidance, recovery and stopping within each case. This changes the BC sampling
condition, not the PPO environment's scenario probabilities or reward.

The executed budget is:

1. Teacher rollout on all eight predeclared cases. If any teacher case fails, stop
   before new optimization; do not replace it with an easier case.
2. Initial student evaluation on these eight cases.
3. 4k residual-only updates on verified parent data plus new teacher data; closed-loop
   eight-case evaluation.
4. Student-only DAgger on all eight cases, zero teacher overrides.
5. Another 4k updates on the aggregate, preserving optimizer/RNG continuity; closed-loop
   eight-case evaluation. Save both round checkpoints and preserve all old artifacts.
6. Evaluate the final predeclared checkpoint on the unchanged 60 pedestrian and
   60 Stage 1 validation cases; apply unchanged gates. These cases never become labels.

Eight development successes are not promotion. Fixed-suite results and gate decisions
are reported separately in `STAGE2_RESIDUAL_COVERAGE_RESULTS_2026-11-09.md`.

## Commands for reproduction, not a request to repeat completed work

The executed output is `runs/residual_coverage_v2_20261109/`. Inspect it first.
A new explicit reproducibility run uses fresh directories:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/residual_coverage_repro_XXXXXX")
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
  research/escort_rl/diagnostics/stage2_residual_coverage.py train \
  --output "$OUT/training" 2>&1 | tee "$OUT/run.log"
```

Fixed validation (the trainer does not automatically launch it):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
  research/escort_rl/diagnostics/stage2_residual.py evaluate \
  --checkpoint "$OUT/training/residual_coverage_1.pt" \
  --suite pedestrian --output "$OUT/validation_pedestrian"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
  research/escort_rl/diagnostics/stage2_residual.py evaluate \
  --checkpoint "$OUT/training/residual_coverage_1.pt" \
  --suite stage1 --output "$OUT/validation_stage1"
```

A dry metadata-only selection check is available through `cases --output FRESH_PATH`.
Do not pass a residual `.pt` to an SB3 PPO resume CLI. Broader future budgets/parents
require a fresh explicit configuration and matching pinned SHA, not silent resetting,
overwriting this experiment or bypassing provenance. Partial physics episodes are
not restored; the saved network/optimizer/batch stream is the continuation boundary.

## Verification

Six new tests verify deterministic two-side selection and retained anchors, exact
family/side/case sampling mass, real-parent weights/Adam/RNG restoration, residual-only
updates with exact no-pedestrian preservation, explicit missing-state failure and
invalid seed/balance configuration rejection. Full regression: **165 passed**, zero
failed/skipped, 14 third-party deprecation warnings, 100.81 seconds.

No system packages, ROS/DDS, live simulator control, original gait/assets or baseline
world changes are required. Final-test data remains untouched. Scripted proxies are
not physical person-collision bodies, and live Grove numerical parity remains unverified.
