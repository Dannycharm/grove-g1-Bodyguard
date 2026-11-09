# Residual variation v3: targeted training-only coverage

This phase changes the supervised DATA COVERAGE, not the gait, frozen Stage 1
actor, residual architecture, loss, teacher, feature schema, action limits, reward,
PPO scenario mixture or validation gates. It reuses the coverage continuation driver
with a separately versioned YAML: `diagnostics/stage2_residual_variation_v3.yaml`.
“Variation v3” is a training protocol name, not a new reward version or Stage 3.

## Budget and interpretation

The pinned v2 parent has 20,000 residual BC updates and unchanged inherited 5,000
Stage 1 PPO transitions. Its exact residual weights, Adam state, Torch RNG and saved
local NumPy batch RNG are restored through the unchanged strict v1 inference loader.
Missing state is an error, never a fresh initialization. Original archives remain
intact. V2 source snapshots remain available for exact historical reproduction.

This phase retains the previous optimization cap: TWO rounds of 4,000 updates,
batch 256, learning rate 3e-4, gradient cap 0.5. Cumulative residual BC becomes
28,000; residual PPO remains zero. More diverse rollouts cost additional collection
work. This is not simply running longer on the same cases. However, without a
matched same-parent/no-new-data control, any validation change cannot be causally
attributed solely to coverage rather than further optimization.

## Predeclared data strata, selected before outcomes

All eight v2 cases are retained, including head-on 930004 and 930020. They and their
verified prior teacher/student labels remain in the aggregate. Eight NEW cases are
selected using only generator parameters and training-residue seeds:

| Stratum | Seed | Generator side | Pedestrian speed / encounter fraction / angle constraint |
|---|---:|---:|---|
| Crossing slow/early | 960008 | +1 | [0.25,0.35) m/s; [0.35,0.45) route-duration fraction |
| Crossing slow/late | 960060 | +1 | [0.25,0.35); [0.45,0.55) |
| Crossing fast/early | 960004 | +1 | [0.35,0.45); [0.35,0.45) |
| Crossing fast/late | 960076 | +1 | [0.35,0.45); [0.45,0.55) |
| Diagonal shallow | 980000 | -1 | Absolute approach angle [110,127.5) degrees |
| Diagonal oblique | 980012 | -1 | [127.5,145) degrees |
| Diagonal shallow | 980016 | +1 | [110,127.5) degrees |
| Diagonal oblique | 980004 | +1 | [127.5,145) degrees |

Crossing negative-side anchors remain; the additions focus on positive-side diversity
because the prior validation showed an asymmetry. Diagonal additions cover both
sides/angle strata. Head-on examples are retained, not replaced with easy cases.
All parameters remain within the EXISTING scenario ranges. The selector rejects
reserved final-test route combinations and uses seeds divisible by four. It never
uses validation-case IDs or teacher/student outcome success to pick replacements.

These are 16 distinct family/seed cases, not 16 independent policy-training seeds.
Scenario metadata, strata, side and future route quantities are only for generation,
batching and evaluation, never neural inputs or observation normalization.

## Training and gates

The original coverage workflow is reused: teacher collection on all 16 cases; refuse
optimization if any teacher case fails; initial student test; first BC fit and
closed-loop test; student-only DAgger; second fit and closed-loop test. Family/side/
case sampling stays balanced, with inherited causal phase priorities. Thus the two
retained head-on cases keep 1/3 of batch probability despite the larger number of
crossing/diagonal cases. No teacher action is secretly substituted during DAgger.

The final checkpoint is predeclared by the budget. Then run the unchanged 60-case
pedestrian and 60-case Stage 1 suites. Do not train on those traces, lower gates or
run PPO automatically. Stage 1's inactive-pedestrian residual bypass stays exact.

## Commands

Executed output: `runs/residual_variation_v3_20261109/`. Inspect existing results
before repeating any compute. A fresh reproducibility run:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/residual_variation_repro_XXXXXX")
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
  research/escort_rl/diagnostics/stage2_residual_coverage.py train \
  --experiment research/escort_rl/diagnostics/stage2_residual_variation_v3.yaml \
  --output "$OUT/training" 2>&1 | tee "$OUT/run.log"
```

Evaluate final `residual_coverage_1.pt` using the unchanged v1 inference/evaluation
CLI, in fresh output directories:

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

The driver defaults to the historical v2 config; the explicit `--experiment` is
required for v3. Its checkpoints retain v1 inference serialization with the actual
v3 training protocol/config in manifest `extra`. They are not SB3 PPO ZIP archives.
The historical auto-registry string “coverage v2 round” identifies the reused driver;
the saved manifest/config and reviewed registry entry identify this v3 experiment.

## Verification

Full regression: **168 passed**, zero failed/skipped, 14 third-party deprecation
warnings, 101.96 seconds. Added tests confirm every new cell matches its predefined
parameter ranges, both head-on cases are retained, partitions remain training-only,
invalid cells are rejected, and the real 20k parent/Adam/RNG is restored. Existing
v2 case selection and all exact no-pedestrian preservation tests still pass.

The driver now additionally checks recorded summary hashes where present in parent
source provenance. It does not weaken a legacy hash check or edit v1 inference.
No dependencies, original controllers/assets, ROS/DDS or running simulators change.
See the separate v3 results report for actual learning outcomes and limitations;
a coverage/test pass is not evidence that the learned policy meets the safety gates.

Executed results: [v3 results report](STAGE2_RESIDUAL_VARIATION_RESULTS_2026-11-09.md).
The final fixed-suite score is 57/60, but one geometric intrusion still fails the
unchanged gate. This is not a request to rerun completed work or start PPO.
