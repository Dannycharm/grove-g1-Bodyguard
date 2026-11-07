# Imitation-pretrained PPO preservation diagnostic

This experiment imports the existing diagnostic imitation **actor only**. It does not
resume the BC archive, claim PPO discovered the controller, or replace the frozen AGILE
gait. The released `tracking_stop_v3` reward/completion logic, 40 causal observations,
128-by-128 actor/critic, Stage 1 scenarios and full G1 dynamics remain fixed.

Implementation: `escort_rl/imitation_ppo.py`, `tools/imitation_ppo.py`,
`configs/ppo_stage1_imitation_warmstart_v1.yaml`, `tests/test_imitation_ppo.py`.

## Prospective protocol

1. Check pinned BC archive, data and manifest hashes. Check protected environment,
   observation, dynamics, gait, scenario and split contracts against the BC manifest.
   This explicitly allows a new training/reward experiment; it does not weaken resume.
2. Evaluate the unchanged BC mean with independent Gaussian per-axis sigma
   1.0, .37, .10, .05 and .02. Use the same twelve development cases, balanced by family,
   training-residue seeds, excluding reserved final-test combinations. Common Gaussian
   random numbers per case allow paired comparisons. Actor input is only the causal
   observation; metadata is diagnostic. Choose the largest tested sigma with >=90%
   success, >=80% per family, mean formation <=.15 m, and no critical failures. With
   four cases per family, these success thresholds require twelve successes.
3. Create a fresh PPO instance with independent model/worker RNG streams. Copy only
   policy-network and action-head tensors. Critic, optimizer and learning counters are
   fresh. Set log_std to the measured chosen level; do not reimport the BC archive's
   untrained unit variance.
4. Freeze actor and variance. Collect 24 complete training-only stochastic episodes.
   Fit the critic to discounted actual C returns with a separate Adam optimizer,
   6,000 updates, batch 256, learning rate .0003. Whole episodes 0,5,10,... form the
   training-data holdout. At true termination bootstrap zero; at time-limit truncation
   bootstrap the initial critic. Select the minimum holdout-MSE critic and require it
   to beat a constant predictor based on the training return mean. The PPO Adam state
   remains fresh. No held-out validation observations/rewards enter this fit.
5. Require the initialized deterministic actor to retain >=90% strict success,
   >=80% each family, mean formation <=.15 m and no critical failures over the fixed
   sixty-case validation suite. Then run exactly 5,000 new PPO transitions using the
   existing defaults and two spawn workers. A critical training failure aborts with
   a recoverable checkpoint. Repeat the full deterministic suite and preserve both
   initial and final archives. No automatic extension or curriculum promotion.

The Gaussian law is unsquashed SB3 PPO: sample raw mean + sigma*z, then clip to the
existing action Box. Noise sweeps log raw and applied actions separately. PPO continues
to learn log_std normally; the preparation does not freeze it during PPO.

## Commands

Run from the repository root. Each output must be fresh. This performs offline headless
simulation only. `noise` is diagnostic evaluation, not training.

```bash
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
RUN=$(mktemp -d "$PWD/research/escort_rl/runs/imitation_ppo_$(date +%Y%m%d_%H%M%S)_XXXXXX")
"$PY" research/escort_rl/tools/imitation_ppo.py noise --output "$RUN/noise"
"$PY" research/escort_rl/tools/imitation_ppo.py train \
  --noise-report "$RUN/noise/noise_report.json" \
  --seed 42 --workers 2 --timesteps 5000 --output "$RUN/preservation_5k"
```

Resume applies only to newly created imitation-PPO archives and requires identical
source/config/dependency hashes, root seed and worker count. It restores policy,
optimizer, counters and independent RNG streams. Active physics episodes and partial
rollouts are restarted; this is not bit-identical uninterrupted training. The code
clears pending vector reset seeds/options before restoring worker streams using the
existing tested SB3 2.7.0 compatibility adapter. Missing/mismatched sidecars fail explicitly.

A later explicit continuation can use `train --resume .../final.zip --timesteps 5000`
with a fresh output, **only after reviewing the preservation result**. Do not resume
an old C-from-scratch or diagnostic BC archive through this path.

## Limitations

Training-only critic preparation consumes additional full-dynamics samples and teacher
supervision already went into the actor. Report those separately from PPO transitions.
A successful deterministic check does not imply stochastic training success, robustness
outside Stage 1, PPO-from-scratch learning or live Grove numerical dynamics parity.
The final-test split remains untouched. No system dependencies or live interfaces change.

## Executed results

Artifacts: `runs/imitation_ppo_20261106/`. All results below are actual isolated full-G1
MuJoCo runs. Regression log: `regression_tests.txt`: **109 passed, zero skipped/failed**,
73.89 seconds. Fourteen third-party Matplotlib/pyparsing deprecation warnings.

| Per-axis sigma | Development success | Mean formation error |
|---|---:|---:|
| 1.00 | 0/12 | .3422 m |
| .37 | 0/12 | .1038 m |
| .10 | 9/12 | .0614 m |
| **.05** | **12/12** | **.0571 m** |
| .02 | 12/12 | .0562 m |

Sigma 1 produced two supervisor failures; .37 failed all stop holds/recovered endpoint
attempts despite relatively small formation error. The selected .05 was the largest
passing measured level. This finite development suite is calibration, not a statistical
robustness guarantee.

Critic preparation: 24/24 successful training-only episodes, 24 unique seeds, **4,467
additional transitions**, 3,544 fitting samples and 923 whole-episode holdout samples.
Held-out MSE **134.066 -> 10.199**, constant predictor **11.712**, explained variance
**.1306**. Actor/log_std stayed identical; PPO optimizer remained fresh. This is a modest
value fit, not evidence of an accurate critic. Hidden route/timing information remains
unavailable to both actor and critic, as required.

| Common strict validation | Prepared imitation actor | After 5k PPO |
|---|---:|---:|
| Success | **60/60** | **0/60** |
| Straight / turn / stop-start | 20/20 each | 0/20 each |
| Mean formation error | .0591 m | 1.0305 m |
| Mean terminal error norm | .0201 m | 1.0731 m |
| Failures | None | 60 recovery timeouts |

No fall, supervisor or physics failures occurred in these validation suites or the 5k
training run. PPO completed exactly **5,000 transitions / 500,000 active physics steps /
50,000 gait calls**, 200 Adam steps over 30 optimization epochs, in **75.064 s**, or
**66.61 transitions/s** including collection, resets, logging and optimizer work. Optimizer
wall time .317 s. Preparation and all evaluation/plotting time are excluded from that
training-only rate. Reported trainer CPU seconds include final parent-process evaluation
and must not be interpreted as training-only CPU utilization. Resident memory ~2.65 GiB.

Training: 26 scenario resets, **26 unique seeds**, 16 straight/5 turn/5 stop-start;
24 completed episodes: ten successes before actor collapse, fourteen recovery timeouts.
Raw Gaussian action-vector bound crossings were .86%, so broad sampled noise alone does
not explain this particular collapse. Sigma moved only .050 -> approximately .0513.

### Collapse occurs during the first optimizer block

The 2,048-transition checkpoint is saved before optimization: actor drift is exactly zero
and all ten completed collection episodes succeeded. The 4,096-transition checkpoint has
only the **first ten optimization epochs / 80 Adam steps** applied. On the same 2,048
training-data causal observation probes, normalized forward-action RMS drift is already
**1.2127**, with mean absolute raw forward output **1.4803**. First-block approximate KL
was **295.19**, objective clipping fraction **.9875**, value loss **2.73**. Final-block KL
was 11.78, clipping .7517; final forward probe RMS drift 1.5754. This is excessive policy
movement relative to sigma .05. PPO's clipped objective does not enforce a hard KL bound;
this experiment retained the original `target_kl=None`, learning rate .0003, ten epochs.

The final policy shifts initial acceleration forward and subsequently settles at the wrong
formation equilibrium. Mean terminal client-frame error is **(+1.0725,+.0320) m**, so the
robot finishes about **.43 m behind the client**, rather than 1.5 m. Mean terminal body
velocity is approximately (.00088,.00038,-.00009): it does stop. This is not evidence that
the low-level gait cannot stop. The actor architecture demonstrably represents a successful
controller; immediate optimization damage is now directly observed. Weak value estimates
and the reward's learning signal may contribute; this run does not separately identify
all causes or prove reward alignment optimal.

Preserved best archive is `preservation_5k/initial.zip`, **not** the collapsed final model.
The untouched diagnostic BC archive is still the initialization source. No continuation,
100k experiment, reward change or larger network was run. `analysis.json`,
`preparation.png`, `optimizer_diagnostics.png`, `preservation_comparison.png`, and final
per-episode plots/traces document the comparison.

## Next command: guarded 5k only, not 100k

Prepared but **NOT EXECUTED**: `configs/ppo_stage1_imitation_guarded_v1.yaml` changes only
subsequent PPO learning rate to **1e-5** and enables **target_kl=.01**. Keep ten epochs,
architecture, reward, critic preparation, sigma, scenarios and root seed fixed. Start
fresh from the original BC actor; do not resume the failed final policy. A target-KL guard
can stop further minibatch updates after detecting excessive KL; it is not an absolute
mathematical bound on every minibatch's parameter movement.

`tools/prepare_imitation_guarded.py` produced `guarded_calibration/noise_report.json` by
explicitly deriving the same actor-noise evidence for these two PPO-only changes. It
verifies every previously measured source hash, dependencies, and all other configuration
fields. Original evidence is untouched. No new noise trials are claimed, and strict
checkpoint resume is not bypassed. A regression test rejects changed noise/config/source.
New candidate files change the repository-wide provenance inventory; historical archives
retain their original manifests and are not silently made resumable with changed sources.

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/imitation_guarded_$(date +%Y%m%d_%H%M%S)_XXXXXX")
set -o pipefail
"$PY" research/escort_rl/tools/imitation_ppo.py train \
  --config research/escort_rl/configs/ppo_stage1_imitation_guarded_v1.yaml \
  --noise-report research/escort_rl/runs/imitation_ppo_20261106/guarded_calibration/noise_report.json \
  --seed 42 --workers 2 --timesteps 5000 \
  --output "$OUT/preservation_5k" 2>&1 | tee "$OUT/run.log"
printf '\nResults: %s\n' "$OUT/preservation_5k"
```

This command repeats training-only critic preparation and initial/final full validation;
it does not repeat the noise sweep. Inspect `result.json`, `logs/progress.csv`,
`validation_final/summary.json` and plots. Require >=90% strict overall, >=80% per family,
no critical failures, mean formation <=.15 m and policy movement consistent with the KL
guard before considering another short continuation. Do not jump directly to 100k.
Even a passing guarded test establishes preservation of imitation competence, not that
PPO learned escorting from scratch. Live Grove numerical dynamics parity remains unverified.
