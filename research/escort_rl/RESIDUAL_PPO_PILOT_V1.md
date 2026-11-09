# Residual-only PPO research pilot v1

This is a bounded experiment to test whether PPO improves the existing avoidance
residual. Perfect imitation is not a mathematical prerequisite for this experiment.
It is **not Stage 2 promotion**: the existing zero-intrusion, family success,
Stage 1 preservation and two-consecutive-round gates remain unchanged.

## Starting point and transition

The source is `runs/residual_variation_v3_20261109/residual_coverage_1.pt`,
SHA-256 `a8bbd73a5a6aee06f23384822f8b7350e148dfc95fdc22f6cdf863b431ff79da`.
Its recorded pedestrian validation is 57/60: crossing 20/20, head-on 20/20,
diagonal 17/20 (one intrusion, two client losses). Stage 1 is 60/60.
These are earlier BC validation results, not new PPO results.

The exact selected Stage 1 actor remains frozen. Its pedestrian inputs are
masked, and the existing 51→128→128→3 residual weights initialize the trainable
actor. A **new critic and fresh PPO optimizer** are created explicitly: the BC
Adam state is not a PPO optimizer continuation. Counters distinguish 28k residual
BC updates, inherited frozen-base 5k PPO transitions, and new residual PPO
transitions (initially zero).

The causal 40-float external observation, Stage 2 scenarios, reward
`stage2_tracking_clearance_v1`, AGILE, physics, scheduling and physical velocity
caps are unchanged. No extra observation normalizer is introduced.

## Stochastic action law

For a valid pedestrian, sample latent `u ~ Normal(residual_logits(o), 0.01)` and
execute `clip(base(o) + 2*tanh(clip(u,-20,20)), -1,1)`.
The ±20 bound is an internal SB3 latent action bound, not a velocity limit.
The ordinary environment still maps normalized commands through the existing
±0.6 m/s and ±0.9 rad/s caps. PPO stores and scores the **raw Gaussian latent**,
not the clipped final velocity action. This avoids an incorrect Gaussian
likelihood for saturated executed commands.

When no pedestrian is valid, the latent action, log probability and entropy are
zero and execution bypasses the residual exactly. The residual actor receives no
policy gradient from these rows. The critic can still learn from Stage 1 replay.
Only one valid pedestrian is supported; additional valid slots are rejected.
Exploration variance is fixed, not learned. Stage 1 parameters never enter the
optimizer. KL stopping is SB3's approximate batch criterion, not a safety proof;
active-pedestrian latent KL is additionally logged.

**Do not feed raw `model.predict()` output to an ordinary Stage2Env.** It is a
latent action. Use `ExecutedPolicy` or the supplied evaluation CLI, which applies
the frozen base and residual transform.

## Preparation and budget

The command below performs:

1. Source hash checks and read-only action import checks on saved validation
   observations. These observations never enter gradient updates.
2. Twenty-four full-G1 stochastic training-partition episodes, four per family,
   to collect actual discounted returns for critic preparation.
3. Six thousand critic-only fitting updates, holding out one complete episode
   per family. The best holdout model must beat its initial error and a constant
   training-mean return predictor; otherwise PPO does not start.
4. At most **5,000 cumulative new residual PPO transitions**, with periodic
   checkpoints. Falls, supervisor failures or physics failures abort training.
5. Deterministic fixed 60 pedestrian and 60 Stage 1 validation cases.

Preparation/evaluation transitions are separately recorded and do not count as
PPO training transitions. This means the complete command takes longer than
collecting 5,000 transitions alone. Geometric intrusions remain recorded failures
and do not count as successful avoidance; full validation gates are unchanged.

CPU defaults: one worker, learning rate 3e-6, rollout 512, batch 128, five epochs,
clip 0.1, target KL 0.005, gamma 0.99, GAE 0.95, entropy coefficient zero,
value coefficient 0.5, gradient norm 0.25. Configuration is
`diagnostics/residual_ppo_pilot_v1.yaml`.

## Run on the host

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/residual_ppo_pilot_XXXXXX")
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
  research/escort_rl/diagnostics/residual_ppo.py train \
  --config research/escort_rl/diagnostics/residual_ppo_pilot_v1.yaml \
  --timesteps 5000 --output "$OUT/pilot" 2>&1 | tee "$OUT/run.log"
printf '\nResults: %s\n' "$OUT"
```

Keep `result.json`, `critic_fit.json`, `critic_episodes.json`, training JSONL,
`training_curves.png`, `logs/` (CSV/TensorBoard), both validation directories and
all ZIP/manifest/RNG triples. Validation directories contain per-episode XY,
velocity and reward information in `trace.jsonl`; the existing comparison tools
can consume those traces. Initial versus final checkpoint selection prioritizes
critical failures, intrusions, Stage 1 success, pedestrian success and formation,
not reward alone. `best_validation.zip` can therefore be the **zero-update initial
policy**. Read its manifest counter before claiming PPO improvement.

The CLI appends an archive registry entry under `docs/bodyguard/MODEL_WEIGHTS.md`
when an actual experiment is saved. No model automatically becomes selected or
promoted. Review the pilot before authorizing further training.

## Evaluation and interruption

To repeat pedestrian evaluation, use a fresh output path:

```bash
"$PY" research/escort_rl/diagnostics/residual_ppo.py evaluate \
  --checkpoint "$OUT/pilot/final.zip" --suite pedestrian \
  --output "$OUT/recheck_pedestrian"
```

Use `--suite stage1` and another fresh directory for Stage 1. To resume an
interrupted, prepared checkpoint, run `train --resume CHECKPOINT.zip` with the
same config, a fresh output directory and `--timesteps REMAINING`. Total saved
residual PPO counter plus REMAINING must not exceed 5,000; at least two samples
per worker are required. Failed/incomplete critic preparation cannot be bypassed
by resume. ZIP, manifest, RNG sidecar and immutable source BC archive are needed.
Source/config/runtime hashes are strict. Resuming restores policy, PPO optimizer
and scenario/global RNG streams after model setup; it starts new physical
episodes and does not restore partial physics or rollout buffers.

## Verification and limitations

Implementation tests include masked Gaussian likelihoods/gradients, exact
inactive-mask preservation, frozen base/variance, residual and critic updates,
critic-only preparation, strict archive checks, fresh-process reload, resume RNG,
independent spawn workers and actual full-G1 optimizer steps. Read-only import
verification matched all 23,399 saved observations exactly, maximum error zero;
see `runs/residual_ppo_import_check_20261109/import_verification.json`.

The affordable full-G1 test collects 64 transitions (6,400 physics steps and 640
gait calls) and performs actual PPO updates. This validates software execution,
not learned avoidance improvement. The full pilot, its stochastic critic
preparation and its post-training success are **not yet empirically validated**.
Live Grove numerical dynamics parity is still unverified. Final-test cases remain
untouched. No ROS, DDS, Docker or system dependency changes are required.

Executed verification on November 9:

- Full existing RL plus scenario-generator suite: **179 passed**, zero failures
  or skips, 108.21 s (includes eleven new residual PPO tests).
- Final focused suite after adding safety-priority selection coverage:
  **12 passed**, zero failures or skips, 11.72 s. Thus 180 distinct tests have
  passed across these runs; the complete suite was not repeated after adding
  the final selection test. Fourteen third-party deprecation warnings remain.
- No dependencies installed and no 5k research pilot executed.

Created: `diagnostics/residual_ppo.py`, `diagnostics/residual_ppo_policy.py`,
`diagnostics/residual_ppo_pilot_v1.yaml`, `tests/test_residual_ppo.py`, this guide,
and the read-only import report directory. Updated: `README.md`, root
`AGENTS.md`, `docs/bodyguard/MODEL_WEIGHTS.md`, and `docs/_debug_stage2.md` to
record the bounded research protocol. Existing runtime, reward, observation,
scenario, teacher, BC inference and checkpoint sources were left intact.

## Logging correction and prepared-checkpoint recovery (November 9)

The first operator run in `runs/residual_ppo_pilot_SDl2ue/pilot` completed all
24 critic episodes successfully and passed calibration (holdout MSE 126.7488 →
8.67035; constant predictor 24.07173). It failed on the first training callback:
raw environment `info` contains NumPy arrays that ordinary `json.dumps` rejects.
This was a logging defect, not gait instability or PPO learning failure.

`initial.zip` is intact: zero residual PPO transitions, zero optimization epochs,
and no PPO Adam entries. The logger now explicitly converts NumPy arrays/scalars,
and the full-G1 smoke test runs the actual production logging callback for 64
transitions. The focused suite passes 14 tests, including strict recovery checks.
The earlier smoke test exercised a counting callback and missed this defect.

A versioned compatibility path accepts only the known pre-fix driver SHA
`d6e96a96a40e5486a418e33b47a8237c66ec92ac87fd2279416a28b455d02bf1`, unchanged
policy source, exact config/runtime/archive hashes, accepted critic preparation,
and verified zero PPO counters/optimizer state. It records the old/new source
hashes in the resumed manifest. Unknown source changes and trained old archives
remain rejected. Original archives are not modified.

Resume the prepared archive using the standard command above with:

```bash
--resume research/escort_rl/runs/residual_ppo_pilot_SDl2ue/pilot/initial.zip
```

Use a fresh output directory and `--timesteps 5000`. This skips critic collection
and fitting, restores saved RNG streams, starts new physical episodes, and runs
the bounded pilot plus paired validation. No PPO training was launched while
implementing this fix.
