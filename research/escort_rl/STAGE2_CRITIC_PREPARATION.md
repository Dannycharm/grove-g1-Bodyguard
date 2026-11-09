# Stage 2 critic only preparation

This opt-in script tests whether the value network can learn the current policy's new
Stage 2 return distribution without changing its actions. **No PPO actor learning,
new teacher, reward change, risk shaping, scene change, ROS/DDS or dependency installation.**

Implementation: `diagnostics/prepare_stage2_critic.py`.
Settings: `diagnostics/stage2_critic_preparation_v1.json`.
Source checkpoint: experimental Stage 2 first pilot, 5k Stage 2 / 10k cumulative PPO
transitions, listed in `docs/bodyguard/MODEL_WEIGHTS.md`.

## Exact operator command

Run on the host using the existing venv. A full preparation run was NOT launched during
script implementation; the operator command below performs the actual collection/fitting.

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/stage2_critic_XXXXXX")
set -o pipefail

"$PY" research/escort_rl/diagnostics/prepare_stage2_critic.py \
  --checkpoint research/escort_rl/runs/stage2_pilot_YfC5S4/pilot_5k/final.zip \
  --episodes 72 \
  --updates 6000 \
  --output "$OUT/preparation" 2>&1 | tee "$OUT/run.log"

printf '\nPreparation output: %s\n' "$OUT/preparation"
```

Keep the printed directory path. Fresh output is required; existing directories are not
overwritten. Do not automatically run PPO afterward. Inspect `result.json` and
`fit_report.json` first. A failed gate produces a nonzero exit, `not_completed.json`,
data/reports, and **no `prepared.zip`**. Do not edit an old checkpoint manifest to make
an incompatible source/configuration pass.

## What the script collects

- Loads the current Stage 2 checkpoint through the existing exact-provenance loader.
- Holds the original two PPO-worker streams at their saved states. A separate private
  Stage 2 environment samples training-only scenarios using independent preparation RNGs.
- Uses the **current actor's stochastic Gaussian distribution**, including its learned
  action variances and the normal Box clipping. It does not execute the successful reward
  audit controllers or use their positive returns as labels.
- Collects 72 complete episodes using the existing 40/60 replay/interaction episode mix,
  normal terminations and identical full G1 physics. Training namespace residue zero and
  reserved-combination exclusion remain enforced. No fixed validation/final-test cases.
- Uses actual rewards and the model's gamma. Zero bootstrap at true termination; frozen
  parent-critic bootstrap only at a time-limit truncation. Such truncations are recorded.
- Records observations (the SAME causal 40-vector), discounted returns and episode IDs.
  Family/outcome/route metadata are used for splitting and diagnostics only, never added
  to the value-network input or an observation normalizer.

Separate seeds: scenario 73101, action sampling 73102, fitting 73103, episode split 73104.
Preflight of the default sampler gives 23 crossing, 9 straight, 11 gentle-turn, 10 stop/start,
10 head-on and 9 diagonal episodes. This is an episode mixture, not a transition quota:
failures are shorter than successful replay episodes. If a changed run collects fewer
than three episodes of any family, it fails explicitly rather than dropping a family.

Current-policy returns matter: the current actor usually collides in crossings/diagonals,
so those failures must appear in its value targets. Fitting that actor's critic to the
successful hand controller's returns would incorrectly assume a different future policy.

## What changes and what remains identical

Only `mlp_extractor.value_net.*` and `value_net.*` parameters receive gradients. The script
uses a separate Adam optimizer, learning rate 3e-4, batch 256, gradient clipping .5 and up
to 6,000 updates. Actor first layers/head, exploration log_std and all other policy tensors
are frozen. The same separate actor/critic architecture and forty input units remain.

It verifies bitwise identity of protected tensors and actor mean/std outputs over ALL
collected observations, plus actor optimizer state. No actor labels or actor loss are used.
The existing fresh-optimizer imitation fit utility is unsuitable because this source has
trained PPO optimizer state; this value-only implementation explicitly preserves it.

After an accepted fit, stale PPO Adam states are removed **only for value-network
parameters**. Actor/log_std optimizer moments and parameter groups are unchanged. This is
an explicit critic-preparation branch, not an assertion of bit-identical ordinary resume.
The next PPO value update starts fresh Adam state for its changed critic weights.

The original checkpoint, PPO timestep counter, optimization counters and gradient-step
counter remain unchanged. New critic-only optimizer updates and newly collected physics
transitions are accounted separately. Parent global Python/NumPy/Torch RNG states and
both saved PPO-worker scenario streams are restored before checkpoint creation. Active
physical episodes/partial rollout buffers are not restored, consistent with the existing
resume contract.

A better critic **does not make the actor avoid pedestrians by itself**. Its deterministic
commands and stochastic action distribution are unchanged. Do not report improved Stage 2
actor success merely because critic holdout scores improve.

## Whole episode holdout and acceptance

Approximately 20% of complete episodes in EACH of the six scenario families are held out.
No neighbouring samples from the same episode cross the fitting/holdout split. Fitting
uses only the training subset. Every 100 updates, all held-out predictions are measured.
Selection chooses the lowest overall holdout MSE among candidates passing all gates:

1. Overall holdout MSE improves over BOTH the source critic and a constant predictor
   fitted from the training-subset return mean.
2. Crossing and diagonal holdout MSE each improve over their source-critic baselines.
3. Each replay family's MSE remains below the larger of 1.25 times its old MSE and old
   MSE + 1.0. These are explicit initial experimental tolerances, not universal guarantees.
4. At least 80% of held-out states in the final two seconds before crossing/diagonal
   intrusion predict NEGATIVE remaining value. Missing such held-out samples fails the gate.

The report includes MSE, MAE, explained variance, mean predicted/actual returns and
negative-value fraction overall, per family and near intrusion. A fit can beat a constant
and still be poorly calibrated; inspect magnitudes, not just sign or one pass flag.
Perfect prediction is not demanded from a partially observed state. No successful model
is published if no scored checkpoint passes all gates. Data and failure diagnostics remain.

## Outputs and next decision

- `preparation_manifest.json` and `preparation_source.py`: immutable source/config/parent
  identity, explicit collection/action law and separate preparation settings.
- `preparation_scenarios.jsonl`, `episodes.json`: resolved training scenario and outcome logs.
- `rollouts.npz`: causal observations, return labels, episode IDs and evaluation-only
  family/near-failure labels. No actor or critic consumes the latter as features.
- `collection.json`: actual active physics/gait and warmup counts, independent RNG states,
  transitions, family/outcome counts and collection wall time.
- `fit_report.json`: before/after/constant/per-family scores, full score history, selected
  update, complete split IDs and actor/optimizer invariants.
- On success ONLY: `prepared.zip`, `prepared.manifest.json`, `prepared.rng.pt`, `result.json`.

On success, the script appends the verified new model path, hash, parent and evidence to
`docs/bodyguard/MODEL_WEIGHTS.md` under a file lock. It labels the artifact critic-only,
not promoted, and local until deliberately archived. No original files are overwritten.
Output must reside within the repository so its registry links are valid.

The prepared archive remains compatible with the original Stage 2 training loader because
no existing runtime/config source is edited. The additional preparation provenance is
recorded in the model manifest. Both later A/B experiments should start from the SAME
accepted critic-prepared branch. Implementing an early-risk reward variant, changing
exploration or starting the next PPO run is a separate step; none is automatic here.

## Implemented verification

Five new tests cover stratified disjoint episode holdout, missing-family rejection,
value fitting with actor/variance/optimizer preservation, value-only moment reset,
checkpoint reload, explicit gate rejection, and one real full-G1 stochastic collection
episode. A full save/load/RNG test uses synthetic fitting data in a temporary fixture and
verifies all parent counters and actor states. It is not research training evidence.
The full 72-episode preparation has not been run during implementation.

Full regression verification: `research/escort_rl/tests` and
`research/escort_scenarios/tests` passed **131 tests**, with no failures or skips
(14 third-party deprecation warnings; 96.68 seconds).
