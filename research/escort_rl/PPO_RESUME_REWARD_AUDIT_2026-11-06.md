# PPO restart correction and Stage 1 reward audit — November 6, 2026

**Implemented and verified: restart protocol v2 and an opt-in formation-progress reward
candidate. No follow-up 20k/30k/100k experiment was launched.** Automated optimizer smoke
tests ran; checkpoint evaluations contained no training updates. All old archives and
experiments remain unchanged. Retain AGILE/full-G1 physics; neither a gait replacement
nor Stage 2 promotion is justified.

Evidence root: `runs/resume_reward_audit_20261106/`. All numbers below are measured from
those artifacts or explicitly labeled evaluator counterfactuals. No new dependencies,
ROS/DDS/live simulator, Docker action, commit or push was used.

## 1. Exact RNG failure and restart protocol

SB3 2.7.0 `PPO.load()` invokes `_setup_model()` -> `set_random_seed()` -> `VecEnv.seed()`.
That queues per-worker seeds for the next reset. The old code restored worker NumPy RNG
states, but left those queued seeds intact. `learn()` then reset the vector environment;
`TrainingScenarios.reset(seed=...)` replaced the restored generators. The fresh-model
branch cleared pending seeds; the resume branch did not. No callback performed the
problematic reseed: the first post-load vector reset did.

Old logs: **471 resets / 65 unique seeds** in the added 90k. The first intended versus
observed seeds at the initial resume were:

| Worker | Saved-stream continuation | Observed queued model-seed continuation |
|---|---:|---:|
| 0 | 639056782823807736 | 4374458801877455528 |
| 1 | 3189754318648236116 | 3439179871814419720 |

The same observed worker prefixes recur in all nine chunks. Policy, Adam state and step
counts continued; scenario stream continuation did not. This is a confirmed defect, but
not proof that it independently explains the entire policy collapse.

The new `sb3_restart.py` protocol `sb3_2_7_0_scenario_restart_v2`:

1. Complete model construction/loading first, with `force_reset=True` on load.
2. Validate sidecar timestep, worker count and every worker RNG state before restoring.
3. Cancel queued vector seeds **and options**.
4. Restore each worker's own generator and episode counter; verify read-back equality.
5. Restore Torch, global NumPy and Python RNG states after model setup.
6. Let the first `learn()` perform an **unseeded** reset into a new physical episode.
   Checkpoint saving and initial evaluation do not reset the training workers.

SB3 has no public seed-cancellation operation: `seed(None)` schedules random seeds.
The private `_reset_seeds()` / `_reset_options()` handling is isolated in one adapter,
requires exact installed SB3 **2.7.0**, checks the resulting queues, and fails on unsupported
versions. No private writes are scattered through production training code. Once restored,
a worker rejects any later explicit reseeding; pending nonempty options are also rejected
by the training wrapper. Failures abort rather than silently initialize fresh streams.
No time-derived scenario seed is used.

Partial physical episodes, partial rollout buffers and within-episode phases are still
not restored. A saved RNG is the **next-scenario** stream, not the already-active scenario.
A restart discards the active physical episode and begins the next one. Tests do not
claim chunked optimization trajectories are bit-identical to uninterrupted training.

## 2. Regression and physics evidence

Final complete suite: **92 passed, zero failed/skipped**, 67.32 s. This includes all
45 original environment/gait tests, eight prior PPO tests, 28 generator/adapter tests,
and eleven new restart/reward tests. A focused restart rerun after strengthening global
RNG assertions passed **six tests** in 26.90 s; fourteen third-party Matplotlib/PyParsing
deprecation warnings were non-failing. No installation/upgrade was performed.

New tests use real DummyVecEnv and two-worker SubprocVecEnv with **spawn**. After multiple
completed physical episodes and real Adam updates, they save/load three consecutive
checkpoints and compare subsequent scenarios against independently advanced saved RNGs:

| Workers | Consecutive resumes | Sampled / unique seeds | Unique signatures | Family counts |
|---|---:|---:|---:|---|
| 1 | 3 | 12 / 12 | 12 | straight 4, turns 5, stops 3 |
| 2 | 3 | 24 / 24 | 24 | straight 11, turns 8, stops 5 |

All expected sequences matched, no startup prefixes replayed, workers stayed independent,
and route lengths varied. Every sampled training seed remained in its 0-modulo-4 namespace;
validation/final namespaces remained isolated. Fixed split files were not changed.
A separate **sampler-only** continuation stress audit (not 2,000 dynamics episodes) found
2,000 unique seeds/signatures, no inter-worker collisions, and the following distribution:

- Worker 0: 470 straight / 262 turns / 268 stops; lengths 3.001–4.999 m, peak limits .150–.250 m/s.
- Worker 1: 479 straight / 274 turns / 247 stops; lengths 3.002–4.999 m, peak limits .150–.250 m/s.

Tests also compare exact policy tensors, all Adam state tensors, timestep counts,
Torch/NumPy/Python RNG restoration, the first vector reset, pending option cancellation,
malformed state rejection, reseeding rejection, legacy-provenance rejection, and matching
initial policy weights/observations/scenario streams between A/B reward variants.
The 20k chunk CLI was tested using a non-training stub; the experiments were not executed.

## 3. Refreshed checkpoint evaluation and observed collapse

Re-ran the identical **60-episode fixed validation suite** without training for each
checkpoint (240 actual full-G1 episodes). Success counts match the saved results:

| Checkpoint | Straight | Gentle turn | Stop/start | Overall | Terminal signed longitudinal error mean | Terminal signed lateral error mean |
|---|---:|---:|---:|---:|---:|---:|
| 10k | 0/20 | 0/20 | 0/20 | 0/60 | -1.376 m | +1.433 m |
| 20k | 9/20 | 6/20 | 6/20 | 21/60 | +1.277 m | -.542 m |
| 70k | 0/20 | 2/20 | 0/20 | 2/60 | +1.916 m | +1.460 m |
| 100k | 0/20 | 0/20 | 0/20 | 0/60 | +2.792 m | +.075 m |

Signed error is relative to the desired **-1.5 m** client-frame offset; positive does not
necessarily mean ahead of the client itself. At 20k, 55/60 were ahead of the preferred
point, and 24/60 ahead of the client. At 70k the counts were 60/60 and 43/60. At 100k both
were 60/60, with mean robot position **1.292 m ahead of the client** and terminal error
norm 2.797 m. Overshoot becomes prevalent between the sampled **10k and 20k** checkpoints;
all checkpoints from 30k onward ended ahead of the desired point on all 60 cases.
The exact optimizer update at which it begins is not resolved by 10k-spaced evaluation.

100k failures: **57 recovery timeout, three bounds failures**. None of the four refreshed
suites reported a fall, supervisor failure or physics failure. This does not erase the
15 supervisor failures in the old stochastic 90k training logs.

After client arrival, pooled mean actor-applied command at 100k was
**(.193, .120, .0058)** m/s, m/s, rad/s; measured body velocity stayed
**(.137, .062, .00072)**. Thus continued commands are directly observed. There were no
natural samples with total command norm <.03 in these four policy suites, so their
trajectories alone cannot establish gait behavior under a zero command.

### Separate stop intervention

Three isolated diagnostic replays used the 100k actor until client arrival, then explicitly
sent zero through the unchanged high-level environment channel. The endpoint query is
**evaluator-only**; this intervention is not an escort policy or success claim.

| Family, seed 1000001 | Stopping time | Sampled path length | Net displacement |
|---|---:|---:|---:|
| Straight | .6 s | .0563 m | .0309 m |
| Gentle turn | .6 s | .0500 m | .0483 m |
| Stop/start | .8 s | .0395 m | .0227 m |

Criterion: planar speed <.03 m/s for five consecutive 5Hz samples. Time resolution is .2 s;
path lengths are sampled, not native-substep integrals. Applied zero was observed at the
first .2 s sample; no substep command-latency claim is made. All probes still timed out
because zero alone does not restore the already-lost formation.

**Primary observed cause of sustained post-stop motion: the actor keeps requesting it.**
Finite stopping dynamics exist, but these probes do not support a gait incapable of stopping.
Three cases are not a complete stop-envelope validation or live-Grove parity result.

### PPO distribution and optimizer diagnostics

State-pooled raw normalized Gaussian action means and standard deviations, evaluated
on each checkpoint's saved causal observations:

| Checkpoint | Mean `(vx,vy,yaw)` before Box clipping | Standard deviations |
|---|---|---|
| 10k | (.223,-.180,.057) | (.995,.993,1.025) |
| 20k | (.052,-.172,.034) | (.958,.979,1.018) |
| 70k | (.316,-.153,.010) | (.941,.948,1.000) |
| 100k | (.334,.188,-.003) | (.936,.953,.994) |

These are normalized policy units, not achieved m/s. Exploration stays broad; in the
old added 90k, 69.68% of sampled three-axis vectors crossed a Box limit on at least one
axis. SB3 clips them before the environment, explaining its zero *additional* clipping rate.
This differs from PPO objective clipping. At 100k: explained variance .267, value loss
151.03, entropy loss -4.1364, approximate KL .00543, PPO clip fraction .04082. Improved
critic return prediction is not escort success. The data do not independently prove
optimizer divergence, excessive learning rate, or a single causal explanation for collapse.

Figures/data: `collapse_analysis/checkpoint_stop_comparison.png`, per-episode actor/evaluator
traces, `collapse_analysis/summary.json`, `optimization_100k.png` and
`optimization_time_series.json`. The latter reads actual TensorBoard steps across all
chunks; logged optimization metrics can lag a rollout, with explicit final dumps.

## 4. Original reward equations and counterfactual audit

Let `s` be robot route projection, `c` client route progress, `H` the raw robot highwater,
`e = R(client_heading)^T(robot_xy-client_xy)-[-1.5,0]`, and `u` the applied normalized action.
The original eligibility indicator requires: norm(e)<=2 m, client distance<=3 m, valid
client, and longitudinal robot position **behind or at the client**. Those constraints
already exist and were not assumed absent.

Original per-transition equations (dt=.2 s):

- Formation: `(exp(-(norm(e)/.75)^2)-1)*dt`, in [-.2,0].
- Progress: `4 * eligible * max(0, min(s,c)-H_previous)`.
- Raw highwater: `H=max(H_previous,s)` even when ineligible, consuming racing progress.
- Command change: `-.02 * sum((u-u_previous)^2)`.
- Torso stability: `-.02 * max(0,tilt)*dt`.
- Terminal success: +20. Terminal failure: -100. Timeout truncation has no failure bonus/penalty.

Success still requires arrival, <=.75 m final formation error, actual robot advance and
accompanied credit each >=.6*route length. Client distance >3 m for 3 continuous seconds
terminates; endpoint formation recovery has 5 seconds. Stationary completion, racing ahead
then waiting, and repeated-location credit are already guarded against.

The problem demonstrated is narrower: while behind the client but **ahead of the desired
following point**, closing the gap can still earn progress credit even when the client
makes no new progress. It can increase total credited advance by up to 1.5 m (six reward
units) on a straight route. No repeated-cycle reward farming was demonstrated.

Synthetic evaluator-only transitions use the existing reward implementation with a
matched 25 s / 125-transition horizon. After termination, remaining samples are absorbing
zero reward. Body tilt is zero, and action changes are explicitly logged from the synthetic
motion. These are **not simulated joint/body motion or learned-policy evidence**.
Every transition and discounted/undiscounted return is exported in `reward_comparison_final/`.

During a realistic 3 s stationary interval:

| Behavior | Original total | Candidate total |
|---|---:|---:|
| Hold correct formation | 0 | 0 |
| Move .6 m toward stationary client from correct formation | **+1.8160** | **-.5840** |
| Begin legitimate recovery from .5 m too far behind | -.2104 | -.2104 |

The original moving-toward-client case received +2.4 progress, -.58176 formation and
-.00222 command-change reward. This is a genuine *local* incentive conflict. However,
over the full 25 s, correct following still strongly beats sustained overshoot. We do
not claim the global full-episode objective prefers failure or that reward alone caused collapse.

Legitimate stationary recovery over 25 s earns +.73224 in both versions (+2 progress,
-1.26748 formation, -.00028 command change). Remaining .5 m behind for that entire interval
would cost about -8.97, so recovery remains preferable even though its first 3 s return
is negative. Previously credited backtracking earns formation improvement, not a second
progress payment.

The full component table follows. F, C, T, S and P denote formation, command change, torso,
terminal success and terminal failure. These components are identical between versions
in these cases; only progress changes. Tiny values are rounded in this table, full
precision is retained in JSON.

| Case | F | Progress v1 | Progress v2 | C | T | S | P | Total v1 | Total v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| walking_correct | 0.00000 | 16.00000 | 16.00000 | -0.00222 | 0.00000 | 20.00000 | 0.00000 | 35.99778 | 35.99778 |
| walking_falls_behind | -11.45757 | 6.00000 | 6.00000 | -0.00056 | 0.00000 | 0.00000 | -100.00000 | -105.45812 | -105.45812 |
| walking_closes_gap | -20.41308 | 22.00000 | 16.00000 | -0.01028 | 0.00000 | 0.00000 | -100.00000 | -98.42335 | -104.42335 |
| stopped_correct | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| stopped_recovery | -1.26748 | 2.00000 | 2.00000 | -0.00028 | 0.00000 | 0.00000 | 0.00000 | 0.73224 | 0.73224 |
| stopped_crosses_front | -21.60467 | 5.92000 | 0.00000 | -0.00444 | 0.00000 | 0.00000 | 0.00000 | -15.68912 | -21.60912 |
| resumes_recovers | -1.26748 | 18.00000 | 18.00000 | -0.00250 | 0.00000 | 20.00000 | 0.00000 | 36.73002 | 36.73002 |
| robot_stationary | -7.17690 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | -100.00000 | -107.17690 | -107.17690 |
| races_ahead_waits | -15.84581 | 7.20000 | 1.44000 | -0.04000 | 0.00000 | 0.00000 | -100.00000 | -108.68581 | -114.44581 |
| oscillates | -1.34328 | 1.00000 | 0.00000 | -0.05986 | 0.00000 | 0.00000 | 0.00000 | -0.40314 | -1.40314 |

The table confirms the existing safeguards: stationary robot gets no progress and fails;
race-ahead/wait fails the accompanied-credit completion guard; oscillations receive only
one old highwater payment, not repeated traversal rewards. The candidate removes that
one-off target overshoot credit as well. When client movement resumes and the robot
recovers legitimately, both versions give the same reward in the audited case.

### Credit assignment

At gamma .99 and 5Hz, the approximate geometric horizon is 100 transitions / 20 s.
In the 100k logs, the mean delay from first longitudinal error >.75 m to termination was
**32.3 s** (discount .197 over that mean delay). Mean terminal discount from episode start
was **.1194**. These are evidence of delayed consequences, not proof that -100 is too small.
Formation cost also saturates toward -1 per simulated second; differences between large
errors become small. Changing terminal weights blindly is not justified by this audit.

The synthetic traces set torso tilt to zero; their torso component is therefore zero.
Real checkpoint traces log its nonzero values separately. In the original 100k final
validation, average terminal failure was -100 and mean progress reward about +19.08,
formation cost -31.21, command-change cost -.00670, torso cost -.000072 per episode.
Nominal reward magnitude alone does not explain which early actions PPO assigns credit to.

## 5. Opt-in candidate: formation_progress_v2

Original `rewards.py` is **byte-for-byte unchanged**. `reward_variants.py` provides a
reward-only environment adapter; original `env.py` is unchanged as well. The candidate
changes progress accounting only, with unchanged weights, outcomes, caps and observation.

Let `g_t` be the route projection of the **current desired world escort point**:
`project(client_xy + R(client_heading)*desired_offset)`. This uses evaluator route
geometry only, never a future time query or actor feature. On arcs it projects the actual
offset point rather than assuming a straight-line -1.5 m subtraction.

Define bounded progress `b_t=min(s_t,g_t)`, bounded highwater `B`, and original raw highwater `H`:

```
Δ_t = min(max(0, b_t - B_previous), max(0, s_t - H_previous))
B_t = max(B_previous, b_t)
H_t = max(H_previous, s_t)
r_progress_v2 = 4 * original_eligibility * Δ_t
```

Both ledgers start at initial actual robot progress and advance even for uncredited
motion. Client movement alone while the robot waits cannot pay, because raw robot advance
must also occur. Closing beyond a stationary desired following point cannot pay. Genuine
recovery from behind can pay up to that point. Repeated locations cannot pay again.
A small .1 m ahead formation offset during valid moving accompaniment still earns progress;
it does not require exact-point tracking.

Implementation reuses the unchanged base evaluator for eligibility, weights, highwater,
completion and failures. An algebraically equivalent progress budget passes the approved
increment to that evaluator; diagnostics restore the true client progress and separately
log `desired_formation_progress_m` and `reward_version`.

The first unreleased draft used only the desired-point cap against the raw highwater.
It incorrectly suppressed continuous progress for slightly-ahead followers. **17 completed
follower episodes timed out**, and that draft run was interrupted. The dual-ledger version
corrects this; those artifacts are preserved and explicitly excluded in `EXCLUDED_DRAFTS.json`.
No earlier dataset was overwritten. Do not use `candidate_follower/`, `candidate_zero/`
or `reward_comparison/` as released-v2 evidence.

Final native-physics verification:

- Candidate diagnostic follower: **60/60 successes**, 20/20 each family, no failures.
- Candidate zero agent: **0/60 successes**, 60 client losses.
- Against the saved original-reward follower, **11,592 transitions** had exactly identical
  robot positions, body velocities and applied commands, with identical episode lengths.
- A direct paired-environment test also checked identical qpos/qvel, observations and
  step counters under identical actions. Gymnasium's checker passed on the candidate.

Thus the candidate is a justified **B-arm hypothesis test**, not a proven fix for learned
policy collapse. No candidate PPO policy has been trained or shown superior.

## 6. Checkpoint integrity and recommended start

Read-only inspection of actual 10k/20k/70k/100k archives verified timesteps
10,000/20,000/70,000/100,000, 13 Adam parameter states each, and optimization epochs
50/100/350/500. SHA-256 values and paths are in `checkpoint_integrity.json`.

The initial 10k pilot predates the first faulty resume. Its 98 sampled seeds were all
unique and its fresh-model branch protected the independent worker streams. It is the
**safest legacy warm-start candidate** if a future audited migration is required.
The first 10k continuation (20k checkpoint) generated 65 unique seeds, with **zero overlap
with the initial pilot**. It was affected by the incorrect reseed, but had not yet replayed
previous pilot seeds. Repetition of its startup prefixes is visible from the next chunk
(30k) onward. The 20k model is useful as a diagnostic/salvage comparator, not definitive
corrected-stream evidence.

All legacy archives lack the new restart protocol and have old source/config hashes.
They are intentionally rejected by the fixed training CLI. **No hash bypass or legacy
migration path was implemented.** Reading them for evaluation is allowed; training resume
with changed sources/rewards is a different operation. Old manifests/archives were not edited.
A migration is unnecessary for the clean comparison and would add avoidable uncertainty.

**Recommendation C: fresh-from-scratch A/B training** with the same initialization seed,
architecture, PPO defaults, environment/scenario contract, held-out cases and chunk schedule.
A direct test verified matching initial policy weights, observations and scenario states
between A and B. Do not resume the 100k collapsed model for this comparison.

## 7. Files changed

| File | Change |
|---|---|
| `escort_rl/sb3_restart.py` (new) | Exact-version reset queue compatibility, state validation and ordered restoration |
| `escort_rl/ppo_training.py` | Uses restart protocol and worker reseeding guard; records protocol/reward version; creates reward variants |
| `escort_rl/reward_variants.py` (new) | Opt-in dual-ledger formation-progress evaluator/environment adapter |
| `escort_rl/reward_audit.py` (new) | Ten synthetic matched-horizon reward cases and full per-transition exports |
| `escort_rl/ppo_evaluation.py` | Optional reward version, original remains default, summary labels version |
| `tools/train_stage1_ppo.py` | `--chunks 2/3` for fresh 20k/30k aggregate diagnostics; no 100k automatic budget |
| `tools/evaluate_stage1_ppo.py` | Uses configured reward version |
| `tools/audit_stage1_rewards.py` (new) | Reproducible original/candidate counterfactual CLI |
| `tools/analyze_policy_collapse.py` (new) | Checkpoint/trace metrics, causal policy-array inspection, plots and optional isolated zero-command probes |
| `configs/ppo_stage1_fixed_legacy_v1.yaml` (new) | Experiment A: unchanged reward and PPO hyperparameters |
| `configs/ppo_stage1_fixed_formation_progress_v2.yaml` (new) | Experiment B: same configuration plus opt-in reward version |
| `tests/test_resume_rng.py` (new) | Sequence-level multi-restart, state/provenance/failure and chunk-budget tests |
| `tests/test_reward_variants.py` (new) | Incentive, recovery, anti-farming, full-dynamics equality and shared initialization tests |
| `README.md`, `PPO_STAGE1.md`, this report | Current restart/candidate status and commands |

`unchanged_contracts.json` verifies **16 protected research source/config files** unchanged,
including original env, rewards, gait, controller, runner, observations, generator, Stage 1
configuration, split and old PPO configuration. `load_contract()` verified original gait,
controller and asset hashes, including AGILE SHA-256
`8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b`.
Existing user modifications were preserved. No system dependencies were installed.

## 8. Prepared controlled experiment commands — NOT executed

Use the host venv. These launch isolated full-G1 training only when the operator runs them.
Each candidate starts fresh, runs **two 10k chunks = 20k aggregate**, and performs the
same fixed validation. Use `--chunks 3` for an explicitly chosen 30k aggregate instead.
Both candidates keep all PPO hyperparameters, including initial action noise, unchanged.
Neither is permitted to auto-launch a 100k budget. Outputs must be new directories.

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export ESCORT_PY="$PWD/research/escort_rl/.venv/bin/python"
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios"
ESCORT_AUDIT_TAG=$(date +%Y%m%d_%H%M%S)
```

**A — corrected restart, original reward:**

```bash
"$ESCORT_PY" research/escort_rl/tools/train_stage1_ppo.py \
  --config research/escort_rl/configs/ppo_stage1_fixed_legacy_v1.yaml \
  --seed 42 --workers 2 --timesteps 10000 --chunks 2 \
  --output "research/escort_rl/runs/rng_fixed_A_${ESCORT_AUDIT_TAG}"
```

**B — corrected restart, candidate reward:**

```bash
"$ESCORT_PY" research/escort_rl/tools/train_stage1_ppo.py \
  --config research/escort_rl/configs/ppo_stage1_fixed_formation_progress_v2.yaml \
  --seed 42 --workers 2 --timesteps 10000 --chunks 2 \
  --output "research/escort_rl/runs/rng_fixed_B_${ESCORT_AUDIT_TAG}"
```

Each chunk records initial/final validation traces/plots, step/episode metrics, per-worker
metadata, TensorBoard, optimizer and policy archives, RNG sidecars and strict manifests.
The parent has `comparison_result.json`. Checkpoint comparison should be global across
both chunks, not just the last chunk's `best_validation` label. Re-run the existing plot
CLI on each chunk if additional training-curve PNGs are desired. Do not alter source or
configuration between chunks; strict provenance intentionally rejects that.

This schedule restarts active physical episodes and shortens the final rollout of each
chunk to remain exactly on budget. It is matched between A/B, not identical to one
uninterrupted 20k optimization trajectory. After policy/reward divergence, different
termination timing can consume different numbers of scenarios; initial seed streams,
not every realized transition, are matched. A single training seed is diagnostic evidence,
not a confidence interval or final generalization claim.

Reproduce the audit/tests with fresh output paths:

```bash
"$ESCORT_PY" -m pytest research/escort_rl/tests \
  research/escort_scenarios/tests/test_generator.py \
  research/escort_scenarios/tests/test_mujoco_adapter.py -q
"$ESCORT_PY" research/escort_rl/tools/audit_stage1_rewards.py --candidate \
  --output research/escort_rl/runs/reward_audit_manual_001
```

Measured earlier collection/optimization rates imply roughly 5–6 minutes per 20k candidate
**before validation/plotting overhead**. This is an estimate from prior 60–66 transitions/s,
not a new measured candidate-training rate. Repeated 60-episode validation adds substantial
wall time. No new benchmark or large training was invented.

## 9. Acceptance before another full 100k run

Implemented gates already passed: exact RNG continuation across three resumes for one/two
workers; no replay or namespace leakage; finite data; preserved physics/observation/gait;
original-reward regression tests; candidate recovery/anti-racing fixtures; follower 60/60
and zero 0/60. Pending learning evidence:

- Inspect A/B 10k and 20k (optionally 30k) curves, actual scenario/signature diversity and
  per-worker mix; do not rely only on reward or printed stochastic training success.
- Require repeatable improvement in held-out completion and stopping, with no collapse
  between consecutive validation checkpoints; use the old 35% at 20k as a comparator,
  not a promotion target. No validation falls/emergencies/physics failures; inspect and
  account for all stochastic supervisor failures rather than disabling them.
- Select reward version/checkpoint using success, safety, formation, stopping and
  accompanied progress. Add training seeds before strong scientific claims. Keep final
  tests untouched; additional engineering checks are validation, not final testing.
- Only then authorize a larger budget and its validation/checkpoint schedule. Stage 1
  promotion still requires >=90% overall, >=80% every family and no critical failures
  for two consecutive full suites. Those promotion requirements have **not** passed.

Remaining uncertainty: fixed RNG, reward credit, broad action noise and delayed outcomes
may interact; this milestone does not isolate the causal contribution of each to the old
collapse. Candidate learning superiority is untested. Full numerical parity with the live
asynchronous Grove stack remains unverified. Noncolliding, nonresponsive scripted clients
on flat ground are not perception/navigation-world validation or human safety evidence.
