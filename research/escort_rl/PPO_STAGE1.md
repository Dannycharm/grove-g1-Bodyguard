# Stage 1 PPO training and validation

The actor-capacity and lower-exploration tests, and the opt-in v3 tracking/hold objective,
are documented in [CAPACITY_EXPLORATION_2026-11-06.md](CAPACITY_EXPLORATION_2026-11-06.md).
Use its fresh-run command for that experiment. The imitation archive is diagnostic,
not a PPO resume checkpoint. Old-source archives remain protected by strict provenance;
no migration or hash bypass was added for the objective change.

**Restart correction (November 6):** the original resume implementation restored RNG
values but failed to cancel SB3's queued vector seeds. Historical resume instructions
below describe the intended contract; they are superseded by the tested protocol and
fresh A/B commands in [PPO_RESUME_REWARD_AUDIT_2026-11-06.md](PPO_RESUME_REWARD_AUDIT_2026-11-06.md).
Old archives intentionally fail the new protocol/source compatibility checks. Do not edit
their manifests or bypass those checks. `--chunks 2` now prepares a matched fresh 20k
diagnostic experiment; no follow-up training was executed during this audit.

This adds SB3 PPO around the existing `G1EscortEnv`; it does not change the environment,
reward, scenario generator, runner, frozen gait or low-level controllers. All operation is
isolated, headless and offline. Full numerical dynamics parity with live Grove remains
unverified. Stage 1 uses a noncolliding, nonresponsive client pose proxy on flat ground.

## Dependencies and model

User-approved venv additions: CPU `torch==2.7.1+cpu`, `stable-baselines3==2.7.0`,
`tensorboard==2.20.0`. Existing `numpy==2.2.6`, `mujoco==3.3.6`,
`onnxruntime==1.20.1`, `gymnasium==1.2.2` remain fixed. Exact transitive versions are
recorded in each experiment and the milestone `dependencies.txt`. No CUDA packages,
ROS processes, Docker operations or system installations are needed.

The approved installation commands were (already executed; no reinstall is needed):

```bash
research/escort_rl/.venv/bin/python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cpu
research/escort_rl/.venv/bin/python -m pip install stable-baselines3==2.7.0 \
  tensorboard==2.20.0 numpy==2.2.6 mujoco==3.3.6 onnxruntime==1.20.1 gymnasium==1.2.2
```

`configs/ppo_stage1.yaml` contains experimental defaults: CPU MlpPolicy, separate actor
and critic MLPs [128,128] with SB3 default Tanh activations; learning rate 3e-4,
gamma .99, GAE .95, clip .2, 1024 rollout steps per environment, batch 256,
10 epochs, entropy .001, value coefficient .5, max gradient norm .5. Both heads receive
exactly the same existing 40-float causal vector. No VecNormalize, extra running observation
normalizer, family label, scenario route or evaluator field is passed to either head.
The policy's Gaussian actions are clipped by SB3 to the existing normalized Box; raw
Gaussian action clipping and environment saturation are logged separately. No new command
filter is introduced. Physics/gait rates and current reward coefficients are unchanged.

## Split contract

`configs/stage1_splits.json` defines seed namespaces modulo four: training 0, validation 1,
reserved final test 2. Scenario sampling remains the original 50/25/25 family mixture.
Training rejects the final-test-only conjunction of peak speed >=.235 m/s AND route
length >=4.6 m. Validation also excludes this conjunction and uses twenty fixed unseen
instances of each family, resolved and logged with their full metadata. Numeric seed
separation is supplemented by distinct resolved trajectory signatures and a held-out
parameter combination. Final-test cases are not resolved or run in this milestone.

PPO initialization/action sampling and each worker's scenario RNG use independent
SeedSequence children. Evaluation uses its fixed split and independent environments;
its deterministic actions do not consume training scenario streams. Per-worker episode
metadata is evaluator-only JSONL. New scenarios are sampled on every automatic reset;
no ROS bundle or filesystem mailbox participates in rollout control.

## Training/checkpoint contract

One worker uses DummyVecEnv; two use separate SubprocVecEnv processes with `spawn`.
Each worker owns model/data, ORT histories, controllers, scenario RNG and episode timers.
ORT sessions enforce one intra/inter-op thread; PyTorch uses one CPU thread.

The CLI requires a fresh output directory and accepts at most 10,000 **new** transitions
in this milestone. SB3 normally rounds budgets up to whole rollouts. Here final rollout
length is shortened only when necessary, so optimization completes without exceeding
the authorization. For 10,000 transitions and two workers, four full 2048-transition
rollouts are followed by one 1808-transition rollout (904 per worker). The final PPO
minibatch can be shorter than 256; hyperparameters and algorithm are otherwise unchanged.
`optimizer_updates` / SB3 `_n_updates` counts optimization epochs; separately counted
`gradient_steps_total` counts actual Adam step calls.

Recovery checkpoints occur every 2048 collected transitions, with initial and final
archives. Validation runs before and after the limited pilot; intermediate recovery
snapshots are not mislabeled evaluated checkpoints. Best-validation selection prioritizes
success, then absence of critical failures, lower formation error, then return. Policy and
optimizer are saved by SB3; companion files contain Python/NumPy/Torch RNG state,
worker scenario RNG/counters and immutable config/source/dependency provenance.

`--resume` requires existing checkpoint, RNG and manifest files, the same workers/root
seed, and matching configuration/source/dependencies. Missing or mismatching evidence
raises an error, never initializes a fresh model. Resume restores actor/critic, Adam,
step/epoch counts and RNG streams, but **does not restore partial rollout buffers, the
active physical episode, or within-episode timing**. New episodes begin from saved
next-scenario RNG states. Bit-identical continuation across interruption is not claimed.
Ctrl-C writes a recoverable `interrupted_<step>` checkpoint. Existing outputs are not reused.

Stage promotion requires >=90% overall success, >=80% each family on twenty episodes,
no falls/emergency/physics failures, mean formation error <=.75m and mean accompanied
progress fraction >=.6, on two consecutive full validation rounds. No promotion occurs
automatically. Final-test scenarios and Stages 2–4 are outside this milestone.

## Commands

From repository root:

```bash
export ESCORT_PY="$PWD/research/escort_rl/.venv/bin/python"
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios"

# New output path each time. This measures vectorized full-G1 collection, not PPO updates.
"$ESCORT_PY" research/escort_rl/tools/train_stage1_ppo.py \
  --benchmark --output research/escort_rl/runs/vector_benchmark_manual_001

# Explicitly starts a limited isolated training process and initial/final validation.
"$ESCORT_PY" research/escort_rl/tools/train_stage1_ppo.py \
  --config research/escort_rl/configs/ppo_stage1.yaml --seed 42 --workers 2 \
  --timesteps 10000 --output research/escort_rl/runs/ppo_pilot_manual_001

# Evaluate in a fresh process on the same fixed 60 validation episodes.
"$ESCORT_PY" research/escort_rl/tools/evaluate_stage1_ppo.py \
  --checkpoint research/escort_rl/runs/ppo_pilot_manual_001/final.zip \
  --output research/escort_rl/runs/ppo_validation_manual_001
"$ESCORT_PY" research/escort_rl/tools/evaluate_stage1_ppo.py \
  --policy follower --output research/escort_rl/runs/follower_validation_manual_001
"$ESCORT_PY" research/escort_rl/tools/evaluate_stage1_ppo.py \
  --policy zero --output research/escort_rl/runs/zero_validation_manual_001

"$ESCORT_PY" research/escort_rl/tools/plot_stage1_training.py \
  --run research/escort_rl/runs/ppo_pilot_manual_001 \
  --output research/escort_rl/runs/ppo_pilot_manual_001/plots
"$ESCORT_PY" -m tensorboard.main \
  --logdir research/escort_rl/runs/ppo_pilot_manual_001/logs --host 127.0.0.1

# Example resume into a fresh directory (not executed automatically).
"$ESCORT_PY" research/escort_rl/tools/train_stage1_ppo.py \
  --seed 42 --workers 2 --timesteps 10000 \
  --resume research/escort_rl/runs/ppo_pilot_manual_001/final.zip \
  --output research/escort_rl/runs/ppo_resume_manual_001

"$ESCORT_PY" -m pytest research/escort_rl/tests \
  research/escort_scenarios/tests/test_generator.py \
  research/escort_scenarios/tests/test_mujoco_adapter.py -q
```

The resume example is another training experiment and requires authorization beyond this
milestone's completed pilot. A 100k budget is intentionally not launchable through this
milestone's capped entry point; authorize a new budget before lifting that cap.

Training steps/episode diagnostics, PPO CSV/TensorBoard records, per-worker scenario
metadata, policy archives, provenance and separate validation actor/evaluator traces
are recorded. Episode plots reuse the existing verifier's XY, formation, velocity and
reward-component plots. Evaluation and plotting time are reported separately from
training throughput. Logs contain privileged evaluator metrics; actor input files
contain only the 40-vector and requested action. Future training code must preserve this
boundary when choosing features or normalization.
