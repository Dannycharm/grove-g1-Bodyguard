# Useful model weights and checkpoint registry

Last verified: **2026-11-07**. Paths below are relative to the repository root,
`/home/praise/ECE598_Bodyguard/grove-g1`. Links resolve from this document.

This is the editable, authoritative index of useful model artifacts for this project.
Consult it before loading a policy. Update it whenever a useful checkpoint is created,
selected, superseded or rejected. Keep actual weights in their original experiment
folders; this registry records their locations rather than moving or overwriting them.

## Which model to use now

| Intended use | Registry entry | Decision |
|---|---|---|
| Default Stage 1 evaluation and visualization; selected candidate for a separately implemented Stage 2 transition | Selected guarded PPO 5k | Best validated checkpoint for current seed 42. |
| Continue the latest Stage 1 optimizer and scenario streams | Latest guarded PPO 10k | Valid latest checkpoint, not the validation-selected best. |
| Start a fresh imitation-pretrained PPO experiment or preserve an imitation-only comparator | Original imitation actor | Explicit actor initialization; never treat this as a PPO resume. |
| Reproduce preparation before PPO updates | Prepared actor and critic | Diagnostic prepared initial checkpoint, with zero PPO transitions. |
| Execute low-level walking in every experiment | Frozen AGILE ONNX | Mandatory unchanged gait; not an escort policy. |

Stage 1 preservation passed at both 5k and 10k: **60/60 strict validation successes
(20 per family), no recorded critical failures**, on the same fixed validation suite.
There were 48/48 successful completed stochastic training episodes across both chunks.
This is **imitation-pretrained PPO**, not PPO discovering escorting from scratch.
The learned policy used objective C (`tracking_stop_v3`), 40 causal normalized inputs,
three normalized velocity-command outputs, two 128-unit actor/critic layers, learning
rate `1e-5`, `target_kl=0.01`, initial Gaussian sigma `0.05`, and two independent workers.
Full numerical parity with live Grove remains unverified; final-test data is untouched.

The user accepted readiness to begin controlled Stage 2 work with this seed. Stage 2
is the next separately scoped implementation/training milestone, not already validated
and not authorized to launch automatically. Additional seeds are useful replication,
but do not block Stage 2 development. Preserve Stage 1 replay and regression evaluation.

## Active and reusable weights

### Selected guarded PPO 5k

- **Weights:** [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip)
- **SHA-256:** `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`
- **Saved PPO timestep counter:** 5,000.

**Default policy for demonstrations and matched Stage 1 evaluation.** Also the selected
source candidate for a documented Stage 2 transition. Root PPO seed 42; 5,000 PPO
transitions after imitation and critic preparation. Strict validation: 60/60, mean
formation error 0.05631 m, mean terminal error 0.02442 m. It has trained actor/critic,
PPO optimizer state and scenario/global RNG sidecars.

The selector in the subsequent 10k run retained the initial (5k) model because success
and failures tied and its mean formation error was slightly better. A separately saved
archive of the same selected model may have a different ZIP hash; do not infer timestep
count or selection from folder names alone.

Companion artifacts:

- [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.manifest.json](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.manifest.json)
- [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.rng.pt](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.rng.pt)
- [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/validation_final/summary.json](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/validation_final/summary.json)

Selected aliases (retain provenance and sidecars beside each):

- [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/best_validation.zip](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/best_validation.zip) — saved timestep counter **5,000**, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`.
- [research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/best_validation.zip](../../research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/best_validation.zip) — saved timestep counter **5,000**, SHA-256 `cd4410d6b08b78ff7a43f4192ea08ee99be55d64857a711e0060a8867cc7f9c7`.

**Important:** the `continuation_10k/best_validation.zip` alias contains the selected **5k** policy, not the 10k final policy.

### Latest guarded PPO 10k

- **Weights:** [research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/final.zip](../../research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/final.zip)
- **SHA-256:** `882661fdbf24a8d4f9cfb0961737df58bc538e48d9d9c6ad82ae7ec27c75b6f4`
- **Saved PPO timestep counter:** 10,000.

**Use for continuing the latest Stage 1 training state.** Root seed 42; 10,000 cumulative
PPO transitions (5k + a resumed 5k). Strict validation: 60/60, mean formation error
0.05771 m, mean terminal error 0.02097 m. Actual optimizer-step count: 351. The saved
scenario streams continued correctly; 52/52 distinct sampled training seeds across the
two chunks. Both post-PPO rounds passed preservation criteria.

This remains a valid continuation point even though the selector preferred 5k by mean
formation. Loading this checkpoint does not constitute Stage 2 implementation or promotion.

Companion artifacts:

- [research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/final.manifest.json](../../research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/final.manifest.json)
- [research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/final.rng.pt](../../research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/final.rng.pt)
- [research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/validation_final/summary.json](../../research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/validation_final/summary.json)
- [research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/result.json](../../research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/result.json)

### Original imitation actor

- **Weights:** [research/escort_rl/runs/capacity_exploration_20261106/imitation/diagnostic_imitation_actor.zip](../../research/escort_rl/runs/capacity_exploration_20261106/imitation/diagnostic_imitation_actor.zip)
- **SHA-256:** `ef80e8d20a589c73140c99d1c532a856d7c3d0bb4fbe20f7228aec791fdba116`
- **Saved PPO timestep counter:** 0.

**Keep unchanged as the imitation-only baseline and source for explicit actor initialization.**
The actor was fitted to the causal diagnostic follower's labels, including learner-visited
states: 42 collected episodes, 8,096 labelled rows, and 12,000 supervised actor updates.
The zero PPO counter does **not** mean the actor is untrained. Its critic and unit-variance
Gaussian exploration parameters were not trained by imitation; it has no PPO-resume RNG
sidecars. Do not call ordinary training `--resume` on this archive.

It achieved 60/60 strict objective-C validation successes, mean formation error 0.05906 m.
Initialize a fresh PPO model by copying only actor tensors, calibrating variance and
preparing the critic through the existing explicit initialization path. Its teacher is a
diagnostic causal follower, not the final Reactive Baseline A and not a future-route oracle.

Companion artifacts:

- [research/escort_rl/runs/capacity_exploration_20261106/imitation/manifest.json](../../research/escort_rl/runs/capacity_exploration_20261106/imitation/manifest.json)
- [research/escort_rl/runs/capacity_exploration_20261106/imitation/causal_training_data.npz](../../research/escort_rl/runs/capacity_exploration_20261106/imitation/causal_training_data.npz)
- [research/escort_rl/runs/capacity_exploration_20261106/imitation/supervised_fit.json](../../research/escort_rl/runs/capacity_exploration_20261106/imitation/supervised_fit.json)

### Prepared actor and critic before PPO

- **Weights:** [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/initial.zip](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/initial.zip)
- **SHA-256:** `fb1856ba1f093cd294f10349c09916fce455b4605d0791b77b31111474f33085`
- **Saved PPO timestep counter:** 0.

Diagnostic reproducible starting state after importing the imitation actor, setting sigma
0.05, and fitting the critic on 4,467 training-only full-G1 transitions. Zero PPO updates;
PPO optimizer was fresh. Critic holdout MSE 10.20 versus constant baseline 11.71, explained
variance 0.13. The actor remained unchanged and scored 60/60 before fine-tuning.

Use for preparation/restart investigations under compatible provenance. For a new model
seed or changed PPO configuration, use the explicit initialization tool rather than
bypassing strict resume checks on this prepared archive.

Companion artifacts:

- [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/initial.manifest.json](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/initial.manifest.json)
- [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/initial.rng.pt](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/initial.rng.pt)
- [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/critic_fit.json](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/critic_fit.json)
- [research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/critic_training_data.npz](../../research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/critic_training_data.npz)

### Frozen low level AGILE gait

- **Weights:** [workspace/src/g1_controllers/policy/unitree_g1_velocity_e2e.onnx](../../workspace/src/g1_controllers/policy/unitree_g1_velocity_e2e.onnx)
- **SHA-256:** `8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b`

NVIDIA WBC-AGILE `unitree_g1_velocity_e2e.onnx`. This is the mandatory frozen walking
controller, independent of which high-level escort checkpoint is selected. Never modify,
replace, retrain or rescale it as part of high-level policy work. Preserve its joint
mappings, histories, normalization, timing, gains and velocity caps.

Companion artifacts:

- [workspace/src/g1_controllers/policy/unitree_g1_velocity_e2e.yaml](../../workspace/src/g1_controllers/policy/unitree_g1_velocity_e2e.yaml)
- [research/escort_rl/configs/gait_reference.json](../../research/escort_rl/configs/gait_reference.json)

## Historical diagnostic checkpoints

Preserve these archives for comparisons and failure analysis. **They are not selected
forward-use policies.** Reward versions and success definitions differ; do not compare
old success numbers directly with the strict objective-C stop-hold suite. Historical
source/config incompatibility must never be bypassed to make an old archive resumable.

### Original 10k pilot

- **Weights:** [research/escort_rl/runs/ppo_stage1_20261105/pilot/final.zip](../../research/escort_rl/runs/ppo_stage1_20261105/pilot/final.zip)
- **Saved PPO transitions:** 10,000.
- **SHA-256:** `f11c117df050e908d43bc9199dfc96e34e7490e39a6fea0a602c28e1071f78a3`.
- **Purpose:** Initial PPO pilot; historical pre-fix provenance, not the current imitation actor.

### Old 20k checkpoint

- **Weights:** [research/escort_rl/runs/stage1_100k_20261105_184710_ZZ1bmS/round_01/final.zip](../../research/escort_rl/runs/stage1_100k_20261105_184710_ZZ1bmS/round_01/final.zip)
- **Saved PPO transitions:** 20,000.
- **SHA-256:** `9d800b7cd5fa7465908be54354693a11de60fae60867801ca16f60e51ba0634e`.
- **Purpose:** Old 35% validation result under legacy criteria; already includes faulty resumed scenario replay.

### Old collapsed 100k checkpoint

- **Weights:** [research/escort_rl/runs/stage1_100k_20261105_184710_ZZ1bmS/round_09/final.zip](../../research/escort_rl/runs/stage1_100k_20261105_184710_ZZ1bmS/round_09/final.zip)
- **Saved PPO transitions:** 100,000.
- **SHA-256:** `ed0e87ab1529f4a51bd537d075396e7e0a7afab45863d212efa4d0f8f0f7832d`.
- **Purpose:** Confirmed scenario replay contamination and policy collapse; diagnostic only.

### Fixed RNG experiment A at 20k

- **Weights:** [research/escort_rl/runs/rng_fixed_A_20261106_021910/chunk_02/final.zip](../../research/escort_rl/runs/rng_fixed_A_20261106_021910/chunk_02/final.zip)
- **Saved PPO transitions:** 20,000.
- **SHA-256:** `a9e048ffbaff2378ac24a7abb76dc42071d95a3c05552e538c7e46a882779de6`.
- **Purpose:** Original reward after RNG repair; useful comparator, not selected.

### Fixed RNG experiment A at 30k

- **Weights:** [research/escort_rl/runs/paired_30k_20261106_040735_cSnYrg/A/final.zip](../../research/escort_rl/runs/paired_30k_20261106_040735_cSnYrg/A/final.zip)
- **Saved PPO transitions:** 30,000.
- **SHA-256:** `c0061fc713db2a39ecc8da9966bb4cb73d421540591131d484f6f1851de43d16`.
- **Purpose:** Original-reward comparison continuation, not selected.

### Reward experiment B at 30k

- **Weights:** [research/escort_rl/runs/paired_30k_20261106_040735_cSnYrg/B/final.zip](../../research/escort_rl/runs/paired_30k_20261106_040735_cSnYrg/B/final.zip)
- **Saved PPO transitions:** 30,000.
- **SHA-256:** `af0a7e5845a2952355b9d7ef059ba0419e2e8f2851f083aad6a091a228babd03`.
- **Purpose:** formation_progress_v2 comparison; poor task completion.

### Objective C from scratch at 20k

- **Weights:** [research/escort_rl/runs/tracking_stop_v3_20261106_171411/chunk_02/final.zip](../../research/escort_rl/runs/tracking_stop_v3_20261106_171411/chunk_02/final.zip)
- **Saved PPO transitions:** 20,000.
- **SHA-256:** `d225e997597420a8e5456ca92caf6d33291ee6148cbfd4b1f996d53c7c60d057`.
- **Purpose:** Strict objective-C cold-start comparison; 1/60 validation successes.

### Lower noise from scratch at 20k

- **Weights:** [research/escort_rl/runs/capacity_exploration_20261106/low_std_20k/chunk_02/final.zip](../../research/escort_rl/runs/capacity_exploration_20261106/low_std_20k/chunk_02/final.zip)
- **Saved PPO transitions:** 20,000.
- **SHA-256:** `823fe5e7a5d1865baaa4d55dacbba5fa1d3e48c9ddfc6dd11de12617af2f1b34`.
- **Purpose:** Exploration ablation; lowering noise without a useful actor mean did not solve escorting.

### Unguarded imitation PPO at 5k

- **Weights:** [research/escort_rl/runs/imitation_ppo_20261106/preservation_5k/final.zip](../../research/escort_rl/runs/imitation_ppo_20261106/preservation_5k/final.zip)
- **Saved PPO transitions:** 5,000.
- **SHA-256:** `7358a30cf79ce21fe1a2ea91dbe2f6f91b8ae6ad175f988c3bd61314d595f745`.
- **Purpose:** Collapsed from 60/60 to 0/60; excessive optimizer movement. Never use as the Stage 2 source.

## Git preservation status

The Stage 1 Git checkpoint includes the weights linked here, available adjacent manifests
and RNG sidecars, required imitation/preparation data and compact validation evidence.
Bulk traces, GIFs, the venv and unused periodic checkpoints remain local. See
[Git artifact scope](GIT_ARTIFACT_SCOPE.md) for the exact inclusion boundary.

## Loading and preservation rules

1. Check this registry before choosing a model. Resolve repository-relative paths against
   the repository root, not an assumed Docker path.
2. For PPO training resume, keep the matching `NAME.zip`, `NAME.manifest.json`, and
   `NAME.rng.pt` together. The ZIP contains networks and optimizer state; the sidecar
   contains global and per-worker RNG state. Preserve full experiment directories,
   configuration, logs, validation and initialization-data provenance as well.
3. Current guarded Stage 1 training uses
   `research/escort_rl/configs/ppo_stage1_imitation_guarded_v1.yaml` and
   `research/escort_rl/tools/imitation_ppo.py`. Exact resume requires the same source,
   config, dependencies, root seed and worker count. It restores optimizer/RNG but starts
   new physical episodes; active episodes and partial rollout buffers are not restored.
4. A Stage 2 change needs an explicit, tested stage-transition path. **Do not pass a changed
   stage config into the existing strict Stage 1 resume and bypass its rejection.** Preserve
   the selected actor/critic/optimizer lineage and record the transition configuration,
   scenario mixture, RNG handling, new stage and checkpoint parent in this registry.
5. For inference/visualization, use
   `research/escort_rl/visualization/stage1_viewer.py`, which requires the adjacent manifest
   and checks runtime/gait identity. It does not perform training resume.
6. Never overwrite an original checkpoint or move weights to a vague `latest.zip` without
   recording selection provenance. Keep failed controls labelled as such. Verify that
   artifacts are backed up/versioned as intended; a path in this Markdown file does not
   by itself back up binary weights or make ignored run directories part of Git.
7. When a new useful model is created, evaluated, selected, superseded or recommended,
   **update this file in the same milestone**. Verify the path and hash, record actual saved
   timestep count, evidence, parent checkpoint, reward/stage/seed/config, sidecars and its
   intended use (inference, resume, initialization or diagnostic). Preserve old entries.

## Evidence and guides

- [research/escort_rl/PPO_IMITATION_GUARDED_RESULTS_2026-11-06.md](../../research/escort_rl/PPO_IMITATION_GUARDED_RESULTS_2026-11-06.md)
- [research/escort_rl/PPO_IMITATION_GUARDED_10K_RESULTS_2026-11-06.md](../../research/escort_rl/PPO_IMITATION_GUARDED_10K_RESULTS_2026-11-06.md)
- [research/escort_rl/IMITATION_PPO_2026-11-06.md](../../research/escort_rl/IMITATION_PPO_2026-11-06.md)
- [research/escort_rl/PPO_RESUME_REWARD_AUDIT_2026-11-06.md](../../research/escort_rl/PPO_RESUME_REWARD_AUDIT_2026-11-06.md)
- [docs/bodyguard/RL_TRAINING_EXPLAINED.html](../../docs/bodyguard/RL_TRAINING_EXPLAINED.html)
- [docs/bodyguard/STAGE1_POLICY_VISUALIZATION_GUIDE.html](../../docs/bodyguard/STAGE1_POLICY_VISUALIZATION_GUIDE.html)

## Template for the next useful checkpoint

```markdown
### Model name and stage

- Status and intended use:
- Weights path relative to repository root:
- SHA-256:
- Saved cumulative PPO transitions:
- Parent checkpoint and transition or initialization method:
- Reward version and configuration path:
- Model seed and worker count:
- Validation suite and results, including failures:
- Manifest and RNG sidecar paths:
- Evidence report and experiment directory:
- Compatibility limits and date verified:
```
