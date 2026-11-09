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
| Selected Stage 2 evaluation and visualization | Residual PPO 5k and two-round confirmation below | 60/60 pedestrian and 60/60 Stage 1 in both fixed-suite rounds; selected model seed only. |
| Default Stage 1 evaluation and visualization; selected candidate for a separately implemented Stage 2 transition | Selected guarded PPO 5k | Best validated checkpoint for current seed 42. |
| Begin a bounded Stage 2 pilot through the explicit transition | Stage 2 prepared initialization below | Local, untrained in Stage 2; use Stage 2 tools. |
| Continue the latest Stage 1 optimizer and scenario streams | Latest guarded PPO 10k | Valid latest checkpoint, not the validation-selected best. |
| Start a fresh imitation-pretrained PPO experiment or preserve an imitation-only comparator | Original imitation actor | Explicit actor initialization; never treat this as a PPO resume. |
| Reproduce preparation before PPO updates | Prepared actor and critic | Diagnostic prepared initial checkpoint, with zero PPO transitions. |
| Execute low-level walking in every experiment | Frozen AGILE ONNX | Mandatory unchanged gait; not an escort policy. |
| Evaluate experimental frozen-base residual avoidance | Reviewed variation v3 below; retain v2/v1 comparators | Latest v3: 57/60 pedestrian, exact Stage 1 preservation; one intrusion still fails gate. Diagnostic only, no PPO/promotion. |

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

## Stage 2 prepared initialization with zero pedestrian training

Last verified 2026-11-07. **Local artifacts only; not included in the preceding Stage 1 Git push.**

- **Weights:** [Stage 2 prepared initial checkpoint](../../research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip)
- **SHA-256:** `20c995aa3f41f00bf592b9b28706b1ac59bbf6541832f55e91a43d6845a36c7d`
- **Saved total PPO counter:** 5,000, all inherited from the selected Stage 1 5k parent.
- **Stage 2 training transitions:** **0**. This is not a pedestrian-trained or promoted policy.
- **Parent:** the selected guarded Stage 1 5k `final.zip` above, PPO seed 42, two workers.
- **Configuration:** [Stage 2 YAML](../../research/escort_rl/stage2_config/stage2.yaml).
- **Manifest:** [initial.manifest.json](../../research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.manifest.json).
- **RNG sidecar:** [initial.rng.pt](../../research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.rng.pt).
- **Readiness evidence:** [readiness.json](../../research/escort_rl/runs/stage2_readiness_20261107/readiness.json).
- **Implementation and operator commands:** [Stage 2 guide](../../research/escort_rl/STAGE2_SINGLE_PEDESTRIAN.md).

Use this prepared initialization with `stage2_tools/stage2.py train` for the first bounded
Stage 2 pilot, not the Stage 1 training/visualization loader. The transfer preserves the
actor/critic/optimizer and worker/global RNG lineage, while explicitly zeroing only the
new slot's first-layer columns 22:28 and their Adam moments. This keeps the original
Stage 1 outputs/values exact and starts pedestrian behavior without arbitrary unused-input
weights. Counters and sidecars must remain paired. Subsequent Stage 2 checkpoints record
new-stage transitions separately from inherited cumulative PPO timesteps.

Pretraining validation: Stage 1 **60/60**, mean error **0.0563095368 m**, exactly the parent;
pedestrian **20/60**, consisting of 20/20 head-on lane passes, 0/20 crossings, 0/20 diagonals
and 40 geometric proxy intrusions. This is a baseline requiring learning, not a Stage 2
gate pass. Full test suite: 123 passed. No new optimizer updates were run.

The earlier `runs/stage2_readiness_20261107/transition/initial.zip` was used for baseline
evaluation before a checkpoint-bookkeeping correction. Its policy/value/optimizer tensors
and worker/global RNG states were verified identical to the final prepared checkpoint.
Use **transition_ready**, whose manifest matches the final source, for future training.
Do not bypass a provenance rejection on the earlier diagnostic artifact.

## Stage 2 first 5k pilot experimental checkpoint

Verified 2026-11-07. **Local, not promoted or selected as a successful avoidance policy.**

- **Weights:** [Stage 2 first pilot final.zip](../../research/escort_rl/runs/stage2_pilot_YfC5S4/pilot_5k/final.zip).
- **SHA-256:** `f875b9a2095a0f9e18f131994bde397d3d4a2451cc897754fcb8423a3ff7ba07`.
- **Counters:** 10,000 cumulative PPO transitions = 5,000 inherited Stage 1 + **5,000 Stage 2**.
- **Parent:** `stage2_readiness_20261107/transition_ready/initial.zip` above.
- **Manifest:** [final.manifest.json](../../research/escort_rl/runs/stage2_pilot_YfC5S4/pilot_5k/final.manifest.json).
- **RNG state:** [final.rng.pt](../../research/escort_rl/runs/stage2_pilot_YfC5S4/pilot_5k/final.rng.pt).
- **Result:** [result.json](../../research/escort_rl/runs/stage2_pilot_YfC5S4/pilot_5k/result.json).
- **Review:** [Stage 2 pilot review](../../research/escort_rl/STAGE2_PILOT_REVIEW_2026-11-07.md).
- **Validation:** [pedestrian](../../research/escort_rl/runs/stage2_pilot_YfC5S4/pilot_5k/validation_pedestrian/summary.json),
  [Stage 1 regression](../../research/escort_rl/runs/stage2_pilot_YfC5S4/pilot_5k/validation_stage1/summary.json).

Pedestrian success remains **20/60** (crossing 0/20, head-on 20/20, diagonal 0/20), with
40 geometric intrusions. Stage 1 remains **60/60**, mean formation error .05822 m. No
critical failures. The actual new input weights and optimizer updated, and both complete
worker seed sequences match the expected saved RNG continuation (34/34 new unique seeds).
The actor still has little pedestrian sensitivity and the critic poorly predicts encounter
failures. Retain for diagnosis or an explicitly chosen controlled continuation; do not use
as evidence of learned yielding or Stage 2 promotion. Preserve the prepared initialization
and selected Stage 1 model as comparators. No new teacher was used.

## Evaluation only reward audit without new weights

The [matched Stage 2 reward audit](../../research/escort_rl/STAGE2_REWARD_AUDIT_2026-11-07.md)
ran causal follow/yield/sidestep/stop diagnostics, with code under
`research/escort_rl/diagnostics/`. They are handwritten evaluation controllers, not
neural checkpoints or a teacher used for training. No new weights exist to load and the
experimental Stage 2 pilot above remains at 5k new transitions with its original 20/60
score. Audit diagnostic success must not be reported as improved PPO performance.

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

The Stage 1 Git checkpoint includes the Stage 1 and historical weights listed here,
available adjacent manifests and RNG sidecars, required imitation/preparation data and
compact validation evidence. The newly prepared Stage 2 entry is local and has not been
committed or pushed yet.
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

## Stage 2 critic only preparation research/escort_rl/runs/stage2_critic_zMgnVP/preparation

- **Weights:** [research/escort_rl/runs/stage2_critic_zMgnVP/preparation/prepared.zip](../../research/escort_rl/runs/stage2_critic_zMgnVP/preparation/prepared.zip).
- **SHA-256:** `186ed9e142e880e12f0a37accc35f92835a43860b3ec29ae8561c4291540291c`.
- **Parent:** `research/escort_rl/runs/stage2_pilot_YfC5S4/pilot_5k/final.zip`.
- **PPO transitions:** 10,000 cumulative, 5,000 Stage 2; unchanged by preparation.
- **Actor and variance:** bitwise unchanged; actor optimizer and training RNG streams preserved.
- **Critic:** separately fitted to current stochastic actor's training-only returns; value-only Adam state reset.
- **Report:** [fit_report.json](../../research/escort_rl/runs/stage2_critic_zMgnVP/preparation/fit_report.json).
- **Manifest and RNG:** [prepared.manifest.json](../../research/escort_rl/runs/stage2_critic_zMgnVP/preparation/prepared.manifest.json), [prepared.rng.pt](../../research/escort_rl/runs/stage2_critic_zMgnVP/preparation/prepared.rng.pt).
- **Status:** critic holdout gates passed; no claim of improved actor success or Stage 2 promotion. Local artifact until deliberately archived.
- **Reviewed results:** overall holdout MSE 1508.64 → 219.62; head-on MSE regressed 23.04 → 648.37. Recommended only for a bounded 5k diagnostic continuation with both validation suites; see [critic review](../../research/escort_rl/STAGE2_CRITIC_RESULTS_2026-11-07.md). Actor behavior has not improved through this value-only fit.

## Stage 2 after-critic 5k continuation — diagnostic, not promoted

- **Weights:** [final.zip](../../research/escort_rl/runs/stage2_after_critic_3OEFLy/pilot_5k/final.zip).
- **SHA-256:** `c456882468a00a3dedbf19e192375a8eee7b54d02417c6e07918f98f81d7f8ac`.
- **Counters:** 15,000 cumulative PPO transitions; 10,000 Stage 2. Critic preparation separately collected 10,680 transitions.
- **Parent:** accepted critic-only preparation above; 5,000 new PPO transitions, original `stage2_tracking_clearance_v1`, Stage 2 YAML, seed 42/two workers.
- **Validation:** Stage 1 60/60, mean formation 0.06067 m; pedestrian 20/60 (crossing 0/20, head-on 20/20, diagonal 0/20), forty geometric intrusions. No Stage 2 promotion or long continuation recommended.
- **Companions:** matching `final.manifest.json` and `final.rng.pt` in the same directory. Local artifacts, not backed up by this registry entry.
- **Review:** [2026-11-08 analysis](../../research/escort_rl/STAGE2_AFTER_CRITIC_REVIEW_2026-11-08.md). Critic diagnostics improved, avoidance success did not; retain for controlled comparisons.

## Stage 2 teacher-v2 imitation attempt — no checkpoint saved

- Evidence: `research/escort_rl/runs/stage2_teacher_fit_Wn3DvT/bootstrap/`.
- Initialization: selected Stage 1-derived Stage 2 transition archive, 5,000 inherited PPO transitions. Actor-only supervised fitting: 3,000 updates, no PPO learning.
- Results: Stage 1 60/60; pedestrian 20/60 (crossing 0/20, head-on 20/20, diagonal 0/20), forty geometric intrusions. `saved=false`; there is no `prepared.zip` to resume.
- Status: failed distillation diagnostic, not a forward-use model. Original archives unchanged. See [fit analysis](../../research/escort_rl/STAGE2_TEACHER_FIT_REVIEW_2026-11-08.md). Next investigate intervention-balanced imitation and student-state teacher labeling, not long PPO.
- Follow-up tooling is implemented and tested (150 tests): [v3 guide](../../research/escort_rl/STAGE2_DISTILLATION_V3.md). Subsequent v3 diagnostic weights are reviewed below; actor-only candidates are separate from gate-passing prepared PPO archives.

## Reviewed weighted-v3 actor — DAgger diagnostic source only

- Actor: `research/escort_rl/runs/stage2_distill_v3_yY56tc/weighted/candidate_actor.npz`; matching `candidate.json` required. SHA-256 `70b888fef2a64417681a33e0305627753640ae5b3c9a980fe70c5c0567a4a000`.
- Parent: selected Stage 1-derived Stage 2 transition archive; 5,000 inherited PPO transitions, no new PPO; 3,000 supervised updates, selected update 2,700, v3 intervention-balanced imitation, original reward and dynamics.
- Validation: pedestrian 14/60 (0 crossing, 0 diagonal, 14 head-on), 41 geometric intrusions and five recovery timeouts. Stage 1 60/60 but mean formation 0.17805 m fails preservation gate. No prepared PPO ZIP exists.
- Intended use: bounded student-state teacher labeling only; not forward-use PPO or promotion. The `d4k8UE` and `rSmP7K` runs contain identical actor hashes, not independent replicas. See [review](../../research/escort_rl/STAGE2_DISTILLATION_V3_REVIEW_2026-11-08.md).
- First student-only label collection completed at `research/escort_rl/runs/stage2_dagger_2OalKX/data/`: 18 episodes, 2,770 transitions, zero teacher overrides, verified hashes/partitions/scheduling. No new weights. Ready for one combined supervised fit; see [collection review](../../research/escort_rl/STAGE2_DAGGER_COLLECTION_REVIEW_2026-11-08.md).


## Intervention-balanced imitation candidate — research/escort_rl/runs/stage2_distill_v3_rSmP7K/weighted

- Diagnostic actor only: `research/escort_rl/runs/stage2_distill_v3_rSmP7K/weighted/candidate_actor.npz`, SHA-256 `70b888fef2a64417681a33e0305627753640ae5b3c9a980fe70c5c0567a4a000`; metadata `candidate.json`. NOT a PPO resume archive.
- Parent `research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip`; PPO counter 5000 unchanged, actor-only supervised updates.
- Both validation reports and fitting/data provenance are in this directory. Round pass: False; no promotion.

## Intervention-balanced imitation candidate — research/escort_rl/runs/stage2_distill_v3_d4k8UE/weighted

- Diagnostic actor only: `research/escort_rl/runs/stage2_distill_v3_d4k8UE/weighted/candidate_actor.npz`, SHA-256 `70b888fef2a64417681a33e0305627753640ae5b3c9a980fe70c5c0567a4a000`; metadata `candidate.json`. NOT a PPO resume archive.
- Parent `research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip`; PPO counter 5000 unchanged, actor-only supervised updates.
- Both validation reports and fitting/data provenance are in this directory. Round pass: False; no promotion.

## Intervention-balanced imitation candidate — research/escort_rl/runs/stage2_distill_v3_yY56tc/weighted

- Diagnostic actor only: `research/escort_rl/runs/stage2_distill_v3_yY56tc/weighted/candidate_actor.npz`, SHA-256 `70b888fef2a64417681a33e0305627753640ae5b3c9a980fe70c5c0567a4a000`; metadata `candidate.json`. NOT a PPO resume archive.
- Parent `research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip`; PPO counter 5000 unchanged, actor-only supervised updates.
- Both validation reports and fitting/data provenance are in this directory. Round pass: False; no promotion.

## Intervention-balanced imitation candidate — research/escort_rl/runs/stage2_dagger_fit_XRTTA7/aggregated

- Diagnostic actor only: `research/escort_rl/runs/stage2_dagger_fit_XRTTA7/aggregated/candidate_actor.npz`, SHA-256 `a5e625b21a4ba40cccaab03863fe4f29dd59081bbe043431c8a52293d69eb9ff`; metadata `candidate.json`. NOT a PPO resume archive.
- Parent `research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip`; PPO counter 5000 unchanged, actor-only supervised updates.
- Both validation reports and fitting/data provenance are in this directory. Round pass: False; no promotion.
- Reviewed: pedestrian 20/60 (0 crossing, 0 diagonal, 20 head-on), forty geometric intrusions; Stage 1 47/60 with thirteen recovery timeouts due to failed stopping/settling despite qualified endpoint geometry/progress. No prepared PPO ZIP. Retain only for diagnostics; do not continue PPO or repeat broad fitting automatically. See [aggregated-fit analysis](../../research/escort_rl/STAGE2_AGGREGATED_FIT_REVIEW_2026-11-08.md).

## Frozen-base residual diagnostic — research/escort_rl/runs/residual_overfit_v1b_20261108/residual_0.pt

- Residual weights `research/escort_rl/runs/residual_overfit_v1b_20261108/residual_0.pt`, SHA-256 `d26c392f13c5d207943e0f57130af1d9a6b40dd87ac90a0f2a2de60f3ee01fa3`; preserve adjacent `.manifest.json`.
- Frozen base `research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip`, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`; inherited base PPO counter 5000; residual PPO updates zero.
- Residual BC updates 4000; architecture 51→128→128→3, external 40-vector unchanged; teacher mode `observable_previous_command_v1`.
- Diagnostic only, no Stage 2 promotion or PPO support. Evidence: `research/escort_rl/runs/residual_overfit_v1b_20261108`; two-case development success 1.0; fixed suites pending

## Frozen-base residual diagnostic — research/escort_rl/runs/residual_overfit_v1b_20261108/residual_1.pt

- Residual weights `research/escort_rl/runs/residual_overfit_v1b_20261108/residual_1.pt`, SHA-256 `ca0d335a9b3d191c9115f460769eac0d38352e737480c7069b7c043d5bfefbb5`; preserve adjacent `.manifest.json`.
- Frozen base `research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip`, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`; inherited base PPO counter 5000; residual PPO updates zero.
- Residual BC updates 8000; architecture 51→128→128→3, external 40-vector unchanged; teacher mode `observable_previous_command_v1`.
- Diagnostic only, no Stage 2 promotion or PPO support. Evidence: `research/escort_rl/runs/residual_overfit_v1b_20261108`; two-case development success 1.0; fixed suites pending

## Frozen-base residual diagnostic — research/escort_rl/runs/residual_overfit_v1b_20261108/residual_2.pt

- Residual weights `research/escort_rl/runs/residual_overfit_v1b_20261108/residual_2.pt`, SHA-256 `2ff02f723de84308674a5b4d9ff656e4b7bac5331b1cf530e00a55f13f08d56c`; preserve adjacent `.manifest.json`.
- Frozen base `research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip`, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`; inherited base PPO counter 5000; residual PPO updates zero.
- Residual BC updates 12000; architecture 51→128→128→3, external 40-vector unchanged; teacher mode `observable_previous_command_v1`.
- Diagnostic only, no Stage 2 promotion or PPO support. Evidence: `research/escort_rl/runs/residual_overfit_v1b_20261108`; two-case development success 1.0; fixed suites pending

## Reviewed frozen-base residual v1 — not promoted

- **Residual:** [residual_2.pt](../../research/escort_rl/runs/residual_overfit_v1b_20261108/residual_2.pt), SHA-256 `2ff02f723de84308674a5b4d9ff656e4b7bac5331b1cf530e00a55f13f08d56c`.
- **Companion:** [manifest](../../research/escort_rl/runs/residual_overfit_v1b_20261108/residual_2.pt.manifest.json); the `.pt` includes residual optimizer/RNG state. Preserve source/config snapshots and dataset/trace records in the same experiment folder.
- **Frozen base:** selected guarded Stage 1 5k archive, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`. Every base policy parameter is frozen. Base PPO counter remains 5,000; residual has 12,000 supervised updates and ZERO PPO updates.
- **Method:** exactly mask-gated 51→128→128→3 residual, initial zero correction; current-observation risk features, observable-previous-command teacher, unchanged external 40-vector/reward/AGILE. Seed 902001, two predeclared training cases, BC + two student-only DAgger rounds. This is an explicit architecture transition, not an old PPO resume.
- **Development:** crossing 904000 and diagonal 904004 both succeed after every fit iteration (4k/8k/12k). These are same-case overfit diagnostics.
- **Fixed validation:** crossing 12/20, diagonal 17/20, head-on 9/20; 38/60, 22 geometric intrusions. Stage 1 60/60, all 11,892 observation/action pairs EXACTLY match selected 5k traces. Observable teacher separately passed 60/60. No recorded critical failures; pedestrian gates FAIL.
- **Use:** deterministic residual diagnostics only. NOT an SB3 ZIP; no stochastic/PPO adapter or training-resume CLI. Do not promote or launch PPO. Broaden training-only encounter coverage, especially head-on, in a separately versioned next milestone.
- **Evidence:** [results](../../research/escort_rl/STAGE2_RESIDUAL_RESULTS_2026-11-08.md), [guide/commands](../../research/escort_rl/STAGE2_RESIDUAL_V1.md), [development plot](../../research/escort_rl/runs/residual_overfit_v1b_20261108/development_comparison.png), and `validation_decision.json` / `stage1_exact_replay.json` in the experiment folder. Local artifacts until deliberately archived; this entry alone does not back them up.
- **Historical interruption:** `runs/residual_overfit_20261108/` stopped after 4k due to a corrected relative-path registry bug; original source snapshot and weights preserved. Use the completed `v1b` run above, not that interrupted artifact.

## Frozen-base residual diagnostic — research/escort_rl/runs/residual_coverage_v2_20261109/residual_coverage_0.pt

- Residual weights `research/escort_rl/runs/residual_coverage_v2_20261109/residual_coverage_0.pt`, SHA-256 `2ccea1459a9026c0aee7a62a297f1c6d409b528abfd81331bac9ee56f4a28647`; preserve adjacent `.manifest.json`.
- Frozen base `research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip`, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`; inherited base PPO counter 5000; residual PPO updates zero.
- Residual BC updates 16000; architecture 51→128→128→3, external 40-vector unchanged; teacher mode `observable_previous_command_v1`.
- Diagnostic only, no Stage 2 promotion or PPO support. Evidence: `research/escort_rl/runs/residual_coverage_v2_20261109`; coverage v2 round 0, 1.0 development success; fixed gates pending

## Frozen-base residual diagnostic — research/escort_rl/runs/residual_coverage_v2_20261109/residual_coverage_1.pt

- Residual weights `research/escort_rl/runs/residual_coverage_v2_20261109/residual_coverage_1.pt`, SHA-256 `7410ee0ede3e0c69d16dac065928bc9ca7a775915ed5167b7765631c421404de`; preserve adjacent `.manifest.json`.
- Frozen base `research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip`, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`; inherited base PPO counter 5000; residual PPO updates zero.
- Residual BC updates 20000; architecture 51→128→128→3, external 40-vector unchanged; teacher mode `observable_previous_command_v1`.
- Diagnostic only, no Stage 2 promotion or PPO support. Evidence: `research/escort_rl/runs/residual_coverage_v2_20261109`; coverage v2 round 1, 1.0 development success; fixed gates pending

## Reviewed residual coverage v2 — latest diagnostic, not promoted (2026-11-09)

- **Weights:** [residual_coverage_1.pt](../../research/escort_rl/runs/residual_coverage_v2_20261109/residual_coverage_1.pt), SHA-256 `7410ee0ede3e0c69d16dac065928bc9ca7a775915ed5167b7765631c421404de`.
- **Companion:** adjacent `.pt.manifest.json`; preserve continuation manifest, source/config snapshots, both round weights, label sources and validation/replay records in the run directory.
- **Parent:** reviewed v1 residual_2.pt above; exact weights/Adam/Torch and local NumPy batch RNG restored. Frozen selected Stage 1 base unchanged, inherited PPO counter 5,000; new residual BC 8k, cumulative 20k; ZERO residual PPO updates.
- **Method/config:** `diagnostics/stage2_residual_coverage_v2.yaml`; six novel training-only cases plus two retained anchors, equal family/side/case probability, unchanged causal phase weights/teacher/features/architecture/reward/physics. Both eight-case development rounds pass. This is not a new initialization or a new gait.
- **Fixed validation:** crossing 12/20, head-on 19/20, diagonal 14/20 = 45/60 (75%), fifteen geometric intrusions, no recorded critical terminations. Stage 1 60/60, all 11,892 observation/action pairs exactly match selected 5k traces. Gates FAIL; no promotion/PPO.
- **Selection limits:** improved aggregate/head-on over v1, but diagonal fell 17/20 → 14/20 and crossing case churn hides a net tie. Retain v1 for paired comparison. Only one residual seed, no final-test evaluation. These `.pt` files are not SB3 ZIP resumes.
- **Evidence:** [coverage results](../../research/escort_rl/STAGE2_RESIDUAL_COVERAGE_RESULTS_2026-11-09.md), [guide/commands](../../research/escort_rl/STAGE2_RESIDUAL_COVERAGE_V2.md), and run `validation_decision.json` / `stage1_exact_replay.json`. Local artifacts; registry does not back up weights. Next proposed work is targeted training-only crossing/diagonal variation with head-on retention, not an automatic long run.

## Frozen-base residual diagnostic — research/escort_rl/runs/residual_variation_v3_20261109/residual_coverage_0.pt

- Residual weights `research/escort_rl/runs/residual_variation_v3_20261109/residual_coverage_0.pt`, SHA-256 `3c8b7386cfd7568022738f6681b47ca3194e71c4900d40ec1f927fd0c0bee2da`; preserve adjacent `.manifest.json`.
- Frozen base `research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip`, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`; inherited base PPO counter 5000; residual PPO updates zero.
- Residual BC updates 24000; architecture 51→128→128→3, external 40-vector unchanged; teacher mode `observable_previous_command_v1`.
- Diagnostic only, no Stage 2 promotion or PPO support. Evidence: `research/escort_rl/runs/residual_variation_v3_20261109`; coverage v2 round 0, 0.875 development success; fixed gates pending

## Frozen-base residual diagnostic — research/escort_rl/runs/residual_variation_v3_20261109/residual_coverage_1.pt

- Residual weights `research/escort_rl/runs/residual_variation_v3_20261109/residual_coverage_1.pt`, SHA-256 `a8bbd73a5a6aee06f23384822f8b7350e148dfc95fdc22f6cdf863b431ff79da`; preserve adjacent `.manifest.json`.
- Frozen base `research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip`, SHA-256 `470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`; inherited base PPO counter 5000; residual PPO updates zero.
- Residual BC updates 28000; architecture 51→128→128→3, external 40-vector unchanged; teacher mode `observable_previous_command_v1`.
- Diagnostic only, no Stage 2 promotion or PPO support. Evidence: `research/escort_rl/runs/residual_variation_v3_20261109`; coverage v2 round 1, 1.0 development success; fixed gates pending

## Reviewed residual variation v3 — latest diagnostic, NOT promoted (2026-11-09)

- **Weights:** [residual_coverage_1.pt](../../research/escort_rl/runs/residual_variation_v3_20261109/residual_coverage_1.pt), SHA-256 `a8bbd73a5a6aee06f23384822f8b7350e148dfc95fdc22f6cdf863b431ff79da`.
- **Companion:** adjacent `.pt.manifest.json`; preserve continuation manifest, source/YAML snapshots, parameter-coverage exports, data/trace records and both round checkpoints. Local artifacts; this entry does not back up binary weights.
- **Parent:** reviewed v2 `residual_coverage_1.pt`, SHA-256 `7410ee0ede3e0c69d16dac065928bc9ca7a775915ed5167b7765631c421404de`; true weights/Adam/Torch/local batch RNG continuation. Same frozen Stage 1 base, inherited PPO counter 5,000. Residual BC +8k = 28k cumulative, ZERO PPO updates.
- **Method/config:** `diagnostics/stage2_residual_variation_v3.yaml`: retained eight cases/head-on examples, added eight training-only crossing speed/timing and diagonal side/angle strata. Same 8k update cap as v2, same network/features/teacher/loss/reward/physics/caps. Teacher 16/16; final student 16/16 development; source selection before outcomes.
- **Fixed validation:** crossing 20/20, head-on 20/20, diagonal 17/20, total **57/60 (95%)**. Two diagonal client losses and ONE geometric intrusion. Stage 1 60/60 and exact 11,892 observation/action pairs. All success-rate targets pass, but zero-intrusion gate FAILS; no promotion or passing-round credit.
- **Use:** latest deterministic residual diagnostic and possible future explicitly approved residual-only continuation parent; NOT an SB3 PPO ZIP or approved PPO initialization. Preserve v2/v1 comparators; one training initialization/fixed reused validation suite only, final-test data untouched.
- **Evidence:** [results](../../research/escort_rl/STAGE2_RESIDUAL_VARIATION_RESULTS_2026-11-09.md), [guide](../../research/escort_rl/STAGE2_RESIDUAL_VARIATION_V3.md), and run `validation_decision.json` / `stage1_exact_replay.json`. Next proposed data target is training-only slow positive-side oblique diagonal passing/recovery, retaining crossing/head-on replay; no automatic additional run.

## Residual PPO pilot v1 — implementation, awaiting operator run (2026-11-09)

- Guide: [RESIDUAL_PPO_PILOT_V1.md](../../research/escort_rl/RESIDUAL_PPO_PILOT_V1.md).
- Initial residual: `research/escort_rl/runs/residual_variation_v3_20261109/residual_coverage_1.pt`, SHA-256 `a8bbd73a5a6aee06f23384822f8b7350e148dfc95fdc22f6cdf863b431ff79da`; 28k supervised residual updates, zero residual PPO transitions. Its frozen base retains the selected 5k Stage 1 actor.
- This source remains the imitation comparator, not a promoted Stage 2 policy. Recorded validation: 57/60 pedestrian, 60/60 Stage 1, one intrusion.
- Read-only PPO adapter import check: `research/escort_rl/runs/residual_ppo_import_check_20261109/import_verification.json`; all 23,399 observation/action probes exact, maximum discrepancy zero. This is not a new closed-loop validation.
- No forward-use residual PPO checkpoint exists from this implementation milestone. Temporary automated-test archives are not selected models. The actual pilot CLI records its fresh output directory, ZIP hashes and sidecars here when executed. Read saved counters and validation before selecting any pilot archive.

## Prepared residual PPO critic — recovered logging failure (2026-11-09)

- Archive: `research/escort_rl/runs/residual_ppo_pilot_SDl2ue/pilot/initial.zip`.
- SHA-256: `485a92323e3780daed6973df0172001aac3f4bc35370993ba8993e9ec6e146d1`.
- Saved residual PPO counter **0**, optimization epochs **0**, PPO optimizer entries **0**. This is prepared initialization, not trained PPO or Stage 2 promotion.
- Parent: residual variation v3, SHA `a8bbd73a5a6aee06f23384822f8b7350e148dfc95fdc22f6cdf863b431ff79da`; 28k residual BC updates; exact frozen selected Stage 1 5k base. New critic and fresh PPO optimizer transition.
- Config/seed/reward: `research/escort_rl/diagnostics/residual_ppo_pilot_v1.yaml`, seed 923001, unchanged Stage 2 tracking/clearance v1 runtime/reward; exact metadata in sibling manifest.
- Required companions: `initial.manifest.json`, `initial.rng.pt`; supporting `critic_data.npz`, `critic_episodes.json`, `critic_fit.json`, `import_verification.json`, `experiment.json` in the same directory.
- Actual preparation: 24/24 training-partition critic episodes succeeded. Holdout MSE 126.7488 → 8.67035, constant comparator 24.07173. No new post-PPO validation exists; source BC validation remains 57/60 pedestrian and 60/60 Stage 1.
- Intended use: resume via residual PPO pilot CLI into a fresh directory after logging correction. Exact known-source, zero-update migration only; no generic compatibility bypass. Full-G1 callback regression and read-only actual checkpoint load pass. Preserve original archive unchanged.

## Residual PPO pilot — research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot

- Status: completed pilot; inspect validation, no automatic promotion; research only, no automatic promotion.
- Checkpoint directory `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot`; preserve ZIP, manifest and RNG sidecars.
- New residual PPO transitions 5000; source residual BC 28k and frozen base PPO 5k are separate counts.
- Use only residual PPO CLI, not generic Stage 1/Stage 2 resume. Training/final validation reports in directory.
- `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/best_validation.zip` SHA-256 `67692f7d7586093290a9ae49d952b2e2012e4135214db5079713f28c49f0695b`; counter 5000; manifest `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/best_validation.manifest.json`, RNG `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/best_validation.rng.pt`.
- `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0001024.zip` SHA-256 `d5fa873a006c3310a14df8d54fd83e1db1ad8af498b769fadfc97eff3d3b0d51`; counter 1024; manifest `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0001024.manifest.json`, RNG `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0001024.rng.pt`.
- `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0002048.zip` SHA-256 `e71520c8ab64564e59d4e34287294da0e411529e562d7b52587b2784066e9102`; counter 2048; manifest `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0002048.manifest.json`, RNG `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0002048.rng.pt`.
- `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0003072.zip` SHA-256 `1aba8765184ec1d3d698310fef9d86d96cf2d3c6e79593c054ba3435a06b13ac`; counter 3072; manifest `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0003072.manifest.json`, RNG `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0003072.rng.pt`.
- `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0004096.zip` SHA-256 `5e69fc35707fc79e76117919f5541d2a2096dc990534613ed2f884257ea6becf`; counter 4096; manifest `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0004096.manifest.json`, RNG `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/checkpoint_0004096.rng.pt`.
- `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/final.zip` SHA-256 `67692f7d7586093290a9ae49d952b2e2012e4135214db5079713f28c49f0695b`; counter 5000; manifest `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/final.manifest.json`, RNG `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/final.rng.pt`.
- `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/initial.zip` SHA-256 `e5480d7556de7c2b4ed3068a59052a099607073638daaa41a6add787f2661db0`; counter 0; manifest `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/initial.manifest.json`, RNG `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/initial.rng.pt`.
- Full parent hash, source/configuration hashes, seed, reward/runtime identity and transition method: `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/experiment.json` and checkpoint manifests. Actual validation: `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/validation_pedestrian/summary.json` and `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/validation_stage1/summary.json` when completed; interrupted/calibration-failed archives are diagnostic only.

## Residual PPO 5k — first passing Stage 2 round (2026-11-09)

- Candidate: `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/final.zip`; identical selected archive `best_validation.zip` in the same folder.
- Final ZIP SHA-256: `67692f7d7586093290a9ae49d952b2e2012e4135214db5079713f28c49f0695b`.
- Actual saved residual PPO counter: **5,000**; parent prepared initialization `residual_ppo_pilot_SDl2ue/pilot/initial.zip`, residual variation v3 BC 28k updates and unchanged frozen selected Stage 1 5k actor. These counts are separate.
- Config: `diagnostics/residual_ppo_pilot_v1.yaml`, seed 923001, Stage 2 tracking/clearance v1 reward unchanged. Preserve `final.manifest.json`, `final.rng.pt`, matching `best_validation` companions and original source archive. Strict loader and frozen-base assertions pass.
- Matched initial/final validation: pedestrian **57/60 → 60/60**, diagonal **17/20 → 20/20**, crossing/head-on remain **20/20** each. Geometric intrusions **1 → 0**. Minimum surface clearance **−0.03012 → +0.05524 m**. Mean formation **0.13531 → 0.13269 m**.
- Stage 1 remains **60/60**, mean formation **0.0563095368 m**; complete initial/final Stage 1 summaries match exactly. Base parameters match exactly; residual parameters changed.
- Completed training episodes **23/23 successful**; 500,000 active physics steps, 50,000 gait calls, 200 optimizer steps; measured training throughput **31.97 transitions/s** over **156.38 s**, excluding preparation and evaluation.
- Intended use: leading Stage 2 research candidate for confirmation; **one passing round only, no promotion or additional training authorization**. Results: both validation directories, segment-initial comparisons and `result.json` in the run directory. Final-test cases untouched; live Grove dynamics parity unverified.

### Residual PPO 5k confirmation — two fixed-suite passing rounds

- Same selected candidate `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/final.zip`, SHA `67692f7d7586093290a9ae49d952b2e2012e4135214db5079713f28c49f0695b`; still 5,000 residual PPO transitions, no additional training.
- Second full paired round: `research/escort_rl/runs/residual_ppo_confirmation_fw9EtU/pedestrian/summary.json` and `stage1/summary.json` in that directory. Both complete summaries exactly match the first round; recomputed configured gate passes with no reasons.
- Pedestrian 60/60 (20/20 each), Stage 1 60/60 (20/20 each), zero intrusions/critical failures. Mean formation 0.1326855206 / 0.0563095368 m.
- Status: satisfies configured two-round fixed-suite repeatability criterion for this checkpoint/model seed; selected Stage 2 research candidate. These repeats do not establish new-route or multi-seed generalization; no automatic next-stage training is authorized. Original archives/sidecars unchanged, final tests untouched.

### Selected Stage 2 checkpoint: methodology and visual examples (2026-11-09)

- Beginner-friendly methodology: `docs/bodyguard/STAGE2_RL_TRAINING_EXPLAINED.md`, matching PDF/HTML alongside it.
- Six representative fixed-suite full-G1 visualization episodes: `docs/media/stage2_selected_5k_20261109/index.html`; crossing, head-on, diagonal, straight, gentle-turn and stop/start all succeeded. Final logged info exactly matches the corresponding original validation trace for every clip. No new independent generalization claim.
- Same selected `residual_ppo_recovered_9htyqV/pilot/final.zip`, SHA `67692f7d7586093290a9ae49d952b2e2012e4135214db5079713f28c49f0695b`; counter remains 5,000 residual PPO transitions. No new weights, updates or stage promotion.
- Rendering executed 112,300 active physics steps / 11,230 gait calls, with separate display state. MP4s decode successfully; playback about 2.5x. Humans are decorative proxies, not physical-contact models. Reusable rendering source: `research/escort_rl/visualization/stage2_clips.py`.

## Stage 2 Git archive scope (2026-11-09)

The Stage 2 commit archives the selected residual PPO ZIP/manifest/RNG triples, its recovered initial state, the original prepared critic, the required v3 residual BC source and sidecar, teacher readiness data, and compact results/validation summaries. Stage 1 base weights were already archived in the preceding commit. Raw per-step research traces, intermediate checkpoint sweeps, most failed-run weights, TensorBoard events, local virtual environments and generated caches remain local. Registry paths for those historical diagnostics do not imply their binaries are backed up by Git. Fresh-training read-only import checks need the original v3 validation traces, which remain local; selected-checkpoint inference/evaluation does not need those trace files. Absolute paths in historical manifests are preserved; relocating to another checkout path requires an explicit provenance-aware relocation procedure, not disabling strict checks. See `research/escort_rl/STAGE2_GIT_ARCHIVE.md`.
