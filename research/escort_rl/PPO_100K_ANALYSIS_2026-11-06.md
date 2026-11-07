# Stage 1 100k analysis — November 6, 2026

**Decision: FIX FIRST. Do not promote or extend this run unchanged.** The run reached
100,000 actual training transitions, but final deterministic validation was 0/60.
The strongest evaluated checkpoint was at 20k (21/60). A confirmed resume bug repeatedly
restarted the training scenario streams, substantially reducing scenario diversity.
No new training, simulator interaction, reward change or source-code fix was performed
during this analysis. Original run artifacts remain untouched.

Source run: `runs/stage1_100k_20261105_184710_ZZ1bmS/`.
New analysis data/plot/proof: `runs/analysis_100k_20261106_9kjdgpa2/`.

## Checkpoint comparison

Each checkpoint used the same 60 held-out episodes, 20 per family. Figures below are
validation, not stochastic training statistics. Formation is the mean of per-episode means.

| Total transitions | Success | Formation mean (m) | Accompanied advance mean (m) | Validation failures |
|---:|---:|---:|---:|---|
| 20k | 21/60 (35%) | .954 | 4.575 | 39 recovery timeout |
| 30k | 0/60 | 1.896 | 3.208 | 55 recovery, 3 client loss, 2 bounds |
| 40k | 0/60 | 2.687 | 1.882 | 60 client loss |
| 50k | 0/60 | 1.489 | 4.455 | 57 recovery, 3 bounds |
| 60k | 0/60 | 2.363 | 2.475 | 46 client loss, 14 recovery |
| 70k | 2/60 (3.3%) | 1.281 | 4.963 | 58 recovery |
| 80k | 0/60 | 2.686 | 1.862 | 59 client loss, 1 bounds |
| 90k | 0/60 | 2.625 | 2.000 | 55 client loss, 5 bounds |
| 100k | 0/60 | 1.189 | 4.770 | 57 recovery, 3 bounds |

The 20k checkpoint succeeded on **9/20 straight, 6/20 gentle-turn, 6/20 stop/start**.
It is preserved at `round_01/final.zip`. It is the strongest of the evaluated 10k-spaced
checkpoints, not a claim about unevaluated recovery snapshots. It still falls far below
the 90% overall / 80% per-family gate. Final 100k was 0/20 in every family.
No validation round reported a fall, supervisor failure or physics-failure reason.
The previous diagnostic follower succeeded on all 60 identical validation cases.

`best_validation` in each chunk compares only that chunk's initial and final models.
Thus `round_09/best_validation.zip` is not the global best across the entire experiment.
The plotted comparison is `checkpoint_validation.png` in the new analysis directory.

## Confirmed resume defect

The training wrapper owns a NumPy scenario RNG. The resume path loads the PPO model,
then restores saved wrapper RNG states. However, SB3 model loading calls
`_setup_model()` -> `set_random_seed()` -> vector-environment `seed()`. This queues seeds
for the next vector reset. The following `learn()` resets the environments, passes those
queued seeds to `TrainingScenarios.reset(seed=...)`, and replaces the restored RNGs.
The fresh-model branch clears `vec._seeds`; the resume branch does not.

Actual logs show **471 scenario resets but only 65 unique scenario seeds** across the
additional 90k transitions. Every chunk begins with the same first seeds:

| Worker | Intended next seed from restored state at first resume | Observed first seed in every chunk |
|---|---:|---:|
| 0 | 639056782823807736 | 4374458801877455528 |
| 1 | 3189754318648236116 | 3439179871814419720 |

Reconstructing `default_rng(model.seed + worker)` produces the observed seeds exactly.
`resume_rng_proof.json` contains this read-only checkpoint/log audit. This is direct evidence
of stream replacement, not just an inference from an unusual family distribution.

Sampled family counts were **273 straight, 68 gentle-turn, 130 stop/start** across those
471 resets: approximately 58.0/14.4/27.6%, rather than the intended 50/25/25 mix.
Repeated prefixes restricted coverage and overrepresented the same scenarios. The nominal
sampler probabilities were not changed. These counts include unfinished episodes.

Policy/optimizer/timestep continuation did occur: the run reached 100k, 500 optimization
epochs and 4,000 Adam steps. Each resumed initial validation summary exactly matches the
previous final summary (8/8 chunk boundaries). Therefore the defect is not a silent policy
restart. It compromises scenario-stream continuation and diversity; it does not erase
the executed full-G1 transitions or invalidate the held-out observations. It also does
not prove that this defect alone caused all learning regression.

The prior resume test checked model/optimizer and timestep continuation but did not check
the next sampled scenario after loading. That gap allowed this bug through. The supplied
nine-chunk command exposed the resume defect repeatedly; it should not be reused unchanged.

## Final behavior: passes the client and fails to stop

At termination, **all 60 final episodes were ahead of the client itself**, not merely
ahead of the desired 1.5 m following position. Client-frame longitudinal formation error
was positive in all cases, minimum 1.635 m and mean **2.792 m**. Since the desired offset
is -1.5 m, the mean robot position was about **1.292 m ahead of the client**.
Mean terminal formation-error norm was **2.797 m**. The reported 1.189 m formation mean
averages the whole episode and hides the larger endpoint error.

After client arrival, time-step pooled mean applied command was
**(vx=.193, vy=.120, yaw=.0058)** m/s, m/s, rad/s. Mean measured body velocity remained
**(.137, .062, .00072)**. The policy therefore continued translating while the client
was stationary, instead of recovering and settling behind it. This explains the 57
recovery timeouts; three cases crossed scene bounds before completion. Current/past
client motion is already represented in the causal observation, so this does not require
adding route endpoints or future trajectory labels to the actor.

The final chunk's printed 17% success is **8/47 stochastic training episodes**, not
held-out deterministic success. Across the added 90k, 453 completed training episodes
had: 82 success, 118 client loss, 236 recovery timeout, 15 supervisor failure and two
out-of-bounds; no fall or physics-failure reason codes. Independent validation remained
poor despite occasional training successes.

Explained variance improved from approximately zero in the pilot to .267, but that means
the critic predicts returns somewhat better; it is not task success. Final policy std
remained .961, and **69.68%** of sampled three-axis Gaussian action vectors in the added
90k required clipping on at least one axis. The exploration distribution remained broad.
This is separate from final PPO objective clipping (.0408). The 15 training supervisor
terminations must remain explicit failures; no gain, cap or supervisor change is justified
by this analysis.

## Reward-contract concern to audit before another experiment

The current progress credit is capped at **client route progress**, rather than the
progress of the desired point 1.5 m behind the client. The accompaniment eligibility
permits formation error up to 2 m and being as close as the client's route position.
Consequently, moving from the correct following point up to the client's position can
earn extra progress credit even though it worsens the preferred formation.

For a straight route of length L and initial robot progress -1.5 m, correct endpoint
formation at L-1.5 produces approximately L metres of credited advance. Reaching the
client endpoint at L can produce L+1.5 metres while still satisfying the progress
eligibility at that point: up to **6 additional progress-reward units** at weight 4.
The formation and terminal rewards oppose this behavior, so this is a potential incentive
conflict, not proof that total optimal reward favors failure. Discounting, saturated
formation shaping and delayed completion may make that conflict harder to learn through.
The observed overshoot is consistent with this concern; causality needs a controlled test.
No repeat-traversal/highwater farming was demonstrated, and no reward was changed here.

## Throughput and next steps

The additional 90k transitions took **1,363.40 s (22.72 min)** of timed training,
**66.01 transitions/s** including optimizer/logging/resets/checkpoints, excluding validation
and plotting. The last 10k achieved 66.94/s. Throughput is adequate; additional unmodified
compute is not the immediate remedy.

Recommended next milestone, before another long run:

1. Fix resume by clearing SB3's queued vector reset seeds after loading/restoring worker
   RNGs. Add a regression test comparing the next sampled episode sequence before/after
   checkpoint loading, for both one and two workers. Verify independent streams and
   actual sampled metadata, not only timestep counts.
2. Preserve all old results. A code fix changes source provenance: implement a narrowly
   documented checkpoint migration/compatibility path rather than editing old manifests
   or bypassing asset/reward checks wholesale.
3. Audit formation-preserving progress and stopping with deterministic reward fixtures.
   Treat any reward revision as a new experimental condition, preserving this run's
   frozen reward for comparison. Do not change low-level gait, physics or command caps.
4. Start a short, separately authorized Stage 1 experiment after those checks; compare
   against the 20k checkpoint and the existing follower on the same validation cases.
   Consider reduced initial exploration as a separately recorded ablation, not a silent
   simultaneous change. Require improved stopping and held-out success before scaling up.

**No Stage 2 promotion. No automatic 300k/500k extension.** Retain AGILE/full-G1 physics:
the causal diagnostic follower's 60/60 result still demonstrates scenario feasibility.
The confirmed infrastructure bug and high-level stopping/formation failure should be
addressed before blaming or replacing the gait. Full live-Grove numerical dynamics
parity remains unverified, as previously documented.
