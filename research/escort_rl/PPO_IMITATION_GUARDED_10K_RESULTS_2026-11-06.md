# Guarded imitation-pretrained PPO: 10k continuation

**Second preservation round PASS. Resume RNG continuation also passes direct sequence
verification. Keep the current architecture, objective C and guarded PPO settings.**

Run: `runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/`.
This analysis read existing artifacts only: no training, new dynamics, final-test episodes
or live interfaces were executed. No training/environment source was modified.

| Matched strict validation metric | BC initialization | PPO 5k | PPO 10k |
|---|---:|---:|---:|
| Overall success | 60/60 | 60/60 | 60/60 |
| Straight / gentle-turn / stop-start | 20/20 each | 20/20 each | 20/20 each |
| Mean formation error | .05906 m | .05631 m | .05771 m |
| Mean terminal error norm | .02015 m | .02442 m | .02097 m |
| Mean return | 30.0229 | 30.1416 | 30.1994 |
| Mean completed stop hold | 1.0 s | 1.0 s | 1.0 s |
| Mean terminal planar body speed | .00339 m/s | .00266 m/s | .00183 m/s |
| Critical failures | 0 | 0 | 0 |

Ten-k final terminal signed client-frame formation error mean is (-.01299,+.00140) m.
There is no recurrence of the metre-scale forward offset or inability to stop. Formation
mean increased 1.40 mm from 5k, while terminal error improved 3.45 mm. These small mixed
changes are preservation evidence, not meaningful demonstrated improvement over BC.

## Training and optimization

Exactly 5,000 additional transitions, 5k -> **10,000 total**, with 500,000 active physics
steps and 50,000 gait calls in this continuation. All **24/24 completed stochastic training
episodes succeeded**, no critical failures. Total over both chunks: **48/48** successful
completed training episodes. No Gaussian action-bound crossings, command saturation or
observation clipping were logged in this continuation.

KL by optimizer block: **.00694, .00834, .00441**. PPO objective clipping fractions
.0461, .0637, .0508. One early-stop log at epoch index four due to the KL guard.
Final explained variance **.2497**, value loss **3.302**. Critic learning remains modest;
raw value-loss differences alone do not justify changing network capacity.

Actual Adam steps **194 -> 351**, i.e. **157 new optimizer steps**; optimization-epoch
counter **30 -> 55** because of the early stop. This is real continued optimization.
Sigmas remain near .05 on all axes. Incremental actor-mean probe MSE .00001936. Cumulative
RMS drift relative to original imitation mean on the fixed training observation probe is
(.00383,.00261,.00182) normalized action units. Incremental changes partly reverse earlier
changes; drift need not increase monotonically with more updates.

Training wall **75.122 s**, **66.56 transitions/s**, including collection, resets, logging
and optimizer work, excluding evaluation/preparation/plotting. Optimizer wall .262 s.
Total active PPO collection/optimization over both chunks: 149.898 s, **66.71 transitions/s**.
Training-only CPU utilization cannot be inferred from the current CPU-seconds result field,
which also includes final parent-process validation. RSS ~2.64 GiB.

## Exact resume and provenance checks

- All current source/config/dependency/reward/gait/reference-manifest hashes match.
- Resumed initial validation is exactly equal to preceding final validation.
- Loaded resumed initial checkpoint retains 5k timesteps, 194 Adam steps, nonempty optimizer
  state and exactly matching actor probe outputs; final retains 10k/351.
- For EACH worker, independently advance the preceding final sidecar's saved NumPy RNG,
  including reserved-combination rejection, through every new sampled episode: the entire
  predicted sequence exactly equals the actual new scenario log.
- Worker 0: twelve new resets (3 straight/4 turn/5 stop); worker 1: fourteen
  (5 straight/6 turn/3 stop). **26/26 new unique seeds**, no overlap with preceding chunk.
- Combined chunks: **52/52 distinct training seeds**. Saved final worker counters 25/27
  exactly match cumulative resets. Final RNG sidecar timestep is 10,000.

The old scenario replay bug is not observed. Physical partial episodes still restart at
resume by design; this does not establish bit-identical uninterrupted training trajectories.

## Checkpoint selection and next steps

Both post-PPO rounds pass the prospective preservation thresholds. The checkpoint selector
chose `best_validation=initial` in this chunk, meaning the **5k model**, because success and
critical failures tie and mean formation is slightly better at 5k. `final.zip` contains the
10k state and remains a valid continuation point. A larger return alone did not override
formation in selection. Preserve both checkpoints and the untouched BC baseline.

1. Treat Stage 1 imitation-PPO preservation and tested resume continuity as passed for this
   PPO seed. This does not automatically authorize curriculum promotion.
2. Next, repeat a fresh guarded **5k run with PPO seed 43** from the same immutable BC actor,
   and evaluate another explicitly designated development-validation batch with different
   resolved routes (not final-test data). This checks seed sensitivity and repeated use of
   the current sixty cases. Same BC dataset/teacher and critic-preparation seed settings
   remain shared; report the replication as a different PPO seed, not independent imitation.
3. If that passes, consider a separately scoped Stage 2 implementation, retaining Stage 1
   replay and the same frozen gait/observation-access/safety constraints. No Stage 2 code or
   scenarios were added here. There is no demonstrated need for 100k more Stage 1 samples
   while both BC and PPO already achieve the success ceiling.
4. Keep a BC-only baseline. Current evidence supports stable fine-tuning, not learning
   escort control from scratch, nor a substantial performance gain from PPO yet.

Full numerical live Grove-G1 dynamics parity remains unverified. One training seed,
scripted client proxies and flat-ground scenarios do not establish broad robustness or
real-person safety. The final-test partition remains untouched.
