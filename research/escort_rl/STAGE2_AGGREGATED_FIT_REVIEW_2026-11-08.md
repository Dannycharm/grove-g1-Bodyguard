# First aggregated student-state fit review

Read-only review of `runs/stage2_dagger_fit_XRTTA7/aggregated/`. No new model fit,
rollout or PPO was executed. Offline predictions on the existing dataset were
used for matched holdout comparison.

## Outcome

| Fixed validation | Weighted only | Aggregated DAgger labels |
|---|---:|---:|
| Crossing | 0/20 | 0/20 |
| Diagonal | 0/20 | 0/20 |
| Head-on | 14/20 | 20/20 |
| Pedestrian total | 14/60 | 20/60 |
| Geometric intrusions | 41 | 40 |
| Stage 1 | 60/60 | 47/60 |
| Stage 1 mean formation error | 0.17805 m | 0.10634 m |

All forty pedestrian failures were geometric proxy intrusions. Stage 1 failures
were thirteen recovery timeouts: straight 15/20, turns 17/20, stops 15/20 successes.
No fall/supervisor/physics failure appears in the validation outcome counts.
Stage 1's smaller mean formation error does NOT imply restoration: thirteen
endpoints fail the required stop hold. All thirteen have qualified endpoint
geometry AND accompaniment progress; terminal formation errors are within 0.75 m.
Their maximum recorded terminal hold is 0.8 s versus a required 1.0 s, and mean
terminal instantaneous planar speed is 0.357 m/s versus a 0.15 m/s ceiling.
Commands are often still large, including +/-0.6 m/s examples. This is loss of
stopping/settling behavior, not simply being too far from the client.

Both success and formation preservation gates fail (mean error 0.106343 vs limit
0.106310 m; the thirteen success failures are much more consequential than this
small mean-error excess). No prepared PPO ZIP was saved. The diagnostic actor
remains available but is not approved for PPO continuation.

## What the fit establishes

3,000 supervised updates executed, selected update 2,900, 13,648 rows including
869 substantial intervention labels. PPO counter remains the original 5,000;
there were no Stage 2 PPO updates in this branch. Critic and action variance remain
unchanged by design. The combined holdout objective falls 0.09250 -> 0.03845, but
this alone does not establish useful closed-loop learning.

Because aggregation changes holdout composition, directly comparing its global
score with weighted-only 0.01723 is unfair. On the SAME ORIGINAL teacher holdout,
recomputed from preserved actor parameters:

| Matched holdout metric | Weighted only | Aggregated |
|---|---:|---:|
| Balanced MSE | 0.01723 | 0.02121 |
| Crossing intervention MSE | 0.04589 | 0.05331 |
| Diagonal intervention MSE | 0.07268 | 0.09604 |
| Head-on intervention MSE | 0.04768 | 0.05937 |

Thus the new fit has not improved even the original teacher-state approximation.
Adding one student-state batch did not solve the transfer problem at this budget.
This is not proof that DAgger generally fails, that capacity is insufficient or that
rewards are unsuitable. Rewards are not the supervised fitting objective. More
uncontrolled data/iterations are not justified as the next automatic step.

## Recommended next diagnostic milestone

Stop broad fitting and do not run PPO. Keep the successful Stage 1 archive as the
forward-use base. Before another large experiment:

1. Test teacher-label consistency, including left/right alternatives and causal
   teacher history, and whether the existing causal observation represents the
   decision sufficiently. MSE can average conflicting evasive labels into an
   unsafe middle action; this is a hypothesis to test, not a measured conclusion.
2. Test bounded overfitting of one crossing and one diagonal DEVELOPMENT encounter,
   reporting both same-case closed-loop execution and held-out cases. If those
   cannot be imitated, diagnose mapping/optimization/observability before spending
   more data. Same-case success is a diagnostic, not promotion.
3. Explicitly protect the existing Stage 1 actor's no-pedestrian behavior. Consider
   anchoring outputs to the successful parent, or an opt-in frozen-parent plus
   pedestrian-masked residual architecture that is exactly zero with no pedestrian.
   The latter is a NEW architecture/transition requiring versioned provenance and
   tests, not an existing implementation or a silent checkpoint modification.

Choose a controlled next milestone rather than repeating the same aggregate fit.
Preserve reward/physics/observations/caps and both evaluation gates. No final-test
scenarios were used; live Grove numerical dynamics parity remains unverified.

## Artifact

`runs/stage2_dagger_fit_XRTTA7/aggregated/candidate_actor.npz`

SHA-256 `a5e625b21a4ba40cccaab03863fe4f29dd59081bbe043431c8a52293d69eb9ff`.
Use only with matching `candidate.json` for diagnostics, not PPO resume. Original
archives are untouched. All conclusions are limited to this one initialization,
small development dataset and the existing fixed validation suite.
