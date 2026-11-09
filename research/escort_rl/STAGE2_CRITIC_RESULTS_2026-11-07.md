# Stage 2 critic preparation review

Evidence: `runs/stage2_critic_zMgnVP/preparation/`. This review reads saved results;
it does not launch additional training or claim new policy validation.

Preparation passed its declared gates. The saved archive SHA-256 is
`186ed9e142e880e12f0a37accc35f92835a43860b3ec29ae8561c4291540291c`.
Actor distribution, protected tensors and actor optimizer state remained unchanged.
PPO counters remain 10,000 cumulative / 5,000 Stage 2 transitions. Preparation
separately collected 10,680 transitions over 72 episodes and made 6,000 value updates.
Collection used 1,068,000 active physics steps, 106,800 gait calls and 322.20 seconds.
Forty stochastic collection episodes succeeded and 32 ended in geometric intrusion;
these are training-distribution outcomes, not fixed-suite validation results.

## Value prediction results

Whole-episode fitting/holdout split: 56/16 episodes, 8,377/2,303 transitions.

| Held-out metric | Before | After |
|---|---:|---:|
| Overall MSE | 1508.64 | 219.62 |
| MAE | 22.80 | 8.38 |
| Explained variance | 0.019 | 0.826 |
| Straight MSE | 26.12 | 9.20 |
| Gentle turn MSE | 29.67 | 10.78 |
| Stop/start MSE | 31.64 | 13.33 |
| Crossing MSE | 5347.74 | 216.99 |
| Diagonal MSE | 5321.95 | 673.67 |
| Head-on MSE | 23.04 | 648.37 |

Near-intrusion mean predicted return changed from +7.31 to -60.36, versus
observed discounted return -96.08. All 70 held-out near-intrusion samples now have
negative predictions. Failure severity is still underestimated.

Overall MSE improved 85.4%, but the head-on regression is substantial: mean value
-3.67 versus realized +11.02. The declared gates constrain the three Stage 1
families and crossing/diagonal, but do not impose a head-on regression limit.
This result passes that protocol without proving uniformly calibrated values.
The fitting holdout selected the update and therefore is not an untouched final test.

## Recommendation

Use a bounded 5,000-new-transition PPO continuation with the original reward,
existing learning rate and exploration, starting from this prepared archive.
The existing Stage 2 training command automatically evaluates both fixed 60-case
suites. Preserve the original unprepared pilot as the control. A causal comparison
would additionally run the same bounded continuation from that parent with the
same training streams; critic preparation alone cannot establish an improvement
in subsequent PPO learning.

Review crossing/diagonal success and intrusion counts, head-on preservation,
Stage 1 success/formation, and value/action diagnostics before another chunk.
No longer run or promotion is recommended now. If avoidance remains absent,
investigate safe-action discovery rather than repeating critic fitting or assuming
the reward needs changing. Diagonal feasibility and live Grove dynamics parity
remain unresolved. No final-test scenarios were used.
