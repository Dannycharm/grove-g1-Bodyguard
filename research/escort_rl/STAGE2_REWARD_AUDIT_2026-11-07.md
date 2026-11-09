# Stage 2 matched reward and yielding audit

**The tested reward prefers successful avoidance over collision and permanent stopping.
It nevertheless imposes an immediate cost on early yielding while its dense clearance
feedback arrives later. That learning-signal delay, together with the measured weak
critic and narrow exploration, is a credible difficulty. A globally reversed preference
for collisions is not demonstrated.** No reward weights or training source were changed.

## What actually ran

Four causal diagnostic controllers each ran the exact same sixty validation scenarios:
twenty crossing, twenty opposing-lane head-on and twenty diagonal. This is **240 full-G1
rollouts on sixty distinct scenarios**, not 240 distinct scenarios. A separate six-case
development set was used for controller checks before freezing the validation version.

- **Follow:** the existing diagnostic Stage 1 follower, reused unchanged.
- **Yield:** the same follower with a current-position/current-velocity collision forecast,
  zero translational command while yielding, and return to the original follower afterward.
- **Sidestep:** the same risk detection, braking plus a temporary .65 m lateral target,
  .30 m/s lateral command cap, then return to the original formation target.
- **Stop:** zero commands from the start until the environment ends the episode.

These are evaluation-only controllers, NOT new teacher training, a new PPO policy, or
final Reactive Baseline A. They do not query future pedestrian trajectories, encounter
metadata, route progress or scenario labels. All accept only the causal 40-vector and
fixed configuration; constant-velocity extrapolation from present measurements is a
causal prediction. Temporary diagnostic target changes do not change the environment's
reward target or outcomes.

All strategies use unchanged AGILE, full G1 physics/contacts, caps, scheduling, scenarios,
Stage 2 reward and termination. Actor, critic and optimizer weights were not trained or
updated. The previously trained PPO policy still has its original 20/60 Stage 2 score.

Development version 1 reacted too late. Its closest-approach test wrongly excluded a
risk whose closest time lay beyond the prediction horizon even when its horizon endpoint
was already inside the risk radius. This diagnostic-only condition was corrected and
unit tested. Version 2 was then frozen before the sixty-case comparison; all development
failures and both code snapshots remain preserved. No PPO/environment bug was changed.

## Outcomes

| Strategy | Crossing | Head-on | Diagonal | All successes | Intrusions | Client loss |
|---|---:|---:|---:|---:|---:|---:|
| Follow | 0/20 | 20/20 | 0/20 | 20/60 | 40 | 0 |
| Yield and recover | **20/20** | 7/20 | 3/20 | 30/60 | 11 | 19 |
| Sidestep and recover | 7/20 | **20/20** | 0/20 | 27/60 | 20 | 13 |
| Stop indefinitely | 0/20 | 0/20 | 0/20 | 0/60 | 1 | 59 |

No falls, supervisor or physics failures occurred. Intrusions are planar footprint-proxy
geometry, not physical person contacts. The pedestrian proxies remain noncolliding.

Successful crossing yields demonstrate **20/20 tested crossing cases are feasible under
the existing gait, limits, client-loss and recovery rules**. We have at least one observed
successful diagnostic on all crossing and head-on cases and three diagonal cases. That
is a feasibility observation across different controllers, not a deployable combined policy
score. The other seventeen diagonal cases were not solved by these diagnostics; this does
not prove those cases impossible or identify reward as the reason for their failure.

Overly conservative waiting caused many client losses. On head-on cases the ordinary
follower already passed safely under the configured footprint criterion. Yielding there
was often unnecessary. The sidestep diagnostic avoided all crossing intrusions but lost
the client in thirteen cases; safe passage alone does not complete escorting.

## Exact matched reward preference

The comparison records actual per-transition reward sums and discounted returns with
gamma .99 from the episode start. On the same scenario:

- **23/23 successful yields** whose ordinary follower collided earned both higher total
  reward AND higher discounted return than that collision. All also beat permanent stopping.
- **7/7 successful sidesteps** whose ordinary follower collided likewise beat collision
  and permanent stopping under both measures.
- Successful-yield total reward advantage over the matched collision ranged **+90.11 to
  +114.19**. Discounted advantage was at least **+27.27**, mean **+42.34**.
- Successful-sidestep total advantage ranged **+82.18 to +90.22**. Discounted advantage
  was at least **+24.23**, mean **+32.48**.

Thus all **30 successful-avoidance paired comparisons** favor successful avoidance.
These are thirty comparisons, not necessarily thirty unique cases: some crossing cases
have both a successful yield and a successful sidestep.

| Crossing outcome average | Follow | Yield | Sidestep | Stop |
|---|---:|---:|---:|---:|
| Successes | 0/20 | 20/20 | 7/20 | 0/20 |
| Total episode reward | -97.59 | **+10.72** | -89.03 | -127.26 |
| Discounted return | -43.96 | **-0.95** | -31.21 | -50.99 |

A negative discounted reward is not inherently failure: zero is an arbitrary reference.
The successful crossing yield is much better than the available failed alternatives.
The sidestep averages include its thirteen client-loss failures; they must not be treated
as the return of a successful maneuver.

### Every reward component for crossing episodes

Values below are mean episode component totals across twenty cases, rounded. Exact
per-case values for every family are in `episode_reward_components.csv`.

| Component | Follow | Yield | Sidestep | Stop |
|---|---:|---:|---:|---:|
| Formation | -.091 | -11.725 | -24.283 | -22.122 |
| Progress | +5.001 | +14.485 | +6.878 | +.002 |
| Command change | -.024 | -.069 | -.067 | .000 |
| Torso stability | approximately 0 | approximately 0 | approximately 0 | approximately 0 |
| Terminal success | .000 | +20.000 | +7.000 | .000 |
| Terminal failure | -100.000 | .000 | -65.000 | -100.000 |
| Velocity tracking | -1.572 | -11.607 | -12.713 | -5.139 |
| Pedestrian clearance | -.899 | -.360 | -.845 | .000 |
| Total | -97.585 | **+10.724** | -89.030 | -127.258 |

The successful yielding controller earns more progress and success reward, but pays
substantial formation and velocity-matching costs during its temporary separation.
Indefinite stopping earns essentially no progress and always fails; there is no observed
standing-still success/reward farm. One pedestrian even intrudes into a stationary G1's
footprint, so indefinite stopping is not universally collision-free.

## Where the reward makes learning difficult

For every successful-avoidance/failed-follow pair, we inspect the first two seconds after
the diagnostic intervention, cut short if necessary to exclude the first terminal outcome.
**All 23 yields and all seven sidesteps initially earn less reward than following.**
Mean early reward disadvantage: **-1.57** for yielding, **-2.80** for sidestepping.

Successful interventions begin, on average, **2.76 s** before the ordinary follower's
first nonzero clearance penalty for yields, and **2.77 s** for sidesteps. The current
clearance cost activates only inside a .30 m surface margin (.85 m centre separation
under the footprint discs). Appropriate early yielding starts while that cost is still zero.

The controller therefore has to accept immediate formation/velocity/progress disadvantages
to obtain a later avoidance benefit. In principle a good value estimate handles that.
The pilot's critic instead predicted about +7.5 just before failures whose recorded
remaining returns were about -95. Narrow noise also produced no successful stochastic
crossing/diagonal training episodes. The audit supports a **credit-assignment/discovery
problem**, not merely a penalty that is too small at collision.

This is not proof reward design is optimal or irrelevant. Earlier causal risk shaping
might make learning easier, but it is a separate opt-in experiment, not a justified silent
weight edit. Making -100 arbitrarily larger does not supply successful actions or an
accurate critic, and can increase value-target variance.

## Comfort and failure tradeoffs remain

Current .30 m clearance is a soft preference, not a hard success constraint. On head-on
cases, ordinary follow succeeds 20/20 with mean minimum surface clearance **.171 m**.
Sidestep also succeeds 20/20 and widens that to **.585 m**, but mean total reward falls
**29.15 -> 11.54**, discounted **6.70 -> -.62**. Its extra formation/velocity costs exceed
the modest clearance improvement rewarded by the present weights.

This is a real tradeoff, not evidence that all wider passes are optimal. If .30 m should
be a mandatory comfort/safety criterion, reward and gate need an explicitly versioned
correction. Currently a narrow nonintruding pass can rationally be preferred. The metric
is only a pelvis-centred footprint surrogate, not a whole-body or real-human safety claim.

Intrusion and client-loss failures both carry the same -100 terminal penalty. Longer
failed yielding may accumulate more formation/velocity cost than an early intrusion;
aggregate undiscounted reward need not rank policies exactly by success count or safety
priority. If the intended specification ranks every geometric intrusion below every
client-loss failure, that hierarchy is not explicitly encoded. We should decide that
requirement before changing penalties. None of these qualifications reverses the measured
preference for successfully completed avoidance in the tested pairs.

## What to do next

1. **Keep this reward as the control.** The test rejects a simple explanation that successful
   yielding is globally punished relative to collision. Crossing feasibility is demonstrated;
   no evidence yet supports a larger MLP as the immediate fix.
2. Prepare/check the Stage 2 critic with the **current frozen actor's training-only mixed
   rollouts**, using episode holdout and Stage 1 replay. It should predict that actor's actual
   outcomes. Do not blindly fit the current actor's value function to positive expert returns
   when the current actor would still collide; that is an off-policy target mismatch.
3. Investigate discovery with a bounded comparison. These causal diagnostic trajectories
   demonstrate useful behavior but were not used as labels. Any pedestrian-aware imitation
   warm start, exploration adjustment or easier training encounter timing is a changed,
   recorded experimental condition. Preserve the original Stage 2 pilot and fixed gates.
4. If testing reward shaping, compare one opt-in early-risk variant against the unchanged
   control with matched seeds/budget. Risk must derive from current causal measurements,
   not future scripted trajectories. Retain success, anti-stationary/progress safeguards
   and reward-independent clearance metrics. Decide separately whether comfort margin and
   failure severity should become harder requirements.
5. Improve/test diagonal feasibility before claiming a general diagnostic teacher. No
   diagnostic here meets the full Stage 2 gate; no promotion or longer training was launched.

## Artifacts and reproducibility

Root: `runs/stage2_reward_audit_20261107/`.

- `development_*` and `development_v2_*`: preserved calibration outcomes.
- `audit_source_v1.py`, `audit_source_v2.py`: immutable controller snapshots.
- `validation_{follow,yield,sidestep,stop}/`: manifest, same case definitions, complete
  observations/actions/physics diagnostics/reward traces, per-case and aggregate summaries.
- `comparison_final/summary.json`: matched comparison assertions and counts.
- `comparison_final/episode_reward_components.csv`: all actual episode and discounted
  component totals, outcomes, durations and clearances.
- `comparison_final/five_second_reward_components.csv`: totals in common five-second
  time bins, including actual executed duration for partial final bins.
- `comparison_final/matched_prefix_rewards.csv`: common prefixes before/including the
  earliest task outcome. No padding with invented post-termination rewards.
- `comparison_final/paired_comparisons.csv`: case-wise total/discounted gains, early cost
  and safety-signal timing comparisons.
- `comparison_final/reward_comparison.png`, `paired_crossing.png`: recorded plots.

The summarizer asserts identical cases, configuration and scenario motion across strategies
at common timestamps, and validates every reward-component sum. All task data is isolated
full-G1 evidence, not live ROS/Grove or hardware evidence. Final tests remain untouched.
The original pilot checkpoint still passes strict provenance. No neural weights were created.

Tests: **126 passed, zero failed/skipped**, including three new diagnostic boundary/risk
regressions and the previous 123 tests. The full experimental comparison had 31,401 actual
high-level steps across 240 validation episodes, plus 36 development calibration episodes.
All controllers use unchanged simulated physics; no ROS/DDS, Docker or dependency actions.

Reproduce in fresh directories (the completed audit need not be rerun):

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/reward_audit_XXXXXX")
for STRATEGY in follow yield sidestep stop; do
  "$PY" research/escort_rl/diagnostics/stage2_reward_audit.py \
    --strategy "$STRATEGY" --suite validation --cases-per-family 20 \
    --output "$OUT/validation_$STRATEGY" || break
done
"$PY" research/escort_rl/diagnostics/summarize_stage2_reward_audit.py --root "$OUT"
```

Code is under `diagnostics/`, outside existing training source globs, to preserve exact
checkpoint resume checks. No rewards, dynamics, original policy weights, observation
schemas or existing experiment data were overwritten. No commits/pushes were performed.
