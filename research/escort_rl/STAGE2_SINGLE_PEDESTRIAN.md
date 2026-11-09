# Stage 2 single pedestrian curriculum

## Scope and sampling

This opt-in milestone adds one moving scripted pedestrian to the existing frozen-AGILE,
full-G1 MuJoCo escort environment. It does not change the original Stage 1 modules,
low-level controller, checkpoint, timing, command caps or trajectory generator. It does
not launch ROS/DDS, a live simulator, Docker or training automatically.

The user's proposed mixture is used, with grouped categories split evenly:

| Family | Probability | Pedestrians |
|---|---:|---:|
| Straight replay | 15% | 0 |
| Gentle-turn replay | 12.5% | 0 |
| Stop/start replay | 12.5% | 0 |
| Crossing | 35% | 1 |
| Head-on | 12.5% | 1 |
| Diagonal | 12.5% | 1 |

Forty percent retains Stage 1 competence; sixty percent introduces interactions.
All probabilities, motion ranges, geometry assumptions and gates are in
`stage2_config/stage2.yaml`. There are no Stage 3/4 scenarios.

## Reuse and encounter construction

`escort_rl/stage2/scenarios.py` wraps unchanged `Stage1Episode` client trajectories.
Interaction families initially use a straight client route; replay retains turns and
stops. Client peak speed remains .15–.25 m/s and route length 3–5 m. Pedestrian paths
use the existing standalone generator's analytic quintic interpolation, not a new
trajectory engine, ROS mailbox or filesystem actor commands.

Pedestrian peak speed .25–.45 m/s, acceleration <=.4 m/s², path length 2.4–3.4 m.
The path midpoint is timed around 35–55% of client-route duration, with ±.5 s jitter.
Crossing approaches are ±90 degrees; diagonal approaches are ±110–145 degrees;
head-on approaches oppose client travel with signed .6–.8 m lane offsets. These head-on
cases begin as nearby opposite-direction passes, rather than an unavoidable collinear
client–pedestrian collision. Preferred robot clearance can still require deviation.
Both sides, timing, headings, speeds and initial formations vary deterministically.

Only current state is queried during stepping. Encounter timing, anchor locations,
future routes and metadata remain evaluator-only. Stable pedestrian ID `pedestrian_1`.
Start positions and complete straight pedestrian segments respect original flat-scene
bounds/margin. Initial G1–pedestrian centre distance must be >=1 m. Scripted client and
pedestrian trajectories are checked at .05 s, with a speed-based Lipschitz correction
to bound continuous clearance below the sampled minimum. Their surface clearance lower
bound must be >=.05 m. Invalid configurations fail with an explicit reason after bounded
sampling attempts. A stationary dummy client is required by the generator's schema only;
its trajectory is discarded. The actual client always remains the Stage 1 trajectory.

## Environment and causal observations

`Stage2Env` subclasses the existing objective-C environment, retaining its stabilization,
reset, action mapping, physics stepping and endpoint hold logic. Each environment owns
its model/data/gait/controller. It replaces the initial one-proxy runner once during
construction with an independent two-proxy runner, closing the original. This adds a
one-time initialization cost while preserving original Stage 1 source hashes. Resets
thereafter use only the final runner. It never directly moves the G1 during a step.

Pedestrian-free replay places the unused noncolliding proxy below the scene and leaves
all pedestrian slots zero. The first 22 observation entries and action space are unchanged.
Slot 0, indices 22:28, is now:

| Index | Meaning |
|---|---|
| 22–23 | Current pedestrian position relative to pelvis in robot yaw-aligned axes / 16 m |
| 24–25 | Current pedestrian world velocity minus pelvis world velocity, rotated into robot yaw axes / 2 m/s |
| 26 | Pedestrian footprint radius / 1 m, initially .25 |
| 27 | Validity mask, 1 for the current simulator-pose proxy |

Slots 1 and 2, indices 28:40, remain zero/invalid. Observations remain 40 float32 values
in [-1,1]. The actor and critic receive exactly the same causal vector. There is no
future-waypoint, encounter-time, scenario-family, reward or route-progress input.
These are simulator-pose proxies, not vision-tracked people or social responses.

## Clearance and reward

Pedestrians are mocap proxies with collision disabled. **Proxy intrusion is geometric,
not a MuJoCo physical person contact.** Existing G1/ground contacts remain actual physics.
The initial geometry metric uses a .30 m pelvis-centred planar footprint disc for G1
and a .25 m pedestrian disc. This is a footprint surrogate, not a claim of whole-body,
arm/foot mesh clearance or real-human safety.

Every physics callback checks the swept relative-position segment since the previous
sample. This avoids missing a crossing between .2 s high-level observations; it is a
piecewise-linear swept approximation at the physics grid, not exact continuous mesh
collision checking. Log current and episode-minimum surface clearance separately.
An intrusion below zero latches and ends the episode with `pedestrian_intrusion` and
the existing failure penalty. It never produces a person-contact claim.

Objective C remains intact. One explicit additional component is:

`pedestrian_clearance = -2 * dt * clip((0.30 - step_min_surface_clearance) / 0.30, 0, 1)^2`

There is no pedestrian term during replay. It encourages clearance without rewarding
standing still, arbitrary braking or particular actions directly. Formation costs still
apply during yielding; temporary deviation is permitted and recovery is logged. Original
client-loss grace (3 s above 3 m), endpoint recovery (5 s), bounded-motion hold and progress
safeguards remain unchanged. Intrusion clears any simultaneous success bonus and records
failure. Safety, formation, velocity, progress and command-change components remain visible.

## Transition from the selected model

Use the selected guarded Stage 1 5k model in `docs/bodyguard/MODEL_WEIGHTS.md`. The transition
loads and checks its exact original source/config/gait/dependency provenance, restores
weights, critic, optimizer and per-worker/global RNG streams using the tested SB3 2.7.0
adapter, and cancels pending reset seeds/options.

Stage 1's pedestrian input columns never saw nonzero observations. The transition zeros
only the actor and critic first-layer columns 22:28, plus their Adam moment columns.
All other learned connections, heads, variance and optimizer state are preserved. Stage 1
outputs and values are verified exactly unchanged on stored causal observation probes.
This avoids arbitrary reactions to a newly nonzero mask before learning pedestrian skills.
It is an explicitly recorded initialization of unused inputs, not a supervised refit or
an unrecorded bypass of strict Stage 1 resume checks.

The initial Stage 2 archive still has a **5,000 PPO timestep counter inherited from Stage 1**
and **zero Stage 2 training transitions**. It initially ignores the pedestrian; do not call
it a pedestrian-trained policy. A new Stage 2 5k pilot would end at 10k total along this
selected-5k lineage, not 15k. The separately preserved Stage 1 10k experiment is another
branch of the checkpoint history.

Stage 2 source/config hashes and parent SHA-256 are recorded explicitly. All subsequent
Stage 2 resumes require that exact contract. Original Stage 1 source/config hashes remain
unchanged, so old Stage 1 resumes are not invalidated by adding this opt-in package.

## Regression and promotion

**Keep the exact original sixty-case Stage 1 validation suite.** Replacing it with new
random cases would remove the paired before/after regression comparison. Add a separate
fixed Stage 2 suite: twenty crossing, twenty head-on and twenty diagonal episodes, using
new validation seeds starting at 3000001. The original seed namespaces and reserved
high-speed-and-long-client-route final-test combination remain excluded from training
and validation. Final-test cases are not evaluated.

Proposed stricter promotion gates, stored as experimental settings in YAML:

- At least 80% success overall on the balanced single-pedestrian suite AND in each
  encounter family, preventing a strong family from masking a failed one.
- At least 95% overall on the original Stage 1 suite and at least 90% in each old family.
  From a 60/60 parent this allows at most three overall failures, and at most two per family.
- Stage 1 mean formation error <=.15 m AND no more than .05 m above the parent baseline.
- No falls, supervisor or physics failures; no geometric pedestrian intrusions in promotion
  validation. This is deliberately stricter than merely allowing 20% of trials to collide.
- Two consecutive complete passing rounds. An untrained baseline is not a promotion round.

A fresh development batch and additional PPO seed can supplement these checks. They do
not replace the fixed regression suite, and final tests must not be used for tuning.
Clearance, failure reasons, formation/recovery, commands and achieved velocities must be
reviewed together: an 80% escort score alone does not prove braking or yielding was learned.

## Operator commands

Run from the host repository with the existing venv. No ROS or Docker setup is required.

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
TOOL="$PWD/research/escort_rl/stage2_tools/stage2.py"
```

Export one encounter's resolved evaluator metadata, with a fresh output:

```bash
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/stage2_scenario_XXXXXX")
"$PY" "$TOOL" scenario --family crossing --seed 320000 --output "$OUT/crossing"
```

To construct a new explicitly versioned transition (do not overwrite existing archives):

```bash
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/stage2_transition_XXXXXX")
"$PY" "$TOOL" transition \
  --parent research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip \
  --output "$OUT/transition"
```

For the final prepared checkpoint listed in the registry, the following command is an
**operator-launched, maximum 5k pilot**, not an automatically executed milestone step:

```bash
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/stage2_pilot_XXXXXX")
set -o pipefail
"$PY" "$TOOL" train \
  --checkpoint research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip \
  --timesteps 5000 --output "$OUT/pilot_5k" 2>&1 | tee "$OUT/run.log"
printf '\nResults: %s\n' "$OUT/pilot_5k"
```

This preserves guarded PPO lr=1e-5, target_kl=.01 and the two-worker root-seed-42 lineage.
It collects exactly the requested NEW transitions and runs both complete fixed validation
suites afterward. Intrusion task failures are learnable failures; critical physics/fall/
supervisor failures abort with a checkpoint. No automatic longer budget or promotion.
Expect a diagnostic pilot, not a guaranteed 80% result. Low exploration around a previously
pedestrian-blind actor can make discovery difficult; inspect evidence before changing it.

Evaluation-only commands (no optimizer updates):

```bash
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/stage2_eval_XXXXXX")
CKPT="$PWD/research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip"
"$PY" "$TOOL" evaluate --checkpoint "$CKPT" --suite pedestrian --output "$OUT/pedestrian"
"$PY" "$TOOL" evaluate --checkpoint "$CKPT" --suite stage1 --output "$OUT/stage1"
```

For a quick partial diagnostic, `--episodes-per-encounter 2` runs six interaction cases
and explicitly does not satisfy the full-suite gate. Do not repeat the already recorded
full baseline merely to prepare for training.

Plot an existing case (pick an actual episode directory, fresh PNG path):

```bash
"$PY" research/escort_rl/stage2_tools/plot_episode.py \
  --episode "$OUT/pedestrian/000_crossing_3000001" --output "$OUT/crossing.png"
```

## Evidence and current limitations

The readiness experiment is `runs/stage2_readiness_20261107/`. Its fixed-suite pretraining
results and final test counts are recorded below after completion. Training has not been
launched as part of implementation. No Stage 2 success is claimed before training and
independent validation. The full-G1 dynamics/gait have the same unverified live-Grove
numerical-parity qualification as Stage 1.

### Executed readiness results

- **123 tests passed, zero failed/skipped** (91.40 s; fourteen existing third-party
  Matplotlib/pyparsing deprecation warnings). Tests include Stage 1 bitwise physics/
  observation/reward replay, same-seed reset, independent Stage 2 instances, causal slot
  construction, geometry/speed/acceleration bounds, intrusion outcome, Gymnasium checker,
  gate logic and actor/critic/optimizer checkpoint transition. Original 113 tests remain green.
- **Original Stage 1 suite: 60/60**, 20/20 each family, zero intrusions or critical failures.
  Mean formation error **0.05630953682860855 m**, exactly the parent baseline.
- **Untrained interaction baseline: 20/60**: crossing 0/20, head-on 20/20, diagonal 0/20.
  The forty failures are geometric intrusions, not physical person contacts. No fall,
  supervisor or physics failure was recorded. Zero Stage 2 PPO updates.
- Head-on minimum surface clearances ranged **.0423–.2287 m** under the footprint metric;
  these mild opposing-lane passes already succeed but stay inside the preferred .30 m
  clearance margin. Their success is not evidence of learned avoidance. Crossing and
  diagonal failures show that the pedestrian-blind mean needs new behavior.
- Full balanced interaction gate fails, correctly, on success/per-family success/intrusions.
  The parent baseline is not expected to pass a new skill gate before training.
- Final usable archive is **transition_ready/initial.zip** in the registry. Baseline
  evaluation used an earlier export in `transition/` before correcting subsequent
  training-checkpoint bookkeeping. Complete policy/value/optimizer tensors and worker/global
  RNG states were compared and are identical. The final export has current source provenance;
  the earlier export is diagnostic only. No environment, scenario or reward change separated
  the two exports, so the baseline describes the same initial policy and dynamics.
- `crossing_baseline.png`, `head_on_baseline.png`, `diagonal_baseline.png`, per-case traces,
  summaries and `readiness.json` are in the readiness run folder. No Stage 2 training
  throughput or learned success is claimed. The bounded training command remains unrun.

Next step is the explicitly launched 5k diagnostic pilot, followed by BOTH complete suites.
Keep the 40% replay mix initially. Do not change rewards, exploration or budgets merely
because the untrained baseline fails; inspect actual pilot trajectories and optimizer
behavior first. The current conservative discovery settings may need a separately justified
adjustment if the pilot cannot discover yielding. No longer run or Stage 3 is automatic.
