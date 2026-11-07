# Guarded imitation-pretrained PPO: 5k preservation result

**PASS. Continue with one reviewed 5k resume to 10k total; do not jump to 100k.**

Source: `runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/`.
This analysis read the saved checkpoints, configuration, source hashes, logs, scenario
records, RNG sidecars and complete initial/final sixty-case trajectories. It executed no
training, dynamics rollouts, ROS/DDS commands or final-test evaluation. New analysis
artifacts are in the run's `analysis/` folder. No working training/environment source changed.

## Results on identical strict validation cases

| Metric | Prepared imitation actor | Guarded PPO, 5k |
|---|---:|---:|
| Overall success | 60/60 | **60/60** |
| Straight | 20/20 | 20/20 |
| Gentle turns | 20/20 | 20/20 |
| Stop/start | 20/20 | 20/20 |
| Mean formation error | .05906 m | **.05631 m** |
| Mean terminal error norm | .02015 m | .02442 m |
| Mean completed stop hold | 1.0 s | 1.0 s |
| Mean terminal planar body speed | .00339 m/s | .00266 m/s |
| Mean return | 30.0229 | 30.1416 |
| Falls / supervisor / physics failures | 0 / 0 / 0 | 0 / 0 / 0 |

Final signed terminal error mean: (-.01555,+.00689) m in client heading axes relative
to the desired [-1.5,0] m offset. The earlier metre-scale forward formation shift is absent.
Mean formation improved 2.75 mm (4.66%), while terminal error increased 4.27 mm. These are
small paired differences on one reused validation suite, not robust evidence of PPO
improvement. Successful behavior was learned through imitation before PPO began.

All **24 completed stochastic training episodes succeeded**. Both workers sampled 13
scenarios, 26/26 unique episode seeds globally, with disjoint worker seed sets. Family
counts: sixteen straight, five gentle-turn, five stop/start. This finite sample is compatible
with the configured mixture and is not a reason to retune sampling. No training critical
failures, Gaussian action clipping or observation clipping were logged.

## Optimization diagnostic

Compared with the failed unguarded warm start, only subsequent PPO learning rate
(.0003 -> .00001) and target_kl (None -> .01) changed. The prepared initial actor and critic
fit reproduce the prior preparation exactly. This comparison tests the two guards together;
it does not isolate each one's individual contribution.

Three optimization blocks reported KL **.011535, .004788, .005205**, versus approximately
295.19, 2.45 and 11.78 unguarded. The log records one target-KL early stop in the first block
at epoch index nine (`max kl: 0.02`, rounded logging). There were **194 actual Adam steps**,
so this is real optimization rather than an unchanged or accidentally frozen actor.

PPO clipping fractions .0756, .0290, .0516. Final value loss 2.43; explained variance .2886,
up from .1771 on the first collected rollout. Critic-only preparation heldout EV remains
.1306 and MSE 10.199 versus constant 11.712; value learning is still imperfect.

On the same 2,048 causal training-observation probes, final action-mean RMS drift per axis
is **(.00807,.00530,.00126)** normalized units. Combined MSE **.00003161**, versus .8286 in
the failed unguarded run. Gaussian sigmas remain approximately (.04999,.05000,.04998).
The guard plus lower step size successfully preserved the working controller at this budget.
The PPO clipped objective alone did not do so previously.

This strongly supports excessive optimization movement as an important proximate cause
of the previous warm-start collapse. It does not clear the reward of every possible issue,
prove critic capacity sufficient, or explain all historical cold-start learning failures.
There is no current reason to enlarge the actor or change the objective on this result.

## Integrity and throughput

All current source, configuration, dependency, reward, gait and reference-manifest hashes
match the saved run. Loaded checkpoint confirms lr=1e-5, target_kl=.01, 5,000 timesteps and
nonempty optimizer state. RNG sidecar matches 5,000 timesteps, two workers and 13 sampled
episodes per worker. Resume-compatible under the unchanged sources/configuration.

Exactly **5,000 transitions / 500,000 active physics steps / 50,000 gait calls**.
Training wall **74.776 s**, **66.87 transitions/s**, optimizer wall .316 s, RSS ~2.66 GiB.
Preparation, evaluation, plotting and startup are excluded from this training rate. The
trainer's CPU-seconds field includes final parent-process validation; do not interpret it
as training-only utilization. Full numerical live Grove dynamics parity remains unverified.

## Recommendation

1. Resume `final.zip` for **5,000 additional transitions** with the identical configuration,
   seed and two workers. This tests cumulative drift and actual process restart/RNG
   continuity, reaching 10k total. Do not reinitialize from imitation or refit the critic.
2. Require a second post-PPO passing full validation: >=90% overall, >=80% per family,
   no critical failures and mean formation <=.15 m. Also inspect signed formation,
   endpoint holds, KL, action drift and stochastic training outcomes. Two consecutive
   passing PPO rounds are stronger preservation evidence; initial BC validation does
   not count as a PPO-trained round.
3. If that passes, verify another model/training seed and compare against frozen BC on
   matched development scenarios before a larger budget. Stage 1 is already at a success
   ceiling; 100k more samples have no demonstrated benefit. Additional budget should test
   robustness or a defined improvement, not merely increase timestep count.
4. Preserve the BC-only comparator and label this method imitation-pretrained PPO.
   Do not claim PPO learned escorting from scratch or automatically promote curriculum.

Exact next command (not executed during this analysis):

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/research/escort_rl/.venv/bin/python"
OUT=$(mktemp -d "$PWD/research/escort_rl/runs/imitation_guarded_resume_$(date +%Y%m%d_%H%M%S)_XXXXXX")
set -o pipefail
"$PY" research/escort_rl/tools/imitation_ppo.py train \
  --config research/escort_rl/configs/ppo_stage1_imitation_guarded_v1.yaml \
  --resume research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip \
  --seed 42 --workers 2 --timesteps 5000 \
  --output "$OUT/continuation_10k" 2>&1 | tee "$OUT/run.log"
printf '\nResults: %s\n' "$OUT/continuation_10k"
```

`--timesteps 5000` means 5k NEW transitions, producing 10k TOTAL. Resume retains weights,
optimizer and worker RNG streams, restarts active physical episodes, and does not repeat
critic preparation/noise calibration. Strict provenance failures must be investigated;
do not bypass them. Full initial/final validation still runs. No ROS or Docker is needed.
