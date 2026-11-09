# Stage 2 teacher imitation fit review

Evidence: `runs/stage2_teacher_fit_Wn3DvT/bootstrap/`. Reviewed existing outputs
and queried the causal teacher on saved student observations; no new fitting,
rollouts or PPO were launched for this analysis.

## Outcome

The 3,000-update supervised fit failed its pedestrian validation gates. It did not
save `prepared.zip` (`result.json: saved=false`). No original model was overwritten.
The selected parent retains 5,000 Stage 1 PPO transitions, zero Stage 2 PPO updates;
supervised updates and 10,878 demonstration transitions are separate budgets.

| Fixed validation | Successful teacher | Fitted student |
|---|---:|---:|
| Crossing | 20/20 | 0/20 |
| Head-on | 20/20 | 20/20 |
| Diagonal | 20/20 | 0/20 |
| Geometric intrusions | 0 | 40 |

The student also passed the original Stage 1 suite, 60/60, with mean formation
error 0.08030 m versus selected-parent 0.05631 m (increase about 2.4 cm, within
configured tolerance). All recorded validation failures were geometric pedestrian
intrusions, not physical person contacts. No fall/supervisor/physics failure was
recorded. Stage 2 promotion remains rejected.

## Distillation evidence

Balanced holdout action MSE fell from 0.02300 at update 100 to 0.007252 at update
3,000, the selected and final update. The latter is normalized-action RMSE about
0.0852, not a task-success measure. Critic and log sigma were unchanged; this was
supervised actor fitting, not PPO optimization. Reward changes cannot explain the
supervised update directly because rewards were not the fitting objective.

Only 698/10,878 labels (6.4%) differed from the original follower by more than 0.15
in normalized three-action L2 norm: crossing 294/1,798 (16.4%), diagonal 270/1,798
(15.0%), head-on 134/1,798 (7.5%), and zero in Stage 1 families. This operational
threshold identifies substantial interventions, not every small avoidance action.
Equal-family batches preserve Stage 1 replay but do not balance avoidance and
ordinary following within encounter episodes. Average MSE can improve mainly on
common behavior while failing rare, decisive maneuvers.

On the final two seconds of the failed student cases, replaying the causal teacher
on the SAME saved observations gives:

| Physical command statistic | Crossing | Diagonal |
|---|---:|---:|
| Student mean forward command (m/s) | +0.241 | +0.247 |
| Queried teacher mean forward command (m/s) | -0.156 | -0.278 |
| Student mean absolute lateral command (m/s) | 0.0134 | 0.0126 |
| Queried teacher mean absolute lateral command (m/s) | 0.424 | 0.360 |
| Normalized student-to-teacher action MSE | 0.429 | 0.459 |
| Normalized student-to-original-follower action MSE | 0.0218 | 0.0171 |

The late-state teacher queries are labels, not executed counterfactual trajectories;
they do not prove recovery was still physically possible at those states. They do
show the student continued following rather than learning the teacher's pronounced
avoidance response. The actor keeps issuing forward commands; this result does not
establish a gait refusal to stop.

## Interpretation and recommendation

The next bottleneck is teacher-to-student transfer. The evidence supports rare-event
imbalance and is consistent with covariate shift: a teacher visits successful
avoidance states while the student enters different, increasingly unsafe states.
Covariate shift is not isolated causally by this analysis. Dataset breadth is also
limited to ten seed ranges paired across six families. A larger MLP or more
optimization might help, but neither is established as the primary fix; the fitting
loss was still decreasing at the budget boundary.

Do not launch PPO or another long continuation from this unsaved failed fit.
Recommended next bounded experiment:

1. Keep the selected Stage 1-derived parent, same MLP, rewards, dynamics and suites.
2. Balance meaningful intervention versus ordinary-following labels inside pedestrian
   families, retaining Stage 1 replay. Log per-family/per-intervention fitting and
   whole-episode holdout errors, especially braking/lateral-command errors.
3. If the student remains weak on closed-loop cases, use a small DAgger-style
   development collection: run the current student in isolated full-G1 physics and
   query the causal teacher on encountered observations; explicitly log any teacher
   execution/intervention schedule. Add those labels to the dataset. Do not use
   fixed validation cases as training labels or privileged future state.
4. Evaluate both original suites after each bounded supervised round. Save diagnostic
   candidates separately with full lineage, without presenting them as approved PPO
   initializations. The current failed fit saved no weights, so it cannot be directly
   continued; any new fitting run must reproduce/start explicitly from the parent.

Do not silently change the original fitter/dataset or lower readiness gates. This
is an opt-in versioned distillation change, not a reward redesign. Successful
teacher imitation must be demonstrated before critic preparation or PPO resumes.
Final-test data and live Grove dynamics parity remain untouched/unverified.
