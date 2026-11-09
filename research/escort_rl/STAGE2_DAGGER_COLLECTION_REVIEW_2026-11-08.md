# First student-state DAgger collection review

Evidence: `runs/stage2_dagger_2OalKX/data/`. This review performs read-only dataset
and trace checks, not fitting or another rollout.

## Collection integrity

- All 18 configured episodes completed, three per all six families.
- 2,770 transitions, 277,000 active physics steps and 27,700 gait calls.
- Every recorded transition executed 100 physics steps and ten gait evaluations.
- Zero teacher-executed actions; every executed action equals the recorded student
  action. Original command caps are respected. No hidden teacher overrides.
- 82.23 s collection wall time (about 33.69 transitions/s, including collection
  resets/logging); this is collection throughput, not PPO training throughput.
- Dataset and episode hashes pass the strict loader. Seeds are in the training
  partition, family/seed cases do not overlap the original demonstrations, and
  reserved final-test route combinations are rejected by the loader.
- All original and new observations/actions are finite float32 with shapes 40/3.
  Teacher labels remain causal commands, not rewards/returns or future route state.

## Student outcomes

Straight, gentle turns, stops and head-on: 3/3 successes in each family.
Crossing and diagonal: 0/3 each, all six ending in geometric intrusion.
Total: 12/18 success, six intrusions, no recorded critical termination.
These are only three development cases per family; head-on 3/3 does not supersede
the earlier fixed-suite 14/20. Failure is expected here: the goal was to query the
teacher on actual student states, not demonstrate an improved student. No weights
were updated and there is no new policy-performance result.

## Ready for one aggregated fit

New labels contain 171 substantial interventions out of 2,770 rows, using the
unchanged normalized action-distance threshold 0.15. Combined data:

- 13,648 rows, 869 substantial interventions.
- 10,584 fitting rows; 3,064 whole-episode holdout rows.
- Every required fitting stratum is populated. The original source's holdout stays
  held out; no fixed validation trajectory is included as fitting data.

Recommend one 3,000-update actor-only aggregated fit, keeping the same v3 sampler,
architecture, teacher, rewards, Stage 1 replay and fixed validation suites. Refit
from the SAME verified Stage 1-derived parent; do not treat the diagnostic NPZ as a
PPO resume archive. Compare with the weighted-only control before any further
changes. Preserve both validation gates and inspect Stage 1 formation regression.
The hypotheses are improved student-state coverage and transfer, not that collection
itself learned avoidance. Do not launch PPO or relax gates on the basis of this batch.

Dataset SHA-256:
`c0933491146bed9b2d432c7bc5e80b991ccc43aaf23d310e6126d43301e6ee5b`.
Original archives and final-test data remain untouched. Live Grove dynamics parity
remains unverified; geometric proxy intrusions are not physical person contacts.
