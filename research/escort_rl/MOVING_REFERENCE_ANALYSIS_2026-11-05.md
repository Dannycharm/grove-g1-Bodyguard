# Moving Grove reference: analysis and next step

**Conclusion:** the supplied moving recording supports the same approximately **0.40–0.41 m/s
forward response to a 0.60 m/s command** observed in the isolated frozen-AGILE runner. This
closes the missing moving-reference sanity check. It does not establish a hard physical
maximum or exact full-trajectory parity. **No additional equivalent operator bag is requested.**

## Evidence actually analyzed

Input: `data/escort_logs/moving_reference_20261105T044700_443290167/`, including the original
MCAP, existing robot-state export, parameter/endpoint/checkpoint sidecars, and
`data/escort_logs/reference_motion_20261105T051139_827178917.log`.

Analysis opened the bag offline and decoded the raw-command and ROS log channels omitted
by the existing exporter. It used the existing ROS-capable `/usr/bin/python3`, NumPy and
matplotlib; the isolated venv lacks PyYAML required by the ROS import, so no dependency was
installed or altered. No ROS node, live command, simulator action, reset or container lifecycle
operation was performed. Inputs were hashed before/after and remained unchanged.

Fresh outputs: `runs/moving_reference_analysis_20261105/analysis.json`, `provenance.json`,
`extra_channels.json`, `timing_sensitivity.json`, `tests.txt`, `time_series.png`, `route.png`.
New reusable offline analysis is `tools/analyze_moving_reference.py` with pure conditioning
helpers in `escort_rl/moving_reference.py`. Added three meaningful statistical/causality tests;
**30 tests passed, zero skipped**. Original gait, runner dynamics, controllers, assets and
command caps were not changed.

## 1. Navigation completed, with one recovered local-control failure

The operator action log reports **six SUCCEEDED goals**, with error_code=0: three outbound
and three return goals. The independent bt_navigator log also reports six successes.

There were **144 Nav2 controller-loop missed-rate warnings**. Among those warning events the
reported rate's median was 9.93 Hz versus a desired 20 Hz. This is a Nav2 local-control timing
finding; it is not evidence that the 50 Hz AGILE policy itself ran at 10 Hz.

The last return included 14 collision-ahead warnings, one controller-patience error and one
FollowPath abort. Recovery/replanning continued and the overall navigation goal succeeded.
Thus this was successful navigation with a recovered intermediate failure, not six flawless
local-control executions. Goal success uses Nav2's configured tolerances and is not escort
formation/safety success.

## 2. Conditioned forward response agrees with the isolated result

The whole-recording speed average is not a gait tracking metric: this recording includes
long periods of standing, turning and stopping. Likewise, a maximum finite-difference speed
of 0.630 m/s or a native pelvis-vx peak of **0.660 m/s** is a transient sample, not proof that
G1 can sustain a 0.50 m/s client speed.

To compare relevant conditions, select native pelvis-origin/body-axis velocity samples where
there was a recently recorded `/cmd_vel` forward request of 0.60 m/s, lateral request below
0.01 m/s and a small yaw request. Association is causal, using base-state mapped capture stamps
and post-monitor Twist recorder-arrival timestamps; samples older than the controller's
0.5 s timeout are excluded. These are **candidate-command associations**, not a newly observed
exact controller-consumption trace. Means are weighted by measured sample intervals; gaps
above 50 ms would reset the continuous-window eligibility.

Sensitivity to declared straightness and minimum continuous-condition time:

| Maximum absolute yaw command | Prior continuous condition | Eligible duration | Mean forward velocity |
|---:|---:|---:|---:|
| 0.05 rad/s | 0.5 s | 3.40 s | 0.3972 m/s |
| 0.10 rad/s | 0.5 s | 10.67 s | 0.4036 m/s |
| 0.10 rad/s | 1.0 s | 5.63 s | **0.4074 m/s** |
| 0.20 rad/s | 1.0 s | 16.30 s | **0.4064 m/s** |
| 0.30 rad/s | 1.0 s | 28.22 s | 0.4074 m/s |

The very strict 0.05 rad/s, 1 s filter leaves only 0.79 s of tail data and gives 0.3887 m/s;
it is reported in the JSON rather than hidden. These are exploratory navigation-condition
filters, not standardized open-loop speed trials or independent statistical replicates.

For comparison, the corrected isolated runner's 20 s hold trials produced a final-10-second
mean of **0.40648 m/s** at a 0.60 m/s command. The selected reference mean 0.40744 differs by
about **0.0010 m/s**. Agreement of response means is useful evidence that the earlier
undertracking is not merely a COM measurement error or an obvious Python-port artifact.
It is **not** proof of pointwise dynamics parity.

Using the 0.10 rad/s / 0.5 s filter separately within each navigation goal gives means
0.4041, 0.3826, 0.3980, 0.4101, 0.4004 and 0.4175 m/s. Eligible durations differ substantially
(0.36–2.69 s), so these are not equal-length independent command trials or confidence bounds.

An exploratory +/-0.2 s command-time shift sensitivity gives conditioned means ranging from
0.3971 to 0.4074 m/s. This does not calibrate latency or establish an uncertainty bound, but
the approximately 0.4 m/s conclusion is not dependent on one exact assumed timestamp offset.

## 3. Raw versus post-monitor commands

The bag contains 1,428 raw and 767 post-monitor Twist messages. Both streams' forward requests
only take values 0 and 0.60 m/s; there is no observed intermediate forward scaling value.
For 761/767 post messages, identical raw command content occurred within the preceding 200 ms.

The count difference **must not be interpreted as 661 transport losses or a 46% velocity
reduction**. Streams have different publication behavior and potentially different recording/
processing timing. Content matching without IDs cannot measure exact forwarding latency.
The matched-content time differences are not reported as a physical response latency.

Crucially, the selected response samples use the **post-monitor 0.60 request**, so this
particular sustained-speed shortfall is not explained by the recorded outgoing command
already being reduced to 0.4 m/s. This is still a received-topic request rather than final
motor or policy-consumption telemetry.

`/collision_monitor_state` has **zero recorded messages**. That does not prove that the
monitor was disabled, broken or never intervened; it means no monitor-status events are
available for attribution. Raw/post command data and Nav2's own collision warnings remain
usable. There is no reason to repeat the same bag solely to obtain a topic that did not emit.

The endpoint snapshot advertises two `/cmd_vel` publishers: collision_monitor and
`g1_base_approach`. Two advertised publishers do not prove concurrent active command writes,
but this bag cannot attribute each Twist to its publisher. Exclusivity is not fully verified
by endpoint count alone. Do not add a third publisher; future capture design should resolve
ownership explicitly without changing the G1 policy.

## 4. Stability and stopping

Across 42,486 native base-state samples:

- Minimum pelvis height: **0.7148 m**.
- Minimum upright cosine: **0.99733**.
- No samples crossed the existing fall thresholds (height <0.45 m or upright cosine <0.5).
- Largest observed gait-joint speed: **22.02 rad/s**; largest mean absolute speed across the
  14 gait joints: **3.45 rad/s**, below configured peak/mean thresholds of 35/10 rad/s.

These are sampled stability observations, not physical-contact accounting or an observed
emergency-latch trace. Neither internal latch status nor exact motor targets/gains was recorded.
The native yaw signal has sizable gait/turn transients; instantaneous overshoot is not the
same as sustained yaw tracking or a violation of a software command clamp.

Six completed navigation zero-request episodes reached the 0.05 m/s planar / 0.05 rad/s body-z
threshold for a continuous 0.5 s. Approximate start-of-dwell times after the zero request were
**0.19–0.64 s**, with **0.013–0.057 m** additional planar path; confirmation took another 0.5 s.
Three short zero episodes during the recovery were interrupted before a full stop could be
confirmed and remain censored, not counted as successful stops.

These navigation stops have varying preceding translation/turn states and approximate clock
alignment. They are not matched full-speed stopping tests and cannot pass a stopping-parity gate.

## 5. Timing/data quality and identity

The base-state mapped-header intervals have median 4.40 ms, p95 6.20 ms, p99 10.01 ms and
maximum 35.80 ms. Timestamps were strictly increasing; no base-state gap exceeded 50 ms.
Arrival-minus-header delay was p50 0.82 ms and p95 1.92 ms, but these are relay-mapped stamps,
not a direct measurement of total native-simulation-to-controller latency. The existing export
reports about 2.35 ms variation in relay/native offset.

The single transport-loss warning does not invalidate the entire recording. Actual stream
gaps, content and conditioning are analyzed rather than assuming ideal rates or imputing data.

Saved controller parameters show active AGILE/safety controllers, decimation 4, update_rate
200, timeout 0.5 s, linear cap 0.6 and yaw cap 0.9. `model_path` is empty, selecting package
share; the recorded installed checkpoint-file hash matches the frozen checkout exactly:
`8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b`.
This is configured-path/on-disk identity evidence, not a new in-memory hash measurement.

The actual reference uses the navigation actor bundle. The established isolated speed suite
uses the original flat scene. Initial state, gait/hold history, exact physics/controller event
phase and final targets/gains are not matched. The full-dynamics-parity flag therefore remains
false for valid reasons; it is not evidence that the bag collection failed.

## 6. Decision and next milestone

**Moving-reference response sanity check: passed within this stated scope.** Both systems
show roughly 0.4–0.41 m/s forward response at the unchanged 0.6 m/s request, with stable posture
in the recorded conditions. **Exact dynamics parity: still open.** Previous Python/C++
ten-output inference parity and deterministic isolated resets remain valid.

Recommended next work is **offline matched-scene / recorded-command replay and residual
analysis using this existing dataset**, not more operator round trips:

1. Load the exact recorded bundle/compiled physical configuration in an isolated instance,
   without editing original assets or touching the live stack.
2. Resolve recorded initial state and a documented settled-start/history reconstruction;
   quantify the unobserved-state uncertainty rather than silently pretending it is a cold reset.
3. Replay the recorded post-monitor requests with declared timing variants and unchanged
   gait, caps and physics; compare response distributions, base/joint traces and stops.
4. Use those residuals to determine whether the synchronous approximation is acceptable or
   whether one targeted passive final-motor/native-time trace is necessary. Only then specify
   a purpose-built capture; do not request another equivalent standard ROS bag for hidden signals.

No operator action is needed now. The data collection requested for this check is complete.
No Gymnasium/PPO implementation or live instrumentation is silently included in this analysis.

Retain AGILE for a **provisional Stage 1 client range of 0.15–0.25 m/s**, subject to backend
acceptance and actual escort feasibility tests. A fixed requirement for sustained
0.45–0.50 m/s clients now has a concrete locomotion-capability concern in both tested systems;
that would justify a separately scoped alternative frozen-gait comparison, not raising caps
or assuming that a physics-backend switch solves it.

## Reproduction

From the host repository with the existing sourced ROS environment:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
PYTHONPATH="$PWD/research/escort_rl:$PYTHONPATH" /usr/bin/python3 \
  research/escort_rl/tools/analyze_moving_reference.py \
  --reference data/escort_logs/moving_reference_20261105T044700_443290167 \
  --motion-log data/escort_logs/reference_motion_20261105T051139_827178917.log \
  --output research/escort_rl/runs/moving_reference_analysis_002
```

Use a new output path. The original bag, export and motion log are never overwritten.
The analyzer uses only offline SequentialReader/deserialization, not a ROS graph.

[Velocity/time-series plot](runs/moving_reference_analysis_20261105/time_series.png)
and [native-world route](runs/moving_reference_analysis_20261105/route.png).
