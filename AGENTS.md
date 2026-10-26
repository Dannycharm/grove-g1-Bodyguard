# ECE 598 Predictive Social Navigation

## Objective and platform
- Build and evaluate predictive humanoid escort navigation using learned pedestrian motion and uncertainty.
- Host: Ubuntu 24.04. ROS 2 Jazzy runs in Grove-G1's Docker development container.
- Platform: MuJoCo, Unitree G1, Grove-G1, Nav2, Python, and PyTorch.
- Research configurations: A — reactive distance-threshold escort controller; B — constant-velocity predictive planner; C — GRU trajectory predictor with deterministic planning; D — GRU predictor with calibrated uncertainty and risk-aware planning.
- VLMs, VLAs, RL, and arm gestures are optional extensions, not core deliverables.

## Current milestone
- Original Week 1 robot setup, sensors, TF, odometry, and fixed-goal navigation are working (project-reported progress). Do not repeat completed ROS setup or diagnostics.
- Revised Week 1 still needs a seeded pedestrian scenario generator and trajectory logging. Inspect existing code and assets before implementing either.

## Safety and architecture
- Preserve Grove-G1's existing locomotion, safety, hardware, and DDS architecture; reuse verified navigation and controllers.
- Do not modify low-level G1 motor control, `rt/lowcmd`, the balance policy, or controller ownership. Do not issue physical robot or motor commands.
- Do not bypass Nav2's configured collision-monitor path or create competing velocity-command publishers. Avoid simultaneous Nav2 and direct `/cmd_vel` writers.
- Never automatically launch, start, stop, reset, or reconfigure simulations or controllers. Do not modify running MuJoCo state without explicit approval.
- Preserve a reproducible original simulation scene and existing working assets and scripts.
- Never run Docker commands, `sudo`, destructive shell operations, `git reset`, `git clean`, commits, or pushes without explicit approval.

## Implementation and research integrity
- Inspect existing code before implementing anything. Prefer isolated new research modules/packages over invasive changes to upstream Grove-G1 packages.
- Reuse verified ROS interfaces; confirm actual topics, message types, frame IDs, and installed APIs for new or changed integrations. If inspection requires a restricted action, request approval first.
- Keep simulator ground truth separate from observations supplied to the robot policy; label ground-truth and perception-derived data explicitly in code, interfaces, logs, and documentation.
- Never expose future simulator trajectories to prediction or planning modules; keep evaluation truth isolated from their inputs.
- Maintain explicit coordinate frames, timestamps, track IDs, and observation validity throughout data collection, prediction, planning, and evaluation.
- Use reproducible scenario seeds and configurations. Compare A–D with consistent scenarios and safety mechanisms.
- Avoid adding simulation frameworks unless the current MuJoCo environment cannot meet the requirements.
- Change only files relevant to the requested milestone.

## Verification and reporting
- Every implementation needs unit tests and documented runtime verification. Report actual results, failures, and checks not run; never claim runtime validation from compilation or unit tests alone.
- Explain changes and provide manual ROS 2 verification commands appropriate to the milestone. Providing commands does not authorize running restricted actions.
