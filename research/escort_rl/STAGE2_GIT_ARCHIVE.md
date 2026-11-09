# Stage 2 Git archive

This commit preserves the implemented one-pedestrian environment, diagnostics,
residual BC/DAgger and PPO adapters, tests, methodology and demo videos on
`bodyguard/week1-validation`. No runtime/control source or frozen gait is changed
by archiving. No further training or stage advancement is implied.

Selected checkpoint:
`runs/residual_ppo_recovered_9htyqV/pilot/final.zip`, SHA-256
`67692f7d7586093290a9ae49d952b2e2012e4135214db5079713f28c49f0695b`.
Its ZIP/manifest/RNG companions, selected best archive, initial archive, required
v3 residual BC parent and sidecar, original prepared critic, compact training
metrics and both full paired validation summaries are included. The preceding
Stage 1 commit already includes the selected frozen base and its companions.
Teacher readiness data is included for the teacher audit. See the model registry
for exact lineage and meaning of update counters.

Ignored `runs/` remains ignored by default. Only explicit curated artifacts are
force-added. Large raw per-step logs, full trace trees, most historical failed
weights, TensorBoard events, `.venv`, `.deps`, caches and rendering intermediates
are not part of this archive. Historical registry references remain useful local
records; they are not a guarantee every diagnostic weight is in Git.

The selected checkpoint can be loaded/evaluated with its archived dependencies
and matching runtime. Fresh BC-to-PPO preparation's action-import verification
requires the source v3 validation `trace.jsonl` trees, which remain local and are
not replaced with summaries. Other historical diagnostic workflows can likewise
require their original local datasets. Historical manifests contain absolute
paths; moving to another checkout root needs an explicit provenance-aware
relocation procedure. Do not bypass source/config/hash checks.

Six MP4s, posters and their compact summaries under
`docs/media/stage2_selected_5k_20261109/` are archived. Their raw per-step media
traces and encoder logs remain local. No external assets or live ROS recordings
are included. A pre-existing Stage 1 PDF rename, an unrelated project DOCX and
paused baseline package changes are outside this Stage 2 commit.

Pre-push verification: full RL plus scenario-generator suite **182 passed**, zero
failures/skips, 14 third-party deprecation warnings, 107.61 s. The actual selected
checkpoint loaded with strict provenance checks and its frozen-base assertion
passed. Required BC parent/base artifacts are indexed. Staged whitespace checks
pass with existing CSV CRLF line endings preserved.
