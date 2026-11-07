# Stage 1 Git archive scope

Prepared 2026-11-07 on `bodyguard/week1-validation`, before Stage 2 development.

## Included in this checkpoint

- `AGENTS.md` and the editable [model-weight registry](MODEL_WEIGHTS.md).
- `research/escort_rl/` source, configurations, tests, command-line tools, isolated
  visualization player and milestone reports.
- The two learning/playback HTML guides and their final PDFs under `output/pdf/`.
- Every model archive linked by the registry, including selected 5k, latest 10k,
  original imitation, prepared initial and historical diagnostic controls. Available
  adjacent checkpoint manifests and RNG sidecars are preserved at their original paths.
- The imitation source manifest, causal demonstration dataset and supervised-fit report
  required by the pinned actor-initialization configuration.
- Critic preparation dataset/fit, measured noise calibration and guarded calibration
  derivation, where linked or required for the current preparation workflow.
- Compact run results, available initial/final validation summaries, training episode
  summaries, worker scenario records, optimizer CSV logs and selected comparison plots.

These curated artifacts were explicitly added despite `research/escort_rl/runs/` being
ignored. Their weight files are small ordinary Git blobs; no new LFS configuration is
required. Future useful artifacts still need deliberate selection and registry updates;
the ignore rule does not automatically include new checkpoint files.

## Remains local

- The Python venv, downloaded development dependencies, caches and compiled intermediates.
- Bulk experiment directories: most per-step raw rollouts, full per-episode evaluation
  traces, TensorBoard event streams, unused periodic checkpoints and bulk benchmark data.
- The three generated playback GIFs (roughly 45 MB each). The HTML guide's sample GIF
  links refer to the original development checkout. A fresh clone can generate the clips
  using the guide's host commands; the GIFs are not bundled in this Git checkpoint.
- Temporary document rendering and visual-QA images under `tmp/`.
- Pre-existing untracked `research/escort_baseline_a/` and
  `docs/bodyguard/PROJECT_PROGRESS_AND_REPRODUCTION_2026-10-29.docx`, which were not part
  of the Stage 1 RL work being committed. They remain untouched on the development host.

The full local RL run directory was approximately 3.2 GB. This commit is a curated
reproducibility and model archive, not a backup of every experiment file. Preserve a
separate full-data backup if complete original trajectory evidence is required. A fresh
clone can inspect the included summaries and reproduce new evaluations; it cannot claim
the excluded raw trajectories were recovered from Git.

## Reuse

Start with [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) to choose the correct archive and its
intended inference, initialization or resume path. Keep manifests and RNG sidecars next
to their ZIP files. Do not modify historical provenance to make an incompatible resume
pass. Use the existing environment pins and retrieve simulator submodules/assets through
the repository's documented setup workflow; a checked-in checkpoint does not install
MuJoCo, PyTorch or ROS.

No Stage 2 code, training run, ROS graph action or simulator change is part of this Git
milestone. Full numerical live-Grove dynamics parity remains unverified.
