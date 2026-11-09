# Selected Stage 2 policy: six full-G1 clips

Open [the video gallery](index.html), or select an MP4 below. Blue is the client;
red is the pedestrian. Playback is approximately **2.5x simulated speed**.
These are representative fixed-validation episodes, not six new independent test
families or training runs. All six completed successfully.

| Scenario | Seed | Short clip | Approximate video duration |
|---|---:|---|---:|
| Crossing | 3000001 | [MP4](crossing/episode.mp4) | 19.5 s |
| Head-on pass | 3000001 | [MP4](head_on/episode.mp4) | 19.5 s |
| Diagonal | 3000037 | [MP4](diagonal/episode.mp4) | 15.2 s |
| Straight replay | 1000001 | [MP4](straight/episode.mp4) | 11.6 s |
| Gentle-turn replay | 1000001 | [MP4](gentle_turn/episode.mp4) | 11.6 s |
| Stop/start replay | 1000001 | [MP4](stop_start/episode.mp4) | 12.4 s |

The diagonal example previously intruded under the BC source; the selected PPO
checkpoint succeeds with minimum surface clearance about 0.0654 m on this case.
It was chosen for explanation, not as an unbiased extra validation sample.

Checkpoint: `research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/final.zip`,
SHA-256 `67692f7d7586093290a9ae49d952b2e2012e4135214db5079713f28c49f0695b`,
5,000 residual PPO transitions. No weights or training state were updated.

The G1 executes full MuJoCo physics through frozen AGILE. Human appearance is
render-only geometry on a separate display copy. Scripted client/pedestrian
proxies do not physically collide; clearance uses the existing swept-disc metric.
The clips do not establish real-person safety or live-Grove dynamics parity.

Companions: `summary.json`, per-family `trace.jsonl`/`summary.json`, poster PNGs,
`contact_sheet.png`, and `video_decode_verification.txt`. Six episodes executed
112,300 active physics steps and 11,230 gait calls; warmup is additional. H.264
encoding used existing system libav libraries; no packages were installed. GIF
intermediates are under `tmp/stage2_media/source_gifs/`, outside final delivery.

Regenerate into a **fresh** directory:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios${PYTHONPATH:+:$PYTHONPATH}"
MUJOCO_GL=egl research/escort_rl/.venv/bin/python \
  research/escort_rl/visualization/stage2_clips.py \
  --format mp4 --output docs/media/stage2_selected_5k_NEW_RUN
```

MP4 encoding requires the already-present `cc`, `pkg-config`, libavcodec,
libavformat, libavutil and libswscale development files, including libx264 encoder
support. The tool installs nothing. `--format gif` is an alternative using Pillow
but produces substantially larger files.

Git archive note: raw `trace.jsonl` and encoder logs remain local; the curated
archive includes videos, posters, summaries and verification reports.
