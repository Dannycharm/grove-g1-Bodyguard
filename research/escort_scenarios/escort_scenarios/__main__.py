"""Offline export CLI. No ROS, MuJoCo, controller, or wall-clock interaction."""

import argparse
import sys

import yaml

from .generator import generate, load_config


def plot_xy(scenario, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    for actor in scenario.actors:
        states = [actor.at(t) for t in scenario.sample_times()]
        line, = ax.plot([s.x for s in states], [s.y for s in states], label=f"{actor.actor_id} ({actor.role})")
        ax.scatter(states[0].x, states[0].y, color=line.get_color(), marker="o")
        ax.scatter(states[-1].x, states[-1].y, color=line.get_color(), marker="x")
    bounds = scenario.resolved_config["bounds"]
    ax.set(xlim=bounds["x"], ylim=bounds["y"], aspect="equal", xlabel="x (m)", ylabel="y (m)",
           title=f"Evaluation ground truth: {scenario.resolved_config['family']}\n"
                 f"seed={scenario.resolved_config['seed']}; frame={scenario.resolved_config['frame_id']}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", required=True, help="New or empty ground-truth output directory")
    parser.add_argument("--seed", type=int, help="Explicit override recorded in exported config")
    parser.add_argument("--plot", action="store_true", help="Add XY PNG; requires matplotlib")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.seed is not None:
            if not isinstance(config, dict):
                raise ValueError("Scenario must be a YAML mapping")
            config["seed"] = args.seed
        scenario = generate(config)
        scenario.export(args.output)
        if args.plot:
            from pathlib import Path

            plot_xy(scenario, Path(args.output) / "xy.png")
        print(f"Exported {len(scenario.actors)} actors, {scenario.duration_s:.3f} simulation seconds to {args.output}")
    except (ValueError, OSError, yaml.YAMLError, ImportError) as error:
        print(f"Scenario error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
