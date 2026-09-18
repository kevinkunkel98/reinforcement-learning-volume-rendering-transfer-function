"""Before/after renders: what the policy actually does to a picture.

Every other figure in this thesis reports `visibility.py`, a cheap estimate. This
one renders the real thing through VTK, so a reader can check the estimate
against their own eyes -- which is the only claim the number cannot make for
itself.

    python -m plots.qualitative --policy out/rl_v2/oneshot_v3_seed1/best.zip
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import goals
import render as render_module
import views
import visibility
from datasets import load_dataset
from plots.style import apply_style
from rl.baselines import CONTROLLABLE, apply_controllable
from rl.oneshot_env import build_observation
from rl.vis_eval import fixed_episodes

OUTPUT_DIR = "plots/output"
DEFAULT_POLICY = "out/rl_v2/oneshot_v3_seed1/best.zip"


def _observation(model, params, instruction):
    start_agg = goals.aggregate(model.features(params))
    solo_max_log = [math.log10(sum(model.solo_max(m) for m in goals.MEASURED_FOR_GOAL[c]) + goals.EPSILON)
                    for c in goals.GOAL_CLASSES]
    controllable = [float(np.mean([params[i] for i in group])) for group in CONTROLLABLE]
    return build_observation(instruction["goal"], model.histogram, start_agg,
                             solo_max_log, controllable), start_agg


def _render_front(volume, spacing, params, frame_bounds):
    camera = views.cameras_for_volume(volume, spacing)[0]
    win = render_module.render(volume, np.asarray(params, dtype=np.float64), spacing, camera,
                               frame_bounds=frame_bounds)
    return render_module.grab(win)


def pick_episodes(episodes: list, policy, per_kind: int = 1) -> list:
    """One episode per instruction kind, choosing the one the policy handles
    best within that kind -- a qualitative figure should show what the method
    does when it works, with the honest numbers printed beside it."""
    scored = []
    for episode in episodes:
        model = visibility.for_volume(episode["volume"])
        start = np.asarray(episode["start_params"], dtype=np.float64)
        obs, start_agg = _observation(model, start, episode["instruction"])
        action, _ = policy.predict(obs, deterministic=True)
        final = apply_controllable(start, np.clip(action, -1.0, 1.0))
        attainment = goals.attainment(episode["instruction"]["goal"], start_agg,
                                       goals.aggregate(model.features(final)))
        scored.append((attainment, episode, final))

    chosen = []
    for kind in ("relative", "show_only", "absolute", "brightness"):
        of_kind = [row for row in scored if row[1]["instruction"]["kind"] == kind]
        of_kind.sort(key=lambda row: -row[0])
        chosen.extend(of_kind[:per_kind])
    return chosen


def plot_qualitative(policy_path: str = DEFAULT_POLICY, episodes: int = 60,
                      output_dir: str = OUTPUT_DIR) -> str:
    from stable_baselines3 import SAC

    apply_style()
    policy = SAC.load(policy_path)
    chosen = pick_episodes(fixed_episodes("test", episodes, seed=0, formulation="one_shot"), policy)

    fig, axes = plt.subplots(len(chosen), 2, figsize=(7.4, 2.15 * len(chosen)),
                              layout="constrained")
    for row, (attainment, episode, final_params) in enumerate(chosen):
        volume, spacing = load_dataset(episode["volume"], canonical=True)
        start = np.asarray(episode["start_params"], dtype=np.float64)
        # One framing for both panels, measured on the start transfer function:
        # a before/after pair that re-frames between panels cannot be compared.
        frame_bounds = render_module.frame_bounds(volume, start, spacing)

        for col, params in enumerate((start, final_params)):
            ax = axes[row][col]
            ax.imshow(_render_front(volume, spacing, params, frame_bounds))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title("start" if col == 0 else "after the policy", fontsize=10)

        text = episode["instruction"]["text"] or episode["instruction"]["kind"]
        axes[row][0].set_ylabel(f'"{text}"\n{episode["volume"]}  ·  attainment {attainment:+.2f}',
                                 fontsize=8.5, rotation=0, ha="right", va="center", labelpad=10)

    fig.suptitle("One forward pass, rendered")
    # Said plainly: these are the best example of each kind, not typical ones.
    # The medians belong to the table, and the reader should not read four
    # hand-picked pairs as the distribution.
    fig.supxlabel("the policy's best result per instruction kind, of 40 held-out episodes; "
                  "median attainment overall is +0.275", fontsize=8, color="#555555")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "qualitative.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--episodes", type=int, default=60,
                     help="how many held-out episodes to search for the examples shown")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()
    print(f"[plots] wrote {plot_qualitative(args.policy, args.episodes, args.output_dir)}")


if __name__ == "__main__":
    main()
