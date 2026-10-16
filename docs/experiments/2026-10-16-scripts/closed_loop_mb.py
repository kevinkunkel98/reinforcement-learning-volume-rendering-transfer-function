"""Zero-shot closed loop: apply a one-shot policy repeatedly, each time re-observing the
current state with the *remaining* goal (goal minus progress so far), scaled by k.
Scored from the episode's original start. No search, one render per step."""
import collections, sys, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
from rl.model_based import ModelBasedPolicy
import goals
from rl.baselines import apply_controllable
from rl.oneshot_env import build_observation
from rl.oneshot_train import build_eval_env

N = len(goals.GOAL_CLASSES)
env = build_eval_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                     hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
policy = ModelBasedPolicy.load(sys.argv[1])
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 150
T = 4
VARIANTS = [(k, keep) for k in (1.0, 0.7, 0.5) for keep in (False, True)]


def remaining_goal(goal, start_agg, cur_agg, keep):
    c, b = goals.progress(start_agg, cur_agg)
    g = np.array(goal, dtype=np.float64)
    for i, cls in enumerate(goals.GOAL_CLASSES):
        if g[N + i]:
            g[i] -= c[cls]
        elif keep and abs(c[cls]) > goals.KEEP_TOLERANCE:
            g[i], g[N + i] = -c[cls], 1.0      # ask to undo the drift
        if g[3 * N + i]:
            g[2 * N + i] -= b[cls]
    return g


res = collections.defaultdict(lambda: collections.defaultdict(list))
for seed in range(50_000, 50_000 + EPISODES):
    env.reset(seed=seed)
    kind, model = env._instruction["kind"], env._model
    goal, start_agg = env._instruction["goal"], env._start_agg
    solo = env._solo_max_log(env._volume, model)
    for k, keep in VARIANTS:
        params, agg = env._start_params.copy(), start_agg
        best = 0.0
        for t in range(1, T + 1):
            g = remaining_goal(goal, start_agg, agg, keep)
            ctrl = np.asarray(env._controllable_values(params))
            obs = build_observation(g, model.histogram, agg, solo, ctrl)
            a, _ = policy.predict(np.clip(obs, -10, 10), deterministic=True)
            params = apply_controllable(params, np.clip(ctrl + k * np.clip(a, -1, 1), -1, 1))
            agg = goals.aggregate(goals.features(model, params))
            att = env._attainment(agg)
            name = f"k{k}{'+keep' if keep else ''}"
            res[kind][f"{name} t{t}"].append(att); res["ALL"][f"{name} t{t}"].append(att)
for kind, by in sorted(res.items()):
    print(f"== {kind} n={len(next(iter(by.values())))}")
    for k, keep in VARIANTS:
        name = f"k{k}{'+keep' if keep else ''}"
        print(f"  {name:10s} " + "  ".join(f"t{t} {np.median(by[f'{name} t{t}']):+.3f}" for t in range(1, T + 1)))
