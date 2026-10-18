"""Per-kind gap: model-predicted vs real attainment for the policy action, and for a
500-eval hill_climb action (does the model even know a good action exists?)."""
import collections, json, sys
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, ".")
import goals, datasets
from rl import model_based as mb
from rl.baselines import hill_climb, apply_controllable
from rl.oneshot_env import OneShotEnv

ckpt, out, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
policy, ens = mb.ModelBasedPolicy.load(ckpt), mb.load_dynamics(ckpt)
env = OneShotEnv(datasets.volumes_for_split("val"), **mb.ENV_KWARGS, task_version="v2")
N = mb.N_CLASSES


def predicted(env, obs_ctrl, action, inst, start_agg):
    s0 = torch.as_tensor(mb.state_vector(start_agg))[None]
    hist = torch.as_tensor(env._model.histogram, dtype=torch.float32)[None]
    solo = torch.as_tensor(env._solo_max_log(env._volume, env._model), dtype=torch.float32)[None]
    ctrl = torch.as_tensor(obs_ctrl, dtype=torch.float32)[None]
    a = (ctrl + torch.as_tensor(action, dtype=torch.float32)[None]).clamp(-1, 1) - ctrl
    with torch.no_grad():
        delta = ens(torch.cat([hist, solo, ctrl, s0], 1), a).mean(0)[:, :mb.STATE_SIZE]
    w = inst.get("weights") or {}
    wt = torch.tensor([[w.get(c, 1.0) for c in goals.GOAL_CLASSES] + [w.get("other", 1.0)]])
    g = torch.as_tensor(np.asarray(inst["goal"]), dtype=torch.float32)[None]
    d0 = mb.goal_distance(g, wt, torch.zeros_like(delta))
    return float(1 - mb.goal_distance(g, wt, delta) / d0.clamp_min(1e-6)), delta[0].numpy()


rows = []
for seed in range(lo, hi):
    obs, _ = env.reset(seed=seed)
    inst = env._instruction
    if inst["kind"] not in ("absolute", "show_only", "relative"):
        continue
    start_agg, start = env._start_agg, env._start_params.copy()
    ctrl = np.asarray(env._controllable_values(start))
    action, _ = policy.predict(obs)
    kw = {"weights": inst["weights"]} if inst.get("weights") is not None else {}
    real_params = apply_controllable(start, np.clip(ctrl + action, -1, 1))
    real_agg = goals.aggregate(goals.features(env._model, real_params))
    real = goals.attainment(inst["goal"], start_agg, real_agg, **kw)
    pred, pdelta = predicted(env, ctrl, action, inst, start_agg)
    found = hill_climb(env._model, start, inst, evaluations=300, initial_step=0.2)
    hc_agg = goals.aggregate(goals.features(env._model, found))
    hc_real = goals.attainment(inst["goal"], start_agg, hc_agg, **kw)
    hc_action = np.asarray(env._controllable_values(found)) - ctrl
    hc_pred, _ = predicted(env, ctrl, hc_action, inst, start_agg)
    real_delta = mb.state_vector(real_agg) - mb.state_vector(start_agg)
    rows.append({"seed": seed, "kind": inst["kind"], "real": real, "pred": pred,
                 "hc_real": hc_real, "hc_pred": hc_pred,
                 "targets": {c: t for c, t in inst["targets"].items()},
                 "occluders": inst.get("occluders", []),
                 "vis_err": float(np.abs(pdelta[:N + 1] - real_delta[:N + 1]).mean()),
                 "bright_err": float(np.abs(pdelta[N + 1:2 * N + 1] - real_delta[N + 1:2 * N + 1]).mean())})
    json.dump(rows, open(out, "w"), indent=1)
