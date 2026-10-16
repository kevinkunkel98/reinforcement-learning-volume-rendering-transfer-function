"""Closed-loop RL: T correction steps per episode instead of one blind shot.

Each step re-renders, and the policy sees the current state with the *remaining*
goal (instruction goal minus progress so far; unmentioned classes that drifted past
KEEP_TOLERANCE become "undo this drift" targets). Same observation layout as the
one-shot policy, so a one-shot checkpoint can be fine-tuned directly.
Reward per step = change in true attainment (measured from the episode start), so the
undiscounted return telescopes to the final attainment.

usage: closed_loop_train.py OUT INIT_ZIP|scratch STEPS T SEED [GAMMA]
"""
import collections, csv, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
import goals
from rl.baselines import apply_controllable
from rl.oneshot_env import build_observation, OBSERVATION_BOUND, USELESS_PENALTY
from rl.oneshot_train import build_env, build_eval_env

N = len(goals.GOAL_CLASSES)
KW = dict(policy_version="oneshot-v9", action_mode="residual", reward_mode="target")
TRAIN_KW = dict(KW, hindsight_ratio=0.25, balance_classes=True, hindsight_brightness_share=0.1)
EVAL_KW = dict(KW, hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)


def remaining_goal(goal, start_agg, cur_agg):
    c, b = goals.progress(start_agg, cur_agg)
    g = np.array(goal, dtype=np.float64)
    for i, cls in enumerate(goals.GOAL_CLASSES):
        if g[N + i]:
            g[i] -= c[cls]
        elif abs(c[cls]) > goals.KEEP_TOLERANCE:
            g[i], g[N + i] = -c[cls], 1.0
        if g[3 * N + i]:
            g[2 * N + i] -= b[cls]
    return g


class ClosedLoopEnv(gym.Env):
    def __init__(self, inner, horizon):
        self.inner, self.horizon = inner, horizon
        self.observation_space, self.action_space = inner.observation_space, inner.action_space

    def _obs(self):
        e = self.inner
        g = remaining_goal(e._instruction["goal"], e._start_agg, self.agg)
        ctrl = np.asarray(e._controllable_values(self.params))
        obs = build_observation(g, e._model.histogram, self.agg, self.solo, ctrl)
        return np.clip(obs, -OBSERVATION_BOUND, OBSERVATION_BOUND).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = self.inner.reset(seed=seed)
        e = self.inner
        self.params, self.agg, self.t, self.att = e._start_params.copy(), e._start_agg, 0, 0.0
        self.solo = e._solo_max_log(e._volume, e._model)
        return self._obs(), info

    def step(self, action):
        e = self.inner
        ctrl = np.asarray(e._controllable_values(self.params))
        self.params = apply_controllable(self.params, np.clip(ctrl + np.clip(action, -1, 1), -1, 1))
        raw = goals.features(e._model, self.params)
        self.agg = goals.aggregate(raw)
        att = e._attainment(self.agg)
        reward = float(np.clip(att, -1, 1) - np.clip(self.att, -1, 1))
        useless = goals.is_useless(raw)
        if useless:
            reward -= USELESS_PENALTY
        self.att, self.t = att, self.t + 1
        done = self.t >= self.horizon
        return self._obs(), reward, done, False, {"attainment": att, "kind": e._instruction["kind"],
                                                  "useless": useless}


def evaluate(model, env, n=100):
    final, first = [], []
    for i in range(n):
        obs, _ = env.reset(seed=900_000 + i)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(a)
            if env.t == 1:
                first.append(info["attainment"])
        final.append(info["attainment"])
    return float(np.median(final)), float(np.median(first))


if __name__ == "__main__":
    out, init, steps, horizon, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    gamma = float(sys.argv[6]) if len(sys.argv) > 6 else 0.95
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    os.makedirs(out, exist_ok=True)
    train = ClosedLoopEnv(build_env(**TRAIN_KW), horizon)
    val = ClosedLoopEnv(build_eval_env(**EVAL_KW), horizon)
    train.reset(seed=seed)
    if init == "scratch":
        model = SAC("MlpPolicy", train, seed=seed, gamma=gamma, learning_starts=500, verbose=0)
    else:
        model = SAC.load(init, env=train, device="cpu", seed=seed, gamma=gamma, learning_starts=2000)
    json.dump({"init": init, "steps": steps, "horizon": horizon, "seed": seed, "gamma": gamma},
              open(f"{out}/meta.json", "w"))
    # The loaded critic only knows one-step (terminal) values: train it alone first,
    # so the actor is not pulled around by a critic that has not learned to bootstrap.
    warmup = int(os.environ.get("ACTOR_WARMUP", "0"))
    real_step = model.actor.optimizer.step
    if warmup:
        model.actor.optimizer.step = lambda *a, **k: None
    best = -1e9
    log = open(f"{out}/eval.csv", "w")
    w = csv.writer(log); w.writerow(["step", "median_final", "median_first", "secs"])
    t0, done = time.time(), 0
    med, first = evaluate(model, val)
    w.writerow([0, med, first, 0]); log.flush()
    print(f"step 0: final {med:+.3f} first {first:+.3f}", flush=True)
    while done < steps:
        model.learn(5000, reset_num_timesteps=(done == 0))
        done += 5000
        if warmup and done >= warmup:
            model.actor.optimizer.step = real_step
        med, first = evaluate(model, val)
        w.writerow([done, med, first, round(time.time() - t0)]); log.flush()
        print(f"step {done}: final {med:+.3f} first {first:+.3f} ({time.time()-t0:.0f}s)", flush=True)
        model.save(f"{out}/last.zip")
        if med > best:
            best = med
            model.save(f"{out}/best.zip")
