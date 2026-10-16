"""Expert iteration probe: refine E3's own proposals with warm search (measured from the
episode start), fine-tune the actor on the refined residuals, score the policy alone."""
import collections, sys, time, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
from stable_baselines3 import SAC
import goals
from rl import pretrain
from rl.baselines import hill_climb
from rl.oneshot_train import build_env, build_eval_env

SRC, N_TRAIN, EVALS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
kw = dict(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
          hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
train_env, val_env = build_env(**kw), build_eval_env(**kw)
model = SAC.load(SRC, device="cpu")

def score_policy(m, n=150):
    res = collections.defaultdict(list)
    for seed in range(50_000, 50_000 + n):
        obs, _ = val_env.reset(seed=seed)
        a, _ = m.predict(obs, deterministic=True)
        _, _, _, _, info = val_env.step(a)
        res[val_env._instruction["kind"]].append(info["attainment"]); res["ALL"].append(info["attainment"])
    return "  ".join(f"{k} {np.median(v):+.3f}" for k, v in sorted(res.items()))

print("before:", score_policy(model), flush=True)
obs_l, act_l, gain = [], [], []
t = time.time()
for i in range(N_TRAIN):
    obs, _ = train_env.reset(seed=300_000 + i)
    a, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = train_env.step(a)
    refined = hill_climb(train_env._model, train_env._params.copy(), train_env._instruction,
                         evaluations=EVALS, initial_step=0.1, reference_params=train_env._start_params)
    start_c = np.asarray(train_env._controllable_values(train_env._start_params))
    target = np.asarray(train_env._controllable_values(refined)) - start_c
    after = train_env._attainment(goals.aggregate(goals.features(train_env._model, refined)))
    obs_l.append(obs); act_l.append(target); gain.append(after - info["attainment"])
    if i % 200 == 199:
        print(f"  collected {i+1} ({time.time()-t:.0f}s), median gain {np.median(gain):+.3f}", flush=True)
data = {"obs": np.asarray(obs_l, np.float32), "actions": np.asarray(act_l, np.float32),
        "rewards": np.asarray(gain, np.float32)}
np.savez_compressed(OUT, **data)
for epochs, lr in ((10, 3e-4), (30, 3e-4)):
    m = SAC.load(SRC, device="cpu")
    rep = pretrain.pretrain_actor(m, data, epochs=epochs, learning_rate=lr, seed=0)
    print(f"finetune {epochs}ep lr {lr}: val/zero {rep['val_mse']/rep['zero_mse']:.3f} |", score_policy(m), flush=True)
