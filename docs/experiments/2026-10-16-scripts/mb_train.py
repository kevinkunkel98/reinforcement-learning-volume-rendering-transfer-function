"""Model-based RL with a known reward: learn the renderer's effect (TF change -> per-class
visibility/brightness change) with an MLP ensemble, then train a policy by backpropagating
the exact goals.distance through the model (SVG(0)-style). Evaluated in the real env.

usage: mb_train.py OUT DATA.npz[,DATA2.npz...] [--beta B] [--eval N]
"""
import argparse, collections, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, ".")
import goals
from rl.baselines import apply_controllable

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "3")))
N = len(goals.GOAL_CLASSES)
G = 24
KAPPA, LAMBDA, TOL = goals.KAPPA, goals.LAMBDA_KEEP, goals.KEEP_TOLERANCE


def load(paths):
    parts = [dict(np.load(p, allow_pickle=True)) for p in paths.split(",")]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def mlp(i, o, h=256, depth=3):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class Ensemble(nn.Module):
    """K dynamics models: (hist, solo, ctrl, s0, residual action) -> (delta state, useless logit)."""
    def __init__(self, k=5):
        super().__init__()
        self.nets = nn.ModuleList(mlp(16 + 8 + G + 18 + G, 18 + 1) for _ in range(k))
        self.register_buffer("mu", torch.zeros(90)); self.register_buffer("sd", torch.ones(90))

    def forward(self, ctx, a):
        x = (torch.cat([ctx, a], -1) - self.mu) / self.sd
        return torch.stack([n(x) for n in self.nets])          # K x B x 19


def policy_obs(goal, hist, s0, solo, ctrl):
    """Same layout as rl.oneshot_env.build_observation."""
    return torch.cat([goal, hist, s0[:, :N], s0[:, N + 1:N + 1 + N], solo, ctrl, s0[:, -1:]], -1)


def distance(goal, w, delta):
    c, b, c_other = delta[:, :N], delta[:, N + 1:2 * N + 1], delta[:, N]
    d, m, e, n = goal[:, :N], goal[:, N:2 * N], goal[:, 2 * N:3 * N], goal[:, 3 * N:]
    t = (m * w * (c - d).abs()).sum(1) + KAPPA * (n * w * (b - e).abs()).sum(1)
    t = t + LAMBDA * ((1 - m) * torch.relu(c.abs() - TOL)).sum(1)
    t = t + LAMBDA * KAPPA * ((1 - n) * torch.relu(b.abs() - TOL)).sum(1)
    return t + LAMBDA * torch.relu(c_other.abs() - TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("data")
    ap.add_argument("--beta", type=float, default=1.0, help="ensemble-disagreement penalty")
    ap.add_argument("--useless", type=float, default=1.0)
    ap.add_argument("--model-epochs", type=int, default=60)
    ap.add_argument("--policy-steps", type=int, default=6000)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--init-model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    D = load(args.data)
    S = len(D["s0"]); A = D["act"].shape[1]
    perm = np.random.default_rng(0).permutation(S)
    rng = np.random.default_rng(args.seed); val_idx, tr_idx = perm[:S // 10], perm[S // 10:]
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)
    ctx_all = T(np.concatenate([D["hist"], D["solo"], D["ctrl"], D["s0"]], 1))     # S x 66

    def flat(idx):
        ctx = ctx_all[idx].repeat_interleave(A, 0)
        a = T(D["act"][idx].reshape(-1, G))
        y = T((D["s1"][idx] - D["s0"][idx][:, None]).reshape(-1, 18))
        u = T(D["useless"][idx].reshape(-1).astype(np.float32))
        return ctx, a, y, u

    ctx_tr, a_tr, y_tr, u_tr = flat(tr_idx)
    ctx_va, a_va, y_va, u_va = flat(val_idx)
    model = Ensemble()
    if args.init_model:
        model.load_state_dict(torch.load(args.init_model))
    x = torch.cat([ctx_tr, a_tr], -1)
    model.mu.copy_(x.mean(0)); model.sd.copy_(x.std(0) + 1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.model_epochs)
    t0 = time.time()
    for ep in range(args.model_epochs):
        model.train()
        for k, net in enumerate(model.nets):                     # bootstrap per member
            idx = torch.randint(0, len(a_tr), (len(a_tr),))
            for b in torch.split(idx, 512):
                p = model(ctx_tr[b], a_tr[b])[k]
                loss = nn.functional.smooth_l1_loss(p[:, :18], y_tr[b], beta=0.1) \
                    + 0.1 * nn.functional.binary_cross_entropy_with_logits(p[:, 18], u_tr[b])
                opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 9 or ep == args.model_epochs - 1:
            with torch.no_grad():
                p = model(ctx_va, a_va).mean(0)
                err = (p[:, :18] - y_va).abs().mean(0)
                r2 = 1 - ((p[:, :18] - y_va) ** 2).mean(0) / y_va.var(0).clamp_min(1e-9)
            print(f"model ep {ep+1}: val MAE logvis {err[:N+1].mean():.3f} bright {err[N+1:2*N+1].mean():.3f} "
                  f"R2 logvis {r2[:N+1].mean():.3f} bright {r2[N+1:2*N+1].mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # does the model rank real outcomes? predicted vs real attainment on held-out (start, action, goal)
    with torch.no_grad():
        gi = torch.randint(0, D["goal"].shape[1], (len(val_idx) * A,))
        goal_v = T(D["goal"][val_idx]).repeat_interleave(A, 0)[torch.arange(len(gi)), gi]
        w_v = T(D["weights"][val_idx]).repeat_interleave(A, 0)[torch.arange(len(gi)), gi]
        d0 = distance(goal_v, w_v, torch.zeros_like(y_va))
        ok = d0 > 1e-6
        real = 1 - distance(goal_v, w_v, y_va) / d0.clamp_min(1e-6)
        pred = 1 - distance(goal_v, w_v, model(ctx_va, a_va).mean(0)[:, :18]) / d0.clamp_min(1e-6)
        r, p_ = real[ok].numpy(), pred[ok].numpy()
        rank = lambda v: np.argsort(np.argsort(v))
        print(f"held-out attainment: spearman {np.corrcoef(rank(r), rank(p_))[0, 1]:.3f}  MAE {np.abs(np.clip(r,-2,1)-np.clip(p_,-2,1)).mean():.3f}", flush=True)

    # policy: obs -> residual, trained through the frozen model on (start, instruction) pairs
    pol = mlp(97, G, 256, 2)
    popt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    goal_all, w_all = T(D["goal"]), T(D["weights"])
    hist, solo, ctrl, s0 = T(D["hist"]), T(D["solo"]), T(D["ctrl"]), T(D["s0"])
    I = D["goal"].shape[1]
    for step in range(args.policy_steps):
        si = torch.as_tensor(rng.choice(tr_idx, 256))
        gi = torch.randint(0, I, (256,))
        goal, w = goal_all[si, gi], w_all[si, gi]
        d0 = distance(goal, w, torch.zeros(256, 18))
        keep = d0 > 1e-6
        obs = policy_obs(goal, hist[si], s0[si], solo[si], ctrl[si]).clamp(-10, 10)
        a = torch.tanh(pol(obs))
        a = (ctrl[si] + a).clamp(-1, 1) - ctrl[si]
        preds = model(ctx_all[si], a)                            # K x B x 19
        mean = preds.mean(0)
        dist = distance(goal, w, mean[:, :18]) / d0.clamp_min(1e-6)
        disagree = preds[:, :, :18].std(0).mean(1)
        loss = (dist.clamp(max=3.0) + args.beta * disagree
                + args.useless * torch.sigmoid(mean[:, 18]))[keep].mean()
        popt.zero_grad(); loss.backward(); popt.step()
        if step % 1000 == 999:
            print(f"policy step {step+1}: model attainment median {(1-dist[keep]).median():+.3f} "
                  f"disagree {disagree.mean():.3f}", flush=True)
    torch.save(pol.state_dict(), f"{args.out}/policy.pt")

    # real evaluation on the validation split, same episodes as the earlier probes
    from rl.oneshot_train import build_eval_env
    env = build_eval_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                         hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
    res = collections.defaultdict(list); rows = []
    for seed in range(50_000, 50_000 + args.eval):
        obs, _ = env.reset(seed=seed)
        with torch.no_grad():
            a = torch.tanh(pol(T(obs)[None].clamp(-10, 10)))[0].numpy()
        _, _, _, _, info = env.step(a)
        res[info["kind"]].append(info["attainment"]); res["ALL"].append(info["attainment"])
        rows.append({"seed": seed, "kind": info["kind"], "attainment": info["attainment"],
                     "solvable": info["solvable"], "useless": info["useless"]})
    summary = {k: float(np.median(v)) for k, v in res.items()}
    print("REAL val medians: " + "  ".join(f"{k} {v:+.3f} (n={len(res[k])})" for k, v in sorted(summary.items())), flush=True)
    print(f"useless share {np.mean([r['useless'] for r in rows]):.3f}")
    json.dump({"args": vars(args), "summary": summary, "rows": rows}, open(f"{args.out}/eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
