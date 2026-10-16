"""BC learning curve on search-distilled data: val MSE / zero MSE vs train size."""
import numpy as np, torch
torch.set_num_threads(1)
obs = np.concatenate([np.load(f"out/rl_v4/search_v9_seed{s}.npz")["obs"] for s in (0, 1, 2)])
act = np.clip(np.concatenate([np.load(f"out/rl_v4/search_v9_seed{s}.npz")["actions"] for s in (0, 1, 2)]), -1, 1)
print("total", len(obs), "dup obs rows", len(obs) - len(np.unique(obs.round(5), axis=0)))
rng = np.random.default_rng(0); order = rng.permutation(len(obs))
val, pool = order[:400], order[400:]
X = torch.tensor(obs); Y = torch.tensor(act)
zero = float((Y[val] ** 2).mean())
for n in (300, 600, 1200, 2600):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(97, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, 24), torch.nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = pool[:n]; best = 9
    for ep in range(300):
        for b in np.array_split(rng.permutation(tr), max(1, n // 128)):
            loss = torch.nn.functional.mse_loss(net(X[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            with torch.no_grad(): best = min(best, float(torch.nn.functional.mse_loss(net(X[val]), Y[val])))
    print(f"n={n:5d} best val {best:.4f}  zero {zero:.4f}  ratio {best/zero:.3f}", flush=True)
