import sys, collections, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
import goals, datasets, visibility, transfer
from rl.baselines import CONTROLLABLE
from rl.oneshot_env import START_NOISE
P = transfer.PARAMS_PER_PEAK
H = {c: transfer.ANATOMICAL_PEAK_INDEX[c]*P + 2 for c in goals.GOAL_CLASSES}
def default(rng):
    p = goals.starting_params()
    for g in CONTROLLABLE:
        n = rng.uniform(-START_NOISE, START_NOISE)
        for i in g: p[i] += n
    return np.clip(p, -1, 1)
def heights(rng):  # every class height redrawn uniformly over unit [0,1]
    p = default(rng)
    for c, i in H.items(): p[i] = rng.uniform(-1, 1)
    return p
def softdim(rng):  # soft tissue height drawn low (unit 0..0.15), rest default
    p = default(rng)
    p[H["soft"]] = transfer._from_unit(rng.uniform(0.0, 0.15))
    return p
ORG = ("heart","vessels","liver","kidneys","spleen")
def organs_up(rng):
    p = default(rng)
    for c in ORG: p[H[c]] = transfer._from_unit(rng.uniform(0.25, 0.7))
    return p
def soft_organs(rng):
    p = organs_up(rng)
    p[H["soft"]] = transfer._from_unit(rng.uniform(0.0, 0.15))
    return p
def soft_organs_bone(rng):
    p = soft_organs(rng)
    p[H["skeleton"]] = transfer._from_unit(rng.uniform(0.1, 0.6))
    p[H["lungs"]] = transfer._from_unit(rng.uniform(0.0, 0.05))
    return p
vols = datasets.volumes_for_split("train")
for name, fn in [("organs_up", organs_up), ("soft_organs", soft_organs), ("soft_organs_bone", soft_organs_bone)]:
    vis_ok = collections.Counter(); useless = 0; N = 0
    r = np.random.default_rng(1)
    for k in range(80):
        v = str(r.choice(vols)); m = visibility.for_volume(v)
        f = goals.features(m, fn(r)); a = goals.aggregate(f)
        N += 1; useless += goals.is_useless(f)
        for c in goals.GOAL_CLASSES: vis_ok[c] += a["vis"].get(c, 0) >= goals.EPSILON
    print(name, "useless %.0f%%" % (100*useless/N), " ".join("%s %.0f%%" % (c, 100*vis_ok[c]/N) for c in goals.GOAL_CLASSES), flush=True)
