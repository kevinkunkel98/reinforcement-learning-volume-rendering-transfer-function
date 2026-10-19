import sys; sys.path.insert(0, ".")
import numpy as np, transfer, goals, datasets, visibility
start = goals.starting_params("v2")
for peaks in [("skeleton","lungs","soft"), ("skeleton","lungs"), ("soft",), ("skeleton",), ()]:
    transfer.UNLABELLED_PEAKS = peaks
    out = []
    for vol in datasets.volumes_for_split("val"):
        vis = goals.aggregate(visibility.for_volume(vol).features(start))["vis"]
        out.append(vis)
    keys = ["other","soft","skeleton","lungs","liver","heart","kidneys","spleen","vessels"]
    print(peaks, " ".join(f"{k}={np.mean([o[k] for o in out]):.3f}" for k in keys))
