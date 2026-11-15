"""Copy cold200 (search from the start: deterministic, policy-independent) from
the v3c seed-1 results into v3d result files, matched by episode index."""
import glob, json, sys
split, pattern = sys.argv[1], sys.argv[2]
ref = {}
for f in glob.glob(f"out/eval_extra/s1_{split}_*.json"):
    for r in json.load(open(f)):
        ref[r["i"]] = r
n = 0
for f in glob.glob(pattern):
    rows = json.load(open(f))
    for r in rows:
        src = ref[r["i"]]
        assert (src["volume"], src["kind"]) == (r["volume"], r["kind"]), (f, r["i"])
        r["cold200"], r["cold200_pp"] = src["cold200"], src["cold200_pp"]
        n += 1
    json.dump(rows, open(f, "w"))
print(f"[merge_cold] {n} rows from v3c seed 1")
