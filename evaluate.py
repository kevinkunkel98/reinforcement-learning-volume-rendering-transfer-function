"""A JSONL append helper.

This module used to hold `objective`, the +1/-1 verdict a bespoke coordinate
stepper in `server.py` hill-climbed on, and `_dominance`, the opacity-mass
share it scored with. That search was never the one the thesis measures -- the
baselines are `rl.baselines.hill_climb` on `goals.distance` -- and the viewer
now runs the measured one, so both had no caller and were removed rather than
left to be re-wired. `search.py`, which held that stepper's move generator,
went with them.
"""
import json


def jsonl_append(path: str, entry: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
