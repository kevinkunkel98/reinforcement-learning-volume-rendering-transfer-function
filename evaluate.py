"""+1/-1 verdicts: objective (opacity_mass) and human (console)."""
import json

from transfer import TISSUE_BANDS, opacity_mass

EPS = 1e-4


def _dominance(params, target) -> float:
    """Target band(s)' share of total opacity mass across all bands."""
    targets = target if isinstance(target, list) else [target]
    target_mass = sum(opacity_mass(params, *TISSUE_BANDS[t]) for t in targets)
    total = sum(opacity_mass(params, tlo, thi) for tlo, thi in TISSUE_BANDS.values())
    return target_mass / (total + EPS)


def objective(params_before, params_after, cmd: dict) -> int:
    if "compound" in cmd or cmd["direction"] == "reset":
        # A compound (multi-tissue absolute-level) command directly sets what
        # it says -- there's no "wrong direction" to detect the way there is
        # for a relative increase/decrease, so treat it as always accepted,
        # the same as reset.
        return 1

    if cmd.get("attribute") in ("width", "brightness", "center"):
        # No established exact metric for these attributes (opacity_mass is
        # opacity-specific) -- treat as always accepted, same as compound/reset.
        return 1

    if cmd["direction"] == "show_only":
        # "isolate X" is about X's share of the image, not its raw mass —
        # crushing every other peak can lower X's own mass while still
        # making it far more dominant.
        before_dom = _dominance(params_before, cmd["target"])
        after_dom = _dominance(params_after, cmd["target"])
        return 1 if after_dom > before_dom + EPS else -1

    direction = "increase" if cmd["direction"] == "increase" else "decrease"
    lo, hi = TISSUE_BANDS[cmd["target"]]
    delta_target = opacity_mass(params_after, lo, hi) - opacity_mass(params_before, lo, hi)
    signed = delta_target if direction == "increase" else -delta_target
    if signed <= EPS:
        return -1
    max_other = 0.0
    for tissue, (tlo, thi) in TISSUE_BANDS.items():
        if tissue == cmd["target"]:
            continue
        d = abs(opacity_mass(params_after, tlo, thi) - opacity_mass(params_before, tlo, thi))
        max_other = max(max_other, d)
    return 1 if max_other <= abs(delta_target) else -1


def human(before_png: str, after_png: str) -> int:
    print(f"[human] before: {before_png}")
    print(f"[human] after:  {after_png}")
    while True:
        answer = input("Besser oder schlechter? (b/s): ").strip().lower()
        if answer == "b":
            return 1
        if answer == "s":
            return -1
        print("Bitte 'b' (besser) oder 's' (schlechter) eingeben.")


def jsonl_append(path: str, entry: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
