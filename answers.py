"""How the viewer answers a command: the policy, planning through its model,
the warm search, the keep repair and the colour guard -- and the comparison
panel, which answers one command several ways side by side.

Everything here works on an already-built goal (`goals.goal_from_command` +
`goals.well_posed`, see `Session.instruction_for`) and the visibility model;
nothing renders and nothing touches session state, so the warm search can run
in a worker thread.
"""
import functools
import os
import time

import numpy as np

import goals
from commands import apply_command, parse_command_with_meta
from rl.colour_guard import colour_guard
from rl.episodes import build_observation, controllable_values, solo_max_log
from rl.keep_repair import keep_repair
from rl.search import apply_controllable, hill_climb

NO_POLICY_MESSAGE = "no trained policy checkpoint found -- applied the command directly instead"

# The policy's answer is polished by a hill_climb warm-started from it.
# Task v3, test, 3 seeds (median attainment): policy + planning 0.64, + 30
# evaluations 0.77, + 200 0.91; a 200-evaluation search from the start 0.70.
# 30 answers in ~0.7 s; 0 shows the policy (+ planning) alone.
REFINE_EVALUATIONS = int(os.environ.get("POLICY_REFINE_EVALUATIONS", "200"))
REFINE_INITIAL_STEP = 0.1
# Planning through the checkpoint's dynamics ensemble (~0.2 s, no renders); 0 skips it.
PLAN = os.environ.get("POLICY_PLAN", "1") != "0"
# After the search, revert every peak of a class the command never mentioned
# wherever that costs at most 0.01 attainment (rl.keep_repair): chained in the
# viewer, the free 0.15 drift per command added up (a black heart after 7).
KEEP_REPAIR = os.environ.get("KEEP_REPAIR", "1") != "0"
# Then give every class its colour back unless the command asked for its
# brightness (rl.colour_guard): the policy recolours hidden classes for free.
COLOUR_GUARD = os.environ.get("COLOUR_GUARD", "1") != "0"


def policy_answer(model, policy, instruction, start_agg, start_params, plan: bool = PLAN) -> np.ndarray:
    """The policy's transfer function for `instruction` from `start_params`,
    refined through its dynamics ensemble when `plan`. Raises ValueError when
    there is no policy."""
    if policy is None:
        raise ValueError(NO_POLICY_MESSAGE)
    ceilings = solo_max_log(model)
    controllable = controllable_values(start_params)
    observation = build_observation(instruction["goal"], model.histogram, start_agg, ceilings, controllable)
    action, _ = policy.predict(observation)
    if plan and policy.dynamics is not None:
        from rl.model_based import plan as plan_action
        action = plan_action(policy.dynamics, instruction["goal"], instruction.get("weights"), model.histogram,
                             ceilings, controllable, start_agg, action)
    # Not rl.episodes.apply_residual: a planned residual can exceed [-1, 1]
    # (it is clip(controllable + a) - controllable), and the viewer has always
    # applied it unclipped.
    return apply_controllable(start_params, np.clip(np.asarray(controllable) + action, -1.0, 1.0))


def warm_search(model, proposal, start_params, instruction, start_agg, evaluations) -> np.ndarray:
    """`hill_climb` from `proposal`, measuring the instruction from the
    command's own start, keeping whichever of {proposal, refined} scores higher."""
    refined = hill_climb(model, proposal, instruction, evaluations=evaluations,
                         initial_step=REFINE_INITIAL_STEP, reference_params=start_params)

    def score(params):
        return goals.attainment(instruction["goal"], start_agg, goals.aggregate(model.features(params)),
                                instruction.get("weights"))

    return refined if score(refined) > score(proposal) else proposal


def repaired(job, params) -> np.ndarray:
    """The keep repair, then the colour guard, each behind its switch: how
    both stages of a policy answer end."""
    instruction = job["instruction"]
    if job["keep_repair"]:
        params, _evaluations = keep_repair(job["model"], instruction["goal"], job["start_params"], params,
                                           weights=instruction.get("weights"), start_agg=job["start_agg"])
    if job["colour_guard"]:
        params = colour_guard(instruction["goal"], job["start_params"], params, goals.starting_params(),
                              weights=instruction.get("weights"))
    return params


def policy_job(model, policy, instruction, start_agg, start_params, plan: bool = PLAN,
               evaluations: int = REFINE_EVALUATIONS) -> dict:
    """Everything a policy answer needs, with the policy's proposal in it:
    `fast_answer` gives the first stage, `full_answer` the final one."""
    proposal = policy_answer(model, policy, instruction, start_agg, start_params, plan)
    return {"model": model, "instruction": instruction, "start_agg": start_agg,
            "start_params": np.asarray(start_params, dtype=np.float64).copy(), "proposal": proposal,
            "evaluations": evaluations, "keep_repair": KEEP_REPAIR, "colour_guard": COLOUR_GUARD}


def fast_answer(job) -> np.ndarray:
    """The two-stage answer's first stage: the proposal, repaired, no search."""
    return repaired(job, job["proposal"])


def full_answer(job) -> np.ndarray:
    """The warm search from the proposal, then the repairs -- the final answer.
    Pure visibility-model work, so it may run in a worker thread."""
    params = job["proposal"]
    if job["evaluations"] > 0:
        params = warm_search(job["model"], params, job["start_params"], job["instruction"],
                             job["start_agg"], job["evaluations"])
    return repaired(job, params)


def search_answer(model, start_params, instruction, evaluations) -> np.ndarray:
    """The hill climb from the command's start (mode "search")."""
    return hill_climb(model, start_params, instruction, evaluations=evaluations)


# --- comparison panel -------------------------------------------------------------

# The search budgets the thesis reports beside the policy: 10 (less than half a
# sweep of 12 groups x 2 signs) and 200.
COMPARE_CHEAP = 10
COMPARE_THOROUGH = 200
SWEEP_EPISODES = 20


def compare(model, policy, cmd, instruction, start_agg, start_params, view,
            cheap: int = COMPARE_CHEAP, thorough: int = COMPARE_THOROUGH) -> dict:
    """Answer one parsed command several ways from the same start: `exact`
    (the command applied directly), `search_cheap` / `search_thorough`, the
    bare `policy` and `policy_keep` (after the keep repair). Every arm is
    scored with `goals.attainment` against the shared start; `view(params)`
    gives its image and per-class readout. An arm that cannot answer is
    reported `unavailable` instead of failing the panel; only the answering
    is timed."""
    for goal_class in goals.GOAL_CLASSES:
        model.solo_max(goal_class)   # warm the ceilings before any stopwatch starts

    def finish(params, evaluations, elapsed_ms):
        final = goals.aggregate(model.features(params))
        return {"params": params.tolist(), **view(params),
                "attainment": float(goals.attainment(instruction["goal"], start_agg, final,
                                                     instruction.get("weights"))),
                "unchanged": bool(np.array_equal(params, start_params)),
                "evaluations": evaluations, "elapsed_ms": elapsed_ms}

    def arm(answer, evaluations, cannot):
        started = time.perf_counter()
        try:
            params, extra = answer()
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"[compare] {cannot}: {type(exc).__name__}: {exc}")
            return {"params": None, "unavailable": cannot, "detail": str(exc), "attainment": None,
                    "class_visibility": None, "class_brightness": None, "image_b64": None,
                    "evaluations": None, "unchanged": None,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000)}
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return finish(params, evaluations if extra is None else extra, elapsed_ms)

    def policy_keep():
        proposal = policy_answer(model, policy, instruction, start_agg, start_params, plan=False)
        params, evaluations = keep_repair(model, instruction["goal"], start_params, proposal,
                                          weights=instruction.get("weights"), start_agg=start_agg)
        return params, evaluations

    no_policy = "no trained policy checkpoint is loaded, so this arm has no answer"
    arms = {"exact": arm(lambda: (apply_command(cmd, start_params), None), 0,
                         "this instruction cannot be applied directly")}
    for name, budget in (("search_cheap", cheap), ("search_thorough", thorough)):
        arms[name] = arm(functools.partial(lambda b: (search_answer(model, start_params, instruction, b), None),
                                           budget), budget, "this volume cannot be scored")
    arms["policy"] = arm(lambda: (policy_answer(model, policy, instruction, start_agg, start_params, plan=False),
                                  None), 0, no_policy)
    arms["policy_keep"] = arm(policy_keep, None, no_policy)
    return arms


def sweep(model, policy, volume, start_params, episodes: int = SWEEP_EPISODES, seed: int = 0,
          thorough=None) -> dict:
    """The arms over `episodes` instructions sampled from the task's grammar,
    all from the viewer's current state: what each method does across many
    instructions, not just the one typed."""
    rng = np.random.default_rng(seed)
    start_agg = goals.aggregate(model.features(start_params))
    budgets = {"search_cheap": COMPARE_CHEAP, **({"search_thorough": thorough} if thorough else {})}
    scores = {name: [] for name in ("exact", "policy", *budgets)}
    kinds = []
    for _ in range(episodes):
        instruction = goals.sample_instruction(volume, model, start_agg, rng)
        kinds.append(instruction["kind"])

        def score(params):
            return float(goals.attainment(instruction["goal"], start_agg, goals.aggregate(model.features(params)),
                                          instruction.get("weights")))

        try:
            parsed, _meta = parse_command_with_meta(instruction["text"], parser="rule")
            scores["exact"].append(score(apply_command(parsed, start_params)))
        except (ValueError, KeyError, FileNotFoundError):
            scores["exact"].append(None)
        for name, budget in budgets.items():
            scores[name].append(score(search_answer(model, start_params, instruction, budget)))
        try:
            scores["policy"].append(score(policy_answer(model, policy, instruction, start_agg, start_params,
                                                        plan=False)))
        except (ValueError, KeyError, FileNotFoundError):
            scores["policy"].append(None)
    return {"episodes": episodes, "seed": seed, "volume": volume, "kinds": sorted(set(kinds)),
            "arms": {name: goals.summarise_attainment(values) for name, values in scores.items()}}
