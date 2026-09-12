# Goal-Conditioned Reward Model Design

## Goal

Replace generic image-quality scoring with a goal-conditioned reward model that
sees before/after rendering features, the requested tissue, and requested
direction. Build reproducible pair extraction, objective pretraining, human
fine-tuning, reward-hacking safeguards, and separate evaluation.

## Scope

Changes are limited to reward-model code, reward data extraction/training, and
evaluation. Parser, command grammar, rendering implementation, chat UI, and
`rl/online_train.py` behavior remain unchanged. Existing clients need not be
modified, but a client-neutral record contract is documented for web and VR
clients.

## Representation

`rl/reward_model.py` owns feature encoding and model behavior. Each observation
is encoded as an 18-value `float32` vector:

```text
[features_after(4), features_before(4), after-before(4), target_onehot(5), direction(1)]
```

Feature order is `mean`, `std`, `coverage`, `entropy`; target order is the
existing `rl.env.TISSUES` order: `air`, `fat`, `soft`, `spongy`, `bone`.
`encode_target(cmd)` isolates target encoding. Its docstring states that the
one-hot representation can later be replaced by a text embedding without
changing callers or the rest of the pipeline.

`RewardModel` uses `18 -> 64 -> 32 -> 1`, ReLU hidden layers, and scalar logit
output `R(z)`. Pair training uses weighted Bradley-Terry loss:

```text
loss = weight * -log(sigmoid(R(z_preferred) - R(z_other)))
```

The mapped reward is `r_tilde(z) = 2 * sigmoid(R(z)) - 1`.

## Client-Neutral Pair Contract

Each JSONL pair contains two observations, command context, label, provenance,
and weight:

```json
{
  "observation_a": {"before_features": {}, "after_features": {}, "before_image": null, "after_image": null},
  "observation_b": {"before_features": {}, "after_features": {}, "before_image": null, "after_image": null},
  "command": {"attribute": "opacity", "target": "bone", "direction": "increase"},
  "target_tissue": "bone",
  "direction": "increase",
  "label": 1,
  "source": "branch",
  "weight": 1.0
}
```

`label=1` means A is preferred; `label=-1` means B is preferred. Canonical
client logs may include `session_id`, `episode_id`, `step_id`,
`parent_step_id`, `branch_id`, `carried_forward`, `accepted`, and `ended`.
Feature dictionaries are authoritative when present. PNG/JPEG image paths are
optional audit artifacts and permit feature recovery for legacy rows; training
does not consume pixels.

## Pair Extraction

`rl/extract_pairs.py` reads current logs and canonical web/VR records without
mutating them, then writes `out/pairs.jsonl`.

1. Branch pairs have strongest signal and weight `1.0`. Only explicit branch
   metadata or histories with parent/child and carried-forward information are
   accepted. Current logs lack this information, so extractor reports zero
   recoverable branch pairs instead of inferring them.
2. Within-episode pairs have weight `0.7`. For an accepted or ended state `k`,
   pair it against earlier state `j` from the same command only when
   `k - j >= 2`.
3. Absolute thumbs pairs have weight `0.4`. Join feedback to rendered steps by
   session/step, with unique-parameter fallback. Recover missing features from
   optional image paths through Pillow; skip unavailable artifacts and report
   skipped rows.

Extractor deduplicates pair events, normalizes command aliases, preserves image
paths, and prints counts by source, target, and total.

## Training

`rl/pretrain_reward.py` creates configurable tens of thousands of synthetic
pairs. It samples random starting TF vectors, target/direction, and two random
deltas, renders both, extracts features, and labels using signed target
`mass_fraction`. Code includes a `TODO` identifying the replacement with a
project visibility measure if one becomes available. It saves a distinct
`reward_pretrained.pt` artifact and, by default, five seeded ensemble members.

`rl/finetune_reward.py` loads pretrained members and trains on real pairs at
roughly one tenth the pretraining learning rate. It uses a fixed seeded split,
does not train on evaluation records, and saves separate
`reward_finetuned.pt` plus member artifacts. Pretrained artifacts are never
overwritten.

## Reward-Hacking Defenses

`RewardModelTFEnv` accepts configurable `alpha` (default `0.7`) and supports
`0.0`/`1.0` ablations. For an ensemble, model reward is mean minus standard
deviation. Final reward is:

```text
r = alpha * r_model_ensemble + (1 - alpha) * r_objective
```

Coverage below a configurable near-black threshold or mean opacity above a
configurable near-opaque threshold receives a hard penalty independent of model
output. Existing automatic metric info remains available for comparisons.

## Evaluation

`rl/eval_reward.py` keeps three levels separate:

1. Reward-only accuracy on a fixed seeded held-out human split, reported
   overall and per source. Both pretrained and fine-tuned models use exactly
   the same test rows.
2. RLHF policy performance against the objective metric, as a sanity check for
   collapse/reward hacking.
3. Blind head-to-head rendering of RLHF policy versus hill-climber in random
   order. Verdicts go to a separate result file and never return to training.

Evaluation writes objective/human disagreement records including image paths,
target, direction, and labels. Missing logs, models, or render artifacts cause
clear errors or empty summaries; scripts never fabricate results.

## Testing

Tests cover 18-value ordering and target encoding, weighted Bradley-Terry loss,
checkpoint round trips, extractor source fixtures and missing-image behavior,
pretraining generation with a renderer seam, fixed split/no leakage, ensemble
mean-minus-standard-deviation, anchor ablations, and independent degenerate
state penalties. Existing reward tests are updated for the intentional input
and API change; unrelated parser, UI, rendering, and online-training tests stay
untouched.
