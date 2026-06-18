# Local Outcome Prediction Plan

## Goal

Move outcomes that are naturally local to rooms, doors, frontiers, or room parts
away from purely pooled/global prediction. Use the most local valid information
available at each generation state:

- Unplaced entities are predicted from the post-pooling global embedding.
- Placed entities are predicted from local node embeddings when the relevant
  local node exists.
- Outcomes that are already known are wired directly into the prediction output
  so generation reward uses the known value and training gradients are masked
  for that entry.

## Current First Step

Split the current room-part save/refill distance outcomes into directed
components.

The current save/refill distance outcome is a combined round-trip value:

- nearest save/refill to room part
- room part to nearest save/refill

Replace this with separate directed outcomes:

- distance from nearest save to room part
- distance from room part to nearest save
- distance from nearest refill to room part
- distance from room part to nearest refill

The training and generation config can keep one weight for save distance and
one weight for refill distance. Each weight applies to both directions.

## Known Directed Distance Outcomes

A directed save/refill distance can be finalized before episode end when future
steps cannot improve it.

For room part `p`:

- `p -> nearest save` is finalized when the current finite `p -> save` distance
  is less than or equal to the current `p -> nearest frontier` distance.
- `nearest save -> p` is finalized when the current finite `save -> p` distance
  is less than or equal to the current `nearest frontier -> p` distance.
- The same rules apply for refill distances.

Unreachable outcomes can also be finalized:

- `p -> save` is finalized as unreachable when there is no current path from
  `p` to any save and no current path from `p` to any frontier.
- `save -> p` is finalized as unreachable when there is no current path from
  any save to `p` and no current path from any frontier to `p`.
- The same rules apply for refill distances.

The finite known values should be substituted in model forward, using the same
numeric scale as the target. This makes generation and training consume the same
model output, while cutting gradients through finalized entries.

Unreachable finalized outcomes need separate reward semantics. The current
distance reward is an expected distance conditioned on reachability, which is
ill-defined when reachability probability is zero. Do not force unreachable
states into the existing distance scalar without first deciding the objective.

## First Implementation Sequence

1. Rename and split Rust outcome generation.
   - Replace combined save/refill room-part outcome vectors with directed
     `to` and `from` vectors.
   - Keep masks direction-specific.
   - Add direction-specific finalized masks and known values.
   - Preserve strict required fields across Python/Rust bindings.

2. Split current room-part distance features.
   - Replace combined save/refill feature encodings with directed feature
     encodings.
   - Split frontier distance features into `room_part -> frontier` and
     `frontier -> room_part`.
   - Keep the existing global-feature path initially; do not introduce
     room-part nodes in this step.

3. Update Python data plumbing.
   - Update `EndOutcomes`, feature dataclasses, buffer allocation, and transfer
     code to carry directed fields.
   - Update training batch construction, generated outcome concatenation, and
     metrics.
   - Use named dataclass construction throughout.

4. Update model heads and forward override.
   - Replace each combined save/refill output head with directed heads.
   - In forward, substitute finalized finite known values with `torch.where`.
   - Leave finalized unreachable entries out of the old distance reward/loss
     until the unreachable objective is defined.

5. Update loss and generation reward.
   - Apply the existing save/refill weights to both directed components.
   - For now, compute distance loss/reward only on reachable directed outcomes.
   - Track finalized-unreachable counts as metrics so the behavior is visible.

6. Test and validate.
   - Add Rust tests for directed distances and finalized masks.
   - Add Python smoke tests for shape compatibility and model forward.
   - Run `cargo test`, `maturin develop`, and Python compile/smoke checks in
     the `map-gen` conda environment.

## Unreachable Reward Follow-Up

The current reward estimates conditional expected distance given reachability.
That is not enough once the model can know or predict unreachable states.

Possible alternatives to evaluate:

- Add explicit directed reachability outcomes and reward their log-probability.
- Model expected cost as `P(reachable) * E(distance | reachable) +
  P(unreachable) * unreachable_penalty`.
- Treat unreachable save/refill as a separate invalidity-style outcome when the
  design goal requires every relevant room part to reach save/refill.

This should be decided before unreachable finalized values affect generation
reward.

## Later Room-Part Node Architecture

Add a second sparse node type for placed room parts.

Node types:

- Frontier nodes use the current larger frontier embedding.
- Placed room-part nodes use a smaller room-part embedding.

Message passing:

- Frontier nodes exchange messages with neighboring frontier nodes.
- Room-part nodes exchange messages with neighboring frontier nodes.
- Frontier nodes receive messages from neighboring room-part nodes.
- Room-part nodes do not exchange messages directly with other room-part nodes.

Prediction routing:

- Placed room-part outcomes are predicted from room-part node embeddings.
- Unplaced room-part outcomes are predicted from the post-pooling global
  embedding and routed to the corresponding output indices.
- Finalized known outcomes override learned predictions.

Room-part node features:

- distance from nearest save
- distance to nearest save
- distance from nearest refill
- distance to nearest refill
- distance from furthest room part
- distance to furthest room part

Room-part/frontier pair features:

- distance from room part to frontier
- distance from frontier to room part

Frontier/frontier pair features:

- graph distance from source frontier to destination frontier
- graph distance from destination frontier to source frontier

The existing combined `room_part_frontier_distance` feature should be removed
when the pair-feature route is introduced.

## Open Design Checks

- Decide the unreachable reward objective before using finalized unreachable
  outcomes for generation reward.
- Decide whether directed distance predictions should be represented as four
  separate fields or grouped named objects in Python and PyO3 result classes.
- Decide the sparse room-part row identity scheme before implementing
  room-part nodes.
- Check the cost of frontier/frontier graph-distance pair features before
  enabling dense all-pairs features; prefer sparse edges if dense pairs are too
  expensive.
