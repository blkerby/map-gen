# Local Outcome Architecture Plan

## Goal

Put local outcomes on a sounder modeling footing while keeping generation cost
predictable.

The model should use local representations for outcomes whose truth depends on
specific room parts, missing connections, and nearby frontiers. This local
information must enter the shared frontier representation used for proposal
scores, not only the final output heads, because generation fully scores only a
small number of proposed candidates.

## Guiding Design

- Improve proposal-visible local representations before hard-wiring local
  rewards into model outputs. Prior experiments with deterministic
  finalized-known save/refill overrides regressed generation quality, likely by
  making short-term reward fulfillment too attractive before the model could
  represent long-term consequences accurately.
- Add graph nodes only for unresolved placed room parts. A placed room part is
  included as a local node when at least one attached local outcome can still be
  affected by future placements.
- Keep unplaced room-part outcomes routed through the global state.
- Use bounded, fixed-width neighbor tensors for the first implementation. Rust
  may generate dynamic candidate edges, but Python should receive top-k padded
  tensors so message-passing cost remains predictable.
- Let frontier and room-part nodes exchange messages in both directions at each
  layer. The final frontier node state remains the proposal state.

## Current State: Directed Features And Room-Part Nodes

The global room-part distance features now expose directional information
instead of round-trip/compressed distances.

Per room part, the global features include:

- distance from room part to nearest save
- distance from nearest save to room part
- distance from room part to nearest refill
- distance from nearest refill to room part
- distance from room part to nearest frontier
- distance from nearest frontier to room part
- furthest destination distance
- furthest source distance

These features use the compact distance encoding convention:

- `0` for unreachable or absent
- `distance + 1` for finite distances, saturated as needed

The model still uses the current global-feature route for these signals. No
deterministic finalized-known utilities are substituted into outputs.

The feature pipeline also emits unresolved room-part node rows. Each row stores:

- row-to-snapshot index
- global room-part index
- unresolved objective flags for directed save/refill and missing-connect
  endpoint status

The model constructs internal room-part node states from room-part identity,
unresolved flags, known directed save/refill encodings, and enabled room-part
distance features. These states are now used by part-frontier message passing,
but not yet by local output heads.

## Current State: Configurable Local Widths

The model separates the main local widths so capacity can be tuned without
forcing the whole architecture to scale together.

Current required model config fields include:

- `embedding_width`: pooled/output-state width for global output heads.
- `frontier_embedding_width`: frontier node width and proposal-state width.
- `room_part_embedding_width`: unresolved room-part node width.
- `global_embedding_width`: global context width used by message-passing
  updates.
- `global_room_position_embedding_width`: global room-position feature width.
- `pooling_hidden_width`: hidden width for the pooled-state MLP.
- `frontier_message_hidden_width`: hidden/update width for frontier-to-frontier
  messages.
- `part_from_frontier_message_hidden_width`: hidden/update width for
  `part <- frontier` messages.
- `frontier_from_part_message_hidden_width`: hidden/update width for
  `frontier <- part` messages.
- `proposal_hidden_width`: proposal-score MLP hidden width.

The default configs currently keep the new local widths equal to the previous
shared values, so the split is behavior-neutral until tuned. A smoke check with
different frontier, room-part, pooled, global, and message widths passed.

## Deferred: Finalized-Known Directed Overrides

Finalized-known directed save/refill overrides remain a plausible later
optimization, but should be deferred until proposal-visible local structure is
stronger.

For room part `p`, the candidate rules are:

- `p -> nearest save` is finalized when the current finite `p -> save` distance
  is less than or equal to the current `p -> nearest frontier` distance.
- `nearest save -> p` is finalized when the current finite `save -> p` distance
  is less than or equal to the current `nearest frontier -> p` distance.
- The same rules apply for refill distances.
- Unreachable outcomes are finalized when neither the objective nor any frontier
  is reachable in the relevant direction.

If reintroduced, known values should use the same numeric scale as the target:

- finite finalized distances: `scale / (d + scale)`
- finalized unreachable distances: `0`

Before enabling these overrides for generation, validate that the proposal
representation can model long-term tradeoffs well enough that deterministic
short-term reward improvements do not dominate candidate selection.

## Current State: Bounded Part-Frontier Message Passing

The model now uses bidirectional sparse edges between unresolved room-part nodes
and relevant frontier nodes.

The first implementation uses separate fixed-width top-k bounds for each
direction:

- `part <- frontier`: 2 frontiers per unresolved room-part node.
- `frontier <- part`: 8 room-part nodes per frontier.
- The first 2 `frontier <- part` slots are reserved for missing-connect
  endpoint pressure. Unused reserved slots are backfilled by the general ranking.

The two directions are selected independently; no edge symmetry is required.
Rust builds dynamic candidate edge lists from graph distances, ranks them, and
packs selected top-k edges into fixed-width tensors with `-1` padding. Python
consumes those tensors with the same gather-and-mask style as the
frontier-neighbor message passing.

Edge features include:

- directed graph distances for the edge
- same-component and directional reachability flags
- local objective flags: save, refill, missing-connect endpoint

At each message-passing layer:

- frontier nodes receive frontier-neighbor messages
- frontier nodes receive room-part messages
- room-part nodes receive frontier messages
- both node types update from their current state, incoming messages, and global
  state

The final frontier node state remains `proposal_state`, so proposal scoring
automatically benefits from unresolved local-outcome information.

Aim diagnostics track:

- unresolved room-part node count
- average selected fan-in for both directions
- missing-connect and general selected fan-in for `frontier <- part`
- cap-hit rates for `part <- frontier`, missing-connect-reserved
  `frontier <- part`, and general `frontier <- part`

If truncation is frequent and appears quality-limiting, raise caps or consider a
COO/segment-reduce representation for only the affected edge direction.

## Remaining Step 1 Validation: Message-Passing Tuning

The message-passing path is implemented, but still needs empirical tuning.

Key checks:

- Monitor selected fan-in and cap-hit diagnostics during training.
- Decide whether `part <- frontier = 2`, `frontier <- part = 8`, and
  missing-connect reserved count of 2 are adequate.
- Refine `frontier <- part` ranking if save/refill pressure crowds out
  missing-connect endpoints or if general slots are poorly used.
- Try narrower `room_part_embedding_width` and/or direction-specific message
  hidden widths if room-part message passing is too expensive.
- Revisit frontier-neighbor count separately; extra frontier-frontier neighbors
  may become useful late in training once local nodes are active.

## Current State: Local Output Heads

Local outcomes now route through local node/query representations when matching
local rows exist. Existing pooled/global heads remain as fallbacks, so public
prediction shapes and loss/reward consumers are unchanged.

Save/refill utilities:

- For placed room parts with unresolved nodes, predict directed save/refill
  utilities from the room-part node state using a linear local head.
- Scatter local values over the existing full utility tensors only for row flags
  matching the unresolved objective direction.
- For unplaced, finalized, or otherwise nonlocal entries, keep the pooled/global
  prediction.

Missing-connect outcomes:

- Treat each missing connection as a directed local query:
  `source_part -> destination_part`.
- Predict missing-connect validity and distance from endpoint room-part states
  plus global state using a linear query head.
- Scatter local validity logits into the existing `connection_invalid` tensor.
- Scatter local distances into the existing `missing_connect_distance` tensor.
- Use local predictions only when both endpoints have local rows in the same
  snapshot; one-sided endpoint rows are treated as an internal error.
- If endpoints are not local, keep the pooled/global prediction.

Door/frontier proposal outcomes:

- Keep proposal scoring on final frontier node states.
- Do not add a separate proposal-only integration path unless diagnostics show
  the final frontier state is not carrying local information effectively.

## Tests And Validation

Completed validation:

- `cargo test`
- `conda run -n map-gen maturin develop`
- `conda run -n map-gen python -m compileall python`
- model forward smoke with part-frontier tensors
- model forward smoke with intentionally different frontier, room-part, pooled,
  global, and message hidden widths
- model forward smoke covering local output-head shapes and output metadata

Still useful tests to add:

- part-frontier top-k edge packing and cap/truncation diagnostics
- missing-connect endpoint/query metadata
- model forward with zero room-part nodes and part-frontier edges
- model forward with room-part nodes and part-frontier edges
- output routing for placed, unplaced, finalized, and missing-connect outcomes

## Open Design Checks

- Evaluate whether linear local output heads have enough capacity or need hidden
  layers after training results are available.
- Decide when, if ever, to re-enable finalized-known deterministic overrides
  after proposal-visible local architecture is in place.
