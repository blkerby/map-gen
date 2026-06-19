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

Room-part rows are currently structural only. The model constructs internal
room-part node states from room-part identity, unresolved flags, known directed
save/refill encodings, and enabled room-part distance features, but these states
are not yet used by message passing or output heads.

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

## Step 1: Bounded Part-Frontier Message Passing

Add bidirectional sparse edges between unresolved room-part nodes and relevant
frontier nodes.

Use separate top-k bounds for each direction:

- `part <- frontier`: nearest or most relevant frontiers for each unresolved
  room part, including both graph directions where applicable.
- `frontier <- part`: most relevant unresolved room parts for each frontier,
  ranked by local objective pressure or potential improvement.

Rust should build candidate edge lists from graph-distance caches, rank them,
and pack the selected top-k edges into fixed-width tensors with `-1` padding.
Python should consume those tensors with the same gather-and-mask style as the
current frontier-neighbor message passing.

Edge features should include:

- directed graph distances for the edge
- same-component/reachability flags where useful
- local objective flags: save, refill, missing-connect endpoint
- improvement margin against the current finalized-known threshold where
  applicable

At each message-passing layer:

- frontier nodes receive frontier-neighbor messages
- frontier nodes receive room-part messages
- room-part nodes receive frontier messages
- both node types update from their current state, incoming messages, and global
  state

The final frontier node state remains `proposal_state`, so proposal scoring
automatically benefits from unresolved local-outcome information.

Track truncation diagnostics:

- unresolved room-part node count
- candidate part-frontier edge count before top-k
- fraction of part rows and frontier rows hitting each cap
- average and max selected fan-in/fan-out

If truncation is frequent and appears quality-limiting, raise caps or consider a
COO/segment-reduce representation for only the affected edge direction.

## Step 2: Local Outcome Heads

Route local outcomes through local node/query representations.

Save/refill utilities:

- For placed room parts with unresolved nodes, predict directed save/refill
  utilities from the room-part node state.
- For finalized entries, initially keep learned predictions unless a later
  experiment re-enables deterministic finalized-known overrides.
- For unplaced room parts, keep using the global pooled state.

Missing-connect outcomes:

- Treat each missing connection as a directed local query:
  `source_part -> destination_part`.
- Predict missing-connect validity and distance from endpoint states plus
  directed pair features and global state.
- If endpoints have room-part nodes, use those node states.
- If an endpoint is placed but omitted because all attached outcomes are
  finalized, use deterministic known values where available or a compact
  non-message-passed endpoint embedding.
- If the room is unplaced, route through the global state as today.

Door/frontier proposal outcomes:

- Keep proposal scoring on final frontier node states.
- Do not add a separate proposal-only integration path unless diagnostics show
  the final frontier state is not carrying local information effectively.

## Tests And Validation

Rust tests:

- part-frontier top-k edge packing and cap/truncation diagnostics
- missing-connect endpoint/query metadata

Python tests:

- model forward with zero room-part nodes and part-frontier edges
- model forward with room-part nodes and part-frontier edges
- output routing for placed, unplaced, finalized, and missing-connect outcomes

Validation commands:

- `cargo test`
- `conda run -n map-gen maturin develop`
- Python compile/smoke checks in the `map-gen` conda environment

## Open Design Checks

- Choose initial top-k caps for `part <- frontier` and `frontier <- part`.
- Define the exact ranking score for `frontier <- part` edges so proposal states
  receive the most useful unresolved local pressure.
- Decide whether missing-connect validity and missing-connect distance should
  share one directed query representation or use separate heads from the same
  query state.
- Revisit frontier-neighbor count after local nodes are added; extra
  frontier-frontier neighbors may become more useful late in training but should
  be evaluated separately from the room-part-node change.
- Decide when, if ever, to re-enable finalized-known deterministic overrides
  after proposal-visible local architecture is in place.
