# Neural dual-price balancing cutover

## Goal

Replace the delayed probability-estimation controller with a conditional neural
dual-price controller for all three balance families:

- compatible door pairings;
- Toilet crossing rooms;
- room-to-area assignments.

The generator will subtract prices *after* dividing ordinary rewards by sampling
temperature. This separates balance-loop gain from the temperature schedules that
caused the Zebes oscillation. The balance model remains conditional on the full
generation-configuration vector and emits every price in one forward pass.

This is one backward-incompatible cutover. There will be no compatibility mode,
old/new feature flag, or fallback for old configs and checkpoints.

## Mathematical contract

For a categorical balance group with target distribution `q`, observed outcome
`y`, and learned price vector `lambda_theta(z)`, use the target-normalized sampled
residual

```
r_k = (1[y = k] - q_k) / q_k.
```

The stochastic dual objective for one observation is

```
D(theta) = sum_k r_k * lambda_theta(z)_k.
```

Gradient descent therefore minimizes `-eta * D`. It raises the price of an
overrepresented outcome and lowers the alternatives. For uniform `q_k = 1/K`,
the residual is `K * 1[y = k] - 1`, so a family does not become weaker merely
because it has more categories.

Prices have an arbitrary common offset. Before use, center each group:

- door and Toilet prices: arithmetic mean over feasible outcomes;
- area prices: `q`-weighted mean.

Then clamp applied and supervised prices to `[-price_limit, price_limit]`.
Incompatible outcomes are excluded from targets, centering, updates, metrics,
and generation. Encountering an observed incompatible outcome is an error.

The generation logits become

```
ordinary_reward / temperature - balance_price
```

and proposal logits use the analogous

```
proposal_score / proposal_temperature - balance_price.
```

Thus `eta`, rather than `eta / temperature`, controls the balance feedback.

## Configuration schema

### Balance controller

Keep `balance_model.hidden_width` and `balance_model.num_layers`.

Replace `balance_optimizer` and `balance_train.ema_half_life_episodes` with
required fields in `balance_train`:

```
batch_size
door_eta
toilet_eta
area_eta
price_limit
```

Initial values for all three gains are `0.02`; the initial price limit is `20`.
Use plain SGD with no momentum and one optimizer step per generated round.
`batch_size` only chunks the forward/backward computation, so changing it does
not change the round-level gain.

Remove generation-time `reward_balance`, `reward_toilet_balance`, and
`reward_area_balance`; their role is now intrinsic to the dual prices.

### Area targets

Replace `target_area_tiles` with six required `target_area_rooms` values. Sample
the six values with the existing variable-float mechanism, require them to be
finite and positive, and normalize each sampled row
to sum to the room count. The normalized values are the baseline expected room
counts; their division by the room count gives baseline area probabilities `b`.

Remove `reward_area_tiles` and the main model's `area_tiles` regression output
and loss. Retain hard area-size validity, area bounding-box targets, and their
existing prediction/reward paths. The new balance target controls expected room
counts, not the variance of per-map counts; log actual count error/variance so a
quota mechanism is added only if a run demonstrates it is necessary.

### Heat/water preferences

Replace the heat/water rewards and tile-floor scales with these required objects:

```
"maridia_water_preferred_probability": {
  "active_probability": 0.5,
  "tier_max": [0.75, 0.75, 0.75]
},
"norfair_heat_preferred_probability": {
  "active_probability": 0.5,
  "tier_max": [0.75, 0.75, 0.75]
}
```

For preferred-area baseline probability `b`:

1. Draw one active Bernoulli per environment and family.
2. If inactive, set every tier probability to `b`.
3. If active, draw tier 3 uniformly from `[b, tier_max[2]]`, tier 2 from
   `[b, min(tier3, tier_max[1])]`, and tier 1 from
   `[b, min(tier2, tier_max[0])]`.
4. Fail if a configured tier maximum is below `b`; do not silently reinterpret
   a maximum.

Put the six sampled probabilities into the generation-configuration vector so
both neural models can condition on them. Remove heat/water reward fields,
reward counts, main-model heads, auxiliary losses, and target-area floor logic.

For a tagged room with preferred area `a*` and sampled probability `rho`, build
its area target row as

```
q[a*] = rho
q[a]  = (1 - rho) * b[a] / (1 - b[a*])  for a != a*.
```

Untagged rooms use `q = b`. Effective target area counts are `sum_room q`; they
may grow in Norfair or Maridia in response to heat/water preferences while the
total remains the number of rooms.

### Forced special rooms

When a vanilla-area force flag is active, use a one-hot area row only when
computing effective target counts. Mask that room out of the area dual loss and
apply zero area-balance price to it. The existing validity objective remains
responsible for satisfying the hard placement request. When the flag is
inactive, use the ordinary baseline/preference target and enable its dual row.
This prevents price drift when a forced placement fails.

## Milestone 1: target construction and config plumbing

1. Add strict Pydantic types and validation for the new balance fields,
   `target_area_rooms`, and the two tiered preference objects.
2. Update `GENERATION_VARIABLE_FLOAT_FIELDS` to contain normalized target room
   counts and six sampled preferred probabilities, and remove obsolete reward
   and tile-target fields.
3. Implement one pure tensor helper that constructs per-room `q`, the forced
   dual mask, and effective target counts from room metadata and sampled config.
4. Mask the Toilet room itself from Toilet crossing targets while keeping the
   existing output indexing; assert that every observed crossing is feasible.
5. Update Norfair and Zebes configs. Norfair has no tagged rooms, but still uses
   the same schema; Zebes exercises preference and forced-room paths.
6. Add focused tests for normalization, tier ordering/ranges, heat/water rows,
   effective-count growth, forced rows, and invalid inputs.

Gate: config tests and target-construction tests pass before modifying the
controller or generator.

## Milestone 2: dual controller and generation cutover

1. Reinterpret `BalanceModel` outputs as residual prices, retaining its compact
   direction-local door-variant representation and single-pass output layout.
2. Replace probability/log-odds table construction with centered bounded price
   tables. Add the `-log(q)` area prior before the learned residual so a new
   controller initially represents the requested area distribution.
3. Replace cross-entropy balance-model fitting with the three linear dual
   objectives. Accumulate chunked gradients over the full fresh round, apply
   family-specific etas, validate finite gradients, and take one plain-SGD
   step without hidden gradient clipping.
4. Remove the balance EMA. Generation and main-model price supervision use the
   current balance model; retain the main model EMA unchanged.
5. Change main-model balance supervision from probability log-odds KL losses to
   price regression for doors, Toilet, and areas, preserving masks for known
   outcomes and already placed rooms.
6. Apply balance prices after temperature in final-candidate and proposal
   sampling. Keep immediate known door/area substitutions so the exact table
   price is used when an action determines an outcome.
7. Remove obsolete heat/water and area-tile model outputs, losses, rewards, and
   metrics. Add controller metrics for price RMS/max, saturation fraction,
   target-vs-observed area counts, and main-model price tracking error.
8. Bump training-checkpoint and model-export formats. Store only the main model,
   main EMA, balance model, main optimizer, and the plain balance optimizer;
   reject old formats as intended.
9. Update serving request/config construction and exports to use direct target
   room counts and preferred probabilities.

Gate: all unit tests pass, checkpoint save/load/export round-trips pass, and a
single debug round has finite losses/prices with no saturated initial outputs.

## Milestone 3: cheap integration validation

1. Run formatting, Python tests in the `map-gen` conda environment, Rust tests,
   and static/import checks.
2. Run the debug config long enough to exercise generation followed by a dual
   update and a main-model update.
3. Run a reduced Norfair-shaped smoke test to exercise its real room/door output
   dimensions without paying for a full training run.
4. Run one small Zebes batch to cover tagged-room `q`, forced special-room masks,
   and incompatible door masks.
5. Verify these invariants from logs/tests:
   - each `q` row sums to one and is strictly positive on active outcomes;
   - incompatible/forced entries receive neither gradient nor applied price;
   - effective area counts sum to the room count;
   - one dual optimizer step occurs per round regardless of chunk count;
   - balance price corrections are independent of sampling temperature;
   - checkpoint reload reproduces model outputs.

## Expensive-run protocol

Use Norfair for the first full run. Inspect controller and generation metrics at
roughly rounds 10 and 25, and treat checkpoint 100 as the continuation gate.
Continue toward at least eight million episodes only if prices remain finite,
saturation is negligible, main-model price tracking is improving, and door,
Toilet, and area distributions move toward their targets without a growing
alternating mode. Run Zebes only after that gate because it adds heat/water and
forced-special interactions.

## Explicitly deferred

- No PI derivative/proportional term, leak, adaptive gain, replay training, or
  statistical fallback table.
- No hard per-map area quota; expected counts are the initial contract.
- No separate exact tier-collapse branch; equal `0.75` maxima already allow
  arbitrarily close tier targets.
- No backward compatibility for configs, checkpoints, or serving requests.

Add any deferred mechanism only in response to a measured failure of the
minimal dual controller.
