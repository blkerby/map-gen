# Neural dual-price balancing cutover

## Goal

Replace the delayed probability-estimation controller with a conditional neural
dual-price controller for all three balance families:

- compatible door pairings;
- Toilet crossing rooms;
- room-to-area assignments.

The generator will combine prices with ordinary rewards before dividing by
sampling temperature. This keeps their relative strength constant as temperature
changes. The balance model remains conditional on the full generation-configuration
vector and emits every price in one forward pass.

This is one backward-incompatible cutover. There will be no compatibility mode,
old/new feature flag, or fallback for old configs and checkpoints.

## Mathematical contract

For a categorical balance group with target distribution `q`, observed outcome
`y`, and learned price vector `lambda_theta(z)`, use the probability-error sampled
residual

```
r_k = 1[y = k] - q_k.
```

The stochastic dual objective for one observation is

```
D(theta) = sum_k r_k * lambda_theta(z)_k.
```

The optimizer minimizes `-D`. It raises the price of an overrepresented outcome
and lowers the alternatives. Each sampled coordinate residual is bounded
independently of category count and target probability, so the three families
have comparable loss scales.

Prices have an arbitrary common offset. Before use, center each group:

- door and Toilet prices: arithmetic mean over feasible outcomes;
- area prices: `q`-weighted mean.

Regularize each learned correction with `beta * lambda^2 / 2`. In expectation
under a fixed observed distribution `p`, the objective's stationary point is
`lambda = (p - q) / beta`. The fixed area prior is a reference-distribution
logit, not a learned correction, so beta does not regularize it. Incompatible
outcomes are excluded from targets, centering, updates, metrics, and generation.
Encountering an observed incompatible outcome is an error.

The area target distribution is applied as a temperature-independent base
measure. With `a = -log(q)`, final generation logits become

```
(ordinary_reward - learned_price) / temperature - a
```

and proposal logits use the analogous expression with `proposal_temperature`.
Equivalently, an area price table used inside the numerator contains
`learned_price + temperature * a`. Door and Toilet prices have no non-uniform
prior and remain

```
(ordinary_reward - learned_price) / temperature.
```

Thus temperature changes sampling sharpness without changing either the
price-to-reward ratio or the intended area target distribution `q`.

## Configuration schema

### Balance controller

Keep `balance_model.hidden_width` and `balance_model.num_layers`. Add a required
Adam `balance_optimizer` with `lr`, `beta1`, and `beta2`. Keep these required
fields in `balance_train`:

```
batch_size
door_beta
toilet_beta
area_beta
```

Checked-in optimizer values are `lr = 0.001`, `beta1 = 0`, and `beta2 = 0.99`;
the existing family beta values are retained. A zero optimizer learning rate
freezes all learned balance corrections. Take one Adam step per balance
minibatch, so `batch_size` controls the optimizer update frequency.

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
2. Replace probability/log-odds table construction with centered price
   tables. Add the `-log(q)` area prior before the learned residual so a new
   controller initially represents the requested area distribution.
3. Replace cross-entropy balance-model fitting with three quadratically
   regularized linear dual objectives. Shuffle the fresh samples, form
   minibatches, apply family-specific betas, validate finite gradients, and
   take one Adam step per minibatch without hidden gradient clipping.
4. Remove the balance EMA. Generation and main-model price supervision use the
   current balance model; retain the main model EMA unchanged.
5. Change main-model balance supervision from probability log-odds KL losses to
   price regression for doors, Toilet, and areas, preserving masks for known
   outcomes and already placed rooms. The area output predicts only the learned
   correction; the exactly known `log(q)` prior is not regressed.
6. Combine learned prices before temperature in final-candidate and proposal
   sampling, while adding the area `log(q)` prior afterward. Keep immediate
   known door/area substitutions so the exact value is used when an action
   determines an outcome.
7. Remove obsolete heat/water and area-tile model outputs, losses, rewards, and
   metrics. Add controller metrics for price RMS/max, target-vs-observed area
   counts, and main-model price tracking error.
8. Bump training-checkpoint and model-export formats. Store only the main model,
   main EMA, balance model, main optimizer, and the balance Adam optimizer;
   reject old formats as intended.
9. Update serving request/config construction and exports to use direct target
   room counts and preferred probabilities.

Gate: all unit tests pass, checkpoint save/load/export round-trips pass, and a
single debug round has finite losses/prices.

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
   - one dual optimizer step occurs per balance minibatch;
   - the balance-price-to-ordinary-reward ratio is independent of temperature;
   - checkpoint reload reproduces model outputs.

## Expensive-run protocol

Use Norfair for the first full run. Inspect controller and generation metrics at
roughly rounds 10 and 25, and treat checkpoint 100 as the continuation gate.
Continue toward at least eight million episodes only if prices remain finite,
main-model price tracking is improving, and door,
Toilet, and area distributions move toward their targets without a growing
alternating mode. Run Zebes only after that gate because it adds heat/water and
forced-special interactions.

## Explicitly deferred

- No family loss weights, replay training, alternating inner optimization, or
  statistical fallback table.
- No hard per-map area quota; expected counts are the initial contract.
- No separate exact tier-collapse branch; equal `0.75` maxima already allow
  arbitrarily close tier targets.
- No backward compatibility for configs, checkpoints, or serving requests.

Add any deferred mechanism only in response to a measured failure of the
minimal dual controller.
