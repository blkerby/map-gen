`generation.door_balance_strength` is a required, finite per-map
multiplier on all door balance prices, including unmatched-door prices. One
retains the existing strength; zero disables door balancing. The separate
unmatched-door reward and the other balance families keep their existing weights.

The field supports the existing scalar, range, mixture and schedule syntax. For
example, to train across independently sampled strengths:

```json
"door_balance_strength": {"linear": {"min": 0.0, "max": 1.0}}
```

A scalar `1.0` preserves the existing strength. Generation requests also require
`door_balance_strength`. Train over the range you intend to use so predictions
can adapt to how different strengths affect maps.

The controller and main model both train on unscaled door costs. During
generation, strength multiplies the main model's predicted door costs directly;
known outcomes use the already-scaled price tables. Proposal residuals use these
scaled tables too. This applies strength exactly once to predicted and exact
costs, including failures, and disables door balancing at zero strength.
Price metrics report the scaled prices applied to maps; `main_balance_loss`
measures prediction error against unscaled costs.

Training checkpoints use v30 and model exports use v24. The previous v29/v23
formats predicted scaled door costs and are rejected. Migrate the original v28
checkpoint again before resuming or exporting:

```sh
conda run -n map-gen python scripts/migrate_door_balance_strength.py SOURCE OUTPUT
```

Migration adds zero input columns to the main, EMA and five controller networks
and their optimizer moments. It sets strength to one, retains the existing
weights, counters and run identity, and refuses to overwrite an existing output.
Re-export serving models from the migrated checkpoint. Checkpoints trained with
scaled door-cost targets cannot be converted by changing their version marker.

Experience files advance from v7 to v8. Versioned imports of older experience
assign strength one, preserving the original episodes' full-balancing meaning.
