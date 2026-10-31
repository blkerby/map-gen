`generation.door_balance_strength` is a required, finite per-map
multiplier on all door balance prices, including unmatched-door prices. One
retains the existing strength; zero disables door balancing. The separate
unmatched-door reward and the other balance families keep their existing weights.

The field supports the existing scalar, range, mixture and schedule syntax. For
example, to train across independently sampled strengths:

```json
"door_balance_strength": {"linear": {"min": 0.0, "max": 1.0}}
```

Checked-in configs use `1.0` to preserve their behavior. Generation requests also
require `door_balance_strength`. Train over the range you intend to use; the new
input alone does not teach a migrated model how different strengths affect maps.

The controller trains its unscaled prices with its existing objective. Generation
scales the resulting price tables for proposal residuals, exact candidate costs,
and main-model cost targets. The main model predicts these scaled costs; at zero
strength generation overrides predicted door costs with zero. Price metrics
report the scaled prices applied to maps.

Training checkpoints advance from v28 to v29 and model exports from v22 to v23.
Migrate an existing training checkpoint before resuming or exporting:

```sh
conda run -n map-gen python scripts/migrate_door_balance_strength.py SOURCE OUTPUT
```

Migration adds zero input columns to the main, EMA and five controller networks
and their optimizer moments. It sets strength to one, retains the existing
weights, counters and run identity, and refuses to overwrite an existing output.
Re-export serving models from the migrated checkpoint.

Experience files advance from v7 to v8. Versioned imports of older experience
assign strength one, preserving the original episodes' full-balancing meaning.
