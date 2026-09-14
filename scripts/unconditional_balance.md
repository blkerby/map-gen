The v19 balance model treats unmatched terminal doors, failed Toilet placement,
and unplaced terminal rooms as failure outcomes with target probability zero.
Successful targets retain the existing door compatibility and area constraints.
Every enabled door or room contributes to its controller objective, including
failures; normalization no longer depends on how many outcomes succeeded.

Successful prices are centered by their target probabilities. Each failure output
directly represents a price relative to that zero. All three cost predictions
estimate unconditional terminal cost, including failure. Generation subtracts
these costs directly and overrides known outcomes with their exact prices.

To migrate a v18 checkpoint, run from the repository root in the `map-gen` conda
environment, using experience from the current run:

```sh
python scripts/unconditional_balance.py INPUT.safetensors OUTPUT.safetensors \
  --experience EXPERIENCE.safetensors \
  --calibration-maps 512 --validation-maps 256 \
  --steps 1 4 16 64 128 192 224 248 \
  --batch-size 32 --threads 2 --seed 19 --ridge 0.01 \
  --frontier-samples-per-batch 1024
```

Choose prefix steps within the recorded episode length. The script refuses to
overwrite an existing output and writes a `.migration.json` report beside it.
The checkpoint format becomes v19; model exports become v12.

Migration changes only these parameters:

- The door and area controllers append zero failure rows to their final layers.
  Existing successful prices are preserved, and new failure prices start at zero.
- The Toilet controller's final failure row subtracts the feasible-success mean
  row, preserving its existing centered failure price in the new coordinates.
- The main and EMA global/frontier door cost final layers are fitted independently
  to the old conditional cost multiplied by predicted success probability.
  Their encoders and validity heads remain frozen. Main/EMA Toilet and area cost
  heads already predict unconditional costs and remain unchanged.

The last step is approximate: a linear head cannot generally reproduce the
product of a linear cost and a sigmoid probability exactly. Calibration and
validation use disjoint maps. The report compares held-out errors with unchanged
heads and scalar rescaling, excluding known outcomes and accounting for frontier
overrides when measuring global-head error. A fit that worsens held-out error
relative to unchanged heads is rejected.

Optimizer state is reset for the fitted main door cost final layers and the
Toilet controller final layer. The Toilet coordinate change has no exact Adam
second-moment mapping. Existing door/area optimizer rows are preserved and new
failure rows start with zero moments. All controller hidden layers and their
optimizer states remain unchanged and independent.

Aim now reports mean, RMS, and maximum absolute failure prices for each family as
`balance_{door,toilet,area}_failure_price_{mean,rms,max}`. Existing successful-price
metrics keep their meanings.
