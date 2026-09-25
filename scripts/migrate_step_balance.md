# Room placement-step balancing and checkpoint migration

The controller adds a separate config-conditioned `step_net` to `BalanceModel`.
For N concrete rooms it emits N × N placement-step prices and N failure prices
for rooms never placed. Equivalent rooms have independent rows. Each row is
centered over all N steps, giving the uniform target expected price zero.
The existing quadratic/quartic balance objective trains this network on fresh
experience, including failed episodes. No new experience tensors are required.

Step 1 includes the random initial placement in the target distribution. Prices
do not change initial-room sampling. Later candidates pay their concrete room's
exact current-step price plus the new value head's expected remaining cost.
Previously placed rooms' costs are omitted. At the episode horizon, all remaining
rooms incur their exact failure prices, including in deliberately shorter maps.

Proposals identify door variants rather than concrete rooms. Their immediate
price is the mean over unused concrete members of the corresponding room variant;
resolved candidates then pay their individual prices. This approximation changes
shortlisting without changing the room-resolution sampling algorithm.

The required new settings are:

| Setting | Meaning | Initial setting in checked-in configs |
| --- | --- | ---: |
| `balance_train.step_beta` | Price regularization strength; smaller values allow stronger balancing | 1.0 |
| `balance_train.step_price_scale` | Price magnitude where quadratic and quartic restoring gradients are equal | 1.0 |
| `train.step_balance_weight` | Supervision weight for the main model's remaining-cost prediction | 1.0 |

The first two settings support schedules. `step_balance_weight` is a prediction
loss weight, not a multiplier on the generation reward. Uniformity is a soft
objective; constraints can make individual room/step combinations infeasible.

Metrics include the new prediction loss and contribution, step-price RMS and
maximum magnitude, and failure-price statistics. `room_step_ss` is the mean
over concrete rooms of the finite-sample-corrected sum of squared placement-step
probabilities. It excludes unplaced observations and pools generation configs.
Uniform placement has expected value 1/N (about 0.003953 for Zebes); concentration
at one step gives 1. Finite-sample correction can put estimates below 1/N. This
pooled diagnostic cannot establish uniformity within each conditioning config.

## Migration

Training checkpoints advance from v22 to v23; exported models advance from v15
to v16. Existing checkpoints require explicit migration before loading.

```sh
conda run -n map-gen python scripts/migrate_step_balance.py \
    SOURCE.safetensors OUTPUT.safetensors \
    --step-beta 1 --step-price-scale 1 --step-balance-weight 1 --seed 0
```

The migration accepts only v22 checkpoints and refuses to overwrite an existing
output. It preserves every existing tensor exactly, existing optimizer parameter
IDs and moments, training counters, run identity, and experience history. New
parameters have no optimizer history. The price network's final layer and both
main/EMA remaining-cost heads start at zero, preserving generation scores at the
moment of migration. Hidden layers receive the supplied seed's initialization.
Both Adam and Muon main optimizers are supported; the controller uses Adam.

The script reads the source checkpoint's configuration and adds only the three
new settings. It writes a migration report into checkpoint metadata and prints
the report. The original checkpoint is left intact. Export the migrated checkpoint
again if a serving model is needed.
