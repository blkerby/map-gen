# Per-room area feature and v22 checkpoint migration

`features.room_area` is a required boolean. It is enabled in the checked-in
configs. The model receives six indicators for each concrete room, in room-list
order, with areas 0–5 in order. Unplaced rooms contribute six zeros. The feature
is appended to the global input and reaches both value and proposal predictions.
Unlike frontier-area inputs, it remains available after a room has no open doors.

Rebuild the extension on the machine where training will run:

```sh
conda activate map-gen
maturin develop --release
```

Prepare a current v21 training checkpoint, then migrate to a **new** path:

```sh
python scripts/migrate_room_area.py SOURCE.safetensors OUTPUT.safetensors
```

The migration:

- Sets `features.room_area` to true and changes the training format to v22.
- Appends `6 * num_rooms` zero columns to `global_mlp.weight` in both main and EMA
  models (1,518 columns for Zebes).
- Appends matching zero columns to optimizer moments, retaining all existing
  tensor values, parameter IDs, optimizer scalar state, counters, run identity,
  and replay history. Both Adam and Muon are supported.
- Strictly loads the migrated model and optimizer states and verifies every
  tensor against the source before publishing the output atomically.
- Prints a JSON report and embeds it in the checkpoint metadata as
  `room_area_migration`. It refuses to overwrite an existing file.

Predictions are mathematically unchanged immediately after migration, up to
floating-point rounding from the wider projection. The new input weights receive
nonzero gradients immediately; no warm-up or calibration dataset is required.
The balance controller and its optimizer are unchanged. Existing experience files
already contain room-area actions and reconstruct the feature during replay.

Use a training config with `features.room_area: true` when resuming. Keep the
output in the same run's `checkpoints` directory when retaining that run's replay
files. The normal loaders remain strict: an unmigrated v21 checkpoint or a config
missing the new field is rejected. Exported models now use format v15 and should
be regenerated from the migrated training checkpoint.

## Validation

The focused tests cover room identity, unplaced rooms, disabled features, padding,
candidate/restored/current/replayed states, and closed one-door rooms. Migration
tests use synthetic v21 checkpoints with populated Adam and Muon histories,
compare all prediction heads and proposal outputs, and execute a training update.

```sh
PYTHONPATH=python python -m unittest -v test_room_area_feature test_room_area_migration
cargo test
```

A two-round CPU training check with both `--verify-feature-consistency` and
`--verify-outcome-consistency` completed, including historical replay in round 2;
all 1,144 compared feature tensors (62,275 values) matched exactly.
