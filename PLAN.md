# Compact Active-Row Generation Plan

Goal: continue reducing wave generation work based on measured bottlenecks after compacting candidate rows.

Local `configs/debug.json --profile` after step 1 showed:

- `python.wave.score_candidates`: ~67% of Python generation span.
- `python.wave.prepare_candidates`: ~11%.
- `python.wave.apply_candidates`: ~6%.
- `python.wave.score_proposals`: ~12%.
- `python.wave.prepare_proposal_features`: ~1%.
- `python.wave.prepare_proposal_mask`: less than 1%.

The next work should therefore target candidate-side volume before proposal-mask compaction.

## Steps

1. Complete - Score proposal features only for active environments.
   - Slice the wave proposal mask to active environment rows.
   - Extract state features with `extract_features_for_env_indices`.
   - Index lookahead outcomes and proposal temperature to the same active environment rows.
   - Keep candidate extraction using original environment indices.

2. Complete - Reduce candidate model scoring volume.
   - Identify candidate rows where fewer than `recommended_candidates` real candidates are available.
   - Avoid value-model forward for dummy candidate slots by packing real candidate snapshots before scoring.
   - Rust still prepares rectangular candidate features; reducing that remains separate work.
   - Preserve proposal-training target shape by padding logits after scoring only real slots.

3. Complete - Reduce candidate preparation work in Rust for dummy slots.
   - Inspect `pack_compact_wave_candidates_from_proposals_into` and `get_candidates_from_proposals`.
   - Skip feature plans and feature packing for padded dummy slots.
   - Keep rectangular action/outcome buffers for apply and proposal target shape.
   - Real candidate lookahead/outcome work remains the dominant Rust candidate-prep cost.

4. Pending - Reduce real candidate preparation work in Rust.
   - Inspect whether proposal candidates can be filtered earlier before `apply_lookahead`.
   - Avoid repeated lookahead/outcome work for candidates that cannot survive scoring/application.
   - Keep fallback behavior for rejected candidates that fit geometry.

5. Complete - Compact wave application inputs.
   - Avoid densifying scored candidates back to all environments.
   - Add a Rust apply API that accepts compact env ids plus sorted candidate attempts.
   - Scatter applied actions back to full episode tensors on the Python side.

6. Complete - Avoid recomputing candidate outcomes during wave application.
   - Derive a clean/fallback status mask from candidate-prep `clean_counts`.
   - Send that status through the compact apply API.
   - Apply clean candidates first and rejected fallback candidates second using only current geometry/conflict checks.
   - Remove duplicate lookahead/outcome/feature work from `apply_candidates`.

7. Pending - Re-check finished/stuck environment handling with profiling.
   - Ensure environments with no valid candidate attempts stop generating work immediately.
   - Confirm inactive envs no longer hit proposal/candidate scoring or apply work.

8. Complete - Split wave apply profiling.
   - Attribute CUDA synchronization before CPU candidate sorting separately.
   - Report candidate sorting and Rust apply as separate Python spans.
   - Keep generation behavior unchanged.

9. Complete - Remove Python row loops from wave apply bookkeeping.
   - Sort candidate attempts with vectorized stable sorts rather than per-env full-array scans.
   - Append applied actions with tensor scatter/gather rather than a Python row loop.
   - Preserve sorted-per-environment candidate order and full episode action layout.

10. Pending - Reduce candidate preparation and feature packing cost.
   - Inspect `get_candidates_from_proposals` and `pack_features` for repeated work across candidate rows.
   - Prioritize feature packing hotspots: `frontier_rows`, `frontier_neighbor`, and `frontier_occupancy`.
   - Preserve current proposal/candidate semantics while reducing redundant feature computation.

11. Complete - Add wave dense-work diagnostics.
   - Track active envs, active frontier slots, valid proposal rows, candidate rows, real candidate slots, apply env rows, applied env rows, and applied actions.
   - Normalize rows against dense frontier/env bounds to reveal remaining dense work.
   - Keep generation behavior unchanged.

12. Pending - Move proposal masks and sampling to compact active-row APIs if profiling supports it.
   - Add a Rust API that returns proposal rows only for active environments/frontiers with valid candidates.
   - Replace dense `[env, max_frontiers, door_variant]` sampling with `[proposal_row, door_variant]`.
   - Preserve original environment/frontier identity for candidate extraction and proposal training targets.

13. Pending - Clean up obsolete dense helpers.
   - Remove dense wave reshape/stat helpers that are no longer used.
   - Keep APIs explicit rather than retaining compatibility fallbacks.
