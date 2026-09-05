Review of generation and training, excluding serving. Recorded on 2026-09-04.

The review identified concrete bugs and three sampling/reward flaws that could
affect map quality. The highest priority is the save/refill query bug. No code
changes were made as part of the review.

1. **[P1] Save/refill queries overwrite each other's outputs and masks.**

   Rust legitimately emits separate queries for the same room part when travel
   directions need different frontier contexts. The Python head scatters all
   four outputs from every query, including inactive targets, into the same
   locations. Later queries overwrite earlier predictions and erase their masks.
   In the reduced Zebes run, **9,963 of 100,191 intended query outputs
   disappeared**. Scatter only the entries enabled by each query's target mask.

   Reference: [python/model.py:519](python/model.py#L519).

2. **[P2] Door balancing assigns target probability to impossible same-room connections.**

   Compatibility is based on room variants and geometry, without excluding
   connections between two doors belonging to the same concrete room. Zebes has
   **356 such directed entries affecting 322 doors**. Some Botwoon Quicksand Room
   doors assign two of eight target outcomes to impossible partners. The
   controller therefore cannot achieve its target distribution and learns
   distorted prices. Concrete compatibility needs to exclude identical room IDs.

   Reference: [python/loss.py:437](python/loss.py#L437).

3. **[P2] Proposal training can silently lose all gradients at low temperatures.**

   Invalid teacher logits use the fixed value `-10000`. At the configured
   late-stage temperature of `0.01`, a valid reward below approximately `-100`
   can score worse than that sentinel. Softmax then puts its probability on
   invalid candidates, which the subsequent loss mask discards. A valid reward
   of `-102` reproduced **zero loss and zero gradients**, despite the student
   assigning equal probability to a valid and invalid action. Exclude invalid
   candidates from the teacher's normalization while retaining them in the
   student's denominator.

   Reference: [python/learn.py:1126](python/learn.py#L1126).

4. **[P2] Lookahead-rejected proposals usually provide no negative supervision.**

   Geometrically impossible proposals enter the invalid-example buffers.
   Proposals rejected for breaking door, connectivity, area, or special-room
   constraints only enter the fallback list. When clean alternatives exist,
   these rejected proposals disappear from training. The proposal model
   consequently receives no direct signal to stop wasting its shortlist on
   those known failures. Preserve them as negative examples when clean
   alternatives are available.

   Reference: [src/environment.rs:3798](src/environment.rs#L3798).

5. **[P2, sampling flaw] Applying the area prior in both sampling stages distorts its intended probabilities.**

   The prior biases which candidates enter the shortlist, then biases selection
   within that shortlist again, without accounting for the first selection. In
   an isolated test with all candidates valid, zero rewards/prices, and eight
   shortlisted candidates, an area target of **50% produced approximately 80.5%
   selections**. This makes the requested distribution depend on
   candidate-selection settings and gives the balance controller additional bias
   to counteract.

   References: [proposal sampling, python/generate.py:1689](python/generate.py#L1689),
   [final sampling, python/generate.py:1115](python/generate.py#L1115).

6. **[P2, reward flaw] Save/refill rewards include predictions that failed episodes never supervise.**

   Training masks out room parts that were never placed, while generation sums
   utility predictions over every room part. This trains utility conditional on
   eventual placement, then uses it as unconditional expected utility. Failed
   continuations can therefore retain optimistic rewards for rooms they omit.
   Different candidate rewards were reproduced from omitted-room predictions
   whose training gradients were both zero. Use a consistent treatment of absent
   rooms, such as zero utility targets or explicit placement probabilities.

   References: [training mask, python/learn.py:1437](python/learn.py#L1437),
   [generation reward, python/generate.py:273](python/generate.py#L273).

7. **[P2, reward flaw] Area-position scoring ignores outcome variance.**

   The model learns mean final coordinates, but generation squares the
   difference between that mean and the target. This differs from expected
   squared error. For a target of `0.5`, an action yielding either `0` or `1`
   scores perfectly because its mean is `0.5`; an action consistently yielding
   `0.6` scores worse, despite being much closer on every map. Predicting squared
   error directly, or also predicting second moments, would align scoring with
   per-map accuracy.

   Reference: [python/generate.py:251](python/generate.py#L251).

8. **[P2] Outcome verification crashes on unfinished Zebes maps.**

   `--verify-outcome-consistency` calls the full outcome getter after each
   placement. That getter attempts to convert unknown heat/water outcomes (`-1`)
   into unsigned final outcomes and panics. This was reproduced immediately on
   Zebes. Normal generation works with the flag disabled, but the validation
   path cannot check these maps. Separate intermediate verification from
   final-outcome extraction.

   References: [python/generate.py:1219](python/generate.py#L1219),
   [src/engine.rs:1279](src/engine.rs#L1279).

9. **[P2] A clean training stop does not save the latest model.**

   Checkpoints are saved only on the periodic boundary. Neither normal
   completion nor the handled SIGINT/SIGTERM stop saves the remaining updates.
   With the checked-in period of 100 rounds, a clean stop can discard **up to 99
   rounds of trained model state**.

   Reference: [python/train.py:2281](python/train.py#L2281).

10. **[P3] The selected-candidate probability metric uses different logits from generation.**

    Diagnostics reconstruct probabilities using the proposal's immediate balance
    adjustment. Actual selection includes the complete door, future-area, and
    Toilet balance terms. Consequently, `candidate_selected_probability` can
    misrepresent sampling sharpness and mislead tuning. Record the actual
    sampling logits instead of reconstructing them.

    References: [diagnostics, python/learn.py:495](python/learn.py#L495),
    [actual scoring, python/generate.py:1088](python/generate.py#L1088).

Validation: **99 Rust tests passed; 64 of 66 Python tests passed** using a direct
runner because `pytest` was unavailable. The two failures are outdated
[lookahead fixtures](python/test_lookahead_area_features.py#L19) and
[output-head references](python/test_model_initialization.py#L100). Reduced CPU
generation/training runs for Crateria and Zebes produced finite losses and no
captured feature mismatches; Zebes required disabling the broken
outcome-verification flag. GPU and compiled execution remain untested. The
reduced runs do not establish the magnitude of these issues in a full training
run.

A useful refactor would extend the existing candidate result with rejection
status and actual sampling logits, so generation, supervision, and diagnostics
share the same recorded decisions.
