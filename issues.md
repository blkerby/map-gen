Review of generation and training, excluding serving. Recorded on 2026-09-04.

The review identified concrete bugs and three sampling/reward flaws that could
affect map quality. Completion notes track fixes below; items without a
completion note remain open.

1. **[P1, completed] Save/refill queries overwrite each other's outputs and masks.**

   Rust legitimately emits separate queries for the same room part when travel
   directions need different frontier contexts. The Python head scattered all
   four outputs from every query, including inactive targets, into the same
   locations. Later queries overwrote earlier predictions and erased their masks.
   In the reduced Zebes run, **9,963 of 100,191 intended query outputs
   disappeared**.

   Reference: [python/model.py:519](python/model.py#L519).

   **Completed 2026-09-04.** The head now masks inactive predictions before
   accumulating values and combines target masks with a Boolean OR. Split
   queries preserve both directions' predictions, and gradients flow only
   through active targets. The reductions keep tensor sizes suitable for
   compilation without filtering to a variable-length index.

2. **[P2, completed] Door balancing assigns target probability to impossible same-room connections.**

   Compatibility was based on room variants and geometry, without excluding
   connections between two doors belonging to the same concrete room. Zebes has
   **356 such directed entries affecting 322 doors**. Some Botwoon Quicksand Room
   doors assign two of eight target outcomes to impossible partners. The
   controller therefore cannot achieve its target distribution and learns
   distorted prices.

   **Completed 2026-09-04.** The balance model precomputes and stores concrete
   door-pair compatibility masks for all four directions, excluding identical
   room IDs. Variant compatibility is used only during construction; the balance
   model no longer stores that mask. Variant-index mappings remain for expanding
   predicted prices. Loss validation, price centering, and training diagnostics
   share the cached concrete masks.

   Validation: all 11 balance tests passed, including same-room exclusion,
   compatibility between distinct rooms sharing a variant, zero prices when no
   partner is feasible, and finite gradients toward the corrected distribution.
   Real metadata checks removed exactly 356 directed pairs affecting 322 Zebes
   doors (44 pairs / 40 doors in debug), preserving every other compatibility
   entry. Reduced debug and Zebes generation/training runs produced finite losses
   and gradients with no captured feature mismatches. Zebes outcome verification
   remained disabled for the separate issue 8. CUDA was not tested.

   Checkpoint compatibility: the stored balance-model mask buffers have changed;
   older checkpoints will fail strict loading. No fallback loading was added.

   References: [python/model.py](python/model.py), [python/loss.py](python/loss.py).

3. **[P2, completed] Proposal training can silently lose all gradients at low temperatures.**

   Invalid teacher logits previously used the fixed value `-10000`. At the configured
   late-stage temperature of `0.01`, a valid reward below approximately `-100`
   can score worse than that sentinel. Softmax then puts its probability on
   invalid candidates, which the subsequent loss mask discards. A valid reward
   of `-102` reproduced **zero loss and zero gradients**, despite the student
   assigning equal probability to a valid and invalid action.

   **Completed 2026-09-04.** Increased the shared invalid teacher-logit sentinel
   to `-1_000_000`, retaining finite arithmetic because an earlier `-inf`
   approach reportedly produced gradient NaNs. Invalid examples remain in the
   student's denominator. This increases the numerical margin rather than
   eliminating the finite threshold: at temperature `0.01`, valid rewards near
   `-10_000` can still approach the sentinel, ignoring logit offsets.

   The regression test reproduced zero loss with the old constant. With the new
   constant, reward `-102` at temperature `0.01` gives loss `log(2)` and finite
   score gradients `[-50, 50]`, increasing the valid score and decreasing the
   invalid score under gradient descent. All 18 proposal-action tests and four
   candidate-record tests passed on CPU.

   Reference: [python/learn.py](python/learn.py),
   [regression test](python/test_proposal_action.py).

4. **[P2, completed] Lookahead-rejected proposals usually provide no negative supervision.**

   Proposals failing action resolution entered the invalid-example buffers.
   Proposals rejected for breaking door, connectivity, area, or special-room
   constraints during lookahead only entered the fallback list. When clean
   alternatives existed, these rejected proposals disappeared from training.
   The proposal model consequently received no direct signal to stop wasting
   its shortlist on those known failures.

   Reference: [src/environment.rs:3691](src/environment.rs#L3691).

   **Completed 2026-09-04.** When clean choices exist, action-resolution failures
   and lookahead rejections now share the `num_scored_invalid_candidates` budget
   in evaluation order, including rejections from postponed candidates. The
   shared candidate record carries explicit lookahead-rejection flags through
   Rust packing, Python generation, and training. When no clean choices exist,
   fallback candidates remain selectable and retain reward-based supervision;
   they are not also inserted as contradictory negative examples.

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

6. **[P2, completed] Save/refill rewards include predictions that failed episodes never supervise.**

   Training previously masked out room parts that were never placed, while generation sums
   utility predictions over every room part. This trains utility conditional on
   eventual placement, then uses it as unconditional expected utility. Failed
   continuations can therefore retain optimistic rewards for rooms they omit.
   Different candidate rewards were reproduced from omitted-room predictions
   whose training gradients were both zero.

   **Completed 2026-09-04.** All four save/refill utility targets are explicitly
   zero for room parts absent from the terminal episode. Every room part now
   contributes to utility supervision in both fresh and replay training,
   including the loss normalization. Placed-room targets keep the existing
   distance-based utility; unreachable paths remain zero. Placement in the
   current partial map does not gate supervision, so predictions still account
   for utility from rooms that may be placed later. Generation continues to sum
   these predictions as expected terminal utility.

   Validation: the new training-path regression fails against the old loss mask
   and passes for both fresh and replay training. It checks explicit zero targets
   even with reachable-distance data for absent rooms, unchanged placed-room
   targets, and finite downward gradients for positive absent-room predictions.
   Four utility-query tests passed; one CUDA test was skipped. Reduced CPU debug
   and Zebes generation/training runs had finite losses and gradients and no
   captured feature mismatches (Zebes outcome verification disabled for issue 8).

   References: [training targets and masks, python/learn.py](python/learn.py),
   [regression test](python/test_utility_supervision.py),
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

9. **[P2, completed] A clean training stop does not save the latest model.**

   Checkpoints were saved only on the periodic boundary. Neither normal
   completion nor the handled SIGINT/SIGTERM stop saved the remaining updates.
   With the checked-in period of 100 rounds, a clean stop could discard **up to 99
   rounds of trained model state**.

   Reference: [python/train.py:2229](python/train.py#L2229).

   **Completed 2026-09-04.** Normal completion and the first SIGINT/SIGTERM now
   save the latest completed round before cleanup, without repeating a periodic
   checkpoint already saved for that round. The first interrupt lets the current
   round finish; generation workers ignore terminal SIGINT so they can finish
   their work. A second stop signal kills child processes and exits immediately
   without checkpointing or waiting for cleanup, including during a save.
   Existing checkpoints retain the temporary-file/atomic-replace protection;
   a forced exit during writing can leave an incomplete `.tmp` file. Failed
   partial rounds are not saved as completed checkpoints.

10. **[P3, completed] The selected-candidate probability metric uses different logits from generation.**

    Diagnostics reconstructed probabilities using the proposal's immediate balance
    adjustment. Actual selection includes the complete door, future-area, and
    Toilet balance terms. Consequently, `candidate_selected_probability` can
    misrepresent sampling sharpness and mislead tuning.

    References: [diagnostics, python/learn.py:469](python/learn.py#L469),
    [actual scoring, python/generate.py:905](python/generate.py#L905).

    **Completed 2026-09-04.** Generation now records the final sampling logits,
    including temperature, all balance terms, and the area prior. Diagnostics
    use those recorded logits independently of proposal-training targets.
    Negative-only examples and dummy candidates have `-inf` sampling logits;
    selectable fallback candidates retain their actual logits and probabilities.

The shared-candidate refactor for issues 4 and 10 is complete. Named
`CandidateBatch` and `CandidateSelection` results replace positional unpacking;
`ProposalData` carries rejection flags and sampling logits through transfers,
slicing, and aggregation. `invalid` marks negative-only training examples;
`rejected` records lookahead failures, including selectable fallbacks.

Validation of issues 4 and 10: **101 Rust tests and 21 focused Python tests
passed**. Tests cover the shared budget, postponed rejections, fallback handling,
actual sampler probabilities, and downward gradients for rejected negatives.
Reduced debug/Zebes generation and training runs produced finite losses and
gradients. Two CPU training rounds with two generation groups and two iterations
per round carried **219 lookahead negatives and 93 fallback candidates** through
the pipeline; the resulting checkpoint was verified and reloaded. CUDA was not
available for this validation. The Rust bindings were rebuilt locally.

Investigated observation from issue 9 validation (2026-09-04): the second CPU
training round originally reported 84 mismatched feature values in `room_x`
and `room_y`. These warnings recurred during issues 4/10 validation. The cause
is stale coordinates for **unplaced rooms**, not inconsistent room placements:
[environment reset](src/environment.rs#L1899) clears `room_used` but retains
`room_x`/`room_y`, and [feature extraction](src/environment.rs#L5654) copies
those coordinates unchanged. Generation and training environments can have
different prior placements, so their unused coordinates differ after reset.
[Feature verification](python/learn.py#L628) compares these unused values too.

An instrumented repeat of the issue 9 two-round CPU run reproduced 111 coordinate
mismatches (56 x, 55 y) at step 2. All were for rooms unplaced in both snapshots;
placement masks and all placed-room coordinates matched. The
[room-position feature](python/features.py#L437) masks out unplaced rooms:
its outputs and parameter gradients were exactly equal for the generated and
replayed features. Both training rounds and checkpoint save/reload passed.
This observation is a false-positive diagnostic, with no model-quality effect
from the differing coordinates in the reproduced run. CUDA was not tested.

Proposed follow-up, not yet implemented: clear both coordinate arrays alongside
the placement flags in `Environment::clear`, giving unplaced rooms deterministic
zero coordinates, and add a regression test for reset/replay consistency.

Validation: **99 Rust tests passed; 64 of 66 Python tests passed** using a direct
runner because `pytest` was unavailable. The two failures are outdated
[lookahead fixtures](python/test_lookahead_area_features.py#L19) and
[output-head references](python/test_model_initialization.py#L100). Reduced CPU
generation/training runs for Crateria and Zebes produced finite losses and no
captured feature mismatches; Zebes required disabling the broken
outcome-verification flag. GPU and compiled execution remain untested. The
reduced runs do not establish the magnitude of these issues in a full training
run.
