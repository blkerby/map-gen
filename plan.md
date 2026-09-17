A list of issues that we may want to address, relating to balance model:

1. **The uniform door target is incompatible with matching every door.**

   Each door is trained toward uniform partner choices among its compatible partners. But the pairings must be reciprocal, and each concrete door can only participate in one connection.

   Let \(d_i\) be the number of compatible partners for left door \(i\), and \(e_j\) the equivalent count for right door \(j\). If every door is matched, the two directional uniform targets require the same connection to have both probabilities

   \[
   P(i\leftrightarrow j)=1/d_i
   \quad\text{and}\quad
   P(i\leftrightarrow j)=1/e_j.
   \]

   These disagree whenever \(d_i\ne e_j\).

   This occurs extensively in the actual Zebes metadata:

   | Compatibility matrix | Compatible pairs | Pairs with unequal endpoint degrees | Largest implied use of one target door per map* |
   |---|---:|---:|---:|
   | Horizontal, 245 × 245 | 52,156 | 50,385 | 1.196 |
   | Vertical, 44 × 44 | 724 | 557 | 1.447 |

   *Assuming every source door matches and chooses uniformly among compatible partners.*

   A target door cannot be used 1.447 times per map. Thus exact balancing and complete matching cannot both satisfy these targets, even before considering whole-map geometry.

   Positive regularization permits a compromise, so this does not imply training must diverge. It does mean that some persistent imbalance is structurally unavoidable, and balancing can compete with validity. Because unmatched outcomes are excluded from the observed-price term, changing which doors get matched is also one way the system can change its conditional partner distributions.

   I would reconsider the target as a **joint distribution over feasible pairings with consistent endpoint usage**, rather than independently uniform rows in both directions. The current target construction is in [door price centering](/home/kerby/map-gen/python/loss.py:598); observed matches explicitly populate both directions in [Rust](/home/kerby/map-gen/src/environment.rs:6399).

2. **The area head is trained conditional on eventual placement but used without that condition.**

   Area targets exclude rooms with terminal area `-1`. The prefix mask further excludes already placed and forced rooms. Consequently, an unplaced room’s head learns approximately

   \[
   E[b_{\text{area}}\mid\text{partial map},\text{room eventually placed}].
   \]

   Generation sums that prediction directly. It has no corresponding placement-probability multiplier, and failed continuations provide no zero target for the omitted room. This differs from the door and Toilet treatment. See [area target mask](/home/kerby/map-gen/python/loss.py:692), [prefix supervision](/home/kerby/map-gen/python/learn.py:1379), and [area scoring](/home/kerby/map-gen/python/generate.py:168).

   I reproduced a simple case where an eventually absent room’s prediction of `-4` receives **zero training gradient**, yet contributes **+4 generation reward**.

   This is relevant to actual data: in the 8,192 episodes following checkpoint 1300, only **54.2% placed all 253 rooms**. Most omissions were small—the mean was 252.14 rooms—but the condition is common.

   The cleanest interpretation would be an unconditional expected terminal area penalty, with zero target when the room is never placed. Alternatively, the conditional interpretation needs an explicit placement-probability estimate.

3. **Proposal prices incorrectly vanish for some distinct rooms sharing a variant.**

   Proposal-table construction independently chooses the first concrete representative of each door variant. Sometimes both representatives belong to the same room. The concrete compatibility mask correctly makes that same-room entry zero—but the proposal can actually connect two different instances sharing those variants.

   The resulting zero is therefore incorrect for the real proposal. See [representative selection](/home/kerby/map-gen/python/loss.py:784) and [proposal price construction](/home/kerby/map-gen/python/loss.py:761).

   Comparing proposal prices with exact concrete-pair prices on Zebes reproduced **260 affected directed entries**:

   - 128 left and 128 right entries.
   - Two up and two down entries.

   Examples include Crateria Tube versus Green Brinstar Beetom Room, and West versus East Aqueduct Quicksand Room.

   This persists with trained weights. Across 32 configurations from the saved run, the missing vertical penalty reached **0.79 reward units**, while proposal temperature was approximately **0.064**. That is a substantial change to shortlist probabilities.

   Final candidate scoring uses the correct concrete price, so the two sampling stages disagree. Proposal construction should obtain its value from a compatible concrete pair rather than independently selected representatives.

4. **Your current Adam `beta1` change will be ignored when resuming the inspected checkpoint.**

   The current working config specifies `balance_optimizer.beta1 = 0.5`. Checkpoint 1300 stores actual optimizer betas `(0.9, 0.95)`.

   Loading restores the complete saved optimizer parameter groups, including those betas. Subsequent rounds only reapply the learning rate. I reproduced:

   ```text
   Config beta1:        0.5
   Actual resumed beta1: 0.9
   Learning rate:      updated from the config
   ```

   Moreover, logging reports betas from the config, so it would report `0.5` while Adam actually uses `0.9`. See [optimizer restoration](/home/kerby/map-gen/python/train.py:206), [runtime updates](/home/kerby/map-gen/python/learn.py:1728), and [logging](/home/kerby/map-gen/python/train.py:2002).

   This is especially relevant if recent balance tuning involved resuming existing checkpoints. A fresh run uses the configured betas correctly.

5. **Regularization means the requested probabilities are soft targets, and its exact meaning differs from the written mathematical contract.**

   Even with a perfectly functioning generator and controller, positive beta generally leaves residual imbalance. If the observed distribution equals the target, the observed-price gradient vanishes, while regularization continues pulling any nonzero corrective prices toward zero. Maintaining prices that counteract the generator’s natural bias therefore requires some persistent error.

   There is an additional distinction for nonuniform area targets. The code regularizes the **centered** prices, whereas `plan.md` states the stationary relationship \((p-q)/\beta\).

   For an independently parameterized area row, ignoring the family averaging factors, the implemented constraint is \(q^\mathsf{T}b=0\), giving

   \[
   b^*=
   \frac{1}{\beta}
   \left(p-q\frac{p^\mathsf{T}q}{q^\mathsf{T}q}\right).
   \]

   This generally differs from centering \((p-q)/\beta\). I verified a nonzero gradient at the documented solution for a nonuniform target.

   The implemented objective is mathematically coherent, but beta does not have precisely the documented interpretation. Also, door and area observed terms divide by the number of observed outcomes, whereas their regularizers divide by the number of eligible groups. Incomplete episodes consequently change the effective strength of the data term. See [normalization and regularization](/home/kerby/map-gen/python/loss.py:530) and [the mathematical contract](/home/kerby/map-gen/plan.md:46).

6. **The scalar balance heads must continually chase a moving price function.**

   The controller updates immediately; the main heads learn its newly updated prices; generation uses an EMA of those heads. The actual price tables are not supplied to the main model as inputs.

   Thus generation combines current exact prices with predictions learned under recent historical prices. If a price changes sharply, placing a room can replace an outdated expected price with a different current exact price, distorting candidate comparisons.

   This is a tracking risk rather than proof of an unstable run. My saved-checkpoint checks did not establish that EMA lag is the dominant source of error. Ordinary future-outcome uncertainty also contributes to price-regression MSE.

   One architectural alternative is to predict outcome distributions and compute their expectation against the current price tables. That would separate uncertainty about the future from changes in prices, although full door-partner distributions would be expensive. Supplying a compact representation of current prices is another possibility.

   A related detail: EMA half-life is counted in **processed training episodes**, including replay and repeated passes. With fresh/replay pass factors both 2, an 80,000-example half-life corresponds to roughly 20,000 newly generated episodes once replay is active. See [EMA updates](/home/kerby/map-gen/python/learn.py:1557).

7. **Align the proposal head with full candidate selection (implemented; evaluation pending).**

   Previously its teacher contained ordinary expected reward plus the immediate door/area price adjustment. Final selection additionally considered future door prices, future room-area prices, and Toilet prices.

   The proposal target now uses the full final-selection value. The student retains its external immediate correction; that correction is not added to the teacher a second time. Both distributions use the same target temperature, and KL is multiplied by temperature squared with a cancellation-resistant calculation.

   The motivation is that a candidate with favorable future balance may never reach final scoring, particularly at low proposal temperatures.

   [Offline retraining instructions](scripts/retrain_proposal.md) describe replay distillation of both proposal heads with frozen encoders and balance model, using every episode in the selected rounds. Each replay batch is prepared and trained immediately. Frozen copies of the original heads provide baseline metrics on the same samples, and training agreement metrics reuse the optimization forward passes. The output can be installed into the stopped run. Whether this improves generation still needs evaluation on RTX.

8. **The training population and metrics do not establish balance among delivered valid maps.**

   Balance training uses all fresh episodes, with masks for individual usable outcomes. It does not require the overall map to succeed. Serving subsequently filters maps for validity. Balancing the first population does not guarantee balance in the accepted population. See [fresh training weights](/home/kerby/map-gen/python/learn.py:189) and [serving’s validity filter](/home/kerby/map-gen/python/serve.py:769).

   Several metrics also need careful interpretation:

   - `balance_loss` measures the controller objective, not distance from the requested distribution.
   - Main balance losses include zero-valued replay batches in their round averages, so changing replay proportions changes the reported averages.
   - `avg_area_rooms` is a squared count error, despite its name.
   - Aggregate concentration scores can look good while the model responds poorly to individual requested probabilities.

   I would measure conditional target-versus-observed frequencies, distinguish all generated maps from accepted maps, and evaluate the generation EMA separately from the online training model.

**The saved run shows that preferences influence generation, but strong preferences are substantially underfulfilled.** Using [experience file 1300](/home/kerby/map-gen/runs/2026-09-06T14:27:50.485558-zebes-testing/experience/1300.safetensors), I examined the strongest fifth of active preference settings. Observed rates below are conditional on the tagged room being placed.

| Preference | Mean requested probability | Observed preferred-area rate |
|---|---:|---:|
| Water tier 2 → Maridia | 52.2% | 40.1% |
| Water tier 3 → Maridia | 69.0% | 45.0% |
| Heat tier 2 → Norfair | 52.9% | 44.5% |
| Heat tier 3 → Norfair | 69.2% | 56.9% |

These measurements demonstrate a gap, but do not assign its cause among regularization, competing constraints, approximation errors, and the bugs above. They describe the saved run’s settings; its Adam `beta1` was `0.9`, not the current working config’s `0.5`.

I also found a separate serving regression: [the serving caller](/home/kerby/map-gen/python/serve.py:1095) unpacks six values from `run_generation_groups`, which [returns seven](/home/kerby/map-gen/python/generate.py:2428). A focused reproduction raises `ValueError: too many values to unpack (expected 6)`. This prevents that serving path from completing generation.

My recommended order is to correct optimizer restoration, the missing proposal prices, and unconditional area-head supervision; decide what feasible door balance should mean; then evaluate price tracking and conditional balance with clearer metrics. The 36 existing focused tests passed, but the additional probes reproduced the implementation problems above.
