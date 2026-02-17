# LLM Grading Pipeline - Complete State Machine Documentation

**Last Updated:** Based on current implementation in `refined_grading_pipeline.py`

## Key Constants

- `STABILITY_THRESHOLD = 40.0%` - Threshold for QD versions (flip_rate < this is considered stable/improved)
- `NUM_ITERATIONS = 10` - Number of grading iterations per phase
- **Baseline Stability Rule**: Baseline (rubric) is only stable if `flip_rate == 0%` (100% consensus)
- **QD Stability Rule**: QD versions are stable if `flip_rate < STABILITY_THRESHOLD` (40%)
- **Refinement Attempt Limit**: Maximum 2 refinement attempts (v2 and v3), except for `exact_baseline_match` cases

---

## Phase 1: Baseline Grading (RUBRIC)

```
UNGRADED
  │
  ▼
BASELINE_GRADING
  │ (run NUM_ITERATIONS iterations)
  ▼
BASELINE_COMPLETE
  │
  ├─ Calculate: baseline_metrics (flip_rate, consensus, majority, pass_count, fail_count)
  │
  ├─ Check: baseline_pass_count == baseline_fail_count
  │   └─ → Set: needs_human_review = True (TIE case - majority is ambiguous)
  │
  └─ Check: baseline_flip_rate == 0.0%
      ├─ YES → BASELINE_STABLE [TERMINAL]
      │         - Set: skip_reason = 'baseline_already_stable'
      │         - Record: qd_initial_version (best version from previous responses)
      │         - Return checkpoint (skip QD grading)
      │
      └─ NO → Continue to Phase 2
```

**Terminal State: BASELINE_STABLE**
- Condition: `baseline_flip_rate == 0.0%` (100% consensus)
- Action: Skip all QD grading
- Record: QD version that would have been used (for tracking)

---

## Phase 2: Initial QD Grading

```
BASELINE_COMPLETE (with flip_rate > 0%)
  │
  ├─ Select: best_qd_version = get_best_qd_version(qid, exclude_rid=rid)
  │   (Best version from previous responses, excluding current response)
  │
  ├─ Check: best_version == None
  │   └─ → NO_QDS_AVAILABLE [TERMINAL]
  │         - Set: skip_reason = 'no_qds_available'
  │         - Return checkpoint
  │
  └─ QD_INITIAL_GRADING
      │ (run NUM_ITERATIONS iterations with selected QD version)
      │
      ▼
      QD_INITIAL_COMPLETE
      │
      ├─ Calculate: qd_initial_metrics
      │
      ├─ Check: qd_pass_count == qd_fail_count
      │   └─ → Set: needs_human_review = True (TIE case)
      │
      └─ Compare to Baseline (RUBRIC):
          │
          ├─ Condition: qd_flip_rate > baseline_flip_rate
          │   └─ → NEEDS_REFINEMENT
          │         - Set: qd_result = 'needs_refinement'
          │         - Set: refinement_reason = 'worse_flip_rate'
          │         - Set: refinement_attempt_count = 0
          │         - Return checkpoint (will refine later)
          │
          ├─ Condition: qd_majority != baseline_majority
          │   └─ → NEEDS_REFINEMENT
          │         - Set: qd_result = 'needs_refinement'
          │         - Set: refinement_reason = 'majority_changed'
          │         - Set: refinement_attempt_count = 0
          │         - Return checkpoint (will refine later)
          │
          └─ Condition: qd_flip_rate <= baseline_flip_rate AND 
                        qd_majority == baseline_majority
              └─ → QD_IMPROVED [TERMINAL]
                    - Set: qd_result = 'improved_or_maintained'
                    - Return checkpoint (success!)
```

**Terminal States:**
- **NO_QDS_AVAILABLE**: No QD versions exist for this question
- **QD_IMPROVED**: Initial QD version improved or maintained consistency without changing majority

---

## Phase 3: Refinement Loop

### 3a: Refinement Trigger Check

```
NEEDS_REFINEMENT
  │
  ├─ Check: refinement_attempt_count >= 2 AND 
  │         refinement_result != 'exact_baseline_match'
  │   └─ → MAINTAINED [TERMINAL]
  │         - Set: qd_result = 'maintained'
  │         - Set: abandoned_reason = "After 2 refinement attempts..."
  │         - Stop refinement
  │
  ├─ Check: refinement_result == 'partial_success' AND 
  │         refinement_attempt_count == 1
  │   └─ → Allow one more attempt (v3)
  │
  ├─ Check: refinement_result == 'exact_baseline_match'
  │   └─ → Allow one more attempt (even if attempt_count >= 2)
  │
  └─ Otherwise
      └─ → REFINING (generate new QD version)
```

### 3b: Refinement Generation

```
REFINING
  │
  ├─ Identify problematic responses:
  │   - refinement_reason == 'worse_flip_rate': qd_flip_rate > baseline_flip_rate
  │   - refinement_reason == 'majority_changed': qd_majority != baseline_majority
  │
  ├─ Generate refined QD version (v2, v3, v4, etc.)
  │   - Provide LLM with: current QDs, problematic examples, metrics, previous attempts
  │
  └─ → REFINED_GRADING (regrade with new QD version)
```

### 3c: Refined Grading

```
REFINED_GRADING
  │ (run NUM_ITERATIONS iterations with refined QD version)
  │
  ▼
REFINED_COMPLETE
  │
  ├─ Calculate: refined_metrics
  │
  ├─ Check: refined_pass_count == refined_fail_count
  │   └─ → is_tie = True (majority is ambiguous)
  │
  └─ → REFINEMENT_EVALUATION
```

### 3d: Refinement Evaluation

The evaluation logic depends on `refinement_reason` and whether there's a tie:

#### Evaluation for `refinement_reason == 'majority_changed'`:

```
REFINEMENT_EVALUATION (reason: majority_changed)
  │
  ├─ Check: is_tie == True
  │   ├─ Condition: refined_flip_rate <= baseline_flip_rate + 5.0%
  │   │   └─ → REFINEMENT_SUCCESS [TERMINAL]
  │   │         - Set: refinement_result = 'success'
  │   │         - Set: qd_result = 'improved_or_maintained'
  │   │
  │   └─ Condition: refined_flip_rate > baseline_flip_rate + 5.0%
  │       └─ → REFINEMENT_PARTIAL
  │             - Set: refinement_result = 'partial_success'
  │
  ├─ Check: refined_majority == baseline_majority AND 
  │         refined_flip_rate <= baseline_flip_rate + 5.0%
  │   └─ → REFINEMENT_SUCCESS [TERMINAL]
  │         - Set: refinement_result = 'success'
  │         - Set: qd_result = 'improved_or_maintained'
  │
  ├─ Check: refined_majority == baseline_majority AND 
  │         refined_flip_rate > baseline_flip_rate + 5.0%
  │   └─ → REFINEMENT_PARTIAL
  │         - Set: refinement_result = 'partial_success'
  │
  └─ Condition: refined_majority != baseline_majority
      └─ → REFINEMENT_FAILED
            - Set: refinement_result = 'failed'
```

#### Evaluation for `refinement_reason == 'worse_flip_rate'`:

```
REFINEMENT_EVALUATION (reason: worse_flip_rate)
  │
  ├─ Check: is_tie == True
  │   ├─ Condition: refined_flip_rate <= baseline_flip_rate
  │   │   └─ → REFINEMENT_SUCCESS [TERMINAL]
  │   │         - Set: refinement_result = 'success'
  │   │         - Set: qd_result = 'improved_or_maintained'
  │   │
  │   └─ Condition: refined_flip_rate > baseline_flip_rate
  │       └─ → REFINEMENT_FAILED
  │             - Set: refinement_result = 'failed'
  │
  ├─ Check: refined_flip_rate <= baseline_flip_rate AND 
  │         refined_majority == baseline_majority
  │   └─ → REFINEMENT_SUCCESS [TERMINAL]
  │         - Set: refinement_result = 'success'
  │         - Set: qd_result = 'improved_or_maintained'
  │
  ├─ Check: refined_flip_rate <= baseline_flip_rate AND 
  │         refined_majority != baseline_majority
  │   └─ → REFINEMENT_PARTIAL
  │         - Set: refinement_result = 'partial_success'
  │
  └─ Condition: refined_flip_rate > baseline_flip_rate
      └─ → REFINEMENT_FAILED
            - Set: refinement_result = 'failed'
```

### 3e: Post-Evaluation Processing

After evaluation, check for exact baseline match and handle attempt counting:

```
After REFINEMENT_EVALUATION
  │
  ├─ Check: check_exact_baseline_match(baseline_metrics, refined_metrics, is_tie)
  │   │ (Checks if flip_rate, consensus, and majority all match within 0.1% tolerance)
  │   │
  │   ├─ YES (exact match detected)
  │   │   └─ → Set: refinement_result = 'exact_baseline_match'
  │   │         - Set: qd_result = 'needs_refinement'
  │   │         - Allow one more refinement attempt (even if already at 2 attempts)
  │   │         - Don't increment attempt_count yet (this is detection, not attempt)
  │   │
  │   └─ NO
  │       └─ → Continue to attempt counting logic
  │
  ├─ Check: refinement_result == 'exact_baseline_match' (from previous iteration)
  │   │ (This is the extra attempt after exact baseline match was detected)
  │   │
  │   ├─ Increment: refinement_attempt_count += 1
  │   │
  │   ├─ Check: current_result == 'success' AND 
  │   │         NOT still_exact_match
  │   │   └─ → REFINEMENT_SUCCESS [TERMINAL]
  │   │         - Set: qd_result = 'improved_or_maintained'
  │   │         - Improved beyond baseline!
  │   │
  │   ├─ Check: still_exact_match == True
  │   │   └─ → MAINTAINED [TERMINAL]
  │   │         - Set: qd_result = 'maintained'
  │   │         - Still exactly matches baseline after extra attempt
  │   │
  │   └─ Check: current_result != 'success'
  │       └─ → MAINTAINED [TERMINAL]
  │             - Set: qd_result = 'maintained'
  │             - Not success after extra attempt
  │
  ├─ Check: refinement_result == 'partial_success' (from previous iteration)
  │   │ (This is attempt 2 after partial_success from attempt 1)
  │   │
  │   ├─ Set: refinement_attempt_count = 2
  │   │
  │   └─ Check: current_result != 'success'
  │       └─ → MAINTAINED [TERMINAL]
  │             - Set: qd_result = 'maintained'
  │             - After 2 attempts, still not success
  │
  ├─ Check: refinement_result in ['partial_success', 'failed'] (first attempt)
  │   ├─ Set: refinement_attempt_count = 1
  │   │
  │   └─ Check: current_result != 'success'
  │       └─ → Set: qd_result = 'maintained'
  │             (Will allow one more attempt if attempt_count == 1)
  │
  └─ Check: refinement_result == 'success'
      └─ → REFINEMENT_SUCCESS [TERMINAL]
            - Set: refinement_attempt_count = 0 (reset)
            - Set: qd_result = 'improved_or_maintained'
```

---

## Complete State Transition Diagram

```
┌─────────────┐
│  UNGRADED   │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│BASELINE_GRADING  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│BASELINE_COMPLETE │
└──────┬───────────┘
       │
       ├─ flip_rate == 0% ──→ ┌────────────────┐ [TERMINAL]
       │                      │BASELINE_STABLE │
       │                      └────────────────┘
       │
       └─ flip_rate > 0% ──→ ┌──────────────────┐
                              │QD_INITIAL_GRADING│
                              └──────┬───────────┘
                                     │
                                     ▼
                              ┌──────────────────────┐
                              │QD_INITIAL_COMPLETE   │
                              └──────┬───────────────┘
                                     │
                    ┌────────────────┼──────────────────┐
                    │                │                   │
                    ▼                ▼                   ▼
         worse_flip_rate    majority_changed    improved/maintained
         (qd_flip > base)   (qd_maj != base)    (both conditions met)
                    │                │                   │
                    └────────────────┴──────────────────┘
                                     │
                                     ▼
                              ┌──────────────────┐
                              │NEEDS_REFINEMENT   │
                              └──────┬───────────┘
                                     │
                    ┌────────────────┴──────────────────┐
                    │                                    │
         attempt_count >= 2                    attempt_count < 2
         (and not exact_match)                 (or exact_match)
                    │                                    │
                    ▼                                    ▼
         ┌──────────────────┐              ┌──────────────────┐
         │   MAINTAINED     │              │   REFINING       │
         │   [TERMINAL]     │              └──────┬───────────┘
         └──────────────────┘                    │
                                                 ▼
                                        ┌──────────────────┐
                                        │REFINED_GRADING   │
                                        └──────┬───────────┘
                                               │
                                               ▼
                                        ┌──────────────────┐
                                        │REFINED_COMPLETE  │
                                        └──────┬───────────┘
                                               │
                                               ▼
                                        ┌──────────────────────┐
                                        │REFINEMENT_EVALUATION │
                                        └──────┬───────────────┘
                                               │
                ┌─────────────────────────────┼─────────────────────────────┐
                │                             │                             │
                ▼                             ▼                             ▼
      exact_baseline_match          partial_success              success/failed
      (all metrics match)           (attempt_count==1)            (first attempt)
                │                             │                             │
                └─────────────────────────────┼─────────────────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
         REFINEMENT_SUCCESS          REFINEMENT_PARTIAL        REFINEMENT_FAILED
         [TERMINAL]                  │                        │
                                     │                        │
                    ┌────────────────┴────────────────────────┘
                    │
         attempt_count check
                    │
         ┌───────────┴───────────┐
         │                       │
    count < 2              count >= 2
         │                       │
         ▼                       ▼
  NEEDS_REFINEMENT         MAINTAINED
  (loop back)              [TERMINAL]
```

---

## Terminal States Summary

1. **BASELINE_STABLE** [TERMINAL]
   - Condition: `baseline_flip_rate == 0.0%` (100% consensus)
   - `skip_reason = 'baseline_already_stable'`
   - QD grading skipped

2. **NO_QDS_AVAILABLE** [TERMINAL]
   - Condition: No QD versions exist for this question
   - `skip_reason = 'no_qds_available'`
   - QD grading skipped

3. **QD_IMPROVED** [TERMINAL]
   - Condition: Initial QD version improved or maintained without changing majority
   - `qd_result = 'improved_or_maintained'`
   - No refinement needed

4. **REFINEMENT_SUCCESS** [TERMINAL]
   - Condition: Refined QD version meets success criteria
   - `refinement_result = 'success'`
   - `qd_result = 'improved_or_maintained'`
   - Successfully improved beyond baseline or fixed issues

5. **MAINTAINED** [TERMINAL]
   - Condition: After 2 refinement attempts (v2, v3) or after exact_baseline_match extra attempt
   - `qd_result = 'maintained'`
   - `abandoned_reason` explains why
   - QDs neither improved nor worsened baseline

---

## Key Condition Categories

### 1. Stability Conditions
- **Baseline**: `flip_rate == 0.0%` (must be perfect)
- **QD Versions**: `flip_rate < STABILITY_THRESHOLD` (40%)

### 2. Comparison Conditions
- **Worse Flip Rate**: `qd_flip_rate > baseline_flip_rate`
- **Majority Changed**: `qd_majority != baseline_majority`
- **Improved**: `qd_flip_rate <= baseline_flip_rate AND qd_majority == baseline_majority`

### 3. Tie Conditions
- **Tie Detection**: `pass_count == fail_count`
- **Tie Handling**: Majority is ambiguous, needs human review
- **Tie Evaluation**: Special logic in `evaluate_refinement_result()` with relaxed thresholds

### 4. Refinement Evaluation Conditions
- **Success Criteria** (varies by `refinement_reason`):
  - `majority_changed`: Fixed majority AND `flip_rate <= baseline_flip_rate + 5.0%`
  - `worse_flip_rate`: `flip_rate <= baseline_flip_rate AND majority == baseline_majority`
- **Partial Success**: Fixed one issue but not both
- **Failed**: Did not fix the triggering issue

### 5. Attempt Limit Conditions
- **Maximum Attempts**: 2 refinement attempts (v2 and v3)
- **Exception**: `exact_baseline_match` allows extra attempt even if `attempt_count >= 2`
- **Partial Success Exception**: If `partial_success` on attempt 1, allow attempt 2

### 6. Exact Match Conditions
- **Detection**: All metrics (flip_rate, consensus, majority) match baseline within 0.1% tolerance
- **Tie Case**: Only checks flip_rate and consensus (majority is ambiguous)
- **Action**: Allow one more refinement attempt to try to improve beyond baseline
- **Post-Attempt**: Evaluate if improved beyond baseline or still matches

### 7. Post-Exact-Match Conditions
- **After Extra Attempt**:
  - If `success` AND NOT `still_exact_match` → **SUCCESS** (improved beyond baseline)
  - If `still_exact_match` → **MAINTAINED** (still matches baseline)
  - If NOT `success` → **MAINTAINED** (didn't improve)

---

## State Variables Tracked

### Checkpoint Fields:
- `baseline_iterations`: List of baseline grading iterations
- `qd_initial_iterations`: List of initial QD grading iterations
- `qd_refined_iterations`: List of refined QD grading iterations
- `qd_refined_iterations_v2`, `qd_refined_iterations_v3`, etc.: Per-version iterations
- `baseline_metrics`: Calculated baseline metrics
- `qd_initial_metrics`: Calculated initial QD metrics
- `qd_refined_metrics`: Calculated refined QD metrics
- `qd_refined_metrics_v2`, `qd_refined_metrics_v3`, etc.: Per-version metrics
- `qd_result`: 'improved_or_maintained', 'needs_refinement', 'maintained', 'abandoned'
- `refinement_result`: 'success', 'partial_success', 'failed', 'exact_baseline_match'
- `refinement_reason`: 'worse_flip_rate', 'majority_changed'
- `refinement_attempt_count`: Number of refinement attempts (0, 1, 2)
- `qd_initial_version`: Version used for initial QD grading
- `qd_refined_version`: Latest refined version used
- `skip_reason`: 'baseline_already_stable', 'no_qds_available'
- `needs_human_review`: True if tie case detected
- `version_progression`: List tracking all versions and their metrics

---

## Special Cases

### Tie Cases (pass_count == fail_count)
- **Detection**: Checked in baseline, initial QD, and refined QD phases
- **Action**: Set `needs_human_review = True`
- **Evaluation**: Special logic with relaxed thresholds:
  - For `majority_changed`: `refined_flip_rate <= baseline_flip_rate + 5.0%` → success
  - For `worse_flip_rate`: `refined_flip_rate <= baseline_flip_rate` → success
- **Majority**: Considered ambiguous, not used in exact match checks

### Exact Baseline Match
- **Purpose**: Detect when refined QD exactly matches baseline (within tolerance)
- **Action**: Allow one more refinement attempt to try to improve beyond baseline
- **Exception**: Can proceed even if `refinement_attempt_count >= 2`
- **Post-Attempt**: Evaluate if improved beyond baseline or still matches

### Partial Success
- **Definition**: Fixed one issue but not both (e.g., fixed majority but flip_rate still high)
- **Action**: If on attempt 1, allow attempt 2 (final attempt)
- **After Attempt 2**: If still not success → **MAINTAINED**

---

## Version Tracking

- **v1**: Initial QD version (from `get_best_qd_version()`)
- **v2**: First refinement attempt
- **v3**: Second refinement attempt (or final attempt after partial_success)
- **v4+**: Only possible if `exact_baseline_match` detected (extra attempt)

Each version's iterations and metrics are stored separately:
- `qd_refined_iterations_v2`, `qd_refined_metrics_v2`
- `qd_refined_iterations_v3`, `qd_refined_metrics_v3`
- etc.

This allows tracking the progression and impact of each refinement.

