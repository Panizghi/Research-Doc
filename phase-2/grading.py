"""
Complete Grading Logic - Based on Paper Abstract
=================================================

Implements the exact three-dimensional evaluation described in the paper:
1. Consistency (flip rate)
2. Correctness (instructor alignment)
3. Stability (don't break working responses)
"""

from typing import Dict, List, Tuple
from enum import Enum
import numpy as np


# ============================================================================
# CORE METRICS: THREE DIMENSIONS
# ============================================================================

def calculate_three_dimensions(
    grades: List[int],
    instructor_likert: int = None,
    baseline_grades: List[int] = None
) -> Dict:
    """
    Calculate all three performance dimensions.
    
    Args:
        grades: List of 10-20 binary grades (0=fail, 1=pass)
        instructor_likert: Instructor rating on 1-10 scale
        baseline_grades: Baseline (rubric) grades for stability check
    
    Returns:
        {
            # Dimension 1: Consistency
            'flip_rate': float,              # % grade fluctuation
            'consensus': float,              # % agreement
            'is_consistent': bool,           # flip_rate == 0
            
            # Dimension 2: Correctness
            'majority': int,                 # 0, 1, or -1 (tie)
            'expected_majority': int,        # From instructor (0, 1, or None)
            'is_correct': bool,              # majority == expected
            'pass_count': int,
            'fail_count': int,
            
            # Dimension 3: Stability
            'baseline_was_stable': bool,     # baseline flip_rate == 0
            'stability_maintained': bool,    # didn't break stable baseline
        }
    """
    
    # Dimension 1: CONSISTENCY
    valid_grades = [g for g in grades if g != -1]
    if not valid_grades:
        return create_empty_metrics()
    
    pass_count = sum(valid_grades)
    fail_count = len(valid_grades) - pass_count
    
    # Check for perfect tie (5 pass, 5 fail in 10 runs)
    is_tie = (pass_count == fail_count)
    
    if is_tie:
        majority = -1  # Undefined
        consensus = 50.0
    else:
        majority_count = max(pass_count, fail_count)
        majority = 1 if pass_count > fail_count else 0
        consensus = (majority_count / len(valid_grades)) * 100
    
    flip_rate = 100 - consensus
    is_consistent = (flip_rate == 0.0)
    
    # Dimension 2: CORRECTNESS
    expected_majority = None
    is_correct = None
    
    if instructor_likert is not None:
        if instructor_likert < 5:
            expected_majority = 0  # Should FAIL
        elif instructor_likert > 5:
            expected_majority = 1  # Should PASS
        else:  # instructor_likert == 5
            expected_majority = None  # Uncertain - needs manual review
        
        # Check correctness (only if not tie and instructor is certain)
        if not is_tie and expected_majority is not None:
            is_correct = (majority == expected_majority)
    
    # Dimension 3: STABILITY
    baseline_was_stable = False
    stability_maintained = None
    
    if baseline_grades:
        baseline_metrics = calculate_consistency_only(baseline_grades)
        baseline_was_stable = (baseline_metrics['flip_rate'] == 0.0)
        
        # Stability maintained if:
        # - Baseline was stable AND current is also stable
        # - OR baseline was unstable (nothing to break)
        if baseline_was_stable:
            stability_maintained = is_consistent
        else:
            stability_maintained = True  # Can't break what's already broken
    
    return {
        # Dimension 1
        'flip_rate': flip_rate,
        'consensus': consensus,
        'is_consistent': is_consistent,
        
        # Dimension 2
        'majority': majority,
        'expected_majority': expected_majority,
        'is_correct': is_correct,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'is_tie': is_tie,
        
        # Dimension 3
        'baseline_was_stable': baseline_was_stable,
        'stability_maintained': stability_maintained,
        
        # Raw data
        'sample_size': len(valid_grades),
        'instructor_likert': instructor_likert
    }


def calculate_consistency_only(grades: List[int]) -> Dict:
    """Helper: Just calculate flip rate"""
    valid = [g for g in grades if g != -1]
    if not valid:
        return {'flip_rate': 100, 'consensus': 0}
    
    pass_count = sum(valid)
    fail_count = len(valid) - pass_count
    majority_count = max(pass_count, fail_count)
    consensus = (majority_count / len(valid)) * 100
    
    return {
        'flip_rate': 100 - consensus,
        'consensus': consensus
    }


def create_empty_metrics() -> Dict:
    """Empty metrics when no valid grades"""
    return {
        'flip_rate': 100,
        'consensus': 0,
        'is_consistent': False,
        'majority': -1,
        'expected_majority': None,
        'is_correct': None,
        'pass_count': 0,
        'fail_count': 0,
        'is_tie': False,
        'baseline_was_stable': False,
        'stability_maintained': None,
        'sample_size': 0,
        'instructor_likert': None
    }


# ============================================================================
# DECISION LOGIC: RESPONSE CATEGORIZATION
# ============================================================================

class ResponseOutcome(Enum):
    """Outcome categories from paper Table 1"""
    ALREADY_STABLE_SKIPPED = 'Already Stable (skipped)'
    ALREADY_STABLE_VERIFIED = 'Already Stable (verified)'
    IMPROVED_CONSISTENCY = 'Improved Consistency'
    IMPROVED_BOTH = 'Improved Both (consistency + correctness)'
    FIXED_CORRECTNESS = 'Fixed Correctness'
    NO_CHANGE = 'No Significant Change'
    WORSE = 'Worse'
    NEEDS_MANUAL_REVIEW = 'Unclear Cases (manual review)'


def categorize_response(
    baseline_metrics: Dict,
    qd_metrics: Dict = None
) -> str:
    """
    Categorize response outcome using the paper's three dimensions.
    
    Logic from abstract:
    - 117/136 (86%) already stable at baseline → skip QD
    - 19/136 (14%) unstable at baseline → try QD
    - Track consistency AND correctness separately
    """
    
    # CASE 1: Baseline already stable (flip_rate = 0%)
    # From abstract: "117 of 136 responses (86%) were already consistent at baseline"
    if baseline_metrics['is_consistent']:
        
        # Check if QD was applied or skipped
        if qd_metrics is None:
            # QD was skipped (optimal - no intervention needed)
            return ResponseOutcome.ALREADY_STABLE_SKIPPED.value
        else:
            # QD was applied for verification
            if qd_metrics['is_consistent'] and qd_metrics['majority'] == baseline_metrics['majority']:
                return ResponseOutcome.ALREADY_STABLE_VERIFIED.value
            else:
                # QD broke stable baseline - this is BAD
                return ResponseOutcome.WORSE.value
    
    # CASE 2: Baseline unstable → QD applied
    # From abstract: "19 responses (14%) had baseline grading instability"
    
    if qd_metrics is None:
        # QD should have been applied but wasn't
        return 'ERROR: QD not applied to unstable baseline'
    
    # Handle ties (5 pass, 5 fail) → manual review
    # From abstract: "Likert = 5 indicates uncertainty"
    if qd_metrics['is_tie']:
        return ResponseOutcome.NEEDS_MANUAL_REVIEW.value
    
    if baseline_metrics['expected_majority'] is None:
        # Instructor uncertain (Likert = 5) → manual review
        return ResponseOutcome.NEEDS_MANUAL_REVIEW.value
    
    # Calculate improvements
    flip_improved = (qd_metrics['flip_rate'] < baseline_metrics['flip_rate'])
    flip_delta = baseline_metrics['flip_rate'] - qd_metrics['flip_rate']
    
    # Check correctness change
    baseline_correct = baseline_metrics.get('is_correct')
    qd_correct = qd_metrics.get('is_correct')
    
    correctness_fixed = (baseline_correct == False and qd_correct == True)
    correctness_maintained = (baseline_correct == True and qd_correct == True)
    correctness_broken = (baseline_correct == True and qd_correct == False)
    
    # Decision tree based on paper's three dimensions
    
    # Best case: Improved BOTH dimensions
    if flip_improved and correctness_fixed:
        return ResponseOutcome.IMPROVED_BOTH.value
    
    # Good: Improved consistency, maintained correctness
    if flip_improved and correctness_maintained:
        return ResponseOutcome.IMPROVED_CONSISTENCY.value
    
    # Good: Fixed correctness (even if consistency same/worse)
    if correctness_fixed:
        return ResponseOutcome.FIXED_CORRECTNESS.value
    
    # Neutral: No significant change (within ±5 percentage points)
    if abs(flip_delta) <= 5:
        return ResponseOutcome.NO_CHANGE.value
    
    # Bad: Broke correctness or made consistency worse
    if correctness_broken or (not flip_improved and abs(flip_delta) > 5):
        return ResponseOutcome.WORSE.value
    
    # Default
    return ResponseOutcome.NO_CHANGE.value


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_improvement_subset(
    baseline_flip_rates: List[float],
    qd_flip_rates: List[float]
) -> Dict:
    """
    Calculate statistics for improvement subset.
    
    From abstract:
    - "For responses where QD refinement improved both consistency and 
       correctness (n=8), flip rate decreased 84% (p=0.005, d=1.26)"
    - "Among all responses showing improvement (n=14), effect reached 
       84% relative reduction (d=1.39, p<0.001)"
    """
    
    import scipy.stats as stats
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_flip_rates, qd_flip_rates)
    
    # Cohen's d (paired)
    differences = np.array(baseline_flip_rates) - np.array(qd_flip_rates)
    d = np.mean(differences) / np.std(differences, ddof=1)
    
    # Relative reduction
    mean_baseline = np.mean(baseline_flip_rates)
    mean_qd = np.mean(qd_flip_rates)
    absolute_reduction = mean_baseline - mean_qd
    relative_reduction = (absolute_reduction / mean_baseline) * 100
    
    return {
        'n': len(baseline_flip_rates),
        'mean_baseline_flip': mean_baseline,
        'mean_qd_flip': mean_qd,
        'absolute_reduction': absolute_reduction,
        'relative_reduction_pct': relative_reduction,
        'cohens_d': d,
        'p_value': p_value,
        't_statistic': t_stat
    }


# ============================================================================
# DECISION RULES
# ============================================================================

def should_skip_qd(baseline_metrics: Dict) -> Tuple[bool, str]:
    """
    Decide if QD grading should be skipped.
    
    From abstract: "Critically, 117 of 136 responses (86%) were already 
    consistent at baseline; QD refinement did not change these"
    """
    
    if baseline_metrics['is_consistent']:
        return (True, 'baseline_already_stable')
    
    return (False, '')


def should_get_manual_review(metrics: Dict) -> Tuple[bool, str]:
    """
    Decide if response needs manual review.
    
    Criteria:
    1. Perfect tie (5 pass, 5 fail)
    2. Instructor uncertain (Likert = 5)
    3. Very high flip rate (>40% even after QD)
    """
    
    if metrics['is_tie']:
        return (True, 'perfect_tie')
    
    if metrics['instructor_likert'] == 5:
        return (True, 'instructor_uncertain')
    
    if metrics['flip_rate'] >= 40:
        return (True, 'very_high_flip_rate')
    
    return (False, '')


def needs_qd_refinement(
    baseline_metrics: Dict,
    qd_metrics: Dict
) -> Tuple[bool, List[str]]:
    """
    Decide if QD refinement is needed.
    
    Triggers:
    1. QD made flip rate worse
    2. QD changed majority (broke correctness)
    """
    
    problems = []
    
    # Check consistency dimension
    if qd_metrics['flip_rate'] > baseline_metrics['flip_rate']:
        problems.append('worse_consistency')
    
    # Check correctness dimension
    if qd_metrics['majority'] != baseline_metrics['majority']:
        problems.append('majority_changed')
    
    needs_refinement = len(problems) > 0
    
    return (needs_refinement, problems)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_from_paper():
    """Examples matching the paper's results"""
    
    # Example 1: Already stable (86% of responses)
    baseline_grades = [1,1,1,1,1,1,1,1,1,1]  # Perfect consensus
    baseline = calculate_three_dimensions(baseline_grades, instructor_likert=9)
    
    should_skip, reason = should_skip_qd(baseline)
    print(f"Example 1: Should skip QD? {should_skip} ({reason})")
    print(f"  Flip rate: {baseline['flip_rate']}%")
    print(f"  Category: {categorize_response(baseline)}")
    print()
    
    # Example 2: Unstable baseline, QD improves both (n=8 from paper)
    baseline_grades = [1,0,1,0,1,1,0,1,1,0]  # 60% consensus, 40% flip
    qd_grades = [0,0,0,0,0,0,0,0,0,0]  # Perfect consensus
    
    baseline = calculate_three_dimensions(baseline_grades, instructor_likert=3)  # Should fail
    qd = calculate_three_dimensions(qd_grades, instructor_likert=3, baseline_grades=baseline_grades)
    
    print("Example 2: QD improved both")
    print(f"  Baseline: flip={baseline['flip_rate']}%, majority={baseline['majority']} (expected {baseline['expected_majority']})")
    print(f"  QD:       flip={qd['flip_rate']}%, majority={qd['majority']}")
    print(f"  Correctness: {baseline['is_correct']} → {qd['is_correct']}")
    print(f"  Category: {categorize_response(baseline, qd)}")
    print()
    
    # Example 3: Perfect tie → manual review
    tie_grades = [1,1,1,1,1,0,0,0,0,0]  # 5 pass, 5 fail
    tie = calculate_three_dimensions(tie_grades, instructor_likert=7)
    
    needs_review, review_reason = should_get_manual_review(tie)
    print(f"Example 3: Perfect tie → Manual review? {needs_review} ({review_reason})")
    print(f"  Category: {categorize_response({'is_consistent': False}, tie)}")
    print()
    
    # Example 4: Instructor uncertain (Likert=5)
    uncertain_grades = [1,1,1,0,1,1,1,1,0,1]
    uncertain = calculate_three_dimensions(uncertain_grades, instructor_likert=5)
    
    needs_review, review_reason = should_get_manual_review(uncertain)
    print(f"Example 4: Instructor uncertain (Likert=5) → Manual review? {needs_review} ({review_reason})")
    print(f"  Expected majority: {uncertain['expected_majority']} (None = uncertain)")
    print()


if __name__ == '__main__':
    print("="*70)
    print("COMPLETE GRADING LOGIC - Based on Paper Abstract")
    print("="*70)
    print()
    
    example_from_paper()
    
    print("="*70)
    print("THREE DIMENSIONS TRACKED:")
    print("1. Consistency (flip rate)")
    print("2. Correctness (instructor alignment)")
    print("3. Stability (don't break working responses)")
    print("="*70)