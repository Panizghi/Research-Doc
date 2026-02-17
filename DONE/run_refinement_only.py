#!/usr/bin/env python3
"""
Run Refinement Only
===================
Runs refinement on checkpoints that need it without re-running baseline or initial QD grading.

This script:
1. Scans checkpoints to find those needing refinement
2. Groups them by question ID
3. Runs refinement process (refines QDs and re-grades only those responses)
4. Does NOT re-run baseline or initial QD grading
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Import pipeline functions
from refined_grading_pipeline import (
    load_checkpoint, save_checkpoint,
    refine_problematic_qds, format_rubric,
    CHECKPOINT_DIR, calculate_stability_metrics
)


def load_questions_json(json_path: str) -> dict:
    """Load questions from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def identify_checkpoints_needing_refinement() -> dict:
    """
    Scan all checkpoints and identify which ones need refinement.
    Returns dict mapping question_id -> list of response_ids
    """
    checkpoints_needing_refinement = defaultdict(list)
    
    print("Scanning checkpoints for responses needing refinement...")
    
    for checkpoint_file in CHECKPOINT_DIR.glob('*.json'):
        try:
            checkpoint = json.loads(checkpoint_file.read_text())
            qid = checkpoint.get('question_id')
            rid = checkpoint.get('response_id')
            
            if not qid or not rid:
                continue
            
            # Check if this checkpoint needs refinement
            qd_result = checkpoint.get('qd_result', '')
            refinement_result = checkpoint.get('refinement_result', '')
            refinement_attempt_count = checkpoint.get('refinement_attempt_count', 0)
            
            # Check if it needs refinement
            needs_refinement = qd_result == 'needs_refinement'
            has_partial_success = refinement_result == 'partial_success' and refinement_attempt_count == 1
            has_exact_baseline = refinement_result == 'exact_baseline_match'
            
            # Special case: partial_success with attempt_count=1 should get another try
            # even if marked as 'maintained' (they were marked maintained after first attempt)
            if has_partial_success and qd_result == 'maintained':
                # Reset to needs_refinement so it can be processed
                checkpoint['qd_result'] = 'needs_refinement'
                save_checkpoint(checkpoint, qid, rid)
                needs_refinement = True
            
            # Skip if already abandoned or maintained (unless it's a partial_success case above)
            if qd_result in ['abandoned', 'maintained'] and not has_partial_success:
                continue
            
            # Also check if baseline exists but QD initial doesn't (shouldn't happen, but check)
            baseline_metrics = checkpoint.get('baseline_metrics', {})
            qd_initial_metrics = checkpoint.get('qd_initial_metrics', {})
            qd_refined_metrics = checkpoint.get('qd_refined_metrics', {})
            
            # If baseline exists but no QD initial, might need to check if it was skipped
            skip_reason = checkpoint.get('skip_reason')
            if skip_reason == 'baseline_already_stable':
                # Baseline is stable, no refinement needed
                continue
            
            # Check if refinement_reason is set but no refinement was run
            refinement_reason = checkpoint.get('refinement_reason', '')
            if refinement_reason and not qd_initial_metrics:
                # Has refinement reason but no QD initial - this shouldn't happen
                # But if it does, we can't refine without initial QD
                print(f"  ⚠ {qid}_{rid}: Has refinement_reason but no QD initial metrics - skipping")
                continue
            
            # Also check if refinement_reason is set but no refined metrics exist
            # This means refinement was identified as needed but never run
            has_refinement_reason_but_no_refined = (
                refinement_reason and 
                qd_initial_metrics and 
                not qd_refined_metrics and
                qd_result != 'needs_refinement'
            )
            
            if needs_refinement or has_partial_success or has_exact_baseline or has_refinement_reason_but_no_refined:
                checkpoints_needing_refinement[qid].append(rid)
                reason_str = refinement_reason or 'unknown'
                if has_refinement_reason_but_no_refined:
                    print(f"  ✓ {qid}_{rid}: needs refinement (reason: {reason_str}, but qd_result not set - will fix)")
                elif has_partial_success:
                    print(f"  ✓ {qid}_{rid}: needs refinement (partial_success after attempt {refinement_attempt_count}, trying again)")
                else:
                    print(f"  ✓ {qid}_{rid}: needs refinement (reason: {reason_str})")
            elif refinement_reason and not qd_initial_metrics:
                # Has a reason but no initial QD - might need initial QD first
                print(f"  ⚠ {qid}_{rid}: Has refinement_reason '{refinement_reason}' but no QD initial - may need full pipeline")
            
        except Exception as e:
            print(f"  ✗ Error reading {checkpoint_file.name}: {e}")
            continue
    
    return dict(checkpoints_needing_refinement)


def ensure_refinement_reason_set(checkpoint: dict) -> bool:
    """
    Ensure refinement_reason is set if it should be.
    Returns True if reason was set or already exists.
    """
    refinement_reason = checkpoint.get('refinement_reason', '')
    
    if refinement_reason:
        return True  # Already set
    
    # Check if we can infer the reason from metrics
    baseline_metrics = checkpoint.get('baseline_metrics', {})
    qd_initial_metrics = checkpoint.get('qd_initial_metrics', {})
    
    if not baseline_metrics or not qd_initial_metrics:
        return False  # Can't determine reason without both metrics
    
    baseline_majority = baseline_metrics.get('majority', -1)
    qd_initial_majority = qd_initial_metrics.get('majority', -1)
    baseline_flip = baseline_metrics.get('flip_rate', 100)
    qd_initial_flip = qd_initial_metrics.get('flip_rate', 100)
    
    # Determine reason
    if baseline_majority != -1 and qd_initial_majority != -1 and baseline_majority != qd_initial_majority:
        checkpoint['refinement_reason'] = 'majority_changed'
        checkpoint['baseline_majority'] = baseline_majority
        checkpoint['qd_majority'] = qd_initial_majority
        return True
    elif qd_initial_flip > baseline_flip:
        checkpoint['refinement_reason'] = 'worse_flip_rate'
        return True
    
    return False


def prepare_questions_data(questions_json_path: str, checkpoints_needing_refinement: dict) -> dict:
    """
    Load questions.json and filter to only questions that have checkpoints needing refinement.
    Also filter responses to only those that need refinement.
    """
    print("\nLoading questions.json...")
    all_questions = load_questions_json(questions_json_path)
    
    # Filter to only questions that need refinement
    filtered_questions = {}
    
    for qid, response_ids in checkpoints_needing_refinement.items():
        if qid not in all_questions:
            print(f"  ⚠ Question {qid} not found in questions.json - skipping")
            continue
        
        q_data = all_questions[qid].copy()
        
        # Filter responses to only those needing refinement
        original_responses = q_data.get('responses', [])
        filtered_responses = [
            r for r in original_responses
            if r.get('response_id') in response_ids
        ]
        
        if not filtered_responses:
            print(f"  ⚠ Question {qid}: No matching responses found in questions.json")
            continue
        
        q_data['responses'] = filtered_responses
        filtered_questions[qid] = q_data
        
        print(f"  ✓ {qid}: {len(filtered_responses)}/{len(original_responses)} responses need refinement")
    
    return filtered_questions


def run_refinement_only(questions_json_path: str = './questions.json', specific_questions: list = None) -> None:
    """
    Run refinement on checkpoints that need it.
    
    Args:
        questions_json_path: Path to questions.json file
        specific_questions: Optional list of question IDs to process (e.g., ['L1Q1', 'L1Q2'])
    """
    print("="*80)
    print("REFINEMENT ONLY RUNNER")
    print("="*80)
    print("\nThis script will:")
    print("  1. Identify checkpoints needing refinement")
    print("  2. Refine QDs for those questions")
    print("  3. Re-grade only those responses with refined QDs")
    print("  4. Does NOT re-run baseline or initial QD grading")
    print("="*80)
    
    # Check if questions.json exists
    questions_file = Path(questions_json_path)
    if not questions_file.exists():
        print(f"\n❌ Error: questions.json not found at {questions_json_path}")
        print("Please provide the path to your questions.json file")
        sys.exit(1)
    
    # Identify checkpoints needing refinement
    print("\n" + "="*80)
    print("STEP 1: IDENTIFYING CHECKPOINTS NEEDING REFINEMENT")
    print("="*80)
    checkpoints_needing_refinement = identify_checkpoints_needing_refinement()
    
    if not checkpoints_needing_refinement:
        print("\n✓ No checkpoints need refinement. All done!")
        return
    
    print(f"\nFound {len(checkpoints_needing_refinement)} question(s) with checkpoints needing refinement:")
    for qid, rids in checkpoints_needing_refinement.items():
        print(f"  {qid}: {len(rids)} response(s)")
    
    # Filter to specific questions if requested
    if specific_questions:
        checkpoints_needing_refinement = {
            qid: rids for qid, rids in checkpoints_needing_refinement.items()
            if qid in specific_questions
        }
        if not checkpoints_needing_refinement:
            print(f"\n⚠ No checkpoints found for specified questions: {specific_questions}")
            return
        print(f"\nFiltered to {len(checkpoints_needing_refinement)} question(s): {list(checkpoints_needing_refinement.keys())}")
    
    # Ensure refinement_reason and qd_result are set for all checkpoints
    print("\n" + "="*80)
    print("STEP 2: ENSURING REFINEMENT REASONS AND FLAGS ARE SET")
    print("="*80)
    for qid, rids in checkpoints_needing_refinement.items():
        for rid in rids:
            checkpoint = load_checkpoint(qid, rid)
            if checkpoint:
                updated = False
                # Ensure refinement_reason is set
                if ensure_refinement_reason_set(checkpoint):
                    updated = True
                # Also ensure qd_result is set to needs_refinement
                if checkpoint.get('qd_result') != 'needs_refinement':
                    checkpoint['qd_result'] = 'needs_refinement'
                    updated = True
                if updated:
                    save_checkpoint(checkpoint, qid, rid)
                    print(f"  ✓ {qid}_{rid}: Updated flags")
    
    # Prepare questions data
    print("\n" + "="*80)
    print("STEP 3: PREPARING QUESTIONS DATA")
    print("="*80)
    questions_data = prepare_questions_data(questions_json_path, checkpoints_needing_refinement)
    
    if not questions_data:
        print("\n❌ No questions data prepared. Cannot proceed.")
        return
    
    # Run refinement
    print("\n" + "="*80)
    print("STEP 4: RUNNING REFINEMENT")
    print("="*80)
    print("\nThis will:")
    print("  - Refine QDs for questions with problematic responses")
    print("  - Re-grade only those responses with refined QDs")
    print("  - Skip baseline and initial QD grading (already done)\n")
    
    try:
        refine_problematic_qds(questions_data)
        print("\n✓ Refinement complete!")
    except Exception as e:
        print(f"\n❌ Error during refinement: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nRefinement has been run on all checkpoints that needed it.")
    print("Check the checkpoints directory for updated files.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run refinement on checkpoints that need it without re-running baseline or initial QD grading'
    )
    parser.add_argument(
        '--questions-json',
        type=str,
        default='./questions.json',
        help='Path to questions.json file (default: ./questions.json)'
    )
    parser.add_argument(
        '--questions',
        type=str,
        nargs='+',
        help='Specific question IDs to process (e.g., --questions L1Q1 L1Q2). If not provided, processes all questions with checkpoints needing refinement.'
    )
    
    args = parser.parse_args()
    
    run_refinement_only(
        questions_json_path=args.questions_json,
        specific_questions=args.questions
    )

