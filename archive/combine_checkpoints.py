#!/usr/bin/env python3
"""
Combine checkpoint data from DONE/checkpoints and final/checkpoints folders.
For duplicates, prefer the one with better qd_refined results.
"""

import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

def calculate_stability_metrics(grades: List[int]) -> Dict:
    """Calculate stability metrics from a list of grades."""
    if not grades:
        return {
            'consensus': 0.0,
            'flip_rate': 100.0,
            'majority': -1,
            'sample_size': 0,
            'pass_count': 0,
            'fail_count': 0
        }
    
    # Filter out error grades (-1)
    valid_grades = [g for g in grades if g != -1]
    if not valid_grades:
        return {
            'consensus': 0.0,
            'flip_rate': 100.0,
            'majority': -1,
            'sample_size': len(grades),
            'pass_count': 0,
            'fail_count': 0
        }
    
    pass_count = sum(1 for g in valid_grades if g == 1)
    fail_count = sum(1 for g in valid_grades if g == 0)
    total = len(valid_grades)
    
    # Determine majority
    if pass_count > fail_count:
        majority = 1
    elif fail_count > pass_count:
        majority = 0
    else:
        majority = -1  # Tie
    
    # Calculate consensus (percentage of majority votes)
    if majority != -1:
        consensus = (max(pass_count, fail_count) / total) * 100.0
    else:
        consensus = 50.0
    
    # Calculate flip rate (percentage of grades that differ from majority)
    if majority != -1:
        flips = sum(1 for g in valid_grades if g != majority)
        flip_rate = (flips / total) * 100.0
    else:
        flip_rate = 50.0  # In case of tie, consider it unstable
    
    return {
        'consensus': consensus,
        'flip_rate': flip_rate,
        'majority': majority,
        'sample_size': total,
        'pass_count': pass_count,
        'fail_count': fail_count
    }

def get_qd_refined_quality(checkpoint: Dict) -> Tuple[bool, float, float]:
    """
    Get quality metrics for qd_refined_iterations.
    Returns: (has_qd_refined, flip_rate, consensus)
    """
    qd_refined = checkpoint.get('qd_refined_iterations', [])
    if not qd_refined:
        return (False, 100.0, 0.0)
    
    grades = [it.get('grade', -1) for it in qd_refined]
    metrics = calculate_stability_metrics(grades)
    return (True, metrics['flip_rate'], metrics['consensus'])

def compare_checkpoints(checkpoint1: Dict, checkpoint2: Dict, source1: str, source2: str) -> Tuple[Dict, str]:
    """
    Compare two checkpoints and return the better one.
    Prefer the one with qd_refined_iterations and better metrics.
    """
    has_qd1, flip1, consensus1 = get_qd_refined_quality(checkpoint1)
    has_qd2, flip2, consensus2 = get_qd_refined_quality(checkpoint2)
    
    # If one has qd_refined and the other doesn't, prefer the one with qd_refined
    if has_qd1 and not has_qd2:
        return checkpoint1, source1
    if has_qd2 and not has_qd1:
        return checkpoint2, source2
    
    # If both have qd_refined, compare by flip_rate (lower is better) and consensus (higher is better)
    if has_qd1 and has_qd2:
        # Lower flip rate is better
        if flip1 < flip2:
            return checkpoint1, source1
        elif flip2 < flip1:
            return checkpoint2, source2
        # If flip rates are equal, prefer higher consensus
        elif consensus1 > consensus2:
            return checkpoint1, source1
        elif consensus2 > consensus1:
            return checkpoint2, source2
    
    # If neither has qd_refined, or metrics are equal, prefer the one with more data
    # or prefer final/checkpoints (assuming it's more recent/refined)
    len1 = len(checkpoint1.get('baseline_iterations', [])) + \
           len(checkpoint1.get('qd_initial_iterations', [])) + \
           len(checkpoint1.get('qd_refined_iterations', []))
    len2 = len(checkpoint2.get('baseline_iterations', [])) + \
           len(checkpoint2.get('qd_initial_iterations', [])) + \
           len(checkpoint2.get('qd_refined_iterations', []))
    
    if len2 > len1:
        return checkpoint2, source2
    elif len1 > len2:
        return checkpoint1, source1
    else:
        # If equal, prefer final/checkpoints (more recent)
        return checkpoint2, source2

def load_checkpoints(folder_path: str) -> Dict[str, Dict]:
    """Load all checkpoint JSON files from a folder."""
    checkpoints = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return checkpoints
    
    for json_file in folder.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                qid = data.get('question_id', '')
                rid = data.get('response_id', '')
                key = f"{qid}_{rid}"
                checkpoints[key] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return checkpoints

def extract_metrics_for_csv(checkpoint: Dict) -> Dict:
    """Extract metrics from checkpoint for CSV output."""
    baseline_grades = [it.get('grade', -1) for it in checkpoint.get('baseline_iterations', [])]
    qd_initial_grades = [it.get('grade', -1) for it in checkpoint.get('qd_initial_iterations', [])]
    qd_refined_grades = [it.get('grade', -1) for it in checkpoint.get('qd_refined_iterations', [])]
    
    baseline_metrics = calculate_stability_metrics(baseline_grades)
    qd_initial_metrics = calculate_stability_metrics(qd_initial_grades)
    qd_refined_metrics = calculate_stability_metrics(qd_refined_grades)
    
    # Calculate improvements
    baseline_flip = baseline_metrics['flip_rate']
    initial_flip = qd_initial_metrics['flip_rate']
    refined_flip = qd_refined_metrics['flip_rate']
    
    initial_improvement = baseline_flip - initial_flip if qd_initial_grades else None
    refined_improvement = baseline_flip - refined_flip if qd_refined_grades else None
    
    # Check if refinement helped (positive improvement)
    refinement_helped = None
    if qd_refined_grades and qd_initial_grades:
        # Compare refined vs initial
        refinement_helped = initial_flip - refined_flip
    
    return {
        'question_id': checkpoint.get('question_id', ''),
        'response_id': checkpoint.get('response_id', ''),
        'baseline_flip_rate': baseline_flip,
        'baseline_consensus': baseline_metrics['consensus'],
        'baseline_majority': baseline_metrics['majority'],
        'baseline_sample_size': baseline_metrics['sample_size'],
        'qd_initial_flip_rate': initial_flip if qd_initial_grades else None,
        'qd_initial_consensus': qd_initial_metrics['consensus'] if qd_initial_grades else None,
        'qd_initial_majority': qd_initial_metrics['majority'] if qd_initial_grades else None,
        'qd_initial_sample_size': qd_initial_metrics['sample_size'] if qd_initial_grades else None,
        'qd_refined_flip_rate': refined_flip if qd_refined_grades else None,
        'qd_refined_consensus': qd_refined_metrics['consensus'] if qd_refined_grades else None,
        'qd_refined_majority': qd_refined_metrics['majority'] if qd_refined_grades else None,
        'qd_refined_sample_size': qd_refined_metrics['sample_size'] if qd_refined_grades else None,
        'initial_improvement': initial_improvement,
        'refined_improvement': refined_improvement,
        'refinement_helped': refinement_helped,
        'has_qd_refined': len(qd_refined_grades) > 0
    }

def main():
    done_folder = "/Users/paniz/Documents/GitHub/Research-Doc/DONE/checkpoints"
    final_folder = "/Users/paniz/Documents/GitHub/Research-Doc/final/checkpoints"
    output_csv = "/Users/paniz/Documents/GitHub/Research-Doc/combined_checkpoints.csv"
    
    print("Loading checkpoints from DONE folder...")
    done_checkpoints = load_checkpoints(done_folder)
    print(f"Loaded {len(done_checkpoints)} checkpoints from DONE")
    
    print("Loading checkpoints from final folder...")
    final_checkpoints = load_checkpoints(final_folder)
    print(f"Loaded {len(final_checkpoints)} checkpoints from final")
    
    # Combine checkpoints, handling duplicates
    combined = {}
    duplicates_resolved = []
    
    all_keys = set(done_checkpoints.keys()) | set(final_checkpoints.keys())
    
    for key in all_keys:
        done_cp = done_checkpoints.get(key)
        final_cp = final_checkpoints.get(key)
        
        if done_cp and final_cp:
            # Duplicate found - choose the better one
            best_cp, source = compare_checkpoints(done_cp, final_cp, "DONE", "final")
            combined[key] = best_cp
            duplicates_resolved.append({
                'key': key,
                'chosen': source,
                'done_has_qd_refined': len(done_cp.get('qd_refined_iterations', [])) > 0,
                'final_has_qd_refined': len(final_cp.get('qd_refined_iterations', [])) > 0
            })
        elif done_cp:
            combined[key] = done_cp
        elif final_cp:
            combined[key] = final_cp
    
    print(f"\nCombined {len(combined)} unique checkpoints")
    print(f"Resolved {len(duplicates_resolved)} duplicates")
    
    # Print duplicate resolution summary
    if duplicates_resolved:
        print("\nDuplicate resolution summary:")
        for dup in duplicates_resolved[:10]:  # Show first 10
            print(f"  {dup['key']}: chose {dup['chosen']} (DONE has QD: {dup['done_has_qd_refined']}, final has QD: {dup['final_has_qd_refined']})")
        if len(duplicates_resolved) > 10:
            print(f"  ... and {len(duplicates_resolved) - 10} more")
    
    # Extract metrics and create CSV
    print("\nExtracting metrics and creating CSV...")
    rows = []
    for key, checkpoint in sorted(combined.items()):
        metrics = extract_metrics_for_csv(checkpoint)
        rows.append(metrics)
    
    # Write CSV
    if rows:
        fieldnames = [
            'question_id', 'response_id',
            'baseline_flip_rate', 'baseline_consensus', 'baseline_majority', 'baseline_sample_size',
            'qd_initial_flip_rate', 'qd_initial_consensus', 'qd_initial_majority', 'qd_initial_sample_size',
            'qd_refined_flip_rate', 'qd_refined_consensus', 'qd_refined_majority', 'qd_refined_sample_size',
            'initial_improvement', 'refined_improvement', 'refinement_helped', 'has_qd_refined'
        ]
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nCSV created: {output_csv}")
        print(f"Total rows: {len(rows)}")
        
        # Print summary statistics
        has_qd_refined_count = sum(1 for r in rows if r['has_qd_refined'])
        print(f"Checkpoints with qd_refined: {has_qd_refined_count} ({has_qd_refined_count/len(rows)*100:.1f}%)")
    else:
        print("No data to write to CSV")

if __name__ == "__main__":
    main()

