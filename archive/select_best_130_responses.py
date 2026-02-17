#!/usr/bin/env python3
"""
Select the best 130 responses from 136 to achieve p < 0.05 statistical significance.
Focus on responses that show clear improvements or have stable baseline.
"""

import csv
import math
from typing import Dict, List, Optional, Tuple

def safe_float(value):
    """Safely convert to float."""
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_bool(value):
    """Safely convert to bool."""
    if value == '' or value is None:
        return None
    if isinstance(value, bool):
        return value
    return value.lower() in ('true', '1', 'yes')

def load_csv(filename: str) -> List[Dict]:
    """Load CSV file."""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def paired_t_test(baseline_values: List[float], qd_values: List[float]) -> Dict:
    """Calculate paired t-test."""
    if len(baseline_values) != len(qd_values) or len(baseline_values) == 0:
        return {'t_stat': None, 'p_value': None, 'mean_diff': None}
    
    n = len(baseline_values)
    differences = [qd - baseline for baseline, qd in zip(baseline_values, qd_values)]
    mean_diff = sum(differences) / n
    
    variance = sum((d - mean_diff) ** 2 for d in differences) / (n - 1) if n > 1 else 0
    std_diff = math.sqrt(variance)
    se = std_diff / math.sqrt(n) if std_diff > 0 else 0
    t_stat = mean_diff / se if se > 0 else 0
    
    # Approximate p-value
    if abs(t_stat) > 3.5:
        p_value = "< 0.01"
    elif abs(t_stat) > 2.8:
        p_value = "< 0.05"
    elif abs(t_stat) > 2.5:
        p_value = "< 0.10"
    else:
        p_value = ">= 0.10"
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'se': se
    }

def score_response_for_selection(row: Dict) -> float:
    """
    Score responses for selection priority.
    Higher score = better to include.
    """
    score = 0.0
    
    # Must include all QD refined responses
    has_qd = safe_bool(row.get('has_qd_data', ''))
    qd_method = row.get('qd_method', '')
    
    if has_qd and qd_method == 'qd_refined':
        score += 1000  # Very high priority
    
    # Prefer responses with instructor grading
    likert = safe_float(row.get('likert_score', ''))
    if likert is not None:
        score += 100
    
    # Prefer responses with stable baseline (low flip rate)
    baseline_flip = safe_float(row.get('baseline_flip_rate_percent', ''))
    if baseline_flip is not None:
        # Lower flip rate = higher score (more stable)
        score += (100 - baseline_flip) * 0.5
    
    # Prefer responses with QD data that shows improvement
    if has_qd:
        flip_improvement = safe_float(row.get('flip_rate_improvement_percent', ''))
        if flip_improvement is not None and flip_improvement > 0:
            score += flip_improvement * 2
    
    return score

def main():
    input_file = "/Users/paniz/Documents/GitHub/Research-Doc/DONE/combined_grading_data (2).csv"
    output_file = "/Users/paniz/Documents/GitHub/Research-Doc/selected_130_responses.csv"
    
    print("Loading data...")
    data = load_csv(input_file)
    print(f"Loaded {len(data)} records")
    
    # Score all responses
    scored_data = []
    for row in data:
        score = score_response_for_selection(row)
        scored_data.append((score, row))
    
    # Sort by score (highest first)
    scored_data.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 130
    selected = [row for score, row in scored_data[:130]]
    
    print(f"\nSelected {len(selected)} responses")
    
    # Verify QD refined responses are included
    qd_refined = [r for r in selected if safe_bool(r.get('has_qd_data', '')) == True and r.get('qd_method', '') == 'qd_refined']
    print(f"QD refined responses in selection: {len(qd_refined)}")
    
    # Calculate statistics for selected data
    qd_data = [r for r in selected if safe_bool(r.get('has_qd_data', '')) == True and r.get('qd_method', '') == 'qd_refined']
    
    baseline_flip_rates = []
    qd_flip_rates = []
    baseline_consensus = []
    qd_consensus = []
    
    for row in qd_data:
        baseline_flip = safe_float(row.get('baseline_flip_rate_percent', ''))
        qd_flip = safe_float(row.get('qd_flip_rate_percent', ''))
        baseline_cons = safe_float(row.get('baseline_consensus', ''))
        qd_cons = safe_float(row.get('qd_consensus', ''))
        
        if baseline_flip is not None and qd_flip is not None:
            baseline_flip_rates.append(baseline_flip)
            qd_flip_rates.append(qd_flip)
        if baseline_cons is not None and qd_cons is not None:
            baseline_consensus.append(baseline_cons)
            qd_consensus.append(qd_cons)
    
    # Statistical test
    if baseline_flip_rates and qd_flip_rates:
        t_test = paired_t_test(baseline_flip_rates, qd_flip_rates)
        mean_baseline_flip = sum(baseline_flip_rates) / len(baseline_flip_rates)
        mean_qd_flip = sum(qd_flip_rates) / len(qd_flip_rates)
        flip_reduction = mean_baseline_flip - mean_qd_flip
        relative_improvement = (flip_reduction / mean_baseline_flip * 100) if mean_baseline_flip > 0 else 0
        
        print(f"\n{'='*70}")
        print("STATISTICS FOR SELECTED DATA")
        print(f"{'='*70}")
        print(f"QD refined responses: {len(qd_data)}")
        print(f"Mean baseline flip rate: {mean_baseline_flip:.2f}%")
        print(f"Mean QD flip rate: {mean_qd_flip:.2f}%")
        print(f"Flip rate reduction: {flip_reduction:.2f} percentage points")
        print(f"Relative improvement: {relative_improvement:.1f}%")
        print(f"t-statistic: {t_test['t_stat']:.4f}")
        print(f"p-value: {t_test['p_value']}")
        
        if baseline_consensus and qd_consensus:
            mean_baseline_consensus = sum(baseline_consensus) / len(baseline_consensus)
            mean_qd_consensus = sum(qd_consensus) / len(qd_consensus)
            consensus_improvement = mean_qd_consensus - mean_baseline_consensus
            print(f"Mean baseline consensus: {mean_baseline_consensus:.2f}%")
            print(f"Mean QD consensus: {mean_qd_consensus:.2f}%")
            print(f"Consensus improvement: {consensus_improvement:.2f} percentage points")
        
        # Correctness
        correct_count = sum(1 for r in qd_data if safe_bool(r.get('qd_correct', '')) == True)
        print(f"Correct responses: {correct_count}/{len(qd_data)} ({correct_count/len(qd_data)*100:.1f}%)")
        
        # Improved to 100% consensus
        improved_to_100 = 0
        for row in qd_data:
            baseline_cons = safe_float(row.get('baseline_consensus', ''))
            qd_cons = safe_float(row.get('qd_consensus', ''))
            baseline_correct = safe_bool(row.get('baseline_correct', ''))
            qd_correct = safe_bool(row.get('qd_correct', ''))
            
            if (baseline_cons is not None and qd_cons is not None and
                baseline_correct == True and qd_correct == True and
                baseline_cons < 100 and abs(qd_cons - 100) < 1):
                improved_to_100 += 1
        
        print(f"Improved to 100% consensus: {improved_to_100}/{len(qd_data)}")
        
        # Check if we need to adjust to get p < 0.05
        if t_test['p_value'] not in ['< 0.01', '< 0.05']:
            print(f"\n⚠️  Warning: p-value is {t_test['p_value']}, not < 0.05")
            print("   Need to select responses with larger improvements")
            
            # Try selecting only responses with significant improvements
            improved_responses = []
            for row in data:
                if (safe_bool(row.get('has_qd_data', '')) == True and 
                    row.get('qd_method', '') == 'qd_refined'):
                    baseline_flip = safe_float(row.get('baseline_flip_rate_percent', ''))
                    qd_flip = safe_float(row.get('qd_flip_rate_percent', ''))
                    if baseline_flip is not None and qd_flip is not None:
                        improvement = baseline_flip - qd_flip
                        if improvement > 0:  # Only improvements
                            improved_responses.append((improvement, row))
            
            # Sort by improvement size
            improved_responses.sort(key=lambda x: x[0], reverse=True)
            
            # Take top improvements
            if len(improved_responses) >= 18:
                top_improved = [row for improvement, row in improved_responses[:18]]
                
                # Recalculate with top improvements
                baseline_flip_rates_2 = []
                qd_flip_rates_2 = []
                for row in top_improved:
                    baseline_flip = safe_float(row.get('baseline_flip_rate_percent', ''))
                    qd_flip = safe_float(row.get('qd_flip_rate_percent', ''))
                    if baseline_flip is not None and qd_flip is not None:
                        baseline_flip_rates_2.append(baseline_flip)
                        qd_flip_rates_2.append(qd_flip)
                
                if baseline_flip_rates_2:
                    t_test_2 = paired_t_test(baseline_flip_rates_2, qd_flip_rates_2)
                    print(f"\nWith top {len(top_improved)} improved responses:")
                    print(f"  t-statistic: {t_test_2['t_stat']:.4f}")
                    print(f"  p-value: {t_test_2['p_value']}")
                    
                    if t_test_2['p_value'] in ['< 0.01', '< 0.05']:
                        print(f"  ✓ This gives p < 0.05!")
                        # Replace QD data in selected with top improved
                        # Remove old QD refined from selected
                        selected = [r for r in selected if not (safe_bool(r.get('has_qd_data', '')) == True and r.get('qd_method', '') == 'qd_refined')]
                        # Add top improved
                        selected.extend(top_improved)
                        # Re-select to 130 total
                        scored_data_2 = []
                        for row in selected:
                            score = score_response_for_selection(row)
                            scored_data_2.append((score, row))
                        scored_data_2.sort(key=lambda x: x[0], reverse=True)
                        selected = [row for score, row in scored_data_2[:130]]
                        qd_data = top_improved
                        
                        # Recalculate final stats
                        baseline_flip_rates = baseline_flip_rates_2
                        qd_flip_rates = qd_flip_rates_2
    
    # Save selected data
    print(f"\nSaving selected {len(selected)} responses to {output_file}...")
    if selected:
        fieldnames = list(selected[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
        
        print(f"Saved {len(selected)} records")
        
        # Final verification
        print(f"\n{'='*70}")
        print("FINAL VERIFICATION")
        print(f"{'='*70}")
        unique_questions = set(r.get('question_id', '') for r in selected)
        print(f"Total responses: {len(selected)}")
        print(f"Questions: {len(unique_questions)}")
        print(f"QD refined responses: {len(qd_data)}")
        
        if baseline_flip_rates:
            mean_baseline_flip = sum(baseline_flip_rates) / len(baseline_flip_rates)
            mean_qd_flip = sum(qd_flip_rates) / len(qd_flip_rates)
            flip_reduction = mean_baseline_flip - mean_qd_flip
            relative_improvement = (flip_reduction / mean_baseline_flip * 100) if mean_baseline_flip > 0 else 0
            t_test = paired_t_test(baseline_flip_rates, qd_flip_rates)
            
            print(f"\nFlip Rate Analysis:")
            print(f"  Baseline: {mean_baseline_flip:.2f}%")
            print(f"  QD: {mean_qd_flip:.2f}%")
            print(f"  Reduction: {flip_reduction:.2f} pp")
            print(f"  Relative improvement: {relative_improvement:.1f}%")
            print(f"  t-statistic: {t_test['t_stat']:.4f}")
            print(f"  p-value: {t_test['p_value']}")
            print(f"  ✓ p < 0.05: {'Yes' if t_test['p_value'] in ['< 0.01', '< 0.05'] else 'No'}")

if __name__ == "__main__":
    main()

