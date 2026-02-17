#!/usr/bin/env python3
"""
Combine consistency/correctness analysis with instructor grading data
and create final comprehensive CSV.
"""

import csv
from typing import Dict, List

def safe_float(value: str):
    """Safely convert string to float."""
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_int(value: str):
    """Safely convert string to int."""
    if value == '' or value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None

def safe_bool(value: str):
    """Safely convert string to bool."""
    if value == '' or value is None:
        return None
    if isinstance(value, bool):
        return value
    return value.lower() in ('true', '1', 'yes')

def load_csv(filename: str) -> List[Dict]:
    """Load CSV file and return list of dictionaries."""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def main():
    instructor_file = "/Users/paniz/Documents/GitHub/Research-Doc/DONE/instrcutor_grading.csv"
    analysis_file = "/Users/paniz/Documents/GitHub/Research-Doc/consistency_correctness_analysis.csv"
    output_file = "/Users/paniz/Documents/GitHub/Research-Doc/final_combined_analysis.csv"
    
    print("Loading instructor grading...")
    instructor_data = load_csv(instructor_file)
    print(f"Loaded {len(instructor_data)} instructor grades")
    
    print("Loading consistency/correctness analysis...")
    analysis_data = load_csv(analysis_file)
    print(f"Loaded {len(analysis_data)} analysis records")
    
    # Create lookup dictionaries
    instructor_lookup = {}
    for row in instructor_data:
        key = (row['Question_ID'], row['Response_ID'])
        instructor_lookup[key] = row
    
    analysis_lookup = {}
    for row in analysis_data:
        key = (row['question_id'], row['response_id'])
        analysis_lookup[key] = row
    
    # Combine data
    print("\nCombining data...")
    combined = []
    
    # Start with all instructor grades
    for key, instructor_row in instructor_lookup.items():
        combined_row = {
            'question_id': instructor_row.get('Question_ID', ''),
            'response_id': instructor_row.get('Response_ID', ''),
            'instructor_likert_score': instructor_row.get('Likert_Score', ''),
        }
        
        # Add analysis data if available
        if key in analysis_lookup:
            analysis_row = analysis_lookup[key]
            
            # Add all analysis fields
            combined_row.update({
                'expected_majority': analysis_row.get('expected_majority', ''),
                'baseline_majority': analysis_row.get('baseline_majority', ''),
                'baseline_consensus': analysis_row.get('baseline_consensus', ''),
                'baseline_flip_rate_percent': analysis_row.get('baseline_flip_rate_percent', ''),
                'baseline_correct': analysis_row.get('baseline_correct', ''),
                'baseline_consistent': analysis_row.get('baseline_consistent', ''),
                'baseline_consistency_reason': analysis_row.get('baseline_consistency_reason', ''),
                'qd_majority': analysis_row.get('qd_majority', ''),
                'qd_consensus': analysis_row.get('qd_consensus', ''),
                'qd_flip_rate_percent': analysis_row.get('qd_flip_rate_percent', ''),
                'qd_method': analysis_row.get('qd_method', ''),
                'qd_correct': analysis_row.get('qd_correct', ''),
                'qd_consistent': analysis_row.get('qd_consistent', ''),
                'qd_consistency_reason': analysis_row.get('qd_consistency_reason', ''),
                'consistency_improved': analysis_row.get('consistency_improved', ''),
                'correctness_improved': analysis_row.get('correctness_improved', ''),
                'flip_rate_improvement_percent': analysis_row.get('flip_rate_improvement_percent', ''),
                'consensus_improvement': analysis_row.get('consensus_improvement', ''),
                'has_qd_data': analysis_row.get('has_qd_data', ''),
            })
        else:
            # No analysis data available
            combined_row.update({
                'expected_majority': '',
                'baseline_majority': '',
                'baseline_consensus': '',
                'baseline_flip_rate_percent': '',
                'baseline_correct': '',
                'baseline_consistent': '',
                'baseline_consistency_reason': '',
                'qd_majority': '',
                'qd_consensus': '',
                'qd_flip_rate_percent': '',
                'qd_method': '',
                'qd_correct': '',
                'qd_consistent': '',
                'qd_consistency_reason': '',
                'consistency_improved': '',
                'correctness_improved': '',
                'flip_rate_improvement_percent': '',
                'consensus_improvement': '',
                'has_qd_data': 'False',
            })
        
        combined.append(combined_row)
    
    # Sort by question_id and response_id
    combined.sort(key=lambda x: (x.get('question_id', ''), x.get('response_id', '')))
    
    # Save combined CSV
    print(f"\nSaving combined data to {output_file}...")
    if combined:
        fieldnames = [
            'question_id', 'response_id', 'instructor_likert_score',
            'expected_majority',
            'baseline_majority', 'baseline_consensus', 'baseline_flip_rate_percent',
            'baseline_correct', 'baseline_consistent', 'baseline_consistency_reason',
            'qd_majority', 'qd_consensus', 'qd_flip_rate_percent', 'qd_method',
            'qd_correct', 'qd_consistent', 'qd_consistency_reason',
            'consistency_improved', 'correctness_improved',
            'flip_rate_improvement_percent', 'consensus_improvement',
            'has_qd_data'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)
        
        print(f"Saved {len(combined)} records to {output_file}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        total = len(combined)
        with_qd = sum(1 for r in combined if safe_bool(r.get('has_qd_data', '')) == True)
        baseline_consistent = sum(1 for r in combined if safe_bool(r.get('baseline_consistent', '')) == True)
        baseline_correct = sum(1 for r in combined if safe_bool(r.get('baseline_correct', '')) == True)
        qd_consistent = sum(1 for r in combined if safe_bool(r.get('qd_consistent', '')) == True and safe_bool(r.get('has_qd_data', '')) == True)
        qd_correct = sum(1 for r in combined if safe_bool(r.get('qd_correct', '')) == True and safe_bool(r.get('has_qd_data', '')) == True)
        
        print(f"Total records: {total}")
        print(f"Records with QD data: {with_qd}")
        print(f"Baseline consistent: {baseline_consistent}")
        print(f"Baseline correct: {baseline_correct}")
        print(f"QD consistent (with QD data): {qd_consistent}")
        print(f"QD correct (with QD data): {qd_correct}")
    else:
        print("No data to save")

if __name__ == "__main__":
    main()

