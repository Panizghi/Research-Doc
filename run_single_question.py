#!/usr/bin/env python3
"""
Single Question Experiment Runner
Conservative API usage for testing individual questions
"""

import sys
import os
from config import API_URL, DATA_PATH, MIN_CORPUS_SIZE
from grading_experiment import run_experiment
from qd_extraction import discover_and_refine_qds
import pandas as pd

def run_single_question_conservative(lab_number: int, question_number: int, 
                                   num_trials: int = 3, 
                                   discovery_sample: int = 10):
    """
    Run experiment on single question with conservative API usage
    
    Args:
        lab_number: Lab number (1-4)
        question_number: Question number (1-5)
        num_trials: Number of grading trials (default: 3, was 10)
        discovery_sample: Number of texts for QD discovery (default: 10, was 25)
    """
    print("="*80)
    print(f"CONSERVATIVE SINGLE QUESTION EXPERIMENT")
    print(f"Lab {lab_number}, Question {question_number}")
    print("="*80)
    print(f"📊 Configuration:")
    print(f"  - Grading trials: {num_trials} (reduced from 10)")
    print(f"  - Discovery sample: {discovery_sample} (reduced from 25)")
    print(f"  - API endpoint: {API_URL}")
    print("="*80)
    
    # Load data
    from grading_experiment import load_bad_questions_data, extract_corpus_from_csv, get_question_text
    
    print("\n📁 Loading data...")
    df = load_bad_questions_data(DATA_PATH)
    corpus = extract_corpus_from_csv(df, lab_number, question_number)
    question = get_question_text(lab_number, question_number)
    
    if len(corpus) < MIN_CORPUS_SIZE:
        print(f"❌ Insufficient corpus size ({len(corpus)} < {MIN_CORPUS_SIZE})")
        return None
    
    print(f"✓ Loaded {len(corpus)} unique texts")
    print(f"✓ Question: {question}")
    
    # Estimate API calls
    estimated_calls = estimate_api_calls(len(corpus), discovery_sample, num_trials)
    print(f"\n📞 Estimated API calls: {estimated_calls}")
    
    # Confirm before proceeding
    confirm = input(f"\nProceed with {estimated_calls} API calls? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Experiment cancelled")
        return None
    
    # Temporarily modify config for conservative run
    import qd_extraction
    original_discovery_size = qd_extraction.DISCOVERY_SAMPLE_SIZE
    qd_extraction.DISCOVERY_SAMPLE_SIZE = discovery_sample
    
    try:
        print(f"\n🔬 Starting QD discovery and refinement...")
        print(f"   Using {discovery_sample} texts for discovery")
        
        # Discover and refine QDs
        initial_qds, refined_qds = discover_and_refine_qds(question, corpus)
        
        print(f"✓ Initial QDs: {len(initial_qds)}")
        print(f"✓ Refined QDs: {len(refined_qds)}")
        
        # Run grading comparison
        print(f"\n📝 Running grading comparison...")
        print(f"   Using {num_trials} trials per text")
        
        from grading_experiment import compare_rubrics
        results = compare_rubrics(
            corpus, question, initial_qds, refined_qds, num_trials
        )
        
        # Add metadata
        results['lab'] = lab_number
        results['question'] = question_number
        results['corpus_size'] = len(corpus)
        results['num_trials'] = num_trials
        results['discovery_sample'] = discovery_sample
        
        # Print summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Lab: {lab_number}, Question: {question_number}")
        print(f"Corpus size: {len(corpus)}")
        print(f"Initial QDs: {len(initial_qds)}")
        print(f"Refined QDs: {len(refined_qds)}")
        print(f"Flip rate improvement: {results['improvement']:+.2f}%")
        print(f"Kappa improvement: {results['kappa_improvement']:+.3f}")
        
        return results
        
    finally:
        # Restore original config
        qd_extraction.DISCOVERY_SAMPLE_SIZE = original_discovery_size


def estimate_api_calls(corpus_size: int, discovery_sample: int, num_trials: int) -> int:
    """
    Estimate total API calls for the experiment
    """
    # QD discovery: 1 call (analyzes sample, doesn't call per text)
    discovery_calls = 1
    
    # Annotation: corpus_size × num_qds (each text × each QD)
    # Assume 6 QDs on average
    num_qds = 6
    annotation_calls = corpus_size * num_qds
    
    # QD refinement: up to max_merges calls (coupling decisions)
    # Usually 0-5 merges happen
    refinement_calls = 3  # conservative estimate
    
    # Grading: 2 rubrics × corpus_size × num_trials
    grading_calls = 2 * corpus_size * num_trials
    
    total = discovery_calls + annotation_calls + refinement_calls + grading_calls
    return total


def interactive_mode():
    """
    Interactive mode for selecting question
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SINGLE QUESTION EXPERIMENT RUNNER                        ║
║                                                                              ║
║  Conservative API usage for testing individual questions                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Get lab and question
    try:
        lab = int(input("Enter lab number (1-4): "))
        question = int(input("Enter question number (1-5): "))
    except ValueError:
        print("❌ Invalid input")
        return
    
    if not (1 <= lab <= 4) or not (1 <= question <= 5):
        print("❌ Lab must be 1-4, Question must be 1-5")
        return
    
    # Get conservative settings
    print(f"\n⚙️  Conservative settings:")
    print(f"   - Grading trials: 3 (default)")
    print(f"   - Discovery sample: 10 (default)")
    
    custom = input("Use custom settings? (y/n): ").strip().lower()
    
    num_trials = 3
    discovery_sample = 10
    
    if custom == 'y':
        try:
            num_trials = int(input("Number of grading trials (1-5): "))
            discovery_sample = int(input("Discovery sample size (5-20): "))
            
            if not (1 <= num_trials <= 5) or not (5 <= discovery_sample <= 20):
                print("❌ Invalid ranges, using defaults")
                num_trials = 3
                discovery_sample = 10
        except ValueError:
            print("❌ Invalid input, using defaults")
    
    # Run experiment
    result = run_single_question_conservative(lab, question, num_trials, discovery_sample)
    
    if result:
        print(f"\n✅ Experiment completed successfully!")
        
        # Ask if want to save results
        save = input("Save results to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"results_lab{lab}_q{question}.json"
            import json
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Results saved to {filename}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Command line mode: python run_single_question.py <lab> <question>
        try:
            lab = int(sys.argv[1])
            question = int(sys.argv[2])
            result = run_single_question_conservative(lab, question)
        except (ValueError, IndexError):
            print("Usage: python run_single_question.py <lab> <question>")
            print("Example: python run_single_question.py 1 3")
    else:
        # Interactive mode
        interactive_mode()
