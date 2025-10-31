"""
Grading System and Flip Rate Measurement
Compare baseline vs refined rubrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.metrics import cohen_kappa_score
import requests
import json

BASE_URL = "http://ece-nebula16.eng.uwaterloo.ca:11434"

def generate(prompt: str, reasoning: bool = False) -> str:
    """Call LLM API"""
    payload = {
        "model": "gpt-oss:120b",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(
        f"{BASE_URL}/api/generate", 
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "No response returned")
    else:
        print(f"API Error: {response.status_code} - {response.text}")
        return "API Error occurred"


# ============================================================================
# GRADING WITH QDs
# ============================================================================

def grade_with_qds(text: str, qds: List[Dict], question: str, 
                   use_reasoning: bool = False) -> int:
    """
    Grade a student answer using QDs
    Returns: 0 (does not meet) or 1 (meets expectation)
    
    IMPORTANT: use_reasoning should be FALSE for grading (per your feedback)
    """
    qd_descriptions = "\n".join([
        f"- {qd['name']}: {qd['definition']}"
        for qd in qds
    ])
    
    prompt = f"""Grade this student answer using the quality dimensions below.

QUESTION: {question}

QUALITY DIMENSIONS:
{qd_descriptions}

STUDENT ANSWER:
{text}

Evaluate:
1. Which quality dimensions are present in the answer?
2. Based on QD presence, does the answer meet expectations?

An answer MEETS EXPECTATION if it demonstrates understanding through 
presence of key quality dimensions.

Output format:
Grade: [0 or 1]
Reason: [brief explanation]

Grade must be 0 (does not meet) or 1 (meets expectation)."""
    
    result = generate(prompt, reasoning=use_reasoning)
    
    # Parse grade
    if "Grade: 1" in result or "meets expectation" in result.lower():
        return 1
    else:
        return 0


def grade_corpus_multiple_times(corpus: List[str], 
                                qds: List[Dict],
                                question: str,
                                num_trials: int = 10) -> pd.DataFrame:
    """
    Grade each text multiple times to measure consistency
    
    Returns DataFrame with columns: text_idx, trial_0, trial_1, ..., trial_N
    """
    print(f"\n📝 Grading {len(corpus)} texts {num_trials} times...")
    
    results = []
    
    for idx, text in enumerate(corpus):
        if idx % 5 == 0:
            print(f"  Progress: {idx}/{len(corpus)}")
        
        row = {'text_idx': idx, 'text': text[:100] + "..."}
        
        for trial in range(num_trials):
            grade = grade_with_qds(text, qds, question, use_reasoning=False)
            row[f'trial_{trial}'] = grade
        
        results.append(row)
    
    df = pd.DataFrame(results)
    print(f"✓ Grading complete")
    
    return df


# ============================================================================
# FLIP RATE ANALYSIS
# ============================================================================

def compute_flip_rate(grades_df: pd.DataFrame) -> Dict:
    """
    Compute grade flip rate and reliability metrics
    """
    trial_cols = [col for col in grades_df.columns if col.startswith('trial_')]
    
    # Count texts with flips
    texts_with_flips = 0
    total_texts = len(grades_df)
    
    grade_std_devs = []
    
    for idx, row in grades_df.iterrows():
        grades = [row[col] for col in trial_cols]
        unique_grades = len(set(grades))
        
        if unique_grades > 1:
            texts_with_flips += 1
        
        grade_std_devs.append(np.std(grades))
    
    flip_rate = (texts_with_flips / total_texts * 100) if total_texts > 0 else 0
    
    # Compute Cohen's Kappa (average pairwise)
    kappas = []
    for i in range(len(trial_cols)):
        for j in range(i+1, len(trial_cols)):
            kappa = cohen_kappa_score(
                grades_df[trial_cols[i]], 
                grades_df[trial_cols[j]]
            )
            kappas.append(kappa)
    
    avg_kappa = np.mean(kappas) if kappas else 0
    avg_std_dev = np.mean(grade_std_devs)
    
    return {
        'total_texts': total_texts,
        'texts_with_flips': texts_with_flips,
        'flip_rate': round(flip_rate, 2),
        'cohen_kappa': round(avg_kappa, 3),
        'avg_std_dev': round(avg_std_dev, 3)
    }


def compare_rubrics(corpus: List[str],
                   question: str,
                   initial_qds: List[Dict],
                   refined_qds: List[Dict],
                   num_trials: int = 10) -> Dict:
    """
    Compare flip rates between baseline and refined rubrics
    """
    print("\n" + "="*80)
    print("RUBRIC COMPARISON EXPERIMENT")
    print("="*80)
    
    # Grade with baseline rubric
    print("\n[BASELINE] Grading with initial QDs...")
    baseline_grades = grade_corpus_multiple_times(
        corpus, initial_qds, question, num_trials
    )
    baseline_metrics = compute_flip_rate(baseline_grades)
    
    # Grade with refined rubric
    print("\n[REFINED] Grading with refined QDs...")
    refined_grades = grade_corpus_multiple_times(
        corpus, refined_qds, question, num_trials
    )
    refined_metrics = compute_flip_rate(refined_grades)
    
    # Compute improvement
    improvement = baseline_metrics['flip_rate'] - refined_metrics['flip_rate']
    kappa_improvement = refined_metrics['cohen_kappa'] - baseline_metrics['cohen_kappa']
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nBASELINE (Initial QDs: {len(initial_qds)}):")
    print(f"  Flip Rate: {baseline_metrics['flip_rate']:.2f}%")
    print(f"  Cohen's κ: {baseline_metrics['cohen_kappa']:.3f}")
    print(f"  Avg Std Dev: {baseline_metrics['avg_std_dev']:.3f}")
    print(f"  Texts with flips: {baseline_metrics['texts_with_flips']}/{baseline_metrics['total_texts']}")
    
    print(f"\nREFINED (Refined QDs: {len(refined_qds)}):")
    print(f"  Flip Rate: {refined_metrics['flip_rate']:.2f}%")
    print(f"  Cohen's κ: {refined_metrics['cohen_kappa']:.3f}")
    print(f"  Avg Std Dev: {refined_metrics['avg_std_dev']:.3f}")
    print(f"  Texts with flips: {refined_metrics['texts_with_flips']}/{refined_metrics['total_texts']}")
    
    print(f"\nIMPROVEMENT:")
    print(f"  {'✓' if improvement > 0 else '✗'} Flip Rate: {improvement:+.2f} percentage points")
    print(f"  {'✓' if kappa_improvement > 0 else '✗'} Cohen's κ: {kappa_improvement:+.3f}")
    
    if improvement > 0:
        print(f"\n🎉 SUCCESS: Refined rubric reduces flip rate by {improvement:.2f}%")
    else:
        print(f"\n⚠ Refined rubric did not improve flip rate")
    
    return {
        'baseline': baseline_metrics,
        'refined': refined_metrics,
        'improvement': improvement,
        'kappa_improvement': kappa_improvement,
        'baseline_grades': baseline_grades,
        'refined_grades': refined_grades
    }


# ============================================================================
# LOAD REAL DATA
# ============================================================================

def load_bad_questions_data(csv_path: str) -> pd.DataFrame:
    """
    Load the bad_questions.csv data
    """
    df = pd.read_csv(csv_path, sep='\t')
    return df


def extract_corpus_from_csv(df: pd.DataFrame, 
                            lab_number: int, 
                            question_number: int) -> List[str]:
    """
    Extract unique question texts for a specific lab/question
    """
    filtered = df[
        (df['lab_number'] == lab_number) & 
        (df['question_number'] == question_number)
    ]
    
    # Get unique texts
    corpus = filtered['question_text'].unique().tolist()
    
    print(f"\nExtracted corpus for Lab {lab_number}, Q{question_number}:")
    print(f"  Total submissions: {len(filtered)}")
    print(f"  Unique texts: {len(corpus)}")
    
    # Check for flips in original data
    flip_analysis = filtered.groupby('question_text').agg({
        'grade': lambda x: len(set(x)) > 1
    }).sum()
    
    print(f"  Texts with grade flips: {flip_analysis['grade']}")
    
    return corpus


def get_question_text(lab_number: int, question_number: int) -> str:
    """
    Get question text from rubric files
    """
    # Map from your rubric files
    questions = {
        (1, 1): "In your own words, explain what the HAL is and why it is used",
        (1, 2): "In your own words, explain what the __io_putchar function does and why it is needed",
        (1, 3): "Why did we have to use the debugger in this lab? Why didn't we just use printf?",
        (1, 4): "Explain how the stack allocation method we used works",
        (1, 5): "How to allocate multiple new stacks for multiple threads?",
        (2, 1): "Explain what happened when you changed PC to hold the function pointer",
        (2, 2): "In your own words, explain what a system call is",
        (2, 3): "Explain how the thread's stack was set up",
        (2, 4): "Why would the code crash if we didn't set PSP using __set_PSP?",
        (2, 5): "What would happen if the thread function returned instead of looping?",
        (3, 1): "Explain the process by which multiple *.c files get compiled into a single executable",
        (3, 2): "Explain why you cannot use an #include directive on a *.c file",
        (3, 3): "Explain how to set up and start a thread from the user's perspective",
        (3, 4): "Explain what the stack pool is for",
        (3, 5): "Explain the distinction between user and kernel",
        (4, 1): "Explain how round-robin scheduling can work using a queue",
        (4, 2): "Explain how the PendSV_Handler function executes a context switch",
        (4, 3): "Why is it a bad idea to always choose highest priority task?",
        (4, 4): "Comment on benefits and downsides to calling C functions vs assembly",
        (4, 5): "What would happen with only a single thread?",
    }
    
    return questions.get((lab_number, question_number), f"Lab {lab_number}, Question {question_number}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(lab_number: int, question_number: int, 
                  csv_path: str = './bad_questions.csv',
                  num_trials: int = 10):
    """
    Complete experiment on one question
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: Lab {lab_number}, Question {question_number}")
    print("="*80)
    
    # Load data
    df = load_bad_questions_data(csv_path)
    corpus = extract_corpus_from_csv(df, lab_number, question_number)
    question = get_question_text(lab_number, question_number)
    
    if len(corpus) < 5:
        print(f"⚠ Insufficient corpus size ({len(corpus)}), skipping")
        return None
    
    # Import QD extraction
    from qd_extraction import discover_and_refine_qds
    
    # Discover and refine QDs
    initial_qds, refined_qds = discover_and_refine_qds(question, corpus)
    
    # Compare rubrics
    results = compare_rubrics(
        corpus, question, initial_qds, refined_qds, num_trials
    )
    
    # Save results
    results['lab'] = lab_number
    results['question'] = question_number
    results['corpus_size'] = len(corpus)
    
    return results


if __name__ == "__main__":
    # Run on a single question first
    results = run_experiment(lab_number=1, question_number=3, num_trials=10)