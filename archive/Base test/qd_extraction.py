"""
Quality Dimension Extraction and Coupling Detection
Research goal: Reduce grade flip rate through orthogonality enhancement
"""

import requests
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.metrics import mutual_info_score
import re

from config import DISCOVERY_SAMPLE_SIZE, ANNOTATION_SAMPLE_SIZE  # pyright: ignore[reportMissingImports]

# API Configuration
BASE_URL = "http://ece-nebula16.eng.uwaterloo.ca:11434"  # Replace with actual URL

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


def parse_json_response(text: str) -> any:
    """Extract JSON from LLM response"""
    # Try to find JSON in response
    json_match = re.search(r'\[.*\]|\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # If that fails, try to clean up common issues
    text = text.replace("```json", "").replace("```", "")
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Text was: {text[:200]}")
        return None


def sample_diverse_texts(corpus: List[str], n: int = 20) -> List[str]:
    """Sample diverse texts from corpus"""
    if len(corpus) <= n:
        return corpus
    
    # Simple sampling - take evenly spaced texts
    step = len(corpus) // n
    return [corpus[i * step] for i in range(n)]


# ============================================================================
# STEP 1: CORPUS-DRIVEN QD DISCOVERY
# ============================================================================

def extract_qds_from_corpus(question: str, corpus: List[str]) -> List[Dict]:
    """
    Discover quality dimensions directly from corpus
    NO instructor rubric, NO pre-defined criteria
    """
    sample = sample_diverse_texts(corpus, n=25)
    
    prompt = f"""Analyze these student answers and identify patterns of variation.

QUESTION: {question}

STUDENT ANSWERS:
{chr(10).join(f"{i+1}. {text[:300]}..." for i, text in enumerate(sample))}

Identify 5-8 quality dimensions that capture meaningful variation in responses.

A quality dimension should be:
- BINARY (present or absent in a text)
- SEMANTIC (natural language, not numerical)
- DISTINGUISHABLE (you can reliably detect it)
- VARIABLE (present in some texts, absent in others)

For each dimension:
- Name: short identifier (e.g., "mentions_registers")
- Definition: what "present" means
- Example: text excerpt showing presence

CRITICAL: Focus on CONCEPTUAL differences, not writing style or length.

Output ONLY JSON:
[
  {{
    "name": "dimension_name",
    "definition": "what presence means",
    "example": "excerpt showing dimension"
  }}
]"""
    
    result = generate(prompt, reasoning=False)
    qds = parse_json_response(result)
    
    if qds is None:
        print("⚠ QD extraction failed, retrying...")
        return extract_qds_from_corpus(question, corpus)
    
    print(f"✓ Extracted {len(qds)} initial QDs")
    return qds


# ============================================================================
# STEP 2: ANNOTATE CORPUS WITH QDs
# ============================================================================

def annotate_text_with_qd(text: str, qd: Dict) -> bool:
    """
    Determine if a QD is present in a text
    """
    prompt = f"""Is this quality dimension present in the student answer?

QUALITY DIMENSION: {qd['name']}
Definition: {qd['definition']}

STUDENT ANSWER:
{text}

Answer ONLY: yes or no"""
    
    result = generate(prompt, reasoning=False).strip().lower()
    return 'yes' in result


def annotate_corpus(corpus: List[str], qds: List[Dict]) -> pd.DataFrame:
    """
    Create QD profile for each text in corpus
    Returns DataFrame with columns: text_idx, text, qd1, qd2, ...
    """
    print(f"\n📊 Annotating {len(corpus)} texts with {len(qds)} QDs...")
    
    annotations = []
    
    for idx, text in enumerate(corpus):
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(corpus)}")
        
        qd_profile = {
            'text_idx': idx,
            'text': text[:100] + "..."  # Store abbreviated text
        }
        
        for qd in qds:
            qd_profile[qd['name']] = 1 if annotate_text_with_qd(text, qd) else 0
        
        annotations.append(qd_profile)
    
    df = pd.DataFrame(annotations)
    print(f"✓ Annotation complete")
    
    # Print QD frequencies
    print("\nQD Frequencies:")
    for qd in qds:
        freq = df[qd['name']].mean()
        print(f"  {qd['name']}: {freq:.1%}")
    
    return df


# ============================================================================
# STEP 3: COUPLING ANALYSIS
# ============================================================================

def compute_coupling_matrix(df: pd.DataFrame, qds: List[Dict]) -> pd.DataFrame:
    """
    Compute co-occurrence statistics and mutual information
    """
    print(f"\n🔗 Computing coupling matrix...")
    
    qd_names = [qd['name'] for qd in qds]
    
    # Mutual Information matrix
    mi_matrix = pd.DataFrame(0.0, index=qd_names, columns=qd_names)
    
    # Co-occurrence matrix
    cooc_matrix = pd.DataFrame(0.0, index=qd_names, columns=qd_names)
    
    for qd1_name in qd_names:
        for qd2_name in qd_names:
            if qd1_name == qd2_name:
                mi_matrix.loc[qd1_name, qd2_name] = 1.0
                cooc_matrix.loc[qd1_name, qd2_name] = 1.0
            else:
                # Mutual information
                mi = mutual_info_score(df[qd1_name], df[qd2_name])
                mi_matrix.loc[qd1_name, qd2_name] = mi
                
                # Co-occurrence rate (both present)
                both_present = ((df[qd1_name] == 1) & (df[qd2_name] == 1)).sum()
                cooc_rate = both_present / len(df)
                cooc_matrix.loc[qd1_name, qd2_name] = cooc_rate
    
    print("✓ Coupling matrix computed")
    
    print("\nMutual Information Matrix:")
    print(mi_matrix.round(3))
    
    print("\nCo-occurrence Matrix:")
    print(cooc_matrix.round(3))
    
    return {
        'mutual_info': mi_matrix,
        'cooccurrence': cooc_matrix
    }


def find_coupling_candidates(coupling_matrices: Dict, 
                             mi_threshold: float = 0.2) -> List[Tuple]:
    """
    Identify QD pairs that may be coupled
    Don't use hard threshold - just find highest coupling pairs
    """
    mi_matrix = coupling_matrices['mutual_info']
    cooc_matrix = coupling_matrices['cooccurrence']
    
    candidates = []
    qd_names = mi_matrix.index.tolist()
    
    for i, qd1 in enumerate(qd_names):
        for j, qd2 in enumerate(qd_names):
            if i >= j:
                continue
            
            mi = mi_matrix.loc[qd1, qd2]
            cooc = cooc_matrix.loc[qd1, qd2]
            
            # Only consider pairs with some coupling
            if mi > mi_threshold:
                candidates.append((qd1, qd2, mi, cooc))
    
    # Sort by MI descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n🔍 Found {len(candidates)} coupling candidates (MI > {mi_threshold}):")
    for qd1, qd2, mi, cooc in candidates[:10]:  # Show top 10
        print(f"  {qd1} + {qd2}: MI={mi:.3f}, Co-occur={cooc:.1%}")
    
    return candidates


# ============================================================================
# STEP 4: LLM JUDGMENT ON COUPLING
# ============================================================================

def ask_llm_should_merge(qd1: Dict, qd2: Dict, 
                        mi: float, cooc: float,
                        question: str,
                        df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    CRITICAL FUNCTION: Ask LLM to judge if QDs should be merged
    Use reasoning=True here!
    """
    # Get example texts
    both_present = df[(df[qd1['name']] == 1) & (df[qd2['name']] == 1)]
    only_qd1 = df[(df[qd1['name']] == 1) & (df[qd2['name']] == 0)]
    only_qd2 = df[(df[qd1['name']] == 0) & (df[qd2['name']] == 1)]
    
    prompt = f"""Determine if these quality dimensions should be merged into a compound dimension.

QUESTION: {question}

DIMENSION 1: {qd1['name']}
Definition: {qd1['definition']}

DIMENSION 2: {qd2['name']}
Definition: {qd2['definition']}

STATISTICS FROM CORPUS:
- Mutual Information: {mi:.3f}
- Co-occurrence rate: {cooc:.1%} (both present together)
- Only D1 present: {len(only_qd1)} texts
- Only D2 present: {len(only_qd2)} texts
- Both present: {len(both_present)} texts

EXAMPLES WHERE BOTH PRESENT:
{chr(10).join(both_present['text'].head(3).tolist()) if len(both_present) > 0 else "None"}

EXAMPLES WHERE ONLY D1:
{chr(10).join(only_qd1['text'].head(2).tolist()) if len(only_qd1) > 0 else "None"}

EXAMPLES WHERE ONLY D2:
{chr(10).join(only_qd2['text'].head(2).tolist()) if len(only_qd2) > 0 else "None"}

QUESTION: Should D1 and D2 be merged?

Consider: Are they IRREDUCIBLY COUPLED?
- Irreducibly coupled means: in correct answers to this question, one dimension 
  cannot meaningfully exist without the other
- It means they represent a single unified concept
- NOT just "they often appear together" but "they MUST appear together for correctness"

Think about:
1. Can a student give a correct answer with D1 but not D2?
2. Can a student give a correct answer with D2 but not D1?
3. Do D1 and D2 represent parts of a single concept?

Output JSON:
{{
  "should_merge": true/false,
  "reasoning": "detailed explanation of coupling analysis",
  "confidence": "high/medium/low",
  "merged_qd": {{
    "name": "merged_dimension_name",
    "definition": "combined definition",
    "atomic_components": ["{qd1['name']}", "{qd2['name']}"]
  }}
}}

If should_merge is false, still output the merged_qd as null."""
    
    result = generate(prompt, reasoning=True)  # USE REASONING HERE!
    parsed = parse_json_response(result)
    
    if parsed is None:
        return False, None
    
    should_merge = parsed.get('should_merge', False)
    merged_qd = parsed.get('merged_qd', None)
    
    print(f"\n{'✓ MERGE' if should_merge else '✗ KEEP'}: {qd1['name']} + {qd2['name']}")
    print(f"  Reasoning: {parsed.get('reasoning', 'N/A')[:150]}...")
    print(f"  Confidence: {parsed.get('confidence', 'N/A')}")
    
    return should_merge, merged_qd


# ============================================================================
# STEP 5: RUBRIC REFINEMENT
# ============================================================================

def refine_rubric(qds: List[Dict], 
                 df: pd.DataFrame,
                 coupling_matrices: Dict,
                 question: str,
                 max_merges: int = 10) -> List[Dict]:
    """
    Iteratively merge coupled QDs until no more merges needed
    """
    print(f"\n🔧 Starting rubric refinement...")
    
    refined_qds = qds.copy()
    merged_pairs = set()
    
    candidates = find_coupling_candidates(coupling_matrices)
    
    merge_count = 0
    for qd1_name, qd2_name, mi, cooc in candidates:
        if merge_count >= max_merges:
            print(f"\n⚠ Reached max merges ({max_merges}), stopping")
            break
        
        # Skip if already merged
        if qd1_name in merged_pairs or qd2_name in merged_pairs:
            continue
        
        # Get full QD objects
        qd1 = next(q for q in refined_qds if q['name'] == qd1_name)
        qd2 = next(q for q in refined_qds if q['name'] == qd2_name)
        
        # Ask LLM
        should_merge, merged_qd = ask_llm_should_merge(
            qd1, qd2, mi, cooc, question, df
        )
        
        if should_merge and merged_qd:
            # Remove original QDs
            refined_qds = [q for q in refined_qds 
                          if q['name'] not in [qd1_name, qd2_name]]
            
            # Add merged QD
            refined_qds.append(merged_qd)
            
            merged_pairs.add(qd1_name)
            merged_pairs.add(qd2_name)
            merge_count += 1
            
            print(f"  → Created: {merged_qd['name']}")
    
    print(f"\n✓ Refinement complete: {len(qds)} → {len(refined_qds)} QDs")
    print(f"  Performed {merge_count} merges")
    
    return refined_qds


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def discover_and_refine_qds(question: str, corpus: List[str]) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Complete pipeline: discover → annotate → detect coupling → refine
    
    Returns: (initial_qds, refined_qds, orthogonality_report)
    """
    from orthogonality_metrics import OrthogonalityAnalyzer, quantify_orthogonality_improvement
    
    print("="*80)
    print("QUALITY DIMENSION DISCOVERY AND REFINEMENT")
    print("="*80)
    
    # Step 1: Discover QDs from corpus
    print("\n[STEP 1] Discovering QDs from corpus...")
    initial_qds = extract_qds_from_corpus(question, corpus)
    
    # Step 2: Annotate corpus
    print("\n[STEP 2] Annotating corpus...")
    df_initial = annotate_corpus(corpus, initial_qds)
    
    # Step 3: Compute coupling
    print("\n[STEP 3] Computing coupling matrix...")
    coupling_matrices = compute_coupling_matrix(df_initial, initial_qds)
    
    # Step 3.5: Mathematical orthogonality analysis
    print("\n[STEP 3.5] Mathematical orthogonality analysis...")
    qd_names_initial = [qd['name'] for qd in initial_qds]
    analyzer_initial = OrthogonalityAnalyzer(df_initial, qd_names_initial)
    initial_report = analyzer_initial.generate_report()
    
    # Step 4: Refine via merging
    print("\n[STEP 4] Refining rubric...")
    refined_qds = refine_rubric(initial_qds, df_initial, coupling_matrices, question)
    
    # Step 5: Re-analyze orthogonality after refinement
    print("\n[STEP 5] Re-analyzing orthogonality after refinement...")
    df_refined = annotate_corpus(corpus, refined_qds)
    qd_names_refined = [qd['name'] for qd in refined_qds]
    analyzer_refined = OrthogonalityAnalyzer(df_refined, qd_names_refined)
    refined_report = analyzer_refined.generate_report()
    
    # Step 6: Quantify improvement
    print("\n[STEP 6] Quantifying orthogonality improvement...")
    improvement_report = quantify_orthogonality_improvement(
        qd_names_initial, qd_names_refined,
        df_initial, df_refined
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Initial QDs: {len(initial_qds)}")
    for qd in initial_qds:
        print(f"  - {qd['name']}")
    
    print(f"\nRefined QDs: {len(refined_qds)}")
    for qd in refined_qds:
        if 'atomic_components' in qd and qd['atomic_components']:
            print(f"  - {qd['name']} [MERGED FROM: {', '.join(qd['atomic_components'])}]")
        else:
            print(f"  - {qd['name']}")
    
    print(f"\n📊 ORTHOGONALITY IMPROVEMENT:")
    print(f"  Initial score:  {improvement_report['initial_score']:.3f}")
    print(f"  Refined score:  {improvement_report['refined_score']:.3f}")
    print(f"  Improvement:    {improvement_report['improvement']:+.3f}")
    
    orthogonality_report = {
        'initial': initial_report,
        'refined': refined_report,
        'improvement': improvement_report
    }
    
    return initial_qds, refined_qds, orthogonality_report


if __name__ == "__main__":
    # Test with sample data
    sample_question = "Why did we have to use the debugger in this lab?"
    sample_corpus = [
        "The debugger allows us to view register values during execution.",
        "We needed the debugger to see memory contents that printf cannot access.",
        "Debugger lets us set breakpoints and inspect the program state.",
    ]
    
    initial, refined = discover_and_refine_qds(sample_question, sample_corpus)