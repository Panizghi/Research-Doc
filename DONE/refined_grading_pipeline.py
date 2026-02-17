"""
LLM Grading Consistency Research Pipeline
==========================================

Goal: Analyze stochasticity in LLM grading and explore whether Quality Dimensions (QDs)
can reduce inconsistency through iterative refinement.

Key Concepts:
- **Consensus**: % of iterations agreeing with the majority grade (higher = more stable)
- **Flip Rate**: 100 - consensus (lower = more stable)
- **Baseline**: Grading with original rubric only
- **QD Grading**: Grading using quality dimensions
- **QD Operations**: add, merge, drop, split - operations on quality dimensions

Run Strategy:
- Each question/response: 20 iterations with original rubric
- If answer is already stable (low flip rate), skip QD refinement
- Focus refinement on unstable responses only
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple, Tuple
# Lazy import pandas - only import when needed for analysis
# import pandas as pd
# import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "http://ece-nebula16.eng.uwaterloo.ca:11434"
NUM_ITERATIONS = 10  # Reduced from 20 - sufficient for stability metrics
STABILITY_THRESHOLD = 40.0  # Threshold for QD versions: flip_rate < this % is considered stable/improved
# Note: Baseline (rubric) is only stable if flip_rate == 0% (100% consensus)
REFINEMENT_TRIGGER_THRESHOLD = 30.0  # Minimum % of responses needing refinement to trigger QD refinement

# Directories
CHECKPOINT_DIR = Path('./checkpoints')
QD_DIR = Path('./quality_dimensions')
REFINED_QD_DIR = Path('./refined_qds')
QD_HISTORY_DIR = Path('./qd_history')

for d in [CHECKPOINT_DIR, QD_DIR, REFINED_QD_DIR, QD_HISTORY_DIR]:
    d.mkdir(exist_ok=True)


# ============================================================================
# CORE LLM INTERACTION
# ============================================================================

def call_llm(prompt: str, delay: float = 1.0) -> str:
    """Call LLM with rate limiting and ASCII normalization"""
    if delay > 0:
        time.sleep(delay)
    try:
        response = requests.post(
            f"{BASE_URL}/api/generate",
            json={"model": "gpt-oss:120b", "prompt": prompt, "stream": False},
            timeout=60
        )
        raw_response = response.json().get("response", "") if response.status_code == 200 else f"ERROR_{response.status_code}"
        
        # Normalize to ASCII: convert unicode characters to ASCII equivalents
        # This handles things like \u2013 (en-dash) -> -, \u2019 (right single quote) -> ', etc.
        try:
            # First decode if it's bytes
            if isinstance(raw_response, bytes):
                raw_response = raw_response.decode('utf-8')
            
            # Common unicode to ASCII mappings
            unicode_to_ascii = {
                '\u2013': '-',  # en-dash
                '\u2014': '--',  # em-dash
                '\u2018': "'",  # left single quote
                '\u2019': "'",  # right single quote
                '\u201c': '"',  # left double quote
                '\u201d': '"',  # right double quote
                '\u2026': '...',  # ellipsis
                '\u00a0': ' ',  # non-breaking space
                '\u200b': '',  # zero-width space
                '\u200c': '',  # zero-width non-joiner
                '\u200d': '',  # zero-width joiner
            }
            
            # Replace common unicode characters
            normalized = raw_response
            for unicode_char, ascii_char in unicode_to_ascii.items():
                normalized = normalized.replace(unicode_char, ascii_char)
            
            # Normalize remaining unicode to ASCII using NFKD decomposition
            import unicodedata
            normalized = unicodedata.normalize('NFKD', normalized)
            # Encode to ASCII, replacing any remaining non-ASCII with closest ASCII equivalent
            # Use 'ignore' to skip non-ASCII characters rather than replacing with '?'
            ascii_response = normalized.encode('ascii', 'ignore').decode('ascii')
            
            return ascii_response
        except Exception as e:
            # If normalization fails, return original but log warning
            import logging
            logging.warning(f"Failed to normalize LLM response to ASCII: {e}")
            return raw_response
    except Exception as e:
        return f"ERROR_{e}"


def parse_binary_grade(response: str) -> int:
    """Parse LLM response to binary grade (1=pass, 0=fail, -1=error)"""
    r = response.lower().strip()
    if 'pass' in r and 'fail' not in r:
        return 1
    if 'fail' in r and 'pass' not in r:
        return 0
    if '1' in r:
        return 1
    if '0' in r:
        return 0
    return -1


# ============================================================================
# GRADING FUNCTIONS
# ============================================================================

def grade_with_rubric(response_text: str, rubric: str, question_text: str) -> Dict:
    """Grade response using only the original rubric (baseline)"""
    prompt = f"""Grade this response as PASS (1) or FAIL (0).

QUESTION:
{question_text}

RUBRIC:
{rubric}

STUDENT RESPONSE:
{response_text}

Output: "Grade: [0 or 1]"
"""
    
    llm_response = call_llm(prompt)
    return {
        'grade': parse_binary_grade(llm_response),
        'raw_response': llm_response,
        'timestamp': datetime.now().isoformat(),
        'method': 'baseline'
    }


def grade_with_quality_dimensions(
    response_text: str, 
    rubric: str, 
    quality_dimensions: List[str],
    question_text: str,
    qd_version: Optional[str] = None
) -> Dict:
    """Grade response using quality dimensions
    
    Args:
        response_text: The student response to grade
        rubric: The grading rubric
        quality_dimensions: List of quality dimension descriptions
        question_text: The question the student is answering
        qd_version: Version identifier for the QDs being used (e.g., 'v1', 'v2')
    """
    qd_text = "\n".join([f"{i+1}. {qd}" for i, qd in enumerate(quality_dimensions)])
    
    prompt = f"""Evaluate this response using the quality dimensions provided.

QUESTION:
{question_text}

RUBRIC:
{rubric}

QUALITY DIMENSIONS:
{qd_text}

STUDENT RESPONSE:
{response_text}

Output format:
Dimension 1: [0 or 1]
Dimension 2: [0 or 1]
...
Overall: [0 or 1]
"""
    
    llm_response = call_llm(prompt)
    
    # Parse dimension scores
    dimension_scores = []
    for i in range(len(quality_dimensions)):
        # Try to find "Dimension N: X" pattern
        import re
        pattern = re.compile(rf"^\s*dimension\s+{i+1}\s*:\s*(0|1)\b", flags=re.IGNORECASE | re.MULTILINE)
        match = pattern.search(llm_response)
        if match:
            dimension_scores.append(int(match.group(1)))
        else:
            dimension_scores.append(-1)  # Parsing failed
    
    # Parse overall grade
    overall_match = re.search(r"overall\s*:\s*([01])", llm_response, flags=re.IGNORECASE)
    overall_grade = int(overall_match.group(1)) if overall_match else -1
    
    return {
        'grade': overall_grade,
        'dimension_scores': dimension_scores,
        'quality_dimensions': list(quality_dimensions),
        'qd_version': qd_version,  # Track which version was used
        'raw_response': llm_response,
        'timestamp': datetime.now().isoformat(),
        'method': 'qd'
    }


# ============================================================================
# QUALITY DIMENSION MANAGEMENT
# ============================================================================

def generate_initial_qds(question_text: str, rubric: str) -> Dict:
    """Generate initial quality dimensions for a question"""
    prompt = f"""Generate 3-5 quality dimensions for grading student responses.

QUESTION:
{question_text}

RUBRIC:
{rubric}

Requirements:
- Each dimension should be specific and measurable
- Must be binary (pass/fail)
- Objective criteria only
- Applicable to any response addressing the rubric

Format:
1. [dimension description]
2. [dimension description]
...
"""
    
    llm_response = call_llm(prompt)
    
    # Parse dimensions
    dimensions = []
    for line in llm_response.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            dim = line.split('.', 1)[-1].strip().lstrip('- ')
            if dim:
                dimensions.append(dim)
    
    return {
        'dimensions': dimensions,
        'raw_response': llm_response,
        'timestamp': datetime.now().isoformat(),
        'version': 'v1',
        'operation': 'initial_generation',
        'parent_version': None
    }


def refine_qds(
    qid: str,
    question_text: str,
    rubric: str,
    current_qds: List[str],
    problematic_responses: List[Dict],
    previous_version: str,
    previous_refinement_results: List[Dict] = None
) -> Dict:
    """Refine quality dimensions based on problematic responses
    
    Returns new QDs with operation metadata (add/merge/drop/split)
    """
    if previous_refinement_results is None:
        previous_refinement_results = []
    # Check if we have majority_changed cases
    has_majority_changed = any(r.get('refinement_reason') == 'majority_changed' for r in problematic_responses)
    has_worse_flip = any(r.get('refinement_reason') == 'worse_flip_rate' for r in problematic_responses)
    
    # Build detailed examples with specific metrics
    examples = ""
    for i, resp in enumerate(problematic_responses[:3], 1):
        examples += f"\n--- Example {i} ---\n"
        examples += f"Response: {resp['response_text'][:200]}...\n\n"
        
        baseline_maj = 'Pass' if resp['baseline_majority'] == 1 else 'Fail'
        qd_maj = 'Pass' if resp['qd_majority'] == 1 else 'Fail'
        baseline_flip = resp['baseline_flip_rate']
        qd_flip = resp['qd_flip_rate']
        
        reason = resp.get('refinement_reason', 'unknown')
        examples += f"BASELINE: Grade={baseline_maj}, Flip Rate={baseline_flip:.1f}%\n"
        examples += f"CURRENT QD: Grade={qd_maj}, Flip Rate={qd_flip:.1f}%\n"
        
        if reason == 'majority_changed':
            examples += f"PROBLEM: QD gives WRONG GRADE ({qd_maj} instead of {baseline_maj}) "
            examples += f"even though it's more consistent ({qd_flip:.1f}% vs {baseline_flip:.1f}%)\n"
            examples += f"REQUIRED: Grade must be {baseline_maj} AND flip rate should stay ≤ {baseline_flip + 5:.1f}%\n"
        elif reason == 'worse_flip_rate':
            examples += f"PROBLEM: QD has WORSE CONSISTENCY ({qd_flip:.1f}% vs {baseline_flip:.1f}% flip rate)\n"
            examples += f"REQUIRED: Flip rate must be ≤ {baseline_flip:.1f}% AND grade must stay {baseline_maj}\n"
        else:
            examples += f"PROBLEM: Both grade and consistency issues\n"
    
    # Calculate aggregate metrics for context
    avg_baseline_flip = sum(r['baseline_flip_rate'] for r in problematic_responses) / len(problematic_responses)
    avg_qd_flip = sum(r['qd_flip_rate'] for r in problematic_responses) / len(problematic_responses)
    
    # Build context about previous failed refinements
    previous_attempts_context = ""
    if previous_refinement_results:
        previous_attempts_context = "\n<previous_refinement_attempts>\n"
        previous_attempts_context += "WARNING: Previous refinement attempts had problems. Learn from these failures:\n\n"
        for prev in previous_refinement_results:
            prev_maj = 'Pass' if prev['refined_majority'] == 1 else 'Fail'
            baseline_maj = 'Pass' if prev['baseline_majority'] == 1 else 'Fail'
            if prev['result'] == 'failed':
                previous_attempts_context += f"Version {prev['version']}: FAILED - "
                if prev['refined_majority'] != prev['baseline_majority']:
                    previous_attempts_context += f"Still wrong grade ({prev_maj} instead of {baseline_maj}). "
                if prev['refined_flip'] > prev['baseline_flip'] + 5:
                    previous_attempts_context += f"Worse consistency ({prev['refined_flip']:.1f}% vs {prev['baseline_flip']:.1f}%). "
                previous_attempts_context += "\n"
            elif prev['result'] == 'partial_success':
                previous_attempts_context += f"Version {prev['version']}: PARTIAL SUCCESS - "
                if prev['refined_majority'] == prev['baseline_majority']:
                    previous_attempts_context += f"Fixed grade but worsened consistency ({prev['refined_flip']:.1f}% vs {prev['baseline_flip']:.1f}%). "
                else:
                    previous_attempts_context += f"Improved consistency but still wrong grade. "
                previous_attempts_context += "This is NOT acceptable - we need BOTH correct grade AND good consistency.\n"
        previous_attempts_context += "\nLESSON: You must achieve BOTH goals simultaneously. Do not fix one problem by creating another.\n"
        previous_attempts_context += "</previous_refinement_attempts>\n"
    
    # Build prompt based on the issues
    if has_majority_changed and has_worse_flip:
        issue_description = f"""The current quality dimensions have TWO critical problems:

1. WORSE CONSISTENCY: Average flip rate increased from {avg_baseline_flip:.1f}% (baseline) to {avg_qd_flip:.1f}% (current QD)
2. WRONG GRADE: Some responses get the wrong majority grade compared to baseline

REQUIREMENTS FOR REFINED QDs:
- Must produce the SAME majority grade as baseline
- Must have flip rate ≤ {avg_baseline_flip + 5:.1f}% (baseline + 5% tolerance)
- Should ideally improve consistency (lower flip rate) while maintaining correct grading"""
    elif has_majority_changed:
        issue_description = f"""The current quality dimensions have a SYSTEMATIC BIAS problem:

- CONSISTENCY: Good (flip rate {avg_qd_flip:.1f}% vs baseline {avg_baseline_flip:.1f}%)
- GRADE: WRONG (QDs systematically give different grades than baseline)

This means the QDs are more consistent but systematically biased. They're consistently wrong.

REQUIREMENTS FOR REFINED QDs:
- MUST produce the SAME majority grade as baseline (this is critical)
- MUST maintain flip rate ≤ {avg_baseline_flip + 5:.1f}% (baseline + 5% tolerance)
- The goal is correct grading WITH good consistency, not just consistency alone"""
    else:
        issue_description = f"""The current quality dimensions have a CONSISTENCY problem:

- CONSISTENCY: WORSE (flip rate {avg_qd_flip:.1f}% vs baseline {avg_baseline_flip:.1f}%)
- GRADE: Same as baseline (correct)

REQUIREMENTS FOR REFINED QDs:
- MUST maintain the SAME majority grade as baseline
- MUST improve consistency: flip rate should be ≤ {avg_baseline_flip:.1f}% (baseline level)"""
    
    prompt = f"""You need to fix the current quality dimensions. They have specific problems that must be addressed.

<question>
{question_text}
</question>

<rubric>
{rubric}
</rubric>

<current_quality_dimensions>
{chr(10).join([f"{i+1}. {qd}" for i, qd in enumerate(current_qds)])}
</current_quality_dimensions>
{previous_attempts_context}
<problem_analysis>
{issue_description}
</problem_analysis>

<examples_with_metrics>
{examples}
</examples_with_metrics>

<critical_requirements>
The refined quality dimensions MUST:
1. Produce the CORRECT majority grade (same as baseline) - this is non-negotiable
2. Maintain or improve consistency (flip rate ≤ baseline + 5% tolerance)
3. Be objective, specific, and measurable
4. Apply equally to all responses (no bias)

CRITICAL: You must achieve BOTH requirements simultaneously:
- If you fix the grade but make consistency worse (flip rate > baseline + 5%), that's a FAILURE
- If you improve consistency but get the wrong grade, that's a FAILURE
- You need BOTH correct grading AND good consistency at the same time

Think carefully about how to balance these requirements. The dimensions should be specific enough to align with baseline grading decisions while remaining objective and consistent.
</critical_requirements>

<output_format>
Generate 3-5 refined quality dimensions. Output in numbered list format:
1. [refined dimension description]
2. [refined dimension description]
...
</output_format>
"""
    
    llm_response = call_llm(prompt)
    
    # Parse new dimensions
    new_dimensions = []
    for line in llm_response.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            dim = line.split('.', 1)[-1].strip().lstrip('- ')
            if dim:
                new_dimensions.append(dim)
    
    # Determine operation type with detailed tracking
    operation, operation_details = determine_qd_operation(current_qds, new_dimensions)
    
    # Generate new version number - ensure we don't overwrite existing versions
    # Get all existing versions for this question
    existing_versions = list(QD_HISTORY_DIR.glob(f"{qid}_v*.json"))
    existing_version_nums = set()
    for v_file in existing_versions:
        try:
            v_num = int(v_file.stem.split('_v')[1])
            existing_version_nums.add(v_num)
        except (ValueError, IndexError):
            continue
    
    # Start from previous_version + 1, but ensure we don't overwrite
    prev_version_num = int(previous_version.replace('v', ''))
    new_version_num = prev_version_num + 1
    
    # If the next version already exists (shouldn't happen, but be safe), find next available
    while new_version_num in existing_version_nums:
        new_version_num += 1
    
    new_version = f'v{new_version_num}'
    
    return {
        'dimensions': new_dimensions,
        'raw_response': llm_response,
        'timestamp': datetime.now().isoformat(),
        'version': new_version,
        'operation': operation,
        'operation_details': operation_details,  # Detailed tracking
        'parent_version': previous_version,
        'previous_dimensions': current_qds,
        'problematic_count': len(problematic_responses)
    }


def determine_qd_operation(old_qds: List[str], new_qds: List[str]) -> Tuple[str, Dict]:
    """Determine what operation was performed on the QDs and track specific changes
    
    Returns:
        operation_type: 'split', 'merge', 'add', 'drop', 'refine'
        operation_details: Dictionary with specific changes
    """
    old_count = len(old_qds)
    new_count = len(new_qds)
    
    # Use fuzzy string matching to identify similar dimensions
    from difflib import SequenceMatcher
    
    def similarity(a: str, b: str) -> float:
        """Calculate similarity between two strings (0-1)"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    # Build mapping of old -> new dimensions based on similarity
    dimension_mapping = []
    used_new = set()
    
    for old_idx, old_dim in enumerate(old_qds):
        best_match_idx = None
        best_similarity = 0.0
        
        for new_idx, new_dim in enumerate(new_qds):
            if new_idx in used_new:
                continue
            sim = similarity(old_dim, new_dim)
            if sim > best_similarity and sim > 0.3:  # Threshold for "similar"
                best_similarity = sim
                best_match_idx = new_idx
        
        if best_match_idx is not None:
            dimension_mapping.append({
                'old_index': old_idx,
                'old_text': old_dim,
                'new_index': best_match_idx,
                'new_text': new_qds[best_match_idx],
                'similarity': best_similarity,
                'operation': 'refine' if best_similarity < 0.9 else 'keep'
            })
            used_new.add(best_match_idx)
    
    # Identify dropped dimensions (old dimensions with no match)
    dropped = [
        {'index': i, 'text': dim}
        for i, dim in enumerate(old_qds)
        if not any(m['old_index'] == i for m in dimension_mapping)
    ]
    
    # Identify added dimensions (new dimensions with no match)
    added = [
        {'index': i, 'text': dim}
        for i, dim in enumerate(new_qds)
        if i not in used_new
    ]
    
    # Identify splits (one old -> multiple new)
    splits = []
    for old_idx, old_dim in enumerate(old_qds):
        # Check if this old dimension maps to multiple new dimensions
        matches = [m for m in dimension_mapping if m['old_index'] == old_idx]
        if len(matches) > 1:
            splits.append({
                'old_index': old_idx,
                'old_text': old_dim,
                'new_indices': [m['new_index'] for m in matches],
                'new_texts': [m['new_text'] for m in matches]
            })
    
    # Identify merges (multiple old -> one new)
    merges = []
    for new_idx, new_dim in enumerate(new_qds):
        matches = [m for m in dimension_mapping if m['new_index'] == new_idx]
        if len(matches) > 1:
            merges.append({
                'old_indices': [m['old_index'] for m in matches],
                'old_texts': [m['old_text'] for m in matches],
                'new_index': new_idx,
                'new_text': new_dim
            })
    
    # Determine primary operation
    if len(splits) > 0:
        operation = 'split'
    elif len(merges) > 0:
        operation = 'merge'
    elif len(added) > 0 and len(dropped) == 0:
        operation = 'add'
    elif len(dropped) > 0 and len(added) == 0:
        operation = 'drop'
    elif len(added) > 0 and len(dropped) > 0:
        operation = 'add_drop'  # Both adding and dropping
    else:
        operation = 'refine'  # Just modifications to existing
    
    operation_details = {
        'dimension_mapping': dimension_mapping,
        'added': added,
        'dropped': dropped,
        'splits': splits,
        'merges': merges,
        'old_count': old_count,
        'new_count': new_count
    }
    
    return operation, operation_details


def save_qd_version(qid: str, qd_data: Dict) -> None:
    """Save QD version to history"""
    version = qd_data['version']
    
    # Save to versioned file
    version_path = QD_HISTORY_DIR / f"{qid}_{version}.json"
    version_path.write_text(json.dumps(qd_data, indent=2))
    
    # Update latest pointer for initial version
    if version == 'v1':
        latest_path = QD_DIR / f"{qid}_qds.json"
        latest_path.write_text(json.dumps(qd_data, indent=2))


def load_qd_version(qid: str, version: Optional[str] = None) -> Optional[Dict]:
    """Load specific QD version or latest"""
    if version:
        path = QD_HISTORY_DIR / f"{qid}_{version}.json"
    else:
        path = QD_DIR / f"{qid}_qds.json"
    
    if path.exists():
        return json.loads(path.read_text())
    return None


def get_latest_qd_version(qid: str) -> Optional[str]:
    """Get the latest version number for a question's QDs"""
    versions = list(QD_HISTORY_DIR.glob(f"{qid}_v*.json"))
    if not versions:
        return None
    
    version_numbers = [int(v.stem.split('_v')[1]) for v in versions]
    return f'v{max(version_numbers)}'


def _evaluate_version_performance(checkpoint: Dict, version: str, version_stats: Dict, baseline_majority: int) -> None:
    """Helper function to evaluate a QD version's performance from a checkpoint"""
    if version not in version_stats:
        version_stats[version] = {
            'success_count': 0,
            'total_count': 0,
            'flip_rates': [],
            'correct_majority_count': 0,
            'improvement_over_baseline': []  # Track flip rate improvements
        }
    
    stats = version_stats[version]
    stats['total_count'] += 1
    
    # Get metrics for this version
    if version == checkpoint.get('qd_initial_version'):
        metrics = checkpoint.get('qd_initial_metrics', {})
    else:
        metrics = checkpoint.get('qd_refined_metrics', {})
    
    if metrics:
        flip_rate = metrics.get('flip_rate', 100)
        stats['flip_rates'].append(flip_rate)
        
        # Calculate improvement over baseline
        baseline_metrics = checkpoint.get('baseline_metrics', {})
        baseline_flip = baseline_metrics.get('flip_rate', 100)
        improvement = baseline_flip - flip_rate  # Positive = improvement
        stats['improvement_over_baseline'].append(improvement)
        
        # Check majority match
        qd_majority = metrics.get('majority', -1)
        if qd_majority == baseline_majority and baseline_majority != -1:
            stats['correct_majority_count'] += 1
    
    # Check if this version led to success
    qd_result = checkpoint.get('qd_result', '')
    refinement_result = checkpoint.get('refinement_result', '')
    if qd_result == 'improved_or_maintained' or refinement_result == 'success':
        stats['success_count'] += 1


def get_best_qd_version(qid: str, exclude_rid: Optional[str] = None) -> Optional[str]:
    """Get the best QD version based on success across previous responses for this question
    
    Prioritizes versions with:
    1. Most "success" or "improved_or_maintained" results
    2. Best average flip_rate (lowest)
    3. Most correct majority matches with baseline
    4. Best improvement over baseline
    
    Args:
        qid: Question ID
        exclude_rid: Optional response ID to exclude from analysis (current response being graded)
    """
    # Get all checkpoints for this question
    checkpoint_files = list(CHECKPOINT_DIR.glob(f"{qid}_*.json"))
    if not checkpoint_files:
        # No previous responses, use latest version or v1
        latest = get_latest_qd_version(qid)
        return latest if latest else 'v1'
    
    # Track metrics per version
    version_stats = {}  # version -> {success_count, total_count, avg_flip_rate, correct_majority_count, improvement_over_baseline}
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = json.loads(checkpoint_file.read_text())
            
            # Skip current response if specified
            if exclude_rid and checkpoint.get('response_id') == exclude_rid:
                continue
            
            # Get QD versions used
            qd_initial_version = checkpoint.get('qd_initial_version')
            qd_refined_version = checkpoint.get('qd_refined_version')
            
            baseline_metrics = checkpoint.get('baseline_metrics', {})
            baseline_majority = baseline_metrics.get('majority', -1)
            
            # Evaluate initial QD version
            if qd_initial_version:
                _evaluate_version_performance(checkpoint, qd_initial_version, version_stats, baseline_majority)
            
            # Evaluate refined QD version
            if qd_refined_version:
                _evaluate_version_performance(checkpoint, qd_refined_version, version_stats, baseline_majority)
                    
        except Exception as e:
            # Silently skip invalid checkpoints
            continue
    
    if not version_stats:
        # No stats, use latest version or v1
        latest = get_latest_qd_version(qid)
        return latest if latest else 'v1'
    
    # Calculate aggregate metrics
    for version, stats in version_stats.items():
        if stats['flip_rates']:
            stats['avg_flip_rate'] = sum(stats['flip_rates']) / len(stats['flip_rates'])
        else:
            stats['avg_flip_rate'] = 100.0
        
        if stats['improvement_over_baseline']:
            stats['avg_improvement'] = sum(stats['improvement_over_baseline']) / len(stats['improvement_over_baseline'])
        else:
            stats['avg_improvement'] = 0.0
        
        stats['success_rate'] = stats['success_count'] / stats['total_count'] if stats['total_count'] > 0 else 0.0
    
    # Find best version: prioritize success rate, then improvement, then low flip rate, then correct majority
    best_version = None
    best_score = -1
    
    for version, stats in version_stats.items():
        correct_majority_rate = stats['correct_majority_count'] / stats['total_count'] if stats['total_count'] > 0 else 0.0
        
        # Enhanced scoring: success_rate * 100 + avg_improvement * 2 + (100 - avg_flip_rate) * 0.5 + correct_majority_rate * 50
        # This heavily weights success rate and improvement over baseline
        score = (stats['success_rate'] * 100 + 
                stats['avg_improvement'] * 2 +  # Improvement is important
                (100 - stats['avg_flip_rate']) * 0.5 + 
                correct_majority_rate * 50)
        
        if score > best_score:
            best_score = score
            best_version = version
    
    if best_version:
        stats = version_stats[best_version]
        print(f"  Best QD version for {qid}: {best_version} (success_rate={stats['success_rate']:.1%}, "
              f"avg_flip={stats['avg_flip_rate']:.1f}%, avg_improvement={stats['avg_improvement']:.1f}%)")
        return best_version
    
    # Fallback to latest version
    latest = get_latest_qd_version(qid)
    return latest if latest else 'v1'


# ============================================================================
# HELPER FUNCTIONS FOR REFINEMENT EVALUATION
# ============================================================================

def format_majority_string(majority: int, is_tie: bool = False, pass_count: int = 0, fail_count: int = 0) -> str:
    """Format majority grade as a string"""
    if is_tie:
        return f"TIE (pass={pass_count}, fail={fail_count})"
    return 'Pass' if majority == 1 else 'Fail'


def check_exact_baseline_match(
    baseline_metrics: Dict,
    refined_metrics: Dict,
    is_tie: bool = False
) -> bool:
    """Check if refined metrics exactly match baseline (within tolerance)"""
    baseline_flip = baseline_metrics.get('flip_rate', 0)
    baseline_majority = baseline_metrics.get('majority', -1)
    baseline_consensus = baseline_metrics.get('consensus', 0)
    
    refined_flip = refined_metrics.get('flip_rate', 0)
    refined_majority = refined_metrics.get('majority', -1)
    refined_consensus = refined_metrics.get('consensus', 0)
    
    if is_tie:
        # For ties, only check flip_rate and consensus (majority is ambiguous)
        return (
            abs(refined_flip - baseline_flip) < 0.1 and
            abs(refined_consensus - baseline_consensus) < 0.1
        )
    else:
        # For non-ties, check all metrics including majority
        return (
            abs(refined_flip - baseline_flip) < 0.1 and
            refined_majority == baseline_majority and
            abs(refined_consensus - baseline_consensus) < 0.1
        )


def evaluate_refinement_result(
    refinement_reason: str,
    baseline_metrics: Dict,
    refined_metrics: Dict,
    is_tie: bool = False
) -> Tuple[str, str]:
    """Evaluate refinement result and return (result, message)
    
    Returns:
        Tuple of (result_type, message) where result_type is 'success', 'partial_success', 'failed', or 'needs_evaluation'
    """
    baseline_flip = baseline_metrics.get('flip_rate', 100)
    baseline_majority = baseline_metrics.get('majority', -1)
    refined_flip = refined_metrics.get('flip_rate', 100)
    refined_majority = refined_metrics.get('majority', -1)
    refined_pass_count = refined_metrics.get('pass_count', 0)
    refined_fail_count = refined_metrics.get('fail_count', 0)
    
    baseline_maj_str = format_majority_string(baseline_majority)
    refined_maj_str = format_majority_string(refined_majority, is_tie, refined_pass_count, refined_fail_count)
    
    if refinement_reason == 'majority_changed':
        if is_tie:
            if refined_flip <= baseline_flip + 5.0:
                return ('success', f"Tie case (pass={refined_pass_count}, fail={refined_fail_count}) - good consistency (flip_rate={refined_flip:.1f}% <= {baseline_flip + 5.0:.1f}%)")
            else:
                return ('partial_success', f"Tie case but flip_rate {refined_flip:.1f}% > {baseline_flip + 5.0:.1f}% (baseline + 5%)")
        elif refined_majority == baseline_majority and refined_flip <= baseline_flip + 5.0:
            return ('success', "Fixed both issues (correct majority + good consistency)")
        elif refined_majority == baseline_majority and refined_flip > baseline_flip + 5.0:
            return ('partial_success', f"Fixed majority but flip_rate {refined_flip:.1f}% > {baseline_flip + 5.0:.1f}% (baseline + 5%)")
        else:
            return ('failed', f"Wrong majority ({refined_maj_str} vs {baseline_maj_str})")
    
    elif refinement_reason == 'worse_flip_rate':
        if is_tie:
            if refined_flip <= baseline_flip:
                return ('success', f"Tie case (pass={refined_pass_count}, fail={refined_fail_count}) - good consistency (flip_rate={refined_flip:.1f}% <= {baseline_flip:.1f}%)")
            else:
                return ('failed', f"Tie case but flip_rate {refined_flip:.1f}% > {baseline_flip:.1f}% (baseline)")
        elif refined_flip <= baseline_flip and refined_majority == baseline_majority:
            return ('success', "Fixed both issues (good consistency + correct majority)")
        elif refined_flip <= baseline_flip:
            return ('partial_success', f"Improved consistency but wrong majority ({refined_maj_str} vs {baseline_maj_str})")
        else:
            return ('failed', f"Flip rate {refined_flip:.1f}% > {baseline_flip:.1f}% (baseline)")
    
    else:
        # Unknown reason, compare to baseline
        if is_tie:
            if refined_flip <= baseline_flip:
                return ('success', f"Tie case (pass={refined_pass_count}, fail={refined_fail_count}) - good consistency")
            else:
                return ('needs_evaluation', f"Tie case but flip_rate {refined_flip:.1f}% > {baseline_flip:.1f}%")
        elif refined_flip <= baseline_flip and refined_majority == baseline_majority:
            return ('success', "Both metrics improved")
        else:
            return ('needs_evaluation', "Unknown refinement reason")


# ============================================================================
# STABILITY ANALYSIS
# ============================================================================

def calculate_stability_metrics(grades: List[int]) -> Dict:
    """Calculate consensus, flip rate, and majority for a list of grades"""
    valid_grades = [g for g in grades if g != -1]
    if not valid_grades:
        return {'consensus': 0, 'flip_rate': 100, 'majority': -1, 'sample_size': 0}
    
    pass_count = sum(valid_grades)
    fail_count = len(valid_grades) - pass_count
    majority_count = max(pass_count, fail_count)
    majority_grade = 1 if pass_count > fail_count else 0
    
    consensus = (majority_count / len(valid_grades)) * 100
    flip_rate = 100 - consensus
    
    return {
        'consensus': consensus,
        'flip_rate': flip_rate,
        'majority': majority_grade,
        'sample_size': len(valid_grades),
        'pass_count': pass_count,
        'fail_count': fail_count
    }


def is_baseline_stable(flip_rate: float) -> bool:
    """Check if baseline (rubric) is stable - must be 100% consensus (0% flip rate)"""
    return flip_rate == 0.0


def is_stable(flip_rate: float, threshold: float = STABILITY_THRESHOLD) -> bool:
    """Check if QD grading is stable enough (uses threshold for QD versions)"""
    return flip_rate < threshold


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(data: Dict, qid: str, rid: str) -> None:
    """Save grading checkpoint
    
    Checkpoint structure for graphing and analysis:
    - baseline_iterations: List of baseline (rubric) grading results
    - baseline_metrics: Dict with flip_rate, consensus, majority, pass_count, fail_count
    - qd_initial_iterations: List of initial QD grading results
    - qd_initial_metrics: Dict with metrics for initial QD version
    - qd_initial_version: Version string (e.g., 'v1', 'v6')
    - qd_refined_iterations: List of refined QD grading results (latest version)
    - qd_refined_metrics: Dict with metrics for latest refined QD version
    - qd_refined_version: Version string of latest refined QD
    - qd_refined_iterations_v2, qd_refined_iterations_v3, etc.: Per-version iterations
    - qd_refined_metrics_v2, qd_refined_metrics_v3, etc.: Per-version metrics
    - version_progression: List of all versions with metrics and flip_rate_change_vs_baseline
      Each entry: {version, version_label, metrics, flip_rate_change_vs_baseline, timestamp}
    - flip_rate_change_vs_baseline: Change in flip rate vs baseline (positive = improvement)
    - refinement_result: 'success', 'partial_success', 'failed', 'exact_baseline_match'
    - refinement_reason: 'majority_changed', 'worse_flip_rate'
    """
    path = CHECKPOINT_DIR / f"{qid}_{rid}.json"
    path.write_text(json.dumps(data, indent=2))


def load_checkpoint(qid: str, rid: str) -> Optional[Dict]:
    """Load grading checkpoint"""
    path = CHECKPOINT_DIR / f"{qid}_{rid}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def checkpoint_is_complete(checkpoint: Dict, phase: str, num_iterations: int) -> bool:
    """Check if a grading phase is complete"""
    if not checkpoint:
        return False
    iterations = checkpoint.get(f'{phase}_iterations', [])
    return len(iterations) >= num_iterations


# ============================================================================
# MAIN GRADING PIPELINE
# ============================================================================

def grade_response(
    qid: str,
    rid: str,
    response_text: str,
    rubric: str,
    question_text: str,
    num_iterations: int = NUM_ITERATIONS
) -> Dict:
    """
    Grade a single response through the full pipeline:
    1. Baseline grading (always run)
    2. Initial QD grading (only if baseline is unstable)
    3. Refined QD grading (only if initial QDs made it worse)
    """
    
    # Load or initialize checkpoint
    checkpoint = load_checkpoint(qid, rid)
    if checkpoint is None:
        checkpoint = {
            'question_id': qid,
            'response_id': rid,
            'response_text': response_text,
            'baseline_iterations': [],
            'qd_initial_iterations': [],
            'qd_refined_iterations': []
        }
    
    # === Phase 1: Baseline Grading (RUBRIC) ===
    # IMPORTANT: Baseline = Rubric grading. This is the reference point.
    # All QD versions will be compared to this baseline to measure improvement.
    print(f"  {rid}: Baseline (RUBRIC) grading...", end=' ')
    if not checkpoint_is_complete(checkpoint, 'baseline', num_iterations):
        baseline_count = len(checkpoint.get('baseline_iterations', []))
        for i in range(num_iterations - baseline_count):
            result = grade_with_rubric(response_text, rubric, question_text)
            checkpoint['baseline_iterations'].append(result)
            if (i + 1) % 5 == 0:
                save_checkpoint(checkpoint, qid, rid)
        save_checkpoint(checkpoint, qid, rid)
    
    # Calculate baseline stability
    baseline_grades = [it['grade'] for it in checkpoint['baseline_iterations']]
    baseline_metrics = calculate_stability_metrics(baseline_grades)
    checkpoint['baseline_metrics'] = baseline_metrics
    print(f"flip_rate={baseline_metrics['flip_rate']:.1f}%")
    
    # Check for ties in baseline (pass_count == fail_count) - needs human review
    baseline_pass_count = baseline_metrics.get('pass_count', 0)
    baseline_fail_count = baseline_metrics.get('fail_count', 0)
    if baseline_pass_count == baseline_fail_count and baseline_pass_count > 0:
        checkpoint['needs_human_review'] = True
        checkpoint['human_review_reason'] = f'Baseline tie case: pass_count={baseline_pass_count}, fail_count={baseline_fail_count} (majority is ambiguous)'
        print(f"    ⚠ BASELINE TIE DETECTED (pass={baseline_pass_count}, fail={baseline_fail_count}) - needs human review")
    
    # Initialize version progression tracking for graphing
    if 'version_progression' not in checkpoint:
        checkpoint['version_progression'] = []
    
    # Add baseline (rubric) to progression
    baseline_entry = {
        'version': 'baseline',
        'version_label': 'Baseline (Rubric)',
        'metrics': baseline_metrics.copy(),
        'flip_rate_change_vs_baseline': 0.0,  # Baseline is reference point
        'timestamp': datetime.now().isoformat()
    }
    checkpoint['version_progression'] = [baseline_entry]  # Reset with baseline as first entry
    
    # === Phase 2: Initial QD Grading ===
    # IMPORTANT: Baseline always runs first (above). QD versions are refinements on top of baseline.
    # Determine which QD version would be used (even if we skip grading due to stability)
    # Use the best QD version based on previous responses' success (exclude current response)
    best_version = get_best_qd_version(qid, exclude_rid=rid)
    
    # Record the QD version that would be used (even if we skip)
    if best_version:
        checkpoint['qd_initial_version'] = best_version  # Record which version would be used
        checkpoint['qd_version_selected'] = best_version  # Also record for clarity
    
    # Check if we should skip QD grading (baseline must be 100% stable = 0% flip rate)
    if is_baseline_stable(baseline_metrics['flip_rate']):
        print(f"    → Skipping QD grading (baseline is 100% stable, flip_rate={baseline_metrics['flip_rate']:.1f}%)")
        if best_version:
            print(f"    → Would have used QD version: {best_version}")
        checkpoint['skip_reason'] = 'baseline_already_stable'
        save_checkpoint(checkpoint, qid, rid)
        return checkpoint

    if not best_version:
        print(f"    → No QDs available, skipping")
        checkpoint['skip_reason'] = 'no_qds_available'
        save_checkpoint(checkpoint, qid, rid)
        return checkpoint
    qd_data = load_qd_version(qid, best_version)
    if not qd_data:
        # Fallback to initial version if best version file doesn't exist
        qd_data = load_qd_version(qid)
        if not qd_data:
            print(f"    → No QDs available, skipping")
            checkpoint['skip_reason'] = 'no_qds_available'
            save_checkpoint(checkpoint, qid, rid)
            return checkpoint
        best_version = qd_data.get('version', 'v1')
    
    qd_version = best_version
    print(f"    QD grading (using {qd_version} - best version from previous responses)...", end=' ')
    if not checkpoint_is_complete(checkpoint, 'qd_initial', num_iterations):
        qds = qd_data['dimensions']
        for i in range(num_iterations - len(checkpoint['qd_initial_iterations'])):
            result = grade_with_quality_dimensions(response_text, rubric, qds, question_text, qd_version)
            checkpoint['qd_initial_iterations'].append(result)
            if (i + 1) % 5 == 0:
                save_checkpoint(checkpoint, qid, rid)
        save_checkpoint(checkpoint, qid, rid)
    
    # Confirm/store which QD version was used for initial grading (already set earlier, but confirm here)
    checkpoint['qd_initial_version'] = qd_version
    checkpoint['qd_version_selected'] = qd_version  # Also store for clarity
    
    # Calculate QD stability
    qd_grades = [it['grade'] for it in checkpoint['qd_initial_iterations']]
    qd_metrics = calculate_stability_metrics(qd_grades)
    checkpoint['qd_initial_metrics'] = qd_metrics
    
    # Check for ties in initial QD (pass_count == fail_count) - needs human review
    qd_pass_count = qd_metrics.get('pass_count', 0)
    qd_fail_count = qd_metrics.get('fail_count', 0)
    if qd_pass_count == qd_fail_count and qd_pass_count > 0:
        checkpoint['needs_human_review'] = True
        if checkpoint.get('human_review_reason'):
            checkpoint['human_review_reason'] += f'; Initial QD tie: pass_count={qd_pass_count}, fail_count={qd_fail_count}'
        else:
            checkpoint['human_review_reason'] = f'Initial QD tie case: pass_count={qd_pass_count}, fail_count={qd_fail_count} (majority is ambiguous)'
        print(f"    ⚠ INITIAL QD TIE DETECTED (pass={qd_pass_count}, fail={qd_fail_count}) - needs human review")
    
    # Compare QD to baseline (rubric) - this is the key comparison
    baseline_flip = baseline_metrics.get('flip_rate', 100)
    qd_flip = qd_metrics.get('flip_rate', 100)
    flip_rate_change = baseline_flip - qd_flip  # Positive = improvement
    
    # Add initial QD version to progression tracking
    if 'version_progression' not in checkpoint:
        checkpoint['version_progression'] = []
    
    qd_initial_entry = {
        'version': qd_version,
        'version_label': f'QD {qd_version} (Initial)',
        'metrics': qd_metrics.copy(),
        'flip_rate_change_vs_baseline': flip_rate_change,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check if this version already exists in progression
    existing_idx = None
    for idx, entry in enumerate(checkpoint['version_progression']):
        if entry.get('version') == qd_version and entry.get('version_label', '').endswith('(Initial)'):
            existing_idx = idx
            break
    
    if existing_idx is not None:
        checkpoint['version_progression'][existing_idx] = qd_initial_entry
    else:
        checkpoint['version_progression'].append(qd_initial_entry)
    
    baseline_majority = baseline_metrics.get('majority', -1)
    qd_majority = qd_metrics.get('majority', -1)
    majority_changed = (baseline_majority != -1 and qd_majority != -1 and 
                       baseline_majority != qd_majority)
    
    # Show comparison to baseline (rubric)
    print(f"flip_rate={qd_flip:.1f}% (baseline/rubric: {baseline_flip:.1f}%)")
    if flip_rate_change > 0:
        print(f"    → QD {qd_version} improved: flip_rate reduced by {flip_rate_change:.1f}% vs baseline (rubric)")
    elif flip_rate_change < 0:
        print(f"    → QD {qd_version} worsened: flip_rate increased by {abs(flip_rate_change):.1f}% vs baseline (rubric)")
    else:
        print(f"    → QD {qd_version} maintained: same flip_rate as baseline (rubric)")
    
    # Check if QDs made it worse OR changed the majority
    if qd_metrics['flip_rate'] > baseline_metrics['flip_rate']:
        print(f"    → QDs made it worse (flip_rate increased), will refine later")
        checkpoint['qd_result'] = 'needs_refinement'
        checkpoint['refinement_reason'] = 'worse_flip_rate'
        checkpoint['flip_rate_change_vs_baseline'] = flip_rate_change  # Store for analysis
        save_checkpoint(checkpoint, qid, rid)
        return checkpoint
    elif majority_changed:
        print(f"    → QDs improved consistency but changed majority ({baseline_majority} → {qd_majority}), will refine later")
        checkpoint['qd_result'] = 'needs_refinement'
        checkpoint['refinement_reason'] = 'majority_changed'
        checkpoint['baseline_majority'] = baseline_majority
        checkpoint['qd_majority'] = qd_majority
        checkpoint['flip_rate_change_vs_baseline'] = flip_rate_change  # Store for analysis
        save_checkpoint(checkpoint, qid, rid)
        return checkpoint
    else:
        print(f"    → QDs improved or maintained stability (same majority, flip_rate change: {flip_rate_change:+.1f}%)")
        checkpoint['qd_result'] = 'improved_or_maintained'
        checkpoint['flip_rate_change_vs_baseline'] = flip_rate_change  # Store for analysis
        save_checkpoint(checkpoint, qid, rid)
        return checkpoint


def grade_all_questions(questions: Dict, num_iterations: int = NUM_ITERATIONS) -> None:
    """Grade all questions and responses"""
    total_responses = sum(len(q['responses']) for q in questions.values())
    completed = 0
    
    for qid, q_data in questions.items():
        print(f"\n{qid}")
        rubric = format_rubric(q_data.get('rubric'))
        question_text = q_data.get('question_text', '')
        
        for response in q_data['responses']:
            rid = response['response_id']
            text = response['text']
            
            grade_response(qid, rid, text, rubric, question_text, num_iterations)
            completed += 1
            print(f"  Progress: {completed}/{total_responses}")


def refine_problematic_qds(questions: Dict) -> None:
    """
    Identify questions where QDs made grading worse and refine them.
    Only refine if there are problematic responses.
    """
    
    for qid, q_data in questions.items():
        print(f"\n{qid}: Checking if refinement needed...")
        
        # Collect all checkpoints for this question
        problematic_responses = []
        majority_changed_count = 0
        worse_flip_rate_count = 0
        
        for response in q_data['responses']:
            rid = response['response_id']
            checkpoint = load_checkpoint(qid, rid)
            
            if not checkpoint:
                continue
            
            # Count previous refinement attempts
            refinement_result = checkpoint.get('refinement_result')
            refinement_attempt_count = checkpoint.get('refinement_attempt_count', 0)
            
            # If refinement_result exists but count is 0, this is attempt 1
            if refinement_result and refinement_attempt_count == 0:
                refinement_attempt_count = 1
            
            # Check for partial_success cases that should get another attempt
            # These might be marked as 'maintained' but should get one more try
            has_partial_success_for_retry = (
                refinement_result == 'partial_success' and 
                refinement_attempt_count == 1
            )
            
            # Skip if already abandoned (but allow maintained if it's a partial_success case)
            if checkpoint.get('qd_result') == 'abandoned':
                continue
            if checkpoint.get('qd_result') == 'maintained' and not has_partial_success_for_retry:
                continue
            
            # Check if metrics exactly match baseline (even if marked as success/improved_or_maintained)
            # This allows one more refinement attempt to try to improve beyond baseline
            baseline_metrics = checkpoint.get('baseline_metrics', {})
            qd_refined_metrics = checkpoint.get('qd_refined_metrics', {})
            qd_result_current = checkpoint.get('qd_result', '')
            refinement_result_current = checkpoint.get('refinement_result', '')
            
            # Check for exact baseline match if we have refined metrics and it's marked as success/improved
            if qd_refined_metrics and (qd_result_current in ['improved_or_maintained'] or refinement_result_current == 'success'):
                baseline_flip = baseline_metrics.get('flip_rate', 0)
                baseline_majority = baseline_metrics.get('majority', -1)
                baseline_consensus = baseline_metrics.get('consensus', 0)
                refined_flip = qd_refined_metrics.get('flip_rate', 0)
                refined_majority = qd_refined_metrics.get('majority', -1)
                refined_consensus = qd_refined_metrics.get('consensus', 0)
                refined_pass_count = qd_refined_metrics.get('pass_count', 0)
                refined_fail_count = qd_refined_metrics.get('fail_count', 0)
                is_tie_collection = refined_pass_count == refined_fail_count
                
                # Check if metrics exactly match baseline
                metrics_exact_match = check_exact_baseline_match(baseline_metrics, qd_refined_metrics, is_tie_collection)
                
                if metrics_exact_match and refinement_result != 'exact_baseline_match':
                    # Metrics exactly match baseline - allow one more attempt to try to improve beyond baseline
                    if is_tie_collection:
                        print(f"    {rid}: Exact baseline match detected (TIE: pass={refined_pass_count}, fail={refined_fail_count}, flip_rate={refined_flip:.1f}%) - allowing one more refinement attempt")
                    else:
                        print(f"    {rid}: Exact baseline match detected (flip_rate={refined_flip:.1f}%, majority={'Pass' if refined_majority == 1 else 'Fail'}) - allowing one more refinement attempt")
                    checkpoint['refinement_result'] = 'exact_baseline_match'
                    checkpoint['qd_result'] = 'needs_refinement'
                    # Don't increment attempt count yet - this is the detection, not the attempt
                    refinement_result = 'exact_baseline_match'  # Update local variable
                    save_checkpoint(checkpoint, qid, rid)
            
            # Stop after 2 refinement attempts total (v2 and v3)
            # v1 = initial QDs, v2 = first refinement, v3 = second refinement
            # Exception: allow exact_baseline_match and partial_success (attempt 1) to proceed
            if refinement_attempt_count >= 2 and refinement_result != 'exact_baseline_match' and not has_partial_success_for_retry:
                # Already tried twice (v2 and v3), mark as maintained and skip
                checkpoint['qd_result'] = 'maintained'
                checkpoint['abandoned_reason'] = f"After 2 refinement attempts (v2, v3), still {refinement_result or 'failed'} - marking as maintained"
                save_checkpoint(checkpoint, qid, rid)
                print(f"    {rid}: Already had 2 refinement attempts, marking as maintained")
                continue
            
            # Process responses that need refinement OR have partial_success (for final attempt) OR exact_baseline_match
            needs_refinement = checkpoint.get('qd_result') == 'needs_refinement'
            has_partial_success = refinement_result == 'partial_success' and refinement_attempt_count == 1
            has_exact_baseline = refinement_result == 'exact_baseline_match'
            
            if not (needs_refinement or has_partial_success or has_exact_baseline):
                continue
            
            # If it's a partial_success or exact_baseline_match case, reset qd_result to needs_refinement for processing
            if has_partial_success or has_exact_baseline:
                checkpoint['qd_result'] = 'needs_refinement'
                save_checkpoint(checkpoint, qid, rid)
                if has_partial_success:
                    print(f"    {rid}: Resetting from 'maintained' to 'needs_refinement' for partial_success retry (attempt {refinement_attempt_count})")
            
            baseline_metrics = checkpoint.get('baseline_metrics', {})
            qd_metrics = checkpoint.get('qd_initial_metrics', {})
            refinement_reason = checkpoint.get('refinement_reason', 'unknown')
            
            if refinement_reason == 'majority_changed':
                majority_changed_count += 1
            elif refinement_reason == 'worse_flip_rate':
                worse_flip_rate_count += 1
            
            problematic_responses.append({
                'response_id': rid,
                'response_text': response['text'],
                'baseline_flip_rate': baseline_metrics.get('flip_rate', 0),
                'qd_flip_rate': qd_metrics.get('flip_rate', 0),
                'flip_rate_increase': qd_metrics.get('flip_rate', 0) - baseline_metrics.get('flip_rate', 0),
                'baseline_majority': baseline_metrics.get('majority', -1),
                'qd_majority': qd_metrics.get('majority', -1),
                'refinement_reason': refinement_reason
            })
        
        if not problematic_responses:
            print(f"  No problematic responses found")
            continue
        
        # Calculate percentage of responses that need refinement
        pct_problematic = (len(problematic_responses) / len(q_data['responses'])) * 100
        print(f"  {len(problematic_responses)}/{len(q_data['responses'])} responses need refinement ({pct_problematic:.1f}%)")
        if majority_changed_count > 0:
            print(f"    - {majority_changed_count} with majority changed (better consistency but wrong grade)")
        if worse_flip_rate_count > 0:
            print(f"    - {worse_flip_rate_count} with worse flip rate")
        
        # Only refine if enough responses need refinement (use constant threshold)
        if pct_problematic < REFINEMENT_TRIGGER_THRESHOLD:
            print(f"  Below {REFINEMENT_TRIGGER_THRESHOLD}% threshold, skipping refinement")
            continue
        
        # Get current QD version
        current_version = get_latest_qd_version(qid) or 'v1'
        qd_data = load_qd_version(qid, current_version)
        
        if not qd_data:
            print(f"  No QD data found, skipping")
            continue
        
        # Preserve question data (q_data will be used for question info)
        question_text = q_data.get('question_text', '')
        rubric = format_rubric(q_data.get('rubric'))
        
        if not question_text:
            print(f"  No question_text found in question data, skipping")
            continue
        
        # Check for previous failed refinements to provide context
        previous_refinement_results = []
        for response in q_data['responses']:
            rid = response['response_id']
            checkpoint = load_checkpoint(qid, rid)
            if checkpoint and checkpoint.get('refinement_result'):
                prev_result = checkpoint.get('refinement_result')
                prev_version = checkpoint.get('qd_refined_version')
                prev_metrics = checkpoint.get('qd_refined_metrics', {})
                baseline_metrics = checkpoint.get('baseline_metrics', {})
                
                if prev_result in ['failed', 'partial_success'] and prev_version:
                    previous_refinement_results.append({
                        'version': prev_version,
                        'result': prev_result,
                        'baseline_flip': baseline_metrics.get('flip_rate', 0),
                        'refined_flip': prev_metrics.get('flip_rate', 0),
                        'baseline_majority': baseline_metrics.get('majority', -1),
                        'refined_majority': prev_metrics.get('majority', -1)
                    })
        
        # Refine QDs
        print(f"  Refining QDs (current version: {current_version})...")
        if previous_refinement_results:
            failed_versions = [r['version'] for r in previous_refinement_results if r['result'] == 'failed']
            partial_versions = [r['version'] for r in previous_refinement_results if r['result'] == 'partial_success']
            if failed_versions:
                print(f"  Warning: Previous refinements failed: {', '.join(failed_versions)}")
            if partial_versions:
                print(f"  Warning: Previous refinements were partial: {', '.join(partial_versions)}")
        
        # Sort: prioritize majority_changed cases, then by flip rate increase
        # This ensures we fix systematic bias (majority_changed) before consistency issues
        problematic_responses.sort(key=lambda x: (
            0 if x.get('refinement_reason') == 'majority_changed' else 1,  # majority_changed first
            -x['flip_rate_increase']  # Then by flip rate increase (descending)
        ))
        
        refined_qds = refine_qds(
            qid=qid,
            question_text=question_text,
            rubric=rubric,
            current_qds=qd_data['dimensions'],
            problematic_responses=problematic_responses[:5],  # Top 5 worst
            previous_version=current_version,
            previous_refinement_results=previous_refinement_results
        )
        
        print(f"  New version: {refined_qds['version']}, operation: {refined_qds['operation']}")
        print(f"  Old QDs ({len(refined_qds['previous_dimensions'])}): {refined_qds['previous_dimensions'][:2]}...")
        print(f"  New QDs ({len(refined_qds['dimensions'])}): {refined_qds['dimensions'][:2]}...")
        
        # Save new version
        save_qd_version(qid, refined_qds)
        
        # Re-grade with refined QDs
        regrade_with_refined_qds(qid, q_data, refined_qds)


def regrade_with_refined_qds(qid: str, q_data: Dict, refined_qds: Dict) -> None:
    """Re-grade responses that need refinement with the new QDs"""
    print(f"  Re-grading with refined QDs...")
    rubric = format_rubric(q_data.get('rubric'))
    question_text = q_data.get('question_text', '')
    qds = refined_qds['dimensions']
    qd_version = refined_qds['version']
    
    regrade_count = 0
    for response in q_data['responses']:
        rid = response['response_id']
        checkpoint = load_checkpoint(qid, rid)
        
        if not checkpoint:
            continue
        
        # Process responses that need refinement OR have partial_success (for final attempt) OR exact_baseline_match
        needs_refinement = checkpoint.get('qd_result') == 'needs_refinement'
        has_partial_success = checkpoint.get('refinement_result') == 'partial_success'
        has_exact_baseline = checkpoint.get('refinement_result') == 'exact_baseline_match'
        
        if not (needs_refinement or has_partial_success or has_exact_baseline):
            continue
        
        if has_partial_success:
            print(f"    {rid}: Final attempt (v3) after partial_success in v2...")
        elif has_exact_baseline:
            print(f"    {rid}: One more attempt after exact baseline match...")
        
        # Save iterations per version (v2, v3, v4, etc.) to track changes
        version_key = f'qd_refined_iterations_{qd_version}'
        if version_key not in checkpoint:
            checkpoint[version_key] = []
        
        # Also keep current qd_refined_iterations for backward compatibility
        existing_refined_version = checkpoint.get('qd_refined_version')
        if existing_refined_version and existing_refined_version != qd_version:
            print(f"    {rid}: New QD version ({qd_version} vs {existing_refined_version}), saving previous version iterations")
            # Save previous version iterations before clearing
            prev_version_key = f'qd_refined_iterations_{existing_refined_version}'
            if prev_version_key not in checkpoint:
                checkpoint[prev_version_key] = checkpoint.get('qd_refined_iterations', [])
            checkpoint['qd_refined_iterations'] = []
        
        # Grade with refined QDs
        if 'qd_refined_iterations' not in checkpoint:
            checkpoint['qd_refined_iterations'] = []
        
        text = response['text']
        
        # Get num_iterations from baseline (should match baseline sample size)
        baseline_iterations = checkpoint.get('baseline_iterations', [])
        num_iterations = len(baseline_iterations) if baseline_iterations else NUM_ITERATIONS
        
        # Check if we already have iterations for this version
        existing_version_iterations = checkpoint.get(version_key, [])
        iterations_needed = num_iterations - len(existing_version_iterations)
        
        if iterations_needed > 0:
            print(f"    {rid}: Running {iterations_needed} iterations with {qd_version}...")
            for i in range(iterations_needed):
                result = grade_with_quality_dimensions(text, rubric, qds, question_text, qd_version)
                checkpoint['qd_refined_iterations'].append(result)
                checkpoint[version_key].append(result)
                if (i + 1) % 5 == 0:
                    save_checkpoint(checkpoint, qid, rid)
        else:
            # Use existing iterations for this version
            checkpoint['qd_refined_iterations'] = existing_version_iterations.copy()
            print(f"    {rid}: Using existing {len(existing_version_iterations)} iterations for {qd_version}")
        
        # Calculate refined metrics for this version
        refined_grades = [it['grade'] for it in checkpoint['qd_refined_iterations']]
        refined_metrics = calculate_stability_metrics(refined_grades)
        checkpoint['qd_refined_metrics'] = refined_metrics
        checkpoint[f'qd_refined_metrics_{qd_version}'] = refined_metrics  # Save per version
        checkpoint['qd_refined_version'] = qd_version
        
        # Check for ties (pass_count == fail_count) - needs human review
        refined_pass_count = refined_metrics.get('pass_count', 0)
        refined_fail_count = refined_metrics.get('fail_count', 0)
        if refined_pass_count == refined_fail_count and refined_pass_count > 0:
            checkpoint['needs_human_review'] = True
            if checkpoint.get('human_review_reason'):
                checkpoint['human_review_reason'] += f'; Refined QD {qd_version} tie: pass_count={refined_pass_count}, fail_count={refined_fail_count}'
            else:
                checkpoint['human_review_reason'] = f'Refined QD {qd_version} tie case: pass_count={refined_pass_count}, fail_count={refined_fail_count} (majority is ambiguous)'
            print(f"    {rid}: ⚠ REFINED QD {qd_version} TIE DETECTED (pass={refined_pass_count}, fail={refined_fail_count}) - needs human review")
        
        # Evaluate if refinement actually improved things
        baseline_metrics = checkpoint.get('baseline_metrics', {})
        qd_initial_metrics = checkpoint.get('qd_initial_metrics', {})
        refinement_reason = checkpoint.get('refinement_reason', '')
        
        baseline_flip = baseline_metrics.get('flip_rate', 100)
        qd_initial_flip = qd_initial_metrics.get('flip_rate', 100)
        refined_flip = refined_metrics.get('flip_rate', 100)
        
        # Calculate flip rate change vs baseline for this refined version
        refined_flip_change = baseline_flip - refined_flip  # Positive = improvement
        
        # Track version progression for graphing
        if 'version_progression' not in checkpoint:
            checkpoint['version_progression'] = []
        
        # Add this refined version to progression (avoid duplicates)
        version_entry = {
            'version': qd_version,
            'version_label': f'QD {qd_version} (Refined)',
            'metrics': refined_metrics.copy(),
            'flip_rate_change_vs_baseline': refined_flip_change,
            'refinement_result': checkpoint.get('refinement_result', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if this version already exists in progression
        existing_idx = None
        for idx, entry in enumerate(checkpoint['version_progression']):
            if entry.get('version') == qd_version and entry.get('version_label', '').endswith('(Refined)'):
                existing_idx = idx
                break
        
        if existing_idx is not None:
            checkpoint['version_progression'][existing_idx] = version_entry
        else:
            checkpoint['version_progression'].append(version_entry)
        
        baseline_majority = baseline_metrics.get('majority', -1)
        qd_initial_majority = qd_initial_metrics.get('majority', -1)
        refined_majority = refined_metrics.get('majority', -1)
        
        # Check for tie case (pass_count == fail_count) - majority is ambiguous, so don't penalize
        refined_pass_count = refined_metrics.get('pass_count', 0)
        refined_fail_count = refined_metrics.get('fail_count', 0)
        is_tie = refined_pass_count == refined_fail_count
        
        # Format strings for display
        baseline_maj_str = format_majority_string(baseline_majority)
        refined_maj_str = format_majority_string(refined_majority, is_tie, refined_pass_count, refined_fail_count)
        
        # refined_flip_change already calculated above for version_progression
        
        print(f"    {rid}: Evaluating {qd_version} refinement vs baseline (RUBRIC)...")
        print(f"      Baseline (RUBRIC): majority={baseline_maj_str}, flip_rate={baseline_flip:.1f}%")
        if is_tie:
            print(f"      Refined QD ({qd_version}): majority=TIE (pass={refined_pass_count}, fail={refined_fail_count}), flip_rate={refined_flip:.1f}% (change: {refined_flip_change:+.1f}% vs baseline)")
        else:
            print(f"      Refined QD ({qd_version}): majority={refined_maj_str}, flip_rate={refined_flip:.1f}% (change: {refined_flip_change:+.1f}% vs baseline)")
        
        # Use helper function to evaluate refinement result
        result_type, message = evaluate_refinement_result(
            refinement_reason, baseline_metrics, refined_metrics, is_tie
        )
        
        checkpoint['refinement_result'] = result_type
        checkpoint['flip_rate_change_vs_baseline'] = refined_flip_change  # Store for analysis
        if result_type == 'success':
            checkpoint['qd_result'] = 'improved_or_maintained'
            print(f"      ✓ SUCCESS: {message}")
        elif result_type == 'partial_success':
            print(f"      ⚠ PARTIAL: {message}")
        elif result_type == 'failed':
            print(f"      ✗ FAILED: {message}")
        else:
            print(f"      ? NEEDS EVALUATION: {message}")
        
        # Track refinement attempts
        previous_attempt_count = checkpoint.get('refinement_attempt_count', 0)
        current_result = checkpoint.get('refinement_result')
        has_exact_baseline = checkpoint.get('refinement_result') == 'exact_baseline_match'
        
        # Check if metrics exactly match baseline (within small tolerance)
        metrics_exact_match = check_exact_baseline_match(baseline_metrics, refined_metrics, is_tie)
        
        # If metrics exactly match baseline, allow one more refinement attempt (even if success)
        # This gives us a chance to improve beyond baseline
        if metrics_exact_match:
            # Exact baseline match - allow one more refinement attempt regardless of current result
            if is_tie:
                print(f"      → Exact baseline match detected (TIE: pass={refined_pass_count}, fail={refined_fail_count}, flip_rate={refined_flip:.1f}%) - allowing one more refinement attempt to try to improve beyond baseline")
            else:
                print(f"      → Exact baseline match detected (flip_rate={refined_flip:.1f}%, majority={refined_maj_str}) - allowing one more refinement attempt to try to improve beyond baseline")
            checkpoint['refinement_result'] = 'exact_baseline_match'
            checkpoint['qd_result'] = 'needs_refinement'  # Allow one more try
            checkpoint['refinement_attempt_count'] = previous_attempt_count + 1
        elif has_exact_baseline:
            # This was the extra attempt after exact baseline match
            checkpoint['refinement_attempt_count'] = previous_attempt_count + 1
            # Check if this attempt improved beyond baseline or still matches exactly
            still_exact_match = check_exact_baseline_match(baseline_metrics, refined_metrics, is_tie)
            
            if current_result == 'success' and not still_exact_match:
                # Improved beyond baseline! Keep as success
                checkpoint['qd_result'] = 'improved_or_maintained'
                print(f"      → SUCCESS: Improved beyond baseline (flip_rate={refined_flip:.1f}% vs baseline {baseline_flip:.1f}%)")
            elif still_exact_match:
                # Still exactly matches baseline after extra attempt - mark as maintained
                checkpoint['qd_result'] = 'maintained'
                checkpoint['abandoned_reason'] = f"After exact baseline match + extra attempt, still exactly matches baseline - marking as maintained"
                print(f"      → MAINTAINED: Still exactly matches baseline after extra attempt")
            elif current_result != 'success':
                # Not success and not exact match - mark as maintained
                checkpoint['qd_result'] = 'maintained'
                checkpoint['abandoned_reason'] = f"After exact baseline match + extra attempt, still {current_result} - marking as maintained"
                print(f"      → MAINTAINED: After exact baseline match attempt, marking as maintained (not success)")
        elif has_partial_success:
            # This is attempt 2 (after partial_success from attempt 1)
            checkpoint['refinement_attempt_count'] = 2
            if current_result != 'success':
                # Still failed after 2 attempts, mark as maintained and stop
                checkpoint['qd_result'] = 'maintained'
                checkpoint['abandoned_reason'] = f"After 2 refinement attempts, still {current_result} - marking as maintained"
                print(f"      → MAINTAINED: After 2 attempts, marking as maintained (not success)")
        elif current_result in ['partial_success', 'failed']:
            # First attempt that failed - if not success, mark as maintained
            checkpoint['refinement_attempt_count'] = 1
            if current_result != 'success':
                checkpoint['qd_result'] = 'maintained'
                print(f"      → MAINTAINED: Not success after attempt, marking as maintained")
        elif current_result == 'success':
            # Success! Reset attempt count
            checkpoint['refinement_attempt_count'] = 0
        
        save_checkpoint(checkpoint, qid, rid)
        regrade_count += 1
    
    print(f"  Re-graded {regrade_count} responses")


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results():
    """Analyze all results and create summary DataFrame"""
    import pandas as pd  # Lazy import to avoid slow startup
    records = []
    
    for checkpoint_file in CHECKPOINT_DIR.glob('*.json'):
        checkpoint = json.loads(checkpoint_file.read_text())
        
        qid = checkpoint['question_id']
        rid = checkpoint['response_id']
        
        # Extract metrics
        baseline_metrics = checkpoint.get('baseline_metrics', {})
        qd_initial_metrics = checkpoint.get('qd_initial_metrics', {})
        qd_refined_metrics = checkpoint.get('qd_refined_metrics', {})
        
        record = {
            'question_id': qid,
            'response_id': rid,
            'response_text': checkpoint['response_text'][:100],
            
            # Baseline
            'baseline_flip_rate': baseline_metrics.get('flip_rate'),
            'baseline_consensus': baseline_metrics.get('consensus'),
            'baseline_majority': baseline_metrics.get('majority'),
            
            # Initial QD
            'qd_initial_flip_rate': qd_initial_metrics.get('flip_rate'),
            'qd_initial_consensus': qd_initial_metrics.get('consensus'),
            'qd_initial_majority': qd_initial_metrics.get('majority'),
            
            # Refined QD
            'qd_refined_flip_rate': qd_refined_metrics.get('flip_rate'),
            'qd_refined_consensus': qd_refined_metrics.get('consensus'),
            'qd_refined_majority': qd_refined_metrics.get('majority'),
            'qd_refined_version': checkpoint.get('qd_refined_version'),
            
            # Comparisons
            'qd_result': checkpoint.get('qd_result'),
            'skip_reason': checkpoint.get('skip_reason')
        }
        
        # Calculate improvements
        if record['baseline_flip_rate'] is not None and record['qd_initial_flip_rate'] is not None:
            record['qd_initial_improvement'] = record['baseline_flip_rate'] - record['qd_initial_flip_rate']
        
        if record['baseline_flip_rate'] is not None and record['qd_refined_flip_rate'] is not None:
            record['qd_refined_improvement'] = record['baseline_flip_rate'] - record['qd_refined_flip_rate']
        
        records.append(record)
    
    return pd.DataFrame(records)


def print_summary(df) -> None:
    """Print analysis summary"""
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    total = len(df)
    print(f"\nTotal responses analyzed: {total}")
    
    # Baseline stats
    baseline_mean_flip = df['baseline_flip_rate'].mean()
    print(f"\nBaseline mean flip rate: {baseline_mean_flip:.2f}%")
    
    # How many were stable enough to skip QD?
    skipped_stable = (df['skip_reason'] == 'baseline_already_stable').sum()
    print(f"Responses skipped (already stable): {skipped_stable} ({skipped_stable/total*100:.1f}%)")
    
    # Initial QD stats
    qd_graded = df['qd_initial_flip_rate'].notna().sum()
    if qd_graded > 0:
        qd_initial_mean_flip = df['qd_initial_flip_rate'].mean()
        print(f"\nInitial QD mean flip rate: {qd_initial_mean_flip:.2f}% (n={qd_graded})")
        
        improved = (df['qd_initial_improvement'] > 0).sum()
        maintained = (df['qd_initial_improvement'] == 0).sum()
        worse = (df['qd_initial_improvement'] < 0).sum()
        
        print(f"  Improved: {improved} ({improved/qd_graded*100:.1f}%)")
        print(f"  Maintained: {maintained} ({maintained/qd_graded*100:.1f}%)")
        print(f"  Worse: {worse} ({worse/qd_graded*100:.1f}%)")
    
    # Refined QD stats
    qd_refined_graded = df['qd_refined_flip_rate'].notna().sum()
    if qd_refined_graded > 0:
        qd_refined_mean_flip = df['qd_refined_flip_rate'].mean()
        print(f"\nRefined QD mean flip rate: {qd_refined_mean_flip:.2f}% (n={qd_refined_graded})")
        
        refined_improved = (df['qd_refined_improvement'] > 0).sum()
        print(f"  Improved over baseline: {refined_improved} ({refined_improved/qd_refined_graded*100:.1f}%)")
    
    # Per-question breakdown
    print("\n" + "-"*80)
    print("PER-QUESTION BREAKDOWN")
    print("-"*80)
    
    question_summary = df.groupby('question_id').agg({
        'baseline_flip_rate': 'mean',
        'qd_initial_flip_rate': 'mean',
        'qd_refined_flip_rate': 'mean',
        'response_id': 'count'
    }).round(2)
    question_summary.columns = ['baseline_flip', 'qd_initial_flip', 'qd_refined_flip', 'num_responses']
    
    print(question_summary)
    
    print("\n" + "="*80)


def format_rubric(rubric_data: Optional[Dict]) -> str:
    """Format rubric for LLM prompts"""
    if not rubric_data:
        return "Pass (1): Demonstrates technical correctness, completeness, and clarity\nFail (0): Otherwise"
    
    return f"""Pass (1): {rubric_data['meets_expectation']}
Fail (0): {rubric_data['does_not_meet']}"""


# ============================================================================
# CHECKPOINT EVALUATION (without rerunning)
# ============================================================================

def evaluate_checkpoint_status(checkpoint_path: str) -> None:
    """
    Evaluate and display the final refinement status for a checkpoint without rerunning.
    
    Args:
        checkpoint_path: Path to the checkpoint JSON file
    """
    import json
    from pathlib import Path
    
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    checkpoint = json.loads(checkpoint_file.read_text())
    
    qid = checkpoint.get('question_id', 'Unknown')
    rid = checkpoint.get('response_id', 'Unknown')
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT EVALUATION: {qid} / {rid}")
    print(f"{'='*80}\n")
    
    # Get metrics
    baseline_metrics = checkpoint.get('baseline_metrics', {})
    qd_initial_metrics = checkpoint.get('qd_initial_metrics', {})
    qd_refined_metrics = checkpoint.get('qd_refined_metrics', {})
    
    baseline_flip = baseline_metrics.get('flip_rate', 0)
    baseline_majority = baseline_metrics.get('majority', -1)
    baseline_maj_str = 'Pass' if baseline_majority == 1 else 'Fail'
    
    qd_initial_flip = qd_initial_metrics.get('flip_rate', 0)
    qd_initial_majority = qd_initial_metrics.get('majority', -1)
    qd_initial_maj_str = 'Pass' if qd_initial_majority == 1 else 'Fail'
    qd_initial_version = checkpoint.get('qd_initial_version', 'v1')
    
    qd_refined_flip = qd_refined_metrics.get('flip_rate', 0)
    qd_refined_majority = qd_refined_metrics.get('majority', -1)
    qd_refined_maj_str = 'Pass' if qd_refined_majority == 1 else 'Fail'
    qd_refined_version = checkpoint.get('qd_refined_version', 'Unknown')
    
    refinement_reason = checkpoint.get('refinement_reason', '')
    refinement_result = checkpoint.get('refinement_result', '')
    qd_result = checkpoint.get('qd_result', '')
    refinement_attempt_count = checkpoint.get('refinement_attempt_count', 0)
    
    # Display progression
    print("METRICS PROGRESSION:")
    print(f"  Baseline:           majority={baseline_maj_str:4s}, flip_rate={baseline_flip:5.1f}%")
    print(f"  {qd_initial_version:15s}: majority={qd_initial_maj_str:4s}, flip_rate={qd_initial_flip:5.1f}%")
    if qd_refined_version != 'Unknown':
        print(f"  {qd_refined_version:15s}: majority={qd_refined_maj_str:4s}, flip_rate={qd_refined_flip:5.1f}%")
    print()
    
    # Evaluate final status
    print("FINAL STATUS EVALUATION:")
    print(f"  Refinement Reason: {refinement_reason}")
    print(f"  Refinement Result: {refinement_result}")
    print(f"  QD Result:         {qd_result}")
    print(f"  Attempt Count:     {refinement_attempt_count}")
    print()
    
    # Re-evaluate based on current metrics
    if qd_refined_version != 'Unknown' and refinement_reason:
        print("RE-EVALUATION:")
        if refinement_reason == 'majority_changed':
            if qd_refined_majority == baseline_majority and qd_refined_flip <= baseline_flip + 5.0:
                final_status = '✓ SUCCESS'
                final_note = f"Fixed both issues: correct majority ({qd_refined_maj_str}) + good consistency ({qd_refined_flip:.1f}% <= {baseline_flip + 5.0:.1f}%)"
            elif qd_refined_majority == baseline_majority and qd_refined_flip > baseline_flip + 5.0:
                final_status = '⚠ PARTIAL SUCCESS'
                final_note = f"Fixed majority but flip_rate {qd_refined_flip:.1f}% > {baseline_flip + 5.0:.1f}% (baseline + 5%)"
            else:
                final_status = '✗ FAILED'
                final_note = f"Wrong majority ({qd_refined_maj_str} vs {baseline_maj_str})"
        elif refinement_reason == 'worse_flip_rate':
            if qd_refined_flip <= baseline_flip and qd_refined_majority == baseline_majority:
                final_status = '✓ SUCCESS'
                final_note = f"Fixed both issues: good consistency ({qd_refined_flip:.1f}% <= {baseline_flip:.1f}%) + correct majority"
            elif qd_refined_flip <= baseline_flip:
                final_status = '⚠ PARTIAL SUCCESS'
                final_note = f"Improved consistency but wrong majority ({qd_refined_maj_str} vs {baseline_maj_str})"
            else:
                final_status = '✗ FAILED'
                final_note = f"Flip rate {qd_refined_flip:.1f}% > {baseline_flip:.1f}% (baseline)"
        else:
            if qd_refined_flip <= baseline_flip and qd_refined_majority == baseline_majority:
                final_status = '✓ SUCCESS'
                final_note = "Both metrics improved"
            else:
                final_status = '? NEEDS EVALUATION'
                final_note = "Unknown refinement reason"
        
        print(f"  Status: {final_status}")
        print(f"  Note:   {final_note}")
        print()
        
        # Check if should be abandoned
        if refinement_attempt_count >= 2 and final_status != '✓ SUCCESS':
            print(f"  → Should be ABANDONED (after {refinement_attempt_count} attempts)")
            if qd_result != 'abandoned':
                print(f"  → Current qd_result is '{qd_result}', should be 'abandoned'")
        elif final_status == '✓ SUCCESS':
            print(f"  → Should be marked as 'improved_or_maintained'")
            if qd_result != 'improved_or_maintained':
                print(f"  → Current qd_result is '{qd_result}', should be 'improved_or_maintained'")
    
    print(f"{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # This would be called from a notebook or script that loads your questions
    print("Pipeline module loaded. Import and use functions in your notebook.")
