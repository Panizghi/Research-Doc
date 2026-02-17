"""
QD Operation Analysis Module
=============================

This module provides utilities for analyzing quality dimension operations,
tracking specific dimension changes, and linking grades to QD versions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
# Lazy import pandas - only import when needed
# import pandas as pd

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(log_dir: Path = None, log_level: int = logging.INFO) -> logging.Logger:
    """Set up comprehensive logging with file and console handlers"""
    logger = logging.getLogger('qd_operation_analysis')
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above, if log_dir provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"qd_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger

# Initialize logger
logger = setup_logger(Path('./logs'))


def load_qd_lineage(qid: str, qd_history_dir: Path = Path('./qd_history')) -> List[Dict]:
    """Load complete QD version history for a question
    
    Returns:
        List of QD versions in chronological order (v1, v2, v3, ...)
    """
    logger.info(f"Loading QD lineage for {qid} from {qd_history_dir}")
    versions = sorted(qd_history_dir.glob(f"{qid}_v*.json"))
    logger.debug(f"Found {len(versions)} QD versions for {qid}")
    
    lineage = []
    for version_file in versions:
        try:
            qd_data = json.loads(version_file.read_text())
            lineage.append(qd_data)
            logger.debug(f"Loaded {version_file.name}: version {qd_data.get('version')}, {len(qd_data.get('dimensions', []))} dimensions")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {version_file}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error loading {version_file}: {e}", exc_info=True)
    
    logger.info(f"Loaded {len(lineage)} QD versions for {qid}")
    return lineage


def print_qd_operation_details(qd_data: Dict) -> None:
    """Print detailed information about a QD operation"""
    version = qd_data.get('version', 'unknown')
    operation = qd_data.get('operation', 'unknown')
    logger.info(f"Printing QD operation details for version {version}, operation: {operation}")
    
    print(f"\n{'='*80}")
    print(f"Version: {qd_data['version']}")
    print(f"Operation: {qd_data['operation']}")
    print(f"Parent: {qd_data.get('parent_version', 'none')}")
    print(f"Timestamp: {qd_data['timestamp']}")
    print(f"{'='*80}")
    logger.debug(f"Operation details: {qd_data.get('operation_details', {})}")
    
    op_details = qd_data.get('operation_details', {})
    
    # Show added dimensions
    if op_details.get('added'):
        print(f"\n✚ ADDED DIMENSIONS ({len(op_details['added'])}):")
        for item in op_details['added']:
            print(f"  [{item['index']}] {item['text']}")
    
    # Show dropped dimensions
    if op_details.get('dropped'):
        print(f"\n✖ DROPPED DIMENSIONS ({len(op_details['dropped'])}):")
        for item in op_details['dropped']:
            print(f"  [{item['index']}] {item['text']}")
    
    # Show splits
    if op_details.get('splits'):
        print(f"\n⤴ SPLIT DIMENSIONS ({len(op_details['splits'])}):")
        for item in op_details['splits']:
            print(f"  Old [{item['old_index']}]: {item['old_text'][:60]}...")
            print(f"  → Split into:")
            for idx, text in zip(item['new_indices'], item['new_texts']):
                print(f"    New [{idx}]: {text[:60]}...")
    
    # Show merges
    if op_details.get('merges'):
        print(f"\n⤵ MERGED DIMENSIONS ({len(op_details['merges'])}):")
        for item in op_details['merges']:
            print(f"  Merged from:")
            for idx, text in zip(item['old_indices'], item['old_texts']):
                print(f"    Old [{idx}]: {text[:60]}...")
            print(f"  → New [{item['new_index']}]: {item['new_text'][:60]}...")
    
    # Show refinements (modifications)
    if op_details.get('dimension_mapping'):
        refinements = [m for m in op_details['dimension_mapping'] 
                      if m['operation'] == 'refine']
        if refinements:
            print(f"\n✎ REFINED DIMENSIONS ({len(refinements)}):")
            for item in refinements:
                print(f"  Old [{item['old_index']}]: {item['old_text'][:60]}...")
                print(f"  New [{item['new_index']}]: {item['new_text'][:60]}...")
                print(f"  Similarity: {item['similarity']:.2f}")
    
    # Show current dimensions
    print(f"\n📋 CURRENT DIMENSIONS ({len(qd_data['dimensions'])}):")
    for i, dim in enumerate(qd_data['dimensions']):
        print(f"  {i}: {dim}")


def create_dimension_lineage_df(qid: str, qd_history_dir: Path = Path('./qd_history')):
    """Create a DataFrame showing dimension evolution across versions
    
    Returns:
        DataFrame with columns: version, dimension_index, dimension_text, 
                                operation, came_from, went_to
    """
    import pandas as pd  # Lazy import to avoid slow startup
    logger.info(f"Creating dimension lineage DataFrame for {qid}")
    lineage = load_qd_lineage(qid, qd_history_dir)
    
    if not lineage:
        logger.warning(f"No QD lineage found for {qid}")
        return pd.DataFrame()
    
    records = []
    for qd_data in lineage:
        logger.debug(f"Processing version {qd_data.get('version')} with {len(qd_data.get('dimensions', []))} dimensions")
        version = qd_data['version']
        
        for idx, dim_text in enumerate(qd_data['dimensions']):
            # Track where this dimension came from
            came_from = []
            if qd_data.get('operation_details'):
                op_details = qd_data['operation_details']
                
                # Check dimension mapping
                for mapping in op_details.get('dimension_mapping', []):
                    if mapping['new_index'] == idx:
                        came_from.append(f"{qd_data.get('parent_version')}[{mapping['old_index']}]")
                
                # Check if it was added
                for added in op_details.get('added', []):
                    if added['index'] == idx:
                        came_from.append('newly_added')
                
                # Check if it came from a split
                for split in op_details.get('splits', []):
                    if idx in split['new_indices']:
                        came_from.append(f"{qd_data.get('parent_version')}[{split['old_index']}] (split)")
            
            records.append({
                'question_id': qid,
                'version': version,
                'dimension_index': idx,
                'dimension_text': dim_text,
                'operation': qd_data.get('operation', 'initial'),
                'came_from': ', '.join(came_from) if came_from else 'initial',
                'parent_version': qd_data.get('parent_version', 'none')
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Created lineage DataFrame: {len(df)} records across {len(lineage)} versions")
    return df


def link_grades_to_qd_versions(checkpoint_dir: Path = Path('./checkpoints')):
    """Create DataFrame linking each grading iteration to the QD version used
    
    Returns:
        DataFrame with: question_id, response_id, iteration_number, 
                       phase, grade, qd_version, dimension_scores
    """
    import pandas as pd  # Lazy import to avoid slow startup
    logger.info(f"Linking grades to QD versions from {checkpoint_dir}")
    records = []
    
    checkpoint_files = list(checkpoint_dir.glob('*.json'))
    logger.debug(f"Found {len(checkpoint_files)} checkpoint files")
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = json.loads(checkpoint_file.read_text())
            
            qid = checkpoint['question_id']
            rid = checkpoint['response_id']
            
            # Process each phase
            for phase in ['baseline_iterations', 'qd_initial_iterations', 'qd_refined_iterations']:
                iterations = checkpoint.get(phase, [])
                
                for iter_idx, iteration in enumerate(iterations):
                    qd_version = iteration.get('qd_version', None)
                    
                    records.append({
                        'question_id': qid,
                        'response_id': rid,
                        'phase': phase.replace('_iterations', ''),
                        'iteration_number': iter_idx,
                        'grade': iteration.get('grade'),
                        'qd_version': qd_version,
                        'dimension_scores': str(iteration.get('dimension_scores', [])),
                        'timestamp': iteration.get('timestamp')
                    })
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {checkpoint_file}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing {checkpoint_file}: {e}", exc_info=True)
    
    df = pd.DataFrame(records)
    logger.info(f"Created grades-to-QD DataFrame: {len(df)} records from {len(checkpoint_files)} checkpoints")
    return df


def analyze_dimension_score_stability(
    qid: str,
    rid: str,
    qd_version: str,
    checkpoint_dir: Path = Path('./checkpoints')
):
    """Analyze stability of individual dimension scores for a specific response
    
    Shows which dimensions have high/low agreement across iterations
    """
    import pandas as pd  # Lazy import to avoid slow startup
    logger.info(f"Analyzing dimension score stability for {qid}_{rid} with QD version {qd_version}")
    checkpoint_file = checkpoint_dir / f"{qid}_{rid}.json"
    if not checkpoint_file.exists():
        logger.warning(f"Checkpoint file not found: {checkpoint_file}")
        return pd.DataFrame()
    
    try:
        checkpoint = json.loads(checkpoint_file.read_text())
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_file}: {e}", exc_info=True)
        return pd.DataFrame()
    
    # Find the phase that used this QD version
    phase = None
    if checkpoint.get('qd_initial_version') == qd_version:
        phase = 'qd_initial_iterations'
    elif checkpoint.get('qd_refined_version') == qd_version:
        phase = 'qd_refined_iterations'
    else:
        return pd.DataFrame()
    
    iterations = checkpoint.get(phase, [])
    if not iterations:
        return pd.DataFrame()
    
    # Get dimension scores across iterations
    num_dimensions = len(iterations[0].get('dimension_scores', []))
    
    dimension_results = []
    for dim_idx in range(num_dimensions):
        scores = [
            it['dimension_scores'][dim_idx] 
            for it in iterations 
            if dim_idx < len(it.get('dimension_scores', []))
        ]
        
        # Filter out -1 (parsing errors)
        valid_scores = [s for s in scores if s != -1]
        
        if valid_scores:
            pass_count = sum(valid_scores)
            fail_count = len(valid_scores) - pass_count
            majority = 1 if pass_count > fail_count else 0
            consensus = (max(pass_count, fail_count) / len(valid_scores)) * 100
            flip_rate = 100 - consensus
            
            dimension_results.append({
                'dimension_index': dim_idx,
                'dimension_text': iterations[0]['quality_dimensions'][dim_idx] if dim_idx < len(iterations[0].get('quality_dimensions', [])) else 'N/A',
                'pass_count': pass_count,
                'fail_count': fail_count,
                'majority': majority,
                'consensus': consensus,
                'flip_rate': flip_rate,
                'total_iterations': len(valid_scores)
            })
    
    df = pd.DataFrame(dimension_results)
    if not df.empty:
        logger.info(f"Dimension stability analysis: {len(df)} dimensions, avg flip_rate={df['flip_rate'].mean():.2f}%")
    else:
        logger.warning(f"No dimension stability data found for {qid}_{rid} with version {qd_version}")
    return df


def compare_dimension_stability_across_versions(
    qid: str,
    rid: str,
    checkpoint_dir: Path = Path('./checkpoints')
) -> Dict:
    """Compare dimension-level stability across all QD versions for a response
    
    Returns:
        Dict mapping version -> DataFrame of dimension stability
    """
    logger.info(f"Comparing dimension stability across versions for {qid}_{rid}")
    checkpoint_file = checkpoint_dir / f"{qid}_{rid}.json"
    if not checkpoint_file.exists():
        logger.warning(f"Checkpoint file not found: {checkpoint_file}")
        return {}
    
    try:
        checkpoint = json.loads(checkpoint_file.read_text())
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_file}: {e}", exc_info=True)
        return {}
    
    results = {}
    
    # Analyze initial QD version
    if checkpoint.get('qd_initial_version'):
        version = checkpoint['qd_initial_version']
        logger.debug(f"Analyzing initial QD version: {version}")
        df = analyze_dimension_score_stability(qid, rid, version, checkpoint_dir)
        if not df.empty:
            results[version] = df
            logger.info(f"Added {version} analysis: {len(df)} dimensions")
    
    # Analyze refined QD version
    if checkpoint.get('qd_refined_version'):
        version = checkpoint['qd_refined_version']
        logger.debug(f"Analyzing refined QD version: {version}")
        df = analyze_dimension_score_stability(qid, rid, version, checkpoint_dir)
        if not df.empty:
            results[version] = df
            logger.info(f"Added {version} analysis: {len(df)} dimensions")
    
    logger.info(f"Comparison complete: {len(results)} versions analyzed")
    return results


def print_dimension_comparison(
    qid: str,
    rid: str,
    checkpoint_dir: Path = Path('./checkpoints')
) -> None:
    """Print a comparison of dimension-level stability across QD versions"""
    logger.info(f"Printing dimension comparison for {qid}_{rid}")
    results = compare_dimension_stability_across_versions(qid, rid, checkpoint_dir)
    
    if not results:
        logger.warning(f"No QD grading data for {qid}_{rid}")
        print(f"No QD grading data for {qid}_{rid}")
        return
    
    print(f"\n{'='*100}")
    print(f"Dimension Stability Comparison: {qid}_{rid}")
    print(f"{'='*100}")
    
    for version, df in results.items():
        print(f"\n{version}:")
        print("-" * 100)
        print(df.to_string(index=False))
        print(f"\nAverage flip rate: {df['flip_rate'].mean():.2f}%")


def export_qd_operations_summary(
    qd_history_dir: Path = Path('./qd_history'),
    output_file: str = 'qd_operations_summary.csv'
):
    """Export a summary of all QD operations across all questions
    
    Returns:
        DataFrame with detailed operation information
    """
    import pandas as pd  # Lazy import to avoid slow startup
    logger.info(f"Exporting QD operations summary to {output_file}")
    logger.info(f"Scanning {qd_history_dir} for QD files")
    records = []
    
    qd_files = list(qd_history_dir.glob('*_v*.json'))
    logger.debug(f"Found {len(qd_files)} QD version files")
    
    for qd_file in qd_files:
        if '_v1.json' in str(qd_file):
            logger.debug(f"Skipping initial version: {qd_file.name}")
            continue  # Skip initial versions
        
        try:
            qd_data = json.loads(qd_file.read_text())
            qid = qd_file.stem.rsplit('_', 1)[0]
            
            op_details = qd_data.get('operation_details', {})
            
            records.append({
                'question_id': qid,
                'version': qd_data['version'],
                'parent_version': qd_data.get('parent_version'),
                'operation': qd_data.get('operation'),
                'num_dimensions_before': op_details.get('old_count', 0),
                'num_dimensions_after': op_details.get('new_count', 0),
                'num_added': len(op_details.get('added', [])),
                'num_dropped': len(op_details.get('dropped', [])),
                'num_splits': len(op_details.get('splits', [])),
                'num_merges': len(op_details.get('merges', [])),
                'num_refined': len([m for m in op_details.get('dimension_mapping', []) if m.get('operation') == 'refine']),
                'problematic_responses': qd_data.get('problematic_count', 0),
                'timestamp': qd_data.get('timestamp')
            })
            logger.debug(f"Processed {qd_file.name}: {qd_data.get('operation')} operation")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {qd_file}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing {qd_file}: {e}", exc_info=True)
    
    df = pd.DataFrame(records)
    if not df.empty:
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} QD operations to {output_file}")
            print(f"Exported QD operations summary to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting to {output_file}: {e}", exc_info=True)
            raise
    else:
        logger.warning("No QD operations found to export")
    
    return df


if __name__ == "__main__":
    print("QD Operation Analysis Module loaded.")
    print("\nAvailable functions:")
    print("  - load_qd_lineage(qid)")
    print("  - print_qd_operation_details(qd_data)")
    print("  - create_dimension_lineage_df(qid)")
    print("  - link_grades_to_qd_versions()")
    print("  - analyze_dimension_score_stability(qid, rid, qd_version)")
    print("  - compare_dimension_stability_across_versions(qid, rid)")
    print("  - print_dimension_comparison(qid, rid)")
    print("  - export_qd_operations_summary()")
