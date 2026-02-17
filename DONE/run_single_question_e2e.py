#!/usr/bin/env python3
"""
End-to-End Single Question Runner
==================================
Runs the complete grading pipeline for a single question with multiple responses:
1. Load question from JSON
2. Generate initial QDs (if needed)
3. Grade all responses (baseline + QD)
4. Refine QDs if needed
5. Analyze results
"""
import os 
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

# Import pipeline functions
from refined_grading_pipeline import (
    load_checkpoint, save_checkpoint, 
    grade_response, grade_all_questions, refine_problematic_qds,
    generate_initial_qds, save_qd_version, load_qd_version,
    get_best_qd_version, get_latest_qd_version,
    analyze_results, print_summary, format_rubric,
    NUM_ITERATIONS, calculate_stability_metrics
)

# Import analysis functions
from qd_operation_analysis import (
    print_dimension_comparison,
    load_qd_lineage,
    print_qd_operation_details
)


def load_questions_json(json_path: str) -> dict:
    """Load questions from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(log_dir: Path = None, log_level: int = logging.INFO) -> logging.Logger:
    """Set up comprehensive logging with file and console handlers"""
    logger = logging.getLogger('run_single_question_e2e')
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
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above, if log_dir provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"run_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger

# Initialize logger
logger = setup_logger(Path('./logs'))


def initialize_checkpoint(qid: str, rid: str, response_text: str) -> dict:
    """Initialize a new checkpoint if it doesn't exist"""
    logger.debug(f"Initializing checkpoint for {qid}_{rid}")
    checkpoint = load_checkpoint(qid, rid)
    if checkpoint is None:
        logger.info(f"Creating new checkpoint for {qid}_{rid}")
        checkpoint = {
            'question_id': qid,
            'response_id': rid,
            'response_text': response_text,
            'baseline_iterations': [],
            'qd_initial_iterations': [],
            'qd_refined_iterations': []
        }
        save_checkpoint(checkpoint, qid, rid)
        logger.debug(f"Checkpoint saved for {qid}_{rid}")
    else:
        logger.debug(f"Checkpoint already exists for {qid}_{rid} with {len(checkpoint.get('baseline_iterations', []))} baseline iterations")
    return checkpoint


def ensure_qds_exist(qid: str, question_text: str, rubric: str) -> None:
    """Generate initial QDs if they don't exist, and show which version will be used"""
    logger.info(f"Checking QDs for {qid}")
    qd_data = load_qd_version(qid)
    if qd_data is None:
        logger.info(f"Generating initial QDs for {qid}")
        print(f"\n  Generating initial QDs for {qid}...")
        try:
            qd_data = generate_initial_qds(question_text, rubric)
            save_qd_version(qid, qd_data)
            logger.info(f"Generated {len(qd_data['dimensions'])} QDs for {qid} (version {qd_data['version']})")
            print(f"  ✓ Generated {len(qd_data['dimensions'])} QDs")
            for i, qd in enumerate(qd_data['dimensions'], 1):
                logger.debug(f"QD {i}: {qd}")
                print(f"    {i}. {qd}")
        except Exception as e:
            logger.error(f"Failed to generate QDs for {qid}: {e}", exc_info=True)
            raise
    else:
        # Check what the best version is (for new responses, this will be used)
        best_version = get_best_qd_version(qid)
        latest_version = get_latest_qd_version(qid) or 'v1'
        initial_version = qd_data['version']
        
        logger.info(f"QDs already exist for {qid} (initial version {initial_version}, latest {latest_version}, best {best_version}, {len(qd_data['dimensions'])} dimensions)")
        
        print(f"\n  ✓ QDs already exist")
        print(f"    Initial version: {initial_version} ({len(qd_data['dimensions'])} QDs)")
        if latest_version != initial_version:
            print(f"    Latest version: {latest_version}")
        if best_version and best_version != initial_version:
            best_qd_data = load_qd_version(qid, best_version)
            if best_qd_data:
                print(f"    Best version (will be used for new responses): {best_version} ({len(best_qd_data['dimensions'])} QDs)")
            else:
                print(f"    Best version (will be used for new responses): {best_version}")
        elif best_version == initial_version:
            print(f"    Best version (will be used for new responses): {best_version}")


def run_single_question_e2e(json_path: str, num_iterations: int = None) -> None:
    """
    Run end-to-end pipeline for a single question
    
    Args:
        json_path: Path to JSON file with question data
        num_iterations: Number of grading iterations (default: NUM_ITERATIONS from pipeline)
    """
    logger.info("="*80)
    logger.info("Starting end-to-end pipeline")
    logger.info("="*80)
    logger.info(f"JSON path: {json_path}")
    logger.info(f"Num iterations: {num_iterations or NUM_ITERATIONS}")
    
    if num_iterations is None:
        num_iterations = NUM_ITERATIONS
    
    # Load question data
    print("="*80)
    print("LOADING QUESTION DATA")
    print("="*80)
    logger.info("Loading question data from JSON")
    
    json_file = Path(json_path)
    if not json_file.exists():
        error_msg = f"File not found: {json_path}"
        logger.error(error_msg)
        print(f"❌ Error: {error_msg}")
        sys.exit(1)
    
    try:
        # Use load_questions_json for better compatibility and validation
        questions_data = load_questions_json(json_path)
        logger.info(f"Successfully loaded JSON file: {json_file}")
    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(error_msg)
        print(f"❌ Error: {error_msg}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in {json_path}: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"❌ Error: {error_msg}")
        sys.exit(1)
    except Exception as e:
        error_msg = f"Error reading {json_path}: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"❌ Error: {error_msg}")
        sys.exit(1)
    
    # Get first (and should be only) question
    if len(questions_data) != 1:
        warning_msg = f"Expected 1 question, found {len(questions_data)}"
        logger.warning(warning_msg)
        print(f"⚠ Warning: {warning_msg}")
        print(f"  Using first question: {list(questions_data.keys())[0]}")
    
    qid = list(questions_data.keys())[0]
    q_data = questions_data[qid]
    
    logger.info(f"Processing question: {qid}")
    logger.info(f"Question text: {q_data.get('question_text', '')[:100]}...")
    logger.info(f"Number of responses: {len(q_data['responses'])}")
    logger.info(f"Iterations per response: {num_iterations}")
    
    print(f"\n✓ Loaded question: {qid}")
    print(f"✓ Number of responses: {len(q_data['responses'])}")
    print(f"✓ Iterations per response: {num_iterations}")
    
    # Initialize checkpoints for all responses
    print("\n" + "="*80)
    print("INITIALIZING CHECKPOINTS")
    print("="*80)
    logger.info("Initializing checkpoints for all responses")
    
    for response in q_data['responses']:
        rid = response['response_id']
        try:
            initialize_checkpoint(qid, rid, response['text'])
            logger.debug(f"Initialized checkpoint for {qid}_{rid}")
            print(f"  ✓ {rid}")
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint for {qid}_{rid}: {e}", exc_info=True)
            print(f"  ✗ {rid} (error: {e})")
    
    # Ensure QDs exist
    print("\n" + "="*80)
    print("QUALITY DIMENSIONS")
    print("="*80)
    logger.info("Ensuring QDs exist")
    
    rubric = format_rubric(q_data.get('rubric'))
    try:
        ensure_qds_exist(qid, q_data['question_text'], rubric)
    except Exception as e:
        logger.error(f"Failed to ensure QDs exist: {e}", exc_info=True)
        print(f"❌ Error ensuring QDs: {e}")
        sys.exit(1)
    
    # Grade all responses
    print("\n" + "="*80)
    print("GRADING PHASE")
    print("="*80)
    logger.info("Starting grading phase")
    
    # Grade each response individually (more control)
    question_text = q_data.get('question_text', '')
    for idx, response in enumerate(q_data['responses'], 1):
        rid = response['response_id']
        text = response['text']
        logger.info(f"Grading response {idx}/{len(q_data['responses'])}: {rid}")
        try:
            grade_response(qid, rid, text, rubric, question_text, num_iterations)
            logger.info(f"Completed grading for {rid}")
        except Exception as e:
            logger.error(f"Error grading {rid}: {e}", exc_info=True)
            print(f"  ✗ Error grading {rid}: {e}")
    
    # Refine QDs if needed
    print("\n" + "="*80)
    print("REFINEMENT PHASE")
    print("="*80)
    logger.info("Starting refinement phase")
    
    # Convert to format expected by refine_problematic_qds
    questions_dict = {qid: q_data}
    try:
        refine_problematic_qds(questions_dict)
        logger.info("Refinement phase completed")
    except Exception as e:
        logger.error(f"Error during refinement: {e}", exc_info=True)
        print(f"❌ Error during refinement: {e}")
    
    # Analyze results
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    logger.info("Starting analysis phase")
    
    try:
        df = analyze_results()
        logger.info(f"Analysis complete: {len(df)} records")
        print_summary(df)
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"❌ Error during analysis: {e}")
    
    # Print per-response details using analysis functions
    print("\n" + "="*80)
    print("PER-RESPONSE DETAILS")
    print("="*80)
    logger.info("Generating per-response details")
    
    for response in q_data['responses']:
        rid = response['response_id']
        checkpoint = load_checkpoint(qid, rid)
        
        if not checkpoint:
            logger.warning(f"No checkpoint found for {qid}_{rid}")
            continue
        
        baseline_metrics = checkpoint.get('baseline_metrics', {})
        qd_initial_metrics = checkpoint.get('qd_initial_metrics', {})
        qd_refined_metrics = checkpoint.get('qd_refined_metrics', {})
        
        logger.debug(f"Metrics for {rid}: baseline_flip={baseline_metrics.get('flip_rate', 0):.1f}%, "
                    f"qd_initial_flip={qd_initial_metrics.get('flip_rate', 0) if qd_initial_metrics else 'N/A'}, "
                    f"qd_refined_flip={qd_refined_metrics.get('flip_rate', 0) if qd_refined_metrics else 'N/A'}")
        
        print(f"\n{rid}:")
        baseline_flip = baseline_metrics.get('flip_rate', 0)
        baseline_maj = baseline_metrics.get('majority', -1)
        baseline_cons = baseline_metrics.get('consensus', 0)
        print(f"  Baseline (RUBRIC): flip_rate={baseline_flip:.1f}%, "
              f"majority={baseline_maj}, "
              f"consensus={baseline_cons:.1f}%")
        
        # Show QD version information (even if skipped)
        qd_version_selected = checkpoint.get('qd_version_selected') or checkpoint.get('qd_initial_version')
        
        if qd_initial_metrics:
            qd_initial_version = checkpoint.get('qd_initial_version', 'unknown')
            qd_initial_flip = qd_initial_metrics.get('flip_rate', 0)
            qd_initial_maj = qd_initial_metrics.get('majority', -1)
            qd_initial_cons = qd_initial_metrics.get('consensus', 0)
            flip_change = baseline_flip - qd_initial_flip
            print(f"  QD Initial ({qd_initial_version}): flip_rate={qd_initial_flip:.1f}%, "
                  f"majority={qd_initial_maj}, "
                  f"consensus={qd_initial_cons:.1f}% "
                  f"(change: {flip_change:+.1f}% vs baseline/rubric)")
        elif qd_version_selected:
            # QD grading was skipped, but we know which version would have been used
            print(f"  QD Initial ({qd_version_selected}): SKIPPED (baseline/rubric already stable at {baseline_flip:.1f}%)")
        
        if qd_refined_metrics:
            qd_refined_version = checkpoint.get('qd_refined_version', 'unknown')
            qd_refined_flip = qd_refined_metrics.get('flip_rate', 0)
            qd_refined_maj = qd_refined_metrics.get('majority', -1)
            qd_refined_cons = qd_refined_metrics.get('consensus', 0)
            refined_flip_change = baseline_flip - qd_refined_flip
            print(f"  QD Refined ({qd_refined_version}): flip_rate={qd_refined_flip:.1f}%, "
                  f"majority={qd_refined_maj}, "
                  f"consensus={qd_refined_cons:.1f}% "
                  f"(change: {refined_flip_change:+.1f}% vs baseline/rubric)")
            
            # Show refinement result evaluation
            refinement_result = checkpoint.get('refinement_result')
            if refinement_result:
                if refinement_result == 'success':
                    print(f"  ✓ Refinement: SUCCESS - Fixed issues while maintaining/improving metrics")
                elif refinement_result == 'partial_success':
                    print(f"  ⚠ Refinement: PARTIAL - Fixed one issue but worsened another")
                elif refinement_result == 'failed':
                    print(f"  ✗ Refinement: FAILED - Did not fix the issues")
                else:
                    print(f"  ? Refinement: {refinement_result}")
        
        skip_reason = checkpoint.get('skip_reason')
        if skip_reason:
            logger.debug(f"{rid} skip reason: {skip_reason}")
            print(f"  Skip reason: {skip_reason}")
        
        refinement_reason = checkpoint.get('refinement_reason')
        if refinement_reason:
            if refinement_reason == 'majority_changed':
                baseline_maj = checkpoint.get('baseline_majority', -1)
                qd_maj = checkpoint.get('qd_majority', -1)
                maj_str = f" ({baseline_maj} → {qd_maj})"
                print(f"  Refinement reason: Majority changed{maj_str} - better consistency but wrong grade")
            elif refinement_reason == 'worse_flip_rate':
                print(f"  Refinement reason: Worse flip rate - QDs made consistency worse")
            else:
                print(f"  Refinement reason: {refinement_reason}")
        
        # Use analysis function for detailed dimension-level comparison if QD data exists
        # This provides dimension-by-dimension stability analysis across QD versions
        if qd_initial_metrics or qd_refined_metrics:
            try:
                print_dimension_comparison(qid, rid)
            except Exception as e:
                logger.debug(f"Could not print dimension comparison for {rid}: {e}")
                # Silently continue - dimension comparison is optional
    
    # Show QD evolution if available
    print("\n" + "="*80)
    print("QD EVOLUTION SUMMARY")
    print("="*80)
    try:
        lineage = load_qd_lineage(qid)
        if lineage:
            logger.info(f"Found {len(lineage)} QD versions for {qid}")
            print(f"\nQD versions for {qid}:")
            for qd_data in lineage:
                version = qd_data.get('version', 'unknown')
                operation = qd_data.get('operation', 'unknown')
                num_dims = len(qd_data.get('dimensions', []))
                print(f"  {version}: {operation} ({num_dims} dimensions)")
                if operation != 'initial_generation':
                    print_qd_operation_details(qd_data)
        else:
            print(f"No QD lineage found for {qid}")
    except Exception as e:
        logger.debug(f"Could not load QD lineage: {e}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    logger.info("="*80)
    logger.info("Pipeline execution complete")
    logger.info("="*80)
    logger.info(f"Checkpoints saved in: ./checkpoints/")
    logger.info(f"QDs saved in: ./quality_dimensions/ and ./refined_qds/")
    print(f"\nCheckpoints saved in: ./checkpoints/")
    print(f"QDs saved in: ./quality_dimensions/ and ./refined_qds/")


if __name__ == "__main__":
    import tempfile
    
    questions_file = Path('./questions.json')
    
    # Check if arguments provided
    if len(sys.argv) >= 2:
        # Check if first arg is a JSON file path or question ID
        arg1 = sys.argv[1]
        
        # If it's a JSON file path (contains .json or /)
        if '.json' in arg1 or '/' in arg1 or '\\' in arg1:
            # Old format: JSON file path
            json_path = arg1
            num_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else None
        else:
            # New format: Question ID (e.g., L3Q1)
            qid = arg1.upper()
            rid = None
            num_iterations = None
            
            # Parse remaining arguments
            if len(sys.argv) > 2:
                arg2 = sys.argv[2]
                # Check if arg2 is a response ID (starts with R) or a number (iterations)
                if arg2.upper().startswith('R'):
                    rid = arg2.upper()
                    # Check if there's a third arg for iterations
                    if len(sys.argv) > 3:
                        try:
                            num_iterations = int(sys.argv[3])
                        except ValueError:
                            pass
                else:
                    # arg2 is likely iterations
                    try:
                        num_iterations = int(arg2)
                    except ValueError:
                        pass
            
            # Load questions.json
            if not questions_file.exists():
                print(f"❌ Error: questions.json not found in current directory")
                sys.exit(1)
            
            try:
                with open(questions_file, 'r') as f:
                    all_questions = json.load(f)
            except Exception as e:
                print(f"❌ Error loading questions.json: {e}")
                sys.exit(1)
            
            # Find the question
            if qid not in all_questions:
                print(f"❌ Error: Question ID '{qid}' not found in questions.json")
                print(f"Available questions: {', '.join(sorted(all_questions.keys()))}")
                sys.exit(1)
            
            q_data = all_questions[qid].copy()
            
            # Filter to specific response if provided
            if rid:
                responses = [r for r in q_data.get('responses', []) if r.get('response_id') == rid]
                if not responses:
                    print(f"❌ Error: Response ID '{rid}' not found for question '{qid}'")
                    available_rids = [r.get('response_id') for r in q_data.get('responses', [])]
                    print(f"Available responses: {', '.join(available_rids)}")
                    sys.exit(1)
                q_data['responses'] = responses
                print(f"✓ Running {qid} with response {rid}")
            else:
                print(f"✓ Running {qid} with all {len(q_data.get('responses', []))} responses")
            
            # Create temporary JSON file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump({qid: q_data}, temp_file, indent=2)
            temp_file.close()
            json_path = temp_file.name
    else:
        # Interactive mode: ask for question ID and optionally response ID
        if not questions_file.exists():
            print("❌ Error: questions.json not found in current directory")
            print("Usage: python run_single_question_e2e.py <question_id> [response_id] [num_iterations]")
            print("   or: python run_single_question_e2e.py <path_to_json_file> [num_iterations]")
            sys.exit(1)
        
        # Load all questions
        try:
            with open(questions_file, 'r') as f:
                all_questions = json.load(f)
        except Exception as e:
            print(f"❌ Error loading questions.json: {e}")
            sys.exit(1)
        
        # Display available questions
        print("\n" + "="*80)
        print("AVAILABLE QUESTIONS")
        print("="*80)
        sorted_qids = sorted(all_questions.keys())
        for qid in sorted_qids:
            q_data = all_questions[qid]
            num_responses = len(q_data.get('responses', []))
            lab_num = q_data.get('lab_number', '?')
            q_num = q_data.get('question_number', '?')
            print(f"  {qid}: Lab {lab_num}, Question {q_num} ({num_responses} responses)")
        
        # Ask for question ID
        print("\n" + "="*80)
        while True:
            try:
                qid_input = input("Enter question ID (e.g., L3Q1): ").strip().upper()
                if qid_input in all_questions:
                    qid = qid_input
                    break
                else:
                    print(f"❌ Invalid question ID. Available: {', '.join(sorted_qids)}")
            except KeyboardInterrupt:
                print("\n⚠ Interrupted by user")
                sys.exit(1)
        
        q_data = all_questions[qid].copy()
        
        # Ask for response ID (optional)
        responses = q_data.get('responses', [])
        if len(responses) > 1:
            print(f"\nAvailable responses for {qid}:")
            for r in responses:
                print(f"  - {r.get('response_id')}")
            
            while True:
                try:
                    rid_input = input(f"\nEnter response ID (or press Enter for all): ").strip().upper()
                    if not rid_input:
                        rid = None
                        print(f"✓ Running all {len(responses)} responses")
                        break
                    elif rid_input.startswith('R'):
                        matching = [r for r in responses if r.get('response_id') == rid_input]
                        if matching:
                            rid = rid_input
                            q_data['responses'] = matching
                            print(f"✓ Running response {rid}")
                            break
                        else:
                            print(f"❌ Response ID '{rid_input}' not found")
                    else:
                        print("❌ Response ID should start with 'R' (e.g., R1, R2)")
                except KeyboardInterrupt:
                    print("\n⚠ Interrupted by user")
                    sys.exit(1)
        else:
            rid = None
            print(f"✓ Running {qid} with {len(responses)} response(s)")
        
        # Ask for number of iterations
        print("\n" + "="*80)
        while True:
            try:
                iter_input = input(f"Enter number of iterations (default: {NUM_ITERATIONS}): ").strip()
                if not iter_input:
                    num_iterations = NUM_ITERATIONS
                    break
                num_iterations = int(iter_input)
                if num_iterations > 0:
                    break
                else:
                    print("❌ Number of iterations must be greater than 0")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n⚠ Interrupted by user")
                sys.exit(1)
        
        # Create temporary JSON file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({qid: q_data}, temp_file, indent=2)
        temp_file.close()
        json_path = temp_file.name
        
        print(f"\n✓ Created temporary file: {json_path}")
        print(f"✓ Running {qid}" + (f" with response {rid}" if rid else "") + f" with {num_iterations} iterations")
        print("="*80 + "\n")
    
    try:
        logger.info(f"Running script with JSON path: {json_path} and num_iterations: {num_iterations}")
        run_single_question_e2e(json_path, num_iterations)
        logger.info("Script completed successfully")
        
        # Clean up temporary file if it was created
        if len(sys.argv) >= 2 and not ('.json' in sys.argv[1] or '/' in sys.argv[1] or '\\' in sys.argv[1]):
            if os.path.exists(json_path):
                try:
                    os.unlink(json_path)
                    logger.debug(f"Cleaned up temporary file: {json_path}")
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {json_path}: {e}")
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        print("\n⚠ Interrupted by user")
        # Clean up temporary file
        if len(sys.argv) >= 2 and not ('.json' in sys.argv[1] or '/' in sys.argv[1] or '\\' in sys.argv[1]):
            if os.path.exists(json_path):
                try:
                    os.unlink(json_path)
                except:
                    pass
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        # Clean up temporary file
        if len(sys.argv) >= 2 and not ('.json' in sys.argv[1] or '/' in sys.argv[1] or '\\' in sys.argv[1]):
            if os.path.exists(json_path):
                try:
                    os.unlink(json_path)
                except:
                    pass
        sys.exit(1)

