#!/usr/bin/env python3
"""
Automated Rubric Refinement ML Experiment
Production version with MySQL integration and comprehensive logging
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
from datetime import datetime
import requests
import hashlib
import logging
import argparse
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
import mysql.connector
from mysql.connector import Error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class ExperimentConfig:
    """Centralized configuration"""
    
    # MySQL Database
    # Update these with your actual MySQL credentials
    MYSQL_CONFIG = {
        'host': 'localhost',
        'database': 'mte241_labsdb',
        'user': 'root',  # Change to your MySQL username
        'password': ''    # Change to your MySQL password (or '' if no password)
    }
    
    # LLM API
    API_BASE = 'http://ece-nebula16.eng.uwaterloo.ca:11434'
    MODEL = 'gpt-oss:120b'
    API_TIMEOUT = 30
    API_DELAY = 1.0
    
    # Experiment Parameters
    NUM_TRIALS = 10
    MIN_RESPONSES = 30
    MAX_RESPONSES = 40
    TRAIN_TEST_SPLIT = 0.7
    MAX_REFINEMENTS = 3
    EARLY_STOP_THRESHOLD = 0.0
    
    # Orthogonality
    ORTHOGONAL_THRESHOLD = 0.6
    MI_WEIGHT = 0.4
    CORR_WEIGHT = 0.3
    COOCCUR_WEIGHT = 0.3
    
    # Paths
    EXPERIMENT_ID = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    BASE_DIR = f"experiments/exp_{EXPERIMENT_ID}"
    LOG_DIR = f"{BASE_DIR}/logs"
    RESULTS_DIR = f"{BASE_DIR}/results"
    DATA_DIR = f"{BASE_DIR}/data"
    
    @classmethod
    def setup_directories(cls):
        """Create experiment directory structure"""
        for dir_path in [cls.BASE_DIR, cls.LOG_DIR, cls.RESULTS_DIR, cls.DATA_DIR]:
            os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

class ExperimentLogger:
    """Comprehensive logging system"""
    
    def __init__(self, exp_id: str, log_dir: str):
        self.exp_id = exp_id
        self.log_dir = log_dir
        
        # Main logger
        self.logger = logging.getLogger(f"exp_{exp_id}")
        self.logger.setLevel(logging.INFO)
        
        # File handler - detailed
        fh = logging.FileHandler(f'{log_dir}/experiment.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler - summary
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # JSON loggers for structured data
        self.grading_log = []
        self.metrics_log = []
        self.operations_log = []
        
    def log_grading(self, data: Dict):
        """Log grading result"""
        data['timestamp'] = datetime.now().isoformat()
        data['experiment_id'] = self.exp_id
        self.grading_log.append(data)
        
        # Save incrementally
        if len(self.grading_log) % 100 == 0:
            self.save_grading_log()
    
    def log_metric(self, lab: int, question: int, iteration: int, 
                   metric_name: str, value: float, details: Dict = None):
        """Log metric"""
        metric = {
            'experiment_id': self.exp_id,
            'lab': lab,
            'question': question,
            'iteration': iteration,
            'metric_name': metric_name,
            'value': value,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics_log.append(metric)
        self.logger.info(f"L{lab}Q{question} Iter{iteration}: {metric_name}={value:.2f}")
    
    def log_operation(self, lab: int, question: int, iteration: int,
                      operation: str, qds_before: List, qds_after: List, reason: str):
        """Log QD refinement operation"""
        op_log = {
            'experiment_id': self.exp_id,
            'lab': lab,
            'question': question,
            'iteration': iteration,
            'operation': operation,
            'qds_before': qds_before,
            'qds_after': qds_after,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self.operations_log.append(op_log)
        self.logger.info(f"L{lab}Q{question}: Applied {operation} - {reason}")
    
    def save_grading_log(self):
        """Save grading log to JSON"""
        with open(f'{self.log_dir}/grading_log.json', 'w') as f:
            json.dump(self.grading_log, f, indent=2)
    
    def save_all_logs(self):
        """Save all logs"""
        with open(f'{self.log_dir}/grading_log.json', 'w') as f:
            json.dump(self.grading_log, f, indent=2)
        
        with open(f'{self.log_dir}/metrics_log.json', 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
        
        with open(f'{self.log_dir}/operations_log.json', 'w') as f:
            json.dump(self.operations_log, f, indent=2)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

class DatabaseManager:
    """MySQL database manager"""
    
    def __init__(self, config: dict):
        self.config = config
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
    
    def fetch_data(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Fetch data as DataFrame"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            df = pd.read_sql(query, self.connection, params=params)
            return df
        except Error as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()

# ============================================================================
# DATA MANAGEMENT
# ============================================================================

class DataManager:
    """Handle data loading and splitting"""
    
    def __init__(self, db_manager: DatabaseManager, config: ExperimentConfig):
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(f"exp_{config.EXPERIMENT_ID}.data")
    
    def load_responses(self) -> pd.DataFrame:
        """Load responses from MySQL"""
        query = """
        SELECT username, lab_number, question_number, ts, 
               question_text, grade, term
        FROM question_responses
        WHERE term = 's25'
        ORDER BY lab_number, question_number, username
        """
        
        df = self.db.fetch_data(query)
        self.logger.info(f"Loaded {len(df)} responses from database")
        return df
    
    def load_rubrics(self) -> Dict:
        """Load rubrics from MySQL database"""
        rubrics = {}
        try:
            query = """
            SELECT lab_number, question_number, rubric_text
            FROM rubrics
            WHERE term = 's25'
            ORDER BY lab_number, question_number
            """
            
            rubric_df = self.db.fetch_data(query)
            
            if rubric_df.empty:
                self.logger.warning("No rubrics found in database for term 's25'")
            else:
                for _, row in rubric_df.iterrows():
                    key = (int(row['lab_number']), int(row['question_number']))
                    rubrics[key] = row['rubric_text']
                self.logger.info(f"Loaded {len(rubrics)} rubrics from database")
        except Exception as e:
            self.logger.error(f"Error loading rubrics from database: {e}")
        
        return rubrics
    
    def create_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split ensuring min 30 responses per question"""
        train_list = []
        test_list = []
        
        for (lab, q), group in df.groupby(['lab_number', 'question_number']):
            # Get unique responses
            unique = group.drop_duplicates(subset=['question_text'])
            
            # Always ensure 30-40 responses
            if len(unique) > self.config.MAX_RESPONSES:
                unique = unique.sample(n=self.config.MAX_RESPONSES, random_state=42)
            elif len(unique) < self.config.MIN_RESPONSES:
                # Pad with duplicates if needed
                n_needed = self.config.MIN_RESPONSES - len(unique)
                padding = unique.sample(n=min(n_needed, len(unique)), 
                                       replace=True, random_state=42)
                unique = pd.concat([unique, padding])
            
            # Split with stratification
            if len(unique) >= self.config.MIN_RESPONSES:
                try:
                    train, test = train_test_split(
                        unique,
                        test_size=1-self.config.TRAIN_TEST_SPLIT,
                        stratify=unique['grade'],
                        random_state=42
                    )
                except:
                    # If stratification fails, do random split
                    train, test = train_test_split(
                        unique,
                        test_size=1-self.config.TRAIN_TEST_SPLIT,
                        random_state=42
                    )
            else:
                # Use all for both if still under minimum
                train = unique
                test = unique
            
            train_list.append(train)
            test_list.append(test)
            
            self.logger.info(f"L{lab}Q{q}: {len(train)} train, {len(test)} test")
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        # Save split info
        # Convert tuple keys to strings for JSON serialization
        train_questions_dict = train_df.groupby(['lab_number', 'question_number']).size().to_dict()
        test_questions_dict = test_df.groupby(['lab_number', 'question_number']).size().to_dict()
        
        split_info = {
            'experiment_id': self.config.EXPERIMENT_ID,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_questions': {f"L{k[0]}Q{k[1]}": int(v) for k, v in train_questions_dict.items()},
            'test_questions': {f"L{k[0]}Q{k[1]}": int(v) for k, v in test_questions_dict.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{self.config.DATA_DIR}/train_test_split.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return train_df, test_df

# ============================================================================
# LLM INTERFACE
# ============================================================================

class LLMInterface:
    """Handle all LLM interactions"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(f"exp_{config.EXPERIMENT_ID}.llm")
    
    def call(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.config.API_BASE}/api/generate",
                    json={
                        "model": self.config.MODEL,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=self.config.API_TIMEOUT
                )
                
                if response.status_code == 200:
                    time.sleep(self.config.API_DELAY)
                    return response.json().get('response', '')
                
            except Exception as e:
                self.logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        
        self.logger.error("LLM call failed after all retries")
        return ""
    
    def parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        import re
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {}

# ============================================================================
# GRADING FUNCTIONS
# ============================================================================

class Grader:
    """Handle grading operations"""
    
    def __init__(self, llm: LLMInterface, logger: ExperimentLogger):
        self.llm = llm
        self.logger = logger
    
    def grade_with_rubric(self, answer: str, rubric: str, trial: int,
                         lab: int, question: int, username: str) -> Dict:
        """Grade using rubric with detailed logging"""
        
        prompt = f"""Grade this student answer using the rubric.

RUBRIC:
{rubric}

STUDENT ANSWER:
{answer}

Provide detailed grading as JSON:
{{
  "grade": 0 or 1 (0=fail, 1=pass),
  "is_borderline": true or false,
  "confidence": 0.0 to 1.0,
  "feedback": "detailed explanation of grade",
  "missing_elements": ["list", "of", "missing", "items"],
  "strengths": ["list", "of", "strengths"]
}}

Mark is_borderline=true if the answer is very close to pass/fail threshold.
Output ONLY valid JSON."""
        
        response = self.llm.call(prompt)
        result = self.llm.parse_json_response(response)
        
        # Ensure required fields
        if not result:
            result = {
                "grade": 0,
                "is_borderline": False,
                "confidence": 0.5,
                "feedback": "Failed to parse response",
                "missing_elements": [],
                "strengths": []
            }
        
        # Log the grading
        self.logger.log_grading({
            'lab': lab,
            'question': question,
            'username': username,
            'trial': trial,
            'method': 'rubric',
            'iteration': 0,
            'grade': result.get('grade', 0),
            'is_borderline': result.get('is_borderline', False),
            'confidence': result.get('confidence', 0.5),
            'feedback': result.get('feedback', ''),
            'missing_elements': result.get('missing_elements', []),
            'strengths': result.get('strengths', [])
        })
        
        return result
    
    def grade_with_qds(self, answer: str, qds: List[Dict], trial: int,
                      lab: int, question: int, username: str, iteration: int) -> Dict:
        """Grade using QDs with detailed logging"""
        
        qd_text = "\n".join([
            f"{i+1}. {qd['name']}: {qd['definition']}"
            for i, qd in enumerate(qds)
        ])
        
        prompt = f"""Evaluate this answer against quality dimensions.

QUALITY DIMENSIONS:
{qd_text}

STUDENT ANSWER:
{answer[:1000]}

Provide detailed evaluation as JSON:
{{
  "grade": 0 or 1 (pass if meets majority of QDs),
  "is_borderline": true or false,
  "confidence": 0.0 to 1.0,
  "qd_scores": {{
    "QD_name": 0 or 1,
    ...for each QD
  }},
  "feedback": "explanation of evaluation",
  "qds_met": "X out of Y",
  "threshold_used": "majority/all/specific"
}}

Mark is_borderline=true if exactly at threshold.
Output ONLY valid JSON."""
        
        response = self.llm.call(prompt)
        result = self.llm.parse_json_response(response)
        
        # Ensure required fields
        if not result:
            result = {
                "grade": 0,
                "is_borderline": False,
                "confidence": 0.5,
                "qd_scores": {},
                "feedback": "Failed to parse response"
            }
        
        # Log the grading
        self.logger.log_grading({
            'lab': lab,
            'question': question,
            'username': username,
            'trial': trial,
            'method': 'qd',
            'iteration': iteration,
            'grade': result.get('grade', 0),
            'is_borderline': result.get('is_borderline', False),
            'confidence': result.get('confidence', 0.5),
            'feedback': result.get('feedback', ''),
            'qd_scores': result.get('qd_scores', {}),
            'qds_met': result.get('qds_met', ''),
            'threshold_used': result.get('threshold_used', '')
        })
        
        return result

# ============================================================================
# QD OPERATIONS
# ============================================================================

class QDManager:
    """Manage quality dimensions"""
    
    def __init__(self, llm: LLMInterface, config: ExperimentConfig):
        self.llm = llm
        self.config = config
        self.logger = logging.getLogger(f"exp_{config.EXPERIMENT_ID}.qd")
    
    def extract_qds(self, rubric: str, sample_answers: List[str]) -> List[Dict]:
        """Extract orthogonal QDs"""
        
        samples = "\n".join([
            f"{i+1}. {ans[:200]}..."
            for i, ans in enumerate(sample_answers[:5])
        ])
        
        prompt = f"""Extract quality dimensions from this rubric that can be used for consistent binary evaluation.

RUBRIC:
{rubric}

SAMPLE STUDENT ANSWERS:
{samples}

Your task is to identify 3-5 distinct quality dimensions (QDs) that capture the essential aspects needed to evaluate student answers. Each QD should:

1. Be BINARY: Clearly present or absent - no ambiguity. For example, "Mentions at least one example" is good; "Good explanation" is too vague.

2. Be OBJECTIVELY MEASURABLE: Another grader should be able to evaluate the same answer and reach the same conclusion. Avoid subjective terms like "clear" or "well-written" unless clearly defined.

3. Be DISTINCT: Each QD should capture a different aspect of the answer. Avoid overlap - if two QDs always evaluate the same way, they're redundant.

4. Be CONCISE: Keep definitions under 15 words. Focus on what must be present, not how well it's done (unless "how well" is a specific rubric criterion).

5. Cover KEY REQUIREMENTS: Collectively, your QDs should cover all the important aspects from the rubric that determine pass/fail.

6. Support CONSISTENT GRADING: When evaluated together, your QDs should allow for consistent grading decisions across multiple grading attempts of the same answer.

Guidelines:
- Focus on WHAT is present, not quality judgments
- Make definitions specific enough that two graders would agree
- Ensure each QD helps distinguish between passing and failing answers
- If the rubric has multiple distinct requirements, create a QD for each major one
- Avoid creating QDs that are always present or always absent in typical answers

Output as JSON array:
[
  {{
    "name": "ShortDescriptiveName",
    "definition": "Clear, specific binary criterion that can be objectively evaluated without ambiguity",
    "rationale": "Explain why this QD is important for assessing student answers and how it relates to the rubric requirements"
  }}
]

Output ONLY valid JSON array."""
        
        response = self.llm.call(prompt)
        qds = self.llm.parse_json_response(response)
        
        if not qds or not isinstance(qds, list):
            # Fallback QDs
            qds = [
                {"name": "CoreConcept", "definition": "States the main concept correctly"},
                {"name": "Explanation", "definition": "Provides logical reasoning"},
                {"name": "Details", "definition": "Includes relevant examples or specifics"}
            ]
        
        return qds[:5]  # Max 5 QDs
    
    def refine_qds(self, current_qds: List[Dict], flip_rate: float, 
                  flip_rate_metrics: Dict, rubric: str) -> Tuple[List[Dict], str]:
        """Refine QDs using MERGE/SPLIT/ADD/DROP based on flip rate"""
        
        # Get additional context from metrics
        borderline_rate = flip_rate_metrics.get('borderline_rate', 0)
        avg_confidence = flip_rate_metrics.get('avg_confidence', 0)
        avg_consistency = flip_rate_metrics.get('avg_consistency', 0)
        
        prompt = f"""Refine quality dimensions to improve grading consistency and reduce flip rate.

CURRENT QUALITY DIMENSIONS:
{json.dumps(current_qds, indent=2)}

CURRENT PERFORMANCE METRICS:
- Flip rate: {flip_rate:.1f}% (percentage of answers that get different grades across multiple grading attempts)
- Borderline rate: {borderline_rate:.1f}% (percentage of answers at pass/fail threshold)
- Average confidence: {avg_confidence:.2f} (how confident the grader is, 0-1)
- Average consistency: {avg_consistency:.2f} (how consistent grades are for same answer, 0-1)

ORIGINAL RUBRIC:
{rubric}

Your goal is to improve grading consistency by refining the quality dimensions. The flip rate indicates inconsistency - when the same answer gets graded differently across multiple attempts.

Apply ONE refinement operation:

1. MERGE: Combine QDs that are too similar or overlapping in meaning. If two QDs are frequently evaluated the same way or capture redundant aspects, merge them into a single, clearer QD.

2. SPLIT: Separate a QD that is too broad or ambiguous. If a QD captures multiple distinct concepts that should be evaluated separately, split it into more specific, focused QDs.

3. ADD: Add a new QD to capture an important aspect missing from the current set. Ensure it's distinct from existing QDs and addresses a gap in assessment coverage.

4. DROP: Remove a QD that is not useful for discrimination. If a QD doesn't help distinguish between passing and failing answers, or if it's too vague to evaluate consistently, remove it.

Considerations:
- Each QD should be binary (clearly present or absent)
- QDs should be objectively measurable
- QDs should help distinguish between passing and failing answers
- Lower flip rate means better consistency
- Focus on making QDs clearer and more specific to reduce ambiguity

Output as JSON:
{{
  "operation": "MERGE|SPLIT|ADD|DROP",
  "reason": "Detailed explanation of why this operation will improve grading consistency and reduce flip rate. Explain how the refinement addresses the current performance issues.",
  "refined_qds": [
    {{"name": "ShortDescriptiveName", "definition": "Clear binary criterion that can be objectively evaluated"}}
  ]
}}

Output ONLY valid JSON."""
        
        response = self.llm.call(prompt)
        result = self.llm.parse_json_response(response)
        
        if result and 'refined_qds' in result:
            return result['refined_qds'], result.get('operation', 'UNKNOWN')
        
        return current_qds, 'FAILED'

# ============================================================================
# METRICS CALCULATION
# ============================================================================

class MetricsCalculator:
    """Calculate all experiment metrics"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def calculate_flip_rate(self, grading_results: List[Dict]) -> Dict:
        """Calculate flip rate and related metrics"""
        
        # Group by answer
        by_answer = defaultdict(list)
        for result in grading_results:
            # Create unique key for each answer
            key = f"{result.get('username')}_{result.get('lab')}_{result.get('question')}"
            by_answer[key].append(result)
        
        flips = 0
        borderline_count = 0
        all_confidences = []
        grade_distributions = []
        
        for answer_key, trials in by_answer.items():
            grades = [t.get('grade', 0) for t in trials]
            confidences = [t.get('confidence', 0.5) for t in trials]
            borderlines = [t.get('is_borderline', False) for t in trials]
            
            # Check for flip (inconsistent grades)
            if len(set(grades)) > 1:
                flips += 1
            
            # Count borderline
            if any(borderlines):
                borderline_count += 1
            
            all_confidences.extend(confidences)
            
            # Calculate consistency for this answer
            if grades:
                mode_grade = max(set(grades), key=grades.count)
                consistency = grades.count(mode_grade) / len(grades)
                pass_rate = sum(grades) / len(grades)
            else:
                consistency = 0
                pass_rate = 0
            
            grade_distributions.append({
                'pass_rate': pass_rate,
                'consistency': consistency
            })
        
        total = len(by_answer)
        
        return {
            'flip_rate': (flips / total * 100) if total > 0 else 0,
            'flips': flips,
            'total_answers': total,
            'borderline_rate': (borderline_count / total * 100) if total > 0 else 0,
            'borderline_count': borderline_count,
            'avg_confidence': float(np.mean(all_confidences)) if all_confidences else 0,
            'std_confidence': float(np.std(all_confidences)) if all_confidences else 0,
            'avg_consistency': float(np.mean([g['consistency'] for g in grade_distributions])) if grade_distributions else 0,
            'overall_pass_rate': float(np.mean([g['pass_rate'] for g in grade_distributions])) if grade_distributions else 0
        }
    
    def calculate_orthogonality(self, grading_results: List[Dict]) -> Dict:
        """Calculate orthogonality metrics and composite score"""
        
        # Extract QD scores
        qd_data = []
        qd_names = None
        
        for result in grading_results:
            qd_scores = result.get('qd_scores', {})
            if qd_scores:
                if qd_names is None:
                    qd_names = list(qd_scores.keys())
                
                scores = [qd_scores.get(qd, 0) for qd in qd_names]
                qd_data.append(scores)
        
        if not qd_data or not qd_names:
            return {}
        
        qd_array = np.array(qd_data)
        n_qds = len(qd_names)
        
        if n_qds < 2:
            return {
                'qd_names': qd_names,
                'orthogonality_score': 1.0,
                'avg_mutual_info': 0.0,
                'avg_correlation': 0.0,
                'avg_co_occurrence': 0.0
            }
        
        # 1. Mutual Information Matrix
        mi_matrix = np.zeros((n_qds, n_qds))
        for i in range(n_qds):
            for j in range(i+1, n_qds):
                try:
                    mi = mutual_info_score(qd_array[:, i], qd_array[:, j])
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
                except:
                    pass
        
        # 2. Correlation Matrix
        try:
            corr_matrix = np.corrcoef(qd_array.T)
        except:
            corr_matrix = np.eye(n_qds)
        
        # 3. Co-occurrence Matrix (probability of both present)
        co_matrix = np.zeros((n_qds, n_qds))
        for row in qd_array:
            for i in range(n_qds):
                for j in range(n_qds):
                    if row[i] == 1 and row[j] == 1:
                        co_matrix[i, j] += 1
        
        if len(qd_array) > 0:
            co_matrix = co_matrix / len(qd_array)
        
        # Calculate average pairwise metrics (upper triangle only)
        triu_indices = np.triu_indices(n_qds, k=1)
        avg_mi = float(np.mean(mi_matrix[triu_indices]))
        avg_corr = float(np.mean(np.abs(corr_matrix[triu_indices])))
        avg_co = float(np.mean(co_matrix[triu_indices]))
        
        # Composite orthogonality score
        # Higher score = more orthogonal (independent)
        # Score = 1 - normalized average coupling
        # Normalize each metric to [0, 1] range first
        normalized_mi = min(avg_mi / 1.0, 1.0)  # MI typically [0, 1] for binary
        normalized_corr = avg_corr  # Correlation already [-1, 1], abs gives [0, 1]
        normalized_co = avg_co  # Co-occurrence already [0, 1]
        
        # Weighted average coupling
        avg_coupling = (
            normalized_mi * self.config.MI_WEIGHT +
            normalized_corr * self.config.CORR_WEIGHT +
            normalized_co * self.config.COOCCUR_WEIGHT
        )
        
        # Orthogonality score: 1 = perfectly orthogonal, 0 = completely coupled
        orthogonality_score = 1.0 - avg_coupling
        
        # Identify most coupled pairs for diagnostic purposes only (not for orthogonality calculation)
        diagnostic_pairs = []
        for i in range(n_qds):
            for j in range(i+1, n_qds):
                pair_coupling = (
                    min(mi_matrix[i, j] / 1.0, 1.0) * self.config.MI_WEIGHT +
                    abs(corr_matrix[i, j]) * self.config.CORR_WEIGHT +
                    co_matrix[i, j] * self.config.COOCCUR_WEIGHT
                )
                diagnostic_pairs.append({
                    'qd1': qd_names[i],
                    'qd2': qd_names[j],
                    'mi': float(mi_matrix[i, j]),
                    'corr': float(corr_matrix[i, j]),
                    'co': float(co_matrix[i, j]),
                    'coupling_score': float(pair_coupling)
                })
        
        # Sort by coupling score for diagnostic purposes
        diagnostic_pairs.sort(key=lambda x: x['coupling_score'], reverse=True)
        
        return {
            'qd_names': qd_names,
            'orthogonality_score': orthogonality_score,  # Main orthogonality metric
            'mutual_info_matrix': mi_matrix.tolist(),
            'correlation_matrix': corr_matrix.tolist(),
            'co_occurrence_matrix': co_matrix.tolist(),
            'avg_mutual_info': avg_mi,
            'avg_correlation': avg_corr,
            'avg_co_occurrence': avg_co,
            'avg_coupling': avg_coupling,
            # Diagnostic information (not used for orthogonality calculation)
            'diagnostic_pairs': diagnostic_pairs[:5],  # Top 5 most coupled for reference
            'num_highly_coupled_pairs': len([p for p in diagnostic_pairs if p['coupling_score'] > self.config.ORTHOGONAL_THRESHOLD])
        }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

class ExperimentRunner:
    """Main experiment orchestrator"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = ExperimentLogger(config.EXPERIMENT_ID, config.LOG_DIR)
        self.llm = LLMInterface(config)
        self.grader = Grader(self.llm, self.logger)
        self.qd_manager = QDManager(self.llm, config)
        self.metrics_calc = MetricsCalculator(config)
        
        self.logger.logger.info(f"Initialized experiment {config.EXPERIMENT_ID}")
    
    def run_question(self, lab: int, question: int, 
                    train_df: pd.DataFrame, test_df: pd.DataFrame,
                    rubrics: Dict) -> Dict:
        """Run complete experiment for one question"""
        
        self.logger.logger.info(f"\n{'='*60}")
        self.logger.logger.info(f"Starting Lab {lab}, Question {question}")
        
        # Get rubric
        rubric = rubrics.get((lab, question), "")
        if not rubric:
            self.logger.logger.error(f"No rubric found for L{lab}Q{question}")
            return None
        
        # Get data for this question
        train_q = train_df[(train_df['lab_number'] == lab) & 
                          (train_df['question_number'] == question)]
        test_q = test_df[(test_df['lab_number'] == lab) & 
                        (test_df['question_number'] == question)]
        
        if len(train_q) < self.config.MIN_RESPONSES:
            self.logger.logger.warning(f"Only {len(train_q)} training samples")
        
        # Initialize results
        results = {
            'lab': lab,
            'question': question,
            'num_train': len(train_q),
            'num_test': len(test_q),
            'iterations': []
        }
        
        # BASELINE: Grade with rubric
        self.logger.logger.info("Running baseline grading with rubric...")
        baseline_results = []
        
        for _, row in train_q.iterrows():
            for trial in range(self.config.NUM_TRIALS):
                result = self.grader.grade_with_rubric(
                    row['question_text'], rubric, trial,
                    lab, question, row['username']
                )
                result['username'] = row['username']
                result['lab'] = lab
                result['question'] = question
                baseline_results.append(result)
        
        baseline_metrics = self.metrics_calc.calculate_flip_rate(baseline_results)
        self.logger.log_metric(lab, question, -1, 'flip_rate_baseline', 
                              baseline_metrics['flip_rate'], baseline_metrics)
        
        results['baseline'] = baseline_metrics
        self.logger.logger.info(f"Baseline flip rate: {baseline_metrics['flip_rate']:.1f}%")
        
        # EXTRACT QDs
        self.logger.logger.info("Extracting quality dimensions...")
        sample_answers = train_q['question_text'].tolist()
        current_qds = self.qd_manager.extract_qds(rubric, sample_answers)
        
        # Save initial QDs
        with open(f'{self.config.LOG_DIR}/L{lab}Q{question}_qds_initial.json', 'w') as f:
            json.dump(current_qds, f, indent=2)
        
        best_flip_rate = 100.0
        best_qds = current_qds
        best_iteration = 0
        
        # ITERATIVE REFINEMENT
        for iteration in range(self.config.MAX_REFINEMENTS + 1):
            self.logger.logger.info(f"\nIteration {iteration}: Testing QDs...")
            
            # Grade with current QDs
            qd_results = []
            for _, row in train_q.iterrows():
                for trial in range(self.config.NUM_TRIALS):
                    result = self.grader.grade_with_qds(
                        row['question_text'], current_qds, trial,
                        lab, question, row['username'], iteration
                    )
                    result['username'] = row['username']
                    result['lab'] = lab
                    result['question'] = question
                    qd_results.append(result)
            
            # Calculate metrics
            qd_metrics = self.metrics_calc.calculate_flip_rate(qd_results)
            orthogonality = self.metrics_calc.calculate_orthogonality(qd_results)
            
            # Log ALL metrics
            self.logger.log_metric(lab, question, iteration, 'flip_rate',
                                  qd_metrics['flip_rate'], qd_metrics)
            self.logger.log_metric(lab, question, iteration, 'borderline_rate',
                                  qd_metrics['borderline_rate'])
            
            if orthogonality:
                self.logger.log_metric(lab, question, iteration, 'avg_mutual_info',
                                      orthogonality['avg_mutual_info'], orthogonality)
                self.logger.log_metric(lab, question, iteration, 'avg_correlation',
                                      orthogonality['avg_correlation'])
                self.logger.log_metric(lab, question, iteration, 'avg_co_occurrence',
                                      orthogonality['avg_co_occurrence'])
                self.logger.log_metric(lab, question, iteration, 'num_coupled_pairs',
                                      orthogonality['num_coupled_pairs'])
            
            # Save iteration results
            iteration_result = {
                'iteration': iteration,
                'qds': current_qds,
                'metrics': qd_metrics,
                'orthogonality': orthogonality
            }
            results['iterations'].append(iteration_result)
            
            self.logger.logger.info(f"Flip rate: {qd_metrics['flip_rate']:.1f}%")
            if orthogonality:
                self.logger.logger.info(f"Coupled pairs: {orthogonality['num_coupled_pairs']}")
            
            # Track best
            if qd_metrics['flip_rate'] < best_flip_rate:
                best_flip_rate = qd_metrics['flip_rate']
                best_qds = current_qds.copy()
                best_iteration = iteration
                self.logger.logger.info(f"NEW BEST: {best_flip_rate:.1f}%")
            
            # Early stopping
            if qd_metrics['flip_rate'] <= self.config.EARLY_STOP_THRESHOLD:
                self.logger.logger.info(f"Early stopping: reached {qd_metrics['flip_rate']:.1f}%")
                break
            
            # REFINEMENT (force at least one even if worse)
            if iteration < self.config.MAX_REFINEMENTS:
                if iteration == 0 or qd_metrics['flip_rate'] > 5:
                    self.logger.logger.info("Refining QDs...")
                    
                    refined_qds, operation = self.qd_manager.refine_qds(
                        current_qds, qd_metrics['flip_rate'],
                        qd_metrics, rubric
                    )
                    
                    # Log operation
                    self.logger.log_operation(
                        lab, question, iteration, operation,
                        current_qds, refined_qds,
                        f"Flip={qd_metrics['flip_rate']:.1f}%, Confidence={qd_metrics.get('avg_confidence', 0):.2f}, Consistency={qd_metrics.get('avg_consistency', 0):.2f}"
                    )
                    
                    current_qds = refined_qds
                    self.logger.logger.info(f"Applied operation: {operation}")
        
        # TEST SET VALIDATION
        self.logger.logger.info("\nValidating on test set...")
        
        test_baseline_results = []
        test_qd_results = []
        
        for _, row in test_q.iterrows():
            for trial in range(3):  # Fewer trials for test
                # Test with rubric
                result = self.grader.grade_with_rubric(
                    row['question_text'], rubric, trial,
                    lab, question, row['username']
                )
                result['username'] = row['username']
                result['lab'] = lab
                result['question'] = question
                test_baseline_results.append(result)
                
                # Test with best QDs
                result = self.grader.grade_with_qds(
                    row['question_text'], best_qds, trial,
                    lab, question, row['username'], 999
                )
                result['username'] = row['username']
                result['lab'] = lab
                result['question'] = question
                test_qd_results.append(result)
        
        test_baseline_metrics = self.metrics_calc.calculate_flip_rate(test_baseline_results)
        test_qd_metrics = self.metrics_calc.calculate_flip_rate(test_qd_results)
        
        results['test_validation'] = {
            'baseline': test_baseline_metrics,
            'qd': test_qd_metrics,
            'improvement': test_baseline_metrics['flip_rate'] - test_qd_metrics['flip_rate'],
            'best_iteration': best_iteration
        }
        
        # Save results
        with open(f'{self.config.RESULTS_DIR}/L{lab}Q{question}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.logger.info(f"\nFINAL RESULTS L{lab}Q{question}:")
        self.logger.logger.info(f"Train - Baseline: {baseline_metrics['flip_rate']:.1f}%, Best QD: {best_flip_rate:.1f}%")
        self.logger.logger.info(f"Test - Baseline: {test_baseline_metrics['flip_rate']:.1f}%, QD: {test_qd_metrics['flip_rate']:.1f}%")
        self.logger.logger.info(f"Improvement: {results['test_validation']['improvement']:.1f} pp")
        
        return results
    
    def run_experiment(self, num_questions: int = 5):
        """Run complete experiment"""
        
        self.logger.logger.info(f"\n{'#'*60}")
        self.logger.logger.info(f"STARTING EXPERIMENT {self.config.EXPERIMENT_ID}")
        self.logger.logger.info(f"Questions: {num_questions}")
        self.logger.logger.info(f"Trials: {self.config.NUM_TRIALS}")
        self.logger.logger.info(f"Max refinements: {self.config.MAX_REFINEMENTS}")
        self.logger.logger.info(f"{'#'*60}")
        
        # Setup database
        db = DatabaseManager(self.config.MYSQL_CONFIG)
        if not db.connect():
            self.logger.logger.error("Failed to connect to database")
            return None
        
        # Load data
        data_manager = DataManager(db, self.config)
        df = data_manager.load_responses()
        rubrics = data_manager.load_rubrics()
        
        # Create train/test split
        train_df, test_df = data_manager.create_train_test_split(df)
        
        # Select questions
        available = train_df[['lab_number', 'question_number']].drop_duplicates()
        selected = available.sample(n=min(num_questions, len(available)), random_state=42)
        
        # Run experiments
        all_results = []
        for idx, row in selected.iterrows():
            result = self.run_question(
                int(row['lab_number']),
                int(row['question_number']),
                train_df,
                test_df,
                rubrics
            )
            if result:
                all_results.append(result)
        
        # Generate summary
        summary = self.generate_summary(all_results)
        
        # Save all logs
        self.logger.save_all_logs()
        
        # Close database
        db.close()
        
        self.logger.logger.info(f"\n{'#'*60}")
        self.logger.logger.info(f"EXPERIMENT COMPLETE")
        self.logger.logger.info(f"Results saved to: {self.config.BASE_DIR}")
        self.logger.logger.info(f"{'#'*60}")
        
        return summary
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate experiment summary"""
        
        if not results:
            return {}
        
        summary = {
            'experiment_id': self.config.EXPERIMENT_ID,
            'timestamp': datetime.now().isoformat(),
            'num_questions': len(results),
            'config': {
                'num_trials': self.config.NUM_TRIALS,
                'max_refinements': self.config.MAX_REFINEMENTS,
                'train_test_split': self.config.TRAIN_TEST_SPLIT,
                'orthogonal_threshold': self.config.ORTHOGONAL_THRESHOLD
            },
            'results': {
                'avg_baseline_flip_train': np.mean([r['baseline']['flip_rate'] for r in results]),
                'avg_best_qd_flip_train': np.mean([
                    min([i['metrics']['flip_rate'] for i in r['iterations']])
                    for r in results
                ]),
                'avg_baseline_flip_test': np.mean([
                    r['test_validation']['baseline']['flip_rate'] for r in results
                ]),
                'avg_qd_flip_test': np.mean([
                    r['test_validation']['qd']['flip_rate'] for r in results
                ]),
                'avg_improvement': np.mean([
                    r['test_validation']['improvement'] for r in results
                ]),
                'questions_improved': sum(
                    1 for r in results if r['test_validation']['improvement'] > 0
                )
            },
            'per_question': [
                {
                    'lab': r['lab'],
                    'question': r['question'],
                    'train_improvement': r['baseline']['flip_rate'] - min([
                        i['metrics']['flip_rate'] for i in r['iterations']
                    ]),
                    'test_improvement': r['test_validation']['improvement']
                }
                for r in results
            ]
        }
        
        # Save summary
        with open(f'{self.config.RESULTS_DIR}/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Experiment ID: {summary['experiment_id']}")
        print(f"Questions tested: {summary['num_questions']}")
        print(f"\nTRAIN SET:")
        print(f"  Avg baseline flip rate: {summary['results']['avg_baseline_flip_train']:.1f}%")
        print(f"  Avg best QD flip rate: {summary['results']['avg_best_qd_flip_train']:.1f}%")
        print(f"\nTEST SET:")
        print(f"  Avg baseline flip rate: {summary['results']['avg_baseline_flip_test']:.1f}%")
        print(f"  Avg QD flip rate: {summary['results']['avg_qd_flip_test']:.1f}%")
        print(f"\nIMPROVEMENT:")
        print(f"  Average: {summary['results']['avg_improvement']:.1f} pp")
        print(f"  Questions improved: {summary['results']['questions_improved']}/{summary['num_questions']}")
        
        return summary

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Rubric Refinement ML Experiment')
    parser.add_argument('--questions', type=int, default=5, help='Number of questions to test')
    parser.add_argument('--trials', type=int, default=10, help='Number of grading trials')
    parser.add_argument('--refinements', type=int, default=3, help='Max refinement iterations')
    args = parser.parse_args()
    
    # Setup configuration
    config = ExperimentConfig()
    config.NUM_TRIALS = args.trials
    config.MAX_REFINEMENTS = args.refinements
    config.setup_directories()
    
    # Run experiment
    runner = ExperimentRunner(config)
    summary = runner.run_experiment(args.questions)
    
    return summary

if __name__ == "__main__":
    main()
