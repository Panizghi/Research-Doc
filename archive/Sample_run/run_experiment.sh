#!/bin/bash

# Rubric Refinement Experiment Setup and Run Script

echo "======================================"
echo "Rubric Refinement ML Experiment Setup"
echo "======================================"

# Check Python version
python3 --version

# Install required packages
echo "Installing required packages..."
pip install pandas numpy scikit-learn mysql-connector-python requests

# Create directory structure
echo "Creating experiment directories..."
mkdir -p experiments
mkdir -p data
mkdir -p logs

# Database setup (if needed)
echo "Setting up database connection..."
mysql -u root -p << EOF
CREATE DATABASE IF NOT EXISTS grading_db;
USE grading_db;

-- Create question_responses table if not exists
CREATE TABLE IF NOT EXISTS question_responses (
    username VARCHAR(255) NOT NULL,
    lab_number TINYINT NOT NULL,
    question_number TINYINT NOT NULL,
    ts VARCHAR(19) NOT NULL,
    question_text MEDIUMTEXT NOT NULL,
    grade TINYINT NOT NULL,
    term VARCHAR(10) NOT NULL DEFAULT 's25',
    PRIMARY KEY (username, lab_number, question_number, ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Create rubrics table
CREATE TABLE IF NOT EXISTS rubrics (
    lab_number TINYINT NOT NULL,
    question_number TINYINT NOT NULL,
    rubric TEXT NOT NULL,
    PRIMARY KEY (lab_number, question_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Create experiment tracking tables
CREATE TABLE IF NOT EXISTS experiment_runs (
    experiment_id VARCHAR(16) PRIMARY KEY,
    config JSON,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    status VARCHAR(20),
    summary JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS experiment_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(16),
    lab_number TINYINT,
    question_number TINYINT,
    iteration INT,
    metric_name VARCHAR(50),
    metric_value DOUBLE,
    details JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_experiment (experiment_id),
    INDEX idx_question (lab_number, question_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

GRANT ALL PRIVILEGES ON grading_db.* TO 'grading_user'@'localhost';
FLUSH PRIVILEGES;
EOF

echo "Database setup complete!"

# Run experiment
echo ""
echo "======================================"
echo "Starting Experiment"
echo "======================================"

# Parse command line arguments
QUESTIONS=${1:-5}
TRIALS=${2:-10}
REFINEMENTS=${3:-3}

echo "Configuration:"
echo "  Questions: $QUESTIONS"
echo "  Trials: $TRIALS"
echo "  Max Refinements: $REFINEMENTS"
echo ""

# Run the experiment
python3 ml_experiment_production.py \
    --questions $QUESTIONS \
    --trials $TRIALS \
    --refinements $REFINEMENTS

echo ""
echo "======================================"
echo "Experiment Complete"
echo "======================================"
echo "Results saved in experiments/ directory"
