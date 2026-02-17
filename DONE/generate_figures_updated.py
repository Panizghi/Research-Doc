#!/usr/bin/env python3
"""
Generate publication-quality figures for QD refinement analysis.
Updated for improved compatibility and figure generation.
Run with: python generate_figures_updated.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration - adjust paths as needed
DATA_PATH = './data.csv'  # Update to your data file location
OUTPUT_DIR = './'  # Current directory

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Set publication-quality style
plt.style.use('default')
sns.set_style("whitegrid", {
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': 'gray',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5
})

plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
    'font.family': 'sans-serif'
})

def load_and_prepare_data(filepath):
    """Load and prepare the data."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    # Rename columns for consistency
    rename_map = {
        'baseline_flip_rate_percent %': 'baseline_flip_rate',
        'qd_flip_rate_percent %': 'qd_flip_rate'
    }
    df = df.rename(columns=rename_map)
    
    # Convert numeric columns
    for col in ['expected_majority', 'baseline_majority', 'qd_majority', 
                'baseline_flip_rate', 'qd_flip_rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate flip rate delta
    if 'baseline_flip_rate' in df.columns and 'qd_flip_rate' in df.columns:
        df['flip_rate_delta'] = df['qd_flip_rate'] - df['baseline_flip_rate']
    
    return df

def categorize_responses(df):
    """Categorize all responses based on QD refinement impact."""
    
    def categorize_row(row):
        # Handle missing qd_method column gracefully
        qd_method = row.get('qd_method', 'UNKNOWN') if isinstance(row, dict) else getattr(row, 'qd_method', 'UNKNOWN')
        
        if qd_method == 'SKIPPED' or pd.isna(qd_method):
            return 'Already Stable\n(skipped by QD)'
        
        baseline_flip = row.get('baseline_flip_rate', 0) if isinstance(row, dict) else row['baseline_flip_rate']
        qd_flip = row.get('qd_flip_rate', np.nan) if isinstance(row, dict) else row['qd_flip_rate']
        
        if baseline_flip == 0:
            if pd.isna(qd_flip) or qd_flip == 0:
                return 'Already Stable\n(passed verification)'
            else:
                return 'Was Stable\n(QD introduced instability)'
        
        if qd_method == 'HUMAN_REVIEW':
            return 'Unclear Case\n(needs human review)'
        
        if qd_method == 'NEED_FIX':
            return 'Could Not Process\n(needs fixing)'
        
        flip_delta = row.get('flip_rate_delta', 0) if isinstance(row, dict) else row['flip_rate_delta']
        has_instructor = pd.notna(row.get('expected_majority', np.nan) if isinstance(row, dict) else row['expected_majority'])
        
        improved_correctness = False
        if has_instructor:
            baseline_correct = row.get('baseline_majority', np.nan) if isinstance(row, dict) else row['baseline_majority']
            qd_correct = row.get('qd_majority', np.nan) if isinstance(row, dict) else row['qd_majority']
            expected = row.get('expected_majority', np.nan) if isinstance(row, dict) else row['expected_majority']
            improved_correctness = (baseline_correct != expected) and (qd_correct == expected)
        
        if flip_delta < -5:
            if improved_correctness:
                return 'Improved Both\n(consistency + correctness)'
            else:
                return 'Improved Consistency\n(flip rate decreased)'
        elif flip_delta > 5:
            return 'Worse Consistency\n(flip rate increased)'
        else:
            if improved_correctness:
                return 'Fixed Correctness\n(was consistent but wrong)'
            else:
                return 'No Significant Change\n(flip rate ±5pp)'
    
    df['response_category'] = df.apply(categorize_row, axis=1)
    return df

def create_response_types_figures(df):
    """Create count and percentage bar charts."""
    
    category_counts = df['response_category'].value_counts()
    total = len(df)
    category_pct = (category_counts / total * 100).round(1)
    
    print("\n" + "="*80)
    print("RESPONSE OUTCOME BREAKDOWN")
    print("="*80)
    print(f"Total responses: {total}\n")
    for cat, count in category_counts.items():
        pct = category_pct[cat]
        print(f"  {cat.replace(chr(10), ' ')}: n={count} ({pct:.1f}%)")
    
    colors_map = {
        'Already Stable\n(skipped by QD)': '#4a90e2',
        'Already Stable\n(passed verification)': '#7bccc4',
        'Improved Consistency\n(flip rate decreased)': '#2ecc71',
        'Fixed Correctness\n(was consistent but wrong)': '#3498db',
        'Improved Both\n(consistency + correctness)': '#27ae60',
        'No Significant Change\n(flip rate ±5pp)': '#f39c12',
        'Worse Consistency\n(flip rate increased)': '#e74c3c',
        'Was Stable\n(QD introduced instability)': '#e67e22',
        'Unclear Case\n(needs human review)': '#9b59b6',
        'Could Not Process\n(needs fixing)': '#95a5a6'
    }
    
    categories = category_counts.index.tolist()
    counts = category_counts.values.tolist()
    percentages = [category_pct[cat] for cat in categories]
    colors_list = [colors_map.get(cat, '#808080') for cat in categories]
    
    # Figure 4a: Count version
    fig, ax = plt.subplots(figsize=(13, 8), facecolor='white')
    ax.set_facecolor('white')
    
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, counts, color=colors_list, alpha=0.85, 
                   edgecolor='black', linewidth=1.8, height=0.7)
    
    for i, (cat, count, pct) in enumerate(zip(categories, counts, percentages)):
        ax.text(count + max(counts)*0.03, i, f'n={count} ({pct:.1f}%)', 
               va='center', fontsize=12, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_xlabel('Response Count (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Response Category', fontsize=14, fontweight='bold')
    ax.set_title(f'QD Refinement Impact on Response Outcomes\nTotal unique student responses: N = {total}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(counts) * 1.35)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}4a_response_types_count.png', 
                bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print("\n✓ Saved: 4a_response_types_count.png")
    
    # Figure 4b: Percentage version
    fig, ax = plt.subplots(figsize=(13, 8), facecolor='white')
    ax.set_facecolor('white')
    
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, percentages, color=colors_list, alpha=0.85, 
                   edgecolor='black', linewidth=1.8, height=0.7)
    
    for i, (cat, count, pct) in enumerate(zip(categories, counts, percentages)):
        ax.text(pct + max(percentages)*0.03, i, f'n={count} ({pct:.1f}%)', 
               va='center', fontsize=12, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_xlabel('Response Frequency (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Response Category', fontsize=14, fontweight='bold')
    ax.set_title(f'QD Refinement Impact on Response Outcomes\nTotal unique student responses: N = {total}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(percentages) * 1.35)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}4b_response_types_percent.png', 
                bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print("✓ Saved: 4b_response_types_percent.png")

def create_by_question_figures(df):
    """Create breakdown by question."""
    
    type_by_question = pd.crosstab(df['question_id'], df['response_category'])
    
    print("\n" + "="*80)
    print("RESPONSE TYPES BY QUESTION")
    print("="*80)
    print(type_by_question)
    
    colors_map = {
        'Already Stable\n(skipped by QD)': '#4a90e2',
        'Already Stable\n(passed verification)': '#7bccc4',
        'Improved Consistency\n(flip rate decreased)': '#2ecc71',
        'Fixed Correctness\n(was consistent but wrong)': '#3498db',
        'Improved Both\n(consistency + correctness)': '#27ae60',
        'No Significant Change\n(flip rate ±5pp)': '#f39c12',
        'Worse Consistency\n(flip rate increased)': '#e74c3c',
        'Was Stable\n(QD introduced instability)': '#e67e22',
        'Unclear Case\n(needs human review)': '#9b59b6',
        'Could Not Process\n(needs fixing)': '#95a5a6'
    }
    
    category_order = [cat for cat in colors_map.keys() if cat in type_by_question.columns]
    type_by_question_ordered = type_by_question[category_order]
    
    # Figure 4c: Count version
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    
    type_by_question_ordered.plot(kind='bar', stacked=True, ax=ax,
                                   color=[colors_map[cat] for cat in category_order],
                                   edgecolor='black', linewidth=0.8, width=0.7)
    
    ax.set_facecolor('white')
    ax.set_ylabel('Response Count (n)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Question Identifier', fontsize=14, fontweight='bold')
    ax.set_title('Response Category Distribution by Question\nStacked bars represent total responses per question',
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(title='Response Category', bbox_to_anchor=(1.02, 1), loc='upper left',
             framealpha=1.0, facecolor='white', edgecolor='black', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}4c_response_types_by_question_count.png',
                bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print("✓ Saved: 4c_response_types_by_question_count.png")
    
    # Figure 4d: Percentage version
    type_by_question_pct = (type_by_question_ordered.div(type_by_question_ordered.sum(axis=1), axis=0) * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    
    type_by_question_pct.plot(kind='bar', stacked=True, ax=ax,
                              color=[colors_map[cat] for cat in category_order],
                              edgecolor='black', linewidth=0.8, width=0.7)
    
    ax.set_facecolor('white')
    ax.set_ylabel('Response Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Question Identifier', fontsize=14, fontweight='bold')
    ax.set_title('Response Category Distribution by Question\nPercentage within each question',
                fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.legend(title='Response Category', bbox_to_anchor=(1.02, 1), loc='upper left',
             framealpha=1.0, facecolor='white', edgecolor='black', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}4d_response_types_by_question_percent.png',
                bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print("✓ Saved: 4d_response_types_by_question_percent.png")

def main():
    """Main execution."""
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80)
    
    try:
        # Load data
        print(f"\nLoading data from: {DATA_PATH}")
        df = load_and_prepare_data(DATA_PATH)
        print(f"Loaded {len(df)} responses")
        
        # Categorize responses
        print("\nCategorizing responses...")
        df = categorize_responses(df)
        
        # Generate figures
        print("\nGenerating figures...")
        create_response_types_figures(df)
        create_by_question_figures(df)
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print("\nGenerated files:")
        print("  • 4a_response_types_count.png")
        print("  • 4b_response_types_percent.png")
        print("  • 4c_response_types_by_question_count.png")
        print("  • 4d_response_types_by_question_percent.png")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check:")
        print(f"  1. Data file exists at: {DATA_PATH}")
        print("  2. Required columns: question_id, baseline_flip_rate, qd_flip_rate")
        print("  3. Python libraries installed: pandas, numpy, matplotlib, seaborn")

if __name__ == "__main__":
    main()
