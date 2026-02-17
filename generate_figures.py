#!/usr/bin/env python3
"""
Generate publication-quality figures for QD refinement analysis.
Run with: python generate_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Configuration
DATA_PATH = '/Users/paniz/Desktop/data.csv'
OUTPUT_DIR = '/Users/paniz/Documents/GitHub/Research-Doc/'

# Set publication-quality style
plt.style.use('default')
sns.set_style("whitegrid", {'axes.facecolor': 'white', 'figure.facecolor': 'white'})
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.title_fontsize'] = 12

def load_and_prepare_data():
    """Load and prepare the data."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    
    df = df.rename(columns={
        'baseline_flip_rate_percent %': 'baseline_flip_rate',
        'qd_flip_rate_percent %': 'qd_flip_rate'
    })
    
    for col in ['expected_majority', 'baseline_majority', 'qd_majority']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['flip_rate_delta'] = df['qd_flip_rate'] - df['baseline_flip_rate']
    
    return df

def categorize_all_responses(df):
    """
    Categorize ALL responses including skipped ones.
    Categories based on how QD refinement affected each response.
    """
    
    def categorize_row(row):
        # Check if skipped (already stable, not processed by QD)
        if row['qd_method'] == 'SKIPPED':
            return 'Already Stable\n(skipped by QD)'
        
        # Check if QD was applied but response had no flip at baseline
        if row['baseline_flip_rate'] == 0:
            if pd.isna(row['qd_flip_rate']) or row['qd_flip_rate'] == 0:
                return 'Already Stable\n(passed verification)'
            else:
                return 'Was Stable\n(QD introduced instability)'
        
        # Check if response needs human review
        if row['qd_method'] == 'HUMAN_REVIEW':
            return 'Unclear Case\n(needs human review)'
        
        # Check if needs fix (QD couldn't process)
        if row['qd_method'] == 'NEED_FIX':
            return 'Could Not Process\n(needs fixing)'
        
        # For responses with baseline instability (flip rate > 0)
        flip_delta = row['flip_rate_delta']
        
        # Check if correctness improved
        has_instructor_grade = pd.notna(row['expected_majority'])
        improved_correctness = False
        if has_instructor_grade:
            baseline_correct = row['baseline_majority'] == row['expected_majority']
            qd_correct = row['qd_majority'] == row['expected_majority']
            improved_correctness = (not baseline_correct) and qd_correct
        
        # Categorize based on flip rate change and correctness
        if flip_delta < -5:  # Improved consistency
            if improved_correctness:
                return 'Improved Both\n(consistency + correctness)'
            else:
                return 'Improved Consistency\n(flip rate decreased)'
        elif flip_delta > 5:  # Degraded consistency
            return 'Worse Consistency\n(flip rate increased)'
        else:  # Maintained (change < 5pp)
            if improved_correctness:
                return 'Fixed Correctness\n(was consistent but wrong)'
            else:
                return 'No Significant Change\n(flip rate ±5pp)'
    
    df['response_category'] = df.apply(categorize_row, axis=1)
    return df

def create_response_types_figures(df):
    """Create both count and percentage versions of response types."""
    
    # Get counts
    category_counts = df['response_category'].value_counts()
    total = len(df)
    category_pct = (category_counts / total * 100).round(1)
    
    print("\n" + "="*80)
    print("RESPONSE TYPE BREAKDOWN")
    print("="*80)
    print(f"Total responses: {total}")
    print("\nCategories:")
    for cat, count in category_counts.items():
        pct = category_pct[cat]
        print(f"  {cat.replace(chr(10), ' ')}: {count} ({pct:.1f}%)")
    
    # Color mapping - distinct colors from different parts of spectrum (no similar shades)
    colors_map = {
        'Already Stable\n(skipped by QD)': '#4a90e2',        # Blue
        'Already Stable\n(passed verification)': '#7bccc4',  # Teal/cyan
        'Improved Consistency\n(flip rate decreased)': '#2ecc71',  # Green
        'Fixed Correctness\n(was consistent but wrong)': '#3498db',  # Bright blue
        'Improved Both\n(consistency + correctness)': '#27ae60',  # Dark green
        'No Significant Change\n(flip rate ±5pp)': '#f39c12',  # Orange
        'Worse Consistency\n(flip rate increased)': '#e74c3c',  # Red
        'Was Stable\n(QD introduced instability)': '#e67e22',  # Dark orange
        'Unclear Case\n(needs human review)': '#9b59b6',  # Purple (only one)
        'Could Not Process\n(needs fixing)': '#95a5a6'  # Gray
    }
    
    categories = category_counts.index.tolist()
    counts = category_counts.values.tolist()
    percentages = [category_pct[cat] for cat in categories]
    colors_list = [colors_map.get(cat, '#808080') for cat in categories]
    
    # ========================================================================
    # FIGURE 4a: COUNT VERSION
    # ========================================================================
    fig, ax = plt.subplots(figsize=(13, 8), facecolor='white')
    ax.set_facecolor('white')
    
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, counts, color=colors_list, alpha=0.85, 
                   edgecolor='black', linewidth=1.8, height=0.7)
    
    # Add labels
    for i, (cat, count, pct) in enumerate(zip(categories, counts, percentages)):
        ax.text(count + max(counts)*0.03, i, f'n={count} ({pct:.1f}%)', 
               va='center', fontsize=12, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, fontweight='normal')
    ax.set_xlabel('Response Count (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Response Category', fontsize=14, fontweight='bold')
    ax.set_title(f'QD Refinement Impact on Response Outcomes\nTotal unique student responses: N = {total}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(counts) * 1.35)
    ax.grid(axis='x', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '4a_response_types_count.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 4a_response_types_count.png")
    
    # ========================================================================
    # FIGURE 4b: PERCENTAGE VERSION (with zoomed inset)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(13, 8), facecolor='white')
    ax.set_facecolor('white')
    
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, percentages, color=colors_list, alpha=0.85, 
                   edgecolor='black', linewidth=1.8, height=0.7)
    
    # Add labels
    for i, (cat, count, pct) in enumerate(zip(categories, counts, percentages)):
        ax.text(pct + max(percentages)*0.03, i, f'n={count} ({pct:.1f}%)', 
               va='center', fontsize=12, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, fontweight='normal')
    ax.set_xlabel('Response Frequency (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Response Category', fontsize=14, fontweight='bold')
    ax.set_title(f'QD Refinement Impact on Response Outcomes\nTotal unique student responses: N = {total}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(percentages) * 1.35)
    ax.grid(axis='x', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '4b_response_types_percent.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 4b_response_types_percent.png")
    
    return df

def create_response_types_by_question(df):
    """Create response types breakdown by question."""
    
    # Count by question and category
    type_by_question = pd.crosstab(df['question_id'], df['response_category'])
    
    # Calculate percentages (of total responses, not just that question)
    total_responses = len(df)
    type_by_question_pct = (type_by_question / total_responses * 100).round(1)
    
    print("\n" + "="*80)
    print("RESPONSE TYPES BY QUESTION")
    print("="*80)
    print("Count version:")
    print(type_by_question)
    print("\n% of total responses:")
    print(type_by_question_pct)
    
    # Color mapping - distinct colors from different parts of spectrum (no similar shades)
    colors_map = {
        'Already Stable\n(skipped by QD)': '#4a90e2',        # Blue
        'Already Stable\n(passed verification)': '#7bccc4',  # Teal/cyan
        'Improved Consistency\n(flip rate decreased)': '#2ecc71',  # Green
        'Fixed Correctness\n(was consistent but wrong)': '#3498db',  # Bright blue
        'Improved Both\n(consistency + correctness)': '#27ae60',  # Dark green
        'No Significant Change\n(flip rate ±5pp)': '#f39c12',  # Orange
        'Worse Consistency\n(flip rate increased)': '#e74c3c',  # Red
        'Was Stable\n(QD introduced instability)': '#e67e22',  # Dark orange
        'Unclear Case\n(needs human review)': '#9b59b6',  # Purple (only one)
        'Could Not Process\n(needs fixing)': '#95a5a6'  # Gray
    }
    
    # Clear, scientific legend labels
    legend_labels = {
        'Already Stable\n(skipped by QD)': 'Stable (Skipped)',
        'Already Stable\n(passed verification)': 'Stable (Verified)',
        'Improved Consistency\n(flip rate decreased)': 'Consistency Improved',
        'Fixed Correctness\n(was consistent but wrong)': 'Correctness Fixed',
        'Improved Both\n(consistency + correctness)': 'Both Improved',
        'No Significant Change\n(flip rate ±5pp)': 'No Change',
        'Worse Consistency\n(flip rate increased)': 'Consistency Degraded',
        'Was Stable\n(QD introduced instability)': 'Instability Introduced',
        'Unclear Case\n(needs human review)': 'Requires Review',
        'Could Not Process\n(needs fixing)': 'Processing Failed'
    }
    
    # ========================================================================
    # FIGURE 4c: BY QUESTION - COUNT VERSION
    # ========================================================================
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    
    # Order categories for stacking
    category_order = [cat for cat in colors_map.keys() if cat in type_by_question.columns]
    type_by_question_ordered = type_by_question[category_order]
    
    type_by_question_ordered.plot(kind='bar', stacked=True, ax=ax, 
                                   color=[colors_map[cat] for cat in category_order],
                                   edgecolor='black', linewidth=0.8, width=0.7)
    ax.set_facecolor('white')
    ax.set_ylabel('Response Count (n)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Question Identifier', fontsize=14, fontweight='bold')
    ax.set_title('Response Category Distribution by Question\nStacked bars represent total response count per question', 
                fontsize=15, fontweight='bold', pad=15)
    
    # Apply clear legend labels
    legend = ax.legend(title='Response Category', bbox_to_anchor=(1.02, 1), loc='upper left', 
                      framealpha=1.0, facecolor='white', edgecolor='black', frameon=True, fontsize=10,
                      labels=[legend_labels[cat] for cat in category_order])
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(11)
    
    ax.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '4c_response_types_by_question_count.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 4c_response_types_by_question_count.png")
    
    # ========================================================================
    # FIGURE 4d: BY QUESTION - PERCENTAGE (normalized within each question) VERSION
    # ========================================================================
    # Normalize within each question so each bar sums to 100%
    type_by_question_pct_normalized = (type_by_question.div(type_by_question.sum(axis=1), axis=0) * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    
    type_by_question_pct_normalized_ordered = type_by_question_pct_normalized[category_order]
    
    bars_plot = type_by_question_pct_normalized_ordered.plot(kind='bar', stacked=True, ax=ax, 
                                       color=[colors_map[cat] for cat in category_order],
                                       edgecolor='black', linewidth=0.8, width=0.7)
    ax.set_facecolor('white')
    ax.set_ylabel('Response Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Question Identifier', fontsize=14, fontweight='bold')
    ax.set_title(f'Response Category Distribution by Question\nPercentage within each question', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)  # Set y-axis to 0-100%
    
    # Add percentage labels inside bars if size allows
    # Get all containers (one per category in the stack)
    containers = ax.containers
    for container in containers:
        for rect in container:
            height = rect.get_height()
            # Only add label if the segment is large enough (>5%)
            if height > 5:
                x_pos = rect.get_x() + rect.get_width() / 2
                y_pos = rect.get_y() + height / 2
                pct_value = height
                # Use white text for darker segments, black for lighter ones
                ax.text(x_pos, y_pos, f'{pct_value:.1f}%', 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white' if height > 10 else 'black')
    
    # Apply clear legend labels
    legend = ax.legend(title='Response Category', bbox_to_anchor=(1.02, 1), loc='upper left', 
                      framealpha=1.0, facecolor='white', edgecolor='black', frameon=True, fontsize=10,
                      labels=[legend_labels[cat] for cat in category_order])
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(11)
    
    ax.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '4d_response_types_by_question_percent.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 4d_response_types_by_question_percent.png")

def create_simplified_outcome_types_by_question(df):
    """
    Create simplified figure 5c: Outcome Types Distribution by Question.
    Groups categories into three simplified outcomes with better colors and legend below.
    """
    
    # Create simplified outcome categories
    def simplify_category(row):
        cat = row['response_category']
        
        # Already Stable (0% flip rate) - combine both stable categories
        if 'Already Stable' in cat:
            return 'Already Stable (0% flip rate)'
        
        # Improved Both Consistency & Correctness
        if 'Improved Both' in cat:
            return 'Improved Both Consistency & Correctness'
        
        # Improved Consistency Only (without correctness improvement)
        if 'Improved Consistency' in cat and 'Improved Both' not in cat:
            return 'Improved Consistency Only'
        
        # For other categories, we can either exclude them or map them
        # Based on the image, it seems only these three are shown
        return None
    
    df['simplified_outcome'] = df.apply(simplify_category, axis=1)
    
    # Filter to only include the three main categories
    df_simplified = df[df['simplified_outcome'].notna()].copy()
    
    # Count by question and simplified outcome
    outcome_by_question = pd.crosstab(df_simplified['question_id'], df_simplified['simplified_outcome'])
    
    # Calculate percentages (of total responses)
    total_responses = len(df)
    outcome_by_question_pct = (outcome_by_question / total_responses * 100).round(1)
    
    print("\n" + "="*80)
    print("SIMPLIFIED OUTCOME TYPES BY QUESTION (Figure 5c)")
    print("="*80)
    print("Percentage of total responses:")
    print(outcome_by_question_pct)
    
    # Three distinct colors (different shades, not similar)
    colors_map = {
        'Already Stable (0% flip rate)': '#3498db',           # Blue
        'Improved Both Consistency & Correctness': '#27ae60',  # Green
        'Improved Consistency Only': '#f39c12'                # Orange (distinct from blue/green)
    }
    
    # Order categories
    category_order = ['Already Stable (0% flip rate)', 
                      'Improved Both Consistency & Correctness',
                      'Improved Consistency Only']
    category_order = [cat for cat in category_order if cat in outcome_by_question_pct.columns]
    outcome_by_question_ordered = outcome_by_question_pct[category_order]
    
    # ========================================================================
    # FIGURE 5c: SIMPLIFIED OUTCOME TYPES BY QUESTION
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Create stacked bar chart
    outcome_by_question_ordered.plot(kind='bar', stacked=True, ax=ax, 
                                      color=[colors_map[cat] for cat in category_order],
                                      edgecolor='black', linewidth=1.2, width=0.7)
    
    ax.set_ylabel('Response Frequency (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Question Identifier', fontsize=14, fontweight='bold')
    ax.set_title('Outcome Category Distribution by Question\nPercentage of total unique student responses', 
                fontsize=15, fontweight='bold', pad=15)
    
    # Auto-adjust y-axis: include 100 but with padding above
    max_value = outcome_by_question_ordered.sum(axis=1).max()
    y_max = max(100, max_value * 1.05)  # At least 100, or 5% above max if higher
    y_max = min(y_max, 110)  # Cap at 110 so 100 isn't at the very top
    ax.set_ylim(0, y_max)
    
    # Place legend below the graph with clear labels
    legend_labels_short = {
        'Already Stable (0% flip rate)': 'Stable (0% flip rate)',
        'Improved Both Consistency & Correctness': 'Both Improved',
        'Improved Consistency Only': 'Consistency Improved'
    }
    legend = ax.legend(title='Outcome Category', loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      framealpha=1.0, facecolor='white', edgecolor='black', frameon=True, 
                      fontsize=11, ncol=3, labels=[legend_labels_short.get(cat, cat) for cat in category_order])
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(12)
    
    ax.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    plt.tight_layout()
    # Adjust layout to make room for legend below
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig(OUTPUT_DIR + 'DONE/graphs/5c_response_types_by_question.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: DONE/graphs/5c_response_types_by_question.png")

def main():
    """Main execution."""
    print("\n" + "="*80)
    print("GENERATING RESPONSE TYPE FIGURES")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} responses")
    
    # Categorize all responses
    print("\nCategorizing responses...")
    df = categorize_all_responses(df)
    
    # Create figures
    print("\nGenerating figures...")
    df = create_response_types_figures(df)
    create_response_types_by_question(df)
    create_simplified_outcome_types_by_question(df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • 4a_response_types_count.png - Overall count version")
    print("  • 4b_response_types_percent.png - Overall percentage version")
    print("  • 4c_response_types_by_question_count.png - By question, count version")
    print("  • 4d_response_types_by_question_percent.png - By question, % of total version")
    print("  • DONE/graphs/5c_response_types_by_question.png - Simplified outcome types by question")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

