import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simplified statistical functions")

def main():
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

    # Load data
    df = pd.read_csv('/Users/paniz/Desktop/data.csv')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename columns
    df = df.rename(columns={
        'baseline_flip_rate_percent %': 'baseline_flip_rate',
        'qd_flip_rate_percent %': 'qd_flip_rate'
    })

    # Convert majority columns to numeric
    for col in ['expected_majority', 'baseline_majority', 'qd_majority']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate flip rate change (negative = improvement)
    df['flip_rate_delta'] = df['qd_flip_rate'] - df['baseline_flip_rate']

    # Filter out skipped responses
    df_processed = df[df['qd_majority'] != -1].copy()
    df_processed = df_processed[df_processed['baseline_majority'] != -1].copy()

    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total responses: {len(df)}")
    print(f"Processed responses: {len(df_processed)}")
    print(f"Instructor-graded responses: {df_processed['expected_majority'].notna().sum()}")
    
    # Key statistics
    baseline_stable = (df_processed['baseline_flip_rate'] == 0).sum()
    baseline_unstable = (df_processed['baseline_flip_rate'] > 0).sum()
    print(f"\nBaseline stability:")
    print(f"  Already stable (0% flip): {baseline_stable} responses")
    print(f"  Unstable (>0% flip): {baseline_unstable} responses")
    print(f"  Mean baseline flip rate (unstable only): {df_processed[df_processed['baseline_flip_rate'] > 0]['baseline_flip_rate'].mean():.1f}%")
    print()

    # ============================================================================
    # FIGURE 1: Baseline Instability Distribution
    # ============================================================================
    print("\n" + "=" * 80)
    print("FIGURE 1: BASELINE INSTABILITY")
    print("=" * 80)
    
    stability_by_q = df_processed.groupby('question_id').apply(
        lambda x: pd.Series({
            'Stable': (x['baseline_flip_rate'] == 0).sum(),
            'Unstable': (x['baseline_flip_rate'] > 0).sum(),
            'Pre_QD_Flip': x['baseline_flip_rate'].mean(),
            'Post_QD_Flip': x['qd_flip_rate'].mean()
        })
    ).reset_index()
    
    print("\nBaseline Stability by Question:")
    print(stability_by_q)
    
    # Create figure
    fig = plt.figure(figsize=(16, 6), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # LEFT: Stacked bar - Stable vs Unstable count
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('white')
    
    x = np.arange(len(stability_by_q))
    width = 0.65
    
    bars1 = ax1.bar(x, stability_by_q['Stable'], width,
                    label='Already Stable (0% flip)', color='#66c2a5', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, stability_by_q['Unstable'], width,
                    bottom=stability_by_q['Stable'],
                    label='Unstable (>0% flip)', color='#fc8d59', edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Number of Responses', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Question', fontsize=14, fontweight='bold')
    ax1.set_title('Baseline Stability Distribution\n117/136 responses already stable at baseline', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stability_by_q['question_id'], fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, 
              edgecolor='black', framealpha=0.95, fontsize=12)
    ax1.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()
    ax1.set_ylim(0, max(stability_by_q['Stable'] + stability_by_q['Unstable']) * 1.1)
    
    # Add count labels
    for i, (stable, unstable) in enumerate(zip(stability_by_q['Stable'], stability_by_q['Unstable'])):
        total = stable + unstable
        if stable > 0:
            ax1.text(i, stable/2, f'{int(stable)}', ha='center', va='center', 
                    fontweight='bold', fontsize=11, color='black')
        if unstable > 0:
            ax1.text(i, stable + unstable/2, f'{int(unstable)}', ha='center', va='center', 
                    fontweight='bold', fontsize=11, color='black')
    
    # RIGHT: Grouped bar - Mean flip rates
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('white')
    
    x2 = np.arange(len(stability_by_q))
    width2 = 0.35
    
    bars1 = ax2.bar(x2 - width2/2, stability_by_q['Pre_QD_Flip'], width2,
                    label='Pre-QD', color='#e57373', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x2 + width2/2, stability_by_q['Post_QD_Flip'], width2,
                    label='Post-QD', color='#66bb6a', edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Mean Flip Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Question', fontsize=14, fontweight='bold')
    ax2.set_title('Mean Flip Rate: Pre-QD vs Post-QD\nLower is better (more consistent)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(stability_by_q['question_id'], fontsize=12)
    ax2.legend(loc='upper left', frameon=True, fancybox=False, shadow=False,
              edgecolor='black', framealpha=0.95, fontsize=12)
    ax2.set_ylim(0, max(stability_by_q['Pre_QD_Flip'].max(), stability_by_q['Post_QD_Flip'].max()) * 1.15)
    ax2.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax2.minorticks_on()
    
    plt.tight_layout()
    plt.savefig('/Users/paniz/Documents/GitHub/Research-Doc/1_baseline_instability.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 1_baseline_instability.png")

    # ============================================================================
    # FIGURE 2: Flip Rate Improvements Distribution
    # ============================================================================
    print("\n" + "=" * 80)
    print("FIGURE 2: FLIP RATE IMPROVEMENTS")
    print("=" * 80)
    
    # Categorize changes
    df_processed['improvement_category'] = pd.cut(
        df_processed['flip_rate_delta'],
        bins=[-np.inf, -20, -10, -5, 5, np.inf],
        labels=['Major\nImprovement\n(>20pp↓)', 'Large\nImprovement\n(10-20pp↓)', 
                'Moderate\nImprovement\n(5-10pp↓)', 'No Change\n(±5pp)', 'Degraded\n(>5pp↑)']
    )
    
    change_dist = df_processed['improvement_category'].value_counts().sort_index()
    print("\nFlip Rate Change Distribution:")
    print(change_dist)
    
    # Statistical test on unstable responses only
    if SCIPY_AVAILABLE:
        unstable_only = df_processed[df_processed['baseline_flip_rate'] > 0]['flip_rate_delta']
        if len(unstable_only) > 1:
            t_stat, p_value = stats.ttest_1samp(unstable_only.dropna(), 0)
            mean_improvement = unstable_only.mean()
            print(f"\nStatistics (unstable responses only, n={len(unstable_only)}):")
            print(f"  Mean change: {mean_improvement:.2f} pp")
            print(f"  t = {t_stat:.3f}, p = {p_value:.4f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    colors = ['#1a9850', '#66c2a5', '#a6d96a', '#fee08b', '#d73027']
    bars = ax.bar(range(len(change_dist)), change_dist.values, 
                  color=colors[:len(change_dist)], edgecolor='black', linewidth=2, width=0.7)
    
    ax.set_ylabel('Number of Responses', fontsize=14, fontweight='bold')
    ax.set_xlabel('Improvement Category', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(change_dist)))
    ax.set_xticklabels(change_dist.index, fontsize=11)
    
    # Add title with statistics
    if SCIPY_AVAILABLE and 'p_value' in locals():
        if p_value < 0.001:
            sig_text = f"Unstable responses: Mean Δ = {mean_improvement:.1f}pp, p < 0.001***"
        elif p_value < 0.01:
            sig_text = f"Unstable responses: Mean Δ = {mean_improvement:.1f}pp, p < 0.01**"
        elif p_value < 0.05:
            sig_text = f"Unstable responses: Mean Δ = {mean_improvement:.1f}pp, p < 0.05*"
        else:
            sig_text = f"Unstable responses: Mean Δ = {mean_improvement:.1f}pp, p = {p_value:.3f}"
    else:
        sig_text = ""
    
    ax.set_title(f'Distribution of Flip Rate Changes\nPost-QD minus Pre-QD (negative = improvement)\n{sig_text}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.set_ylim(0, max(change_dist.values) * 1.15)
    
    # Add count labels
    for bar, val in zip(bars, change_dist.values):
        ax.text(bar.get_x() + bar.get_width()/2., val + max(change_dist.values)*0.02,
                f'n={val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/paniz/Documents/GitHub/Research-Doc/2_flip_rate_improvements.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 2_flip_rate_improvements.png")

    # ============================================================================
    # FIGURE 3: Flip Rate Change Scatter
    # ============================================================================
    print("\n" + "=" * 80)
    print("FIGURE 3: FLIP RATE CHANGE PATTERN")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(11, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Color by improvement magnitude
    colors_scatter = []
    for delta in df_processed['flip_rate_delta']:
        if delta < -20:
            colors_scatter.append('#1a9850')  # Major improvement
        elif delta < -10:
            colors_scatter.append('#66c2a5')  # Large improvement
        elif delta < -5:
            colors_scatter.append('#a6d96a')  # Moderate improvement
        elif delta > 5:
            colors_scatter.append('#d73027')  # Degraded
        else:
            colors_scatter.append('#b3b3b3')  # No change
    
    scatter = ax.scatter(df_processed['baseline_flip_rate'], df_processed['flip_rate_delta'],
                        c=colors_scatter, alpha=0.7, edgecolors='black', linewidth=1.2, s=100)
    
    # Reference lines
    ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='No Change Line', zorder=1)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)
    
    # Regression line for unstable responses
    if SCIPY_AVAILABLE:
        unstable_data = df_processed[df_processed['baseline_flip_rate'] > 0][['baseline_flip_rate', 'flip_rate_delta']].dropna()
        if len(unstable_data) > 2:
            slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
                unstable_data['baseline_flip_rate'], unstable_data['flip_rate_delta']
            )
            x_trend = np.linspace(unstable_data['baseline_flip_rate'].min(), 
                                 unstable_data['baseline_flip_rate'].max(), 100)
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'r-', linewidth=3, alpha=0.7, 
                   label=f'Regression (unstable): r={r_value:.2f}, p={p_value_reg:.4f}', zorder=2)
    
    ax.set_xlabel('Baseline Flip Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Flip Rate Change: Post−Pre (%)\n(Negative = Improvement)', fontsize=14, fontweight='bold')
    ax.set_title('QD Refinement Targets High-Instability Responses\nDramatic improvements occur where baseline flip rates were highest', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='lower left', frameon=True, fancybox=False, 
             edgecolor='black', framealpha=0.95)
    ax.grid(alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    
    # Annotations
    ax.text(3, 10, 'Already Stable\n(No intervention\nneeded)', fontsize=11, ha='center', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7, edgecolor='black'))
    ax.text(35, -30, 'Major\nImprovements\n(QD highly effective)', fontsize=11, ha='center', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig('/Users/paniz/Documents/GitHub/Research-Doc/3_flip_rate_change_scatter.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 3_flip_rate_change_scatter.png")

    # ============================================================================
    # FIGURE 4: Response Type Categorization
    # ============================================================================
    print("\n" + "=" * 80)
    print("FIGURE 4: RESPONSE TYPE CATEGORIZATION")
    print("=" * 80)
    
    # Calculate response characteristics
    df_processed['was_stable'] = (df_processed['baseline_flip_rate'] == 0)
    df_processed['improved_consistency'] = (df_processed['flip_rate_delta'] < -5)
    df_processed['maintained_consistency'] = (abs(df_processed['flip_rate_delta']) <= 5)
    df_processed['degraded_consistency'] = (df_processed['flip_rate_delta'] > 5)
    
    # Check correctness
    df_processed['baseline_correct'] = False
    df_processed['qd_correct'] = False
    df_processed['improved_correctness'] = False
    
    mask = df_processed['expected_majority'].notna()
    df_processed.loc[mask, 'baseline_correct'] = (
        df_processed.loc[mask, 'baseline_majority'] == df_processed.loc[mask, 'expected_majority']
    )
    df_processed.loc[mask, 'qd_correct'] = (
        df_processed.loc[mask, 'qd_majority'] == df_processed.loc[mask, 'expected_majority']
    )
    df_processed.loc[mask, 'improved_correctness'] = (
        (~df_processed.loc[mask, 'baseline_correct']) & df_processed.loc[mask, 'qd_correct']
    )
    
    # Categorize - simplified to avoid "Other"
    def categorize_response(row):
        # Priority order for categorization
        if row['was_stable'] and row['maintained_consistency']:
            return 'Already Stable\n(untouched by QD)'
        elif row['improved_consistency'] and row['improved_correctness']:
            return 'Improved Both\n(consistency + correctness)'
        elif row['improved_correctness'] and row['maintained_consistency']:
            return 'Fixed Correctness\n(was consistent but wrong)'
        elif row['improved_consistency']:
            return 'Improved Consistency\n(flip rate decreased)'
        elif row['degraded_consistency']:
            return 'Worse Consistency\n(flip rate increased)'
        elif row['maintained_consistency'] and not row['was_stable']:
            return 'Maintained\n(minor flip rate, no change)'
        else:
            # This catches truly edge cases (should be very few)
            return 'Edge Case\n(complex pattern)'
    
    df_processed['response_type'] = df_processed.apply(categorize_response, axis=1)
    
    # Count and analyze
    type_counts = df_processed['response_type'].value_counts()
    type_pct = (type_counts / len(df_processed) * 100).round(1)
    
    print("\nResponse Type Distribution:")
    for rtype, count in type_counts.items():
        pct = type_pct[rtype]
        print(f"  {rtype.replace(chr(10), ' ')}: {count} responses ({pct:.1f}%)")
    
    # Explain what each category means
    print("\nCategory Definitions:")
    print("  • Already Stable: Baseline flip rate = 0%, QD left unchanged")
    print("  • Improved Both: Flip rate decreased AND fixed incorrect majority decision")
    print("  • Fixed Correctness: Was consistent but wrong, QD fixed the decision")
    print("  • Improved Consistency: Flip rate decreased >5pp")
    print("  • Worse Consistency: Flip rate increased >5pp")
    print("  • Maintained: Flip rate changed <5pp")
    print("  • Edge Case: Responses with complex patterns (should be minimal)")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(13, 8), facecolor='white')
    ax.set_facecolor('white')
    
    categories = type_counts.index.tolist()
    percentages = [type_pct[cat] for cat in categories]
    counts = [type_counts[cat] for cat in categories]
    
    # Color mapping
    colors_map = {
        'Already Stable\n(untouched by QD)': '#b3b3b3',
        'Improved Consistency\n(flip rate decreased)': '#66c2a5',
        'Fixed Correctness\n(was consistent but wrong)': '#4575b4',
        'Improved Both\n(consistency + correctness)': '#1a9850',
        'Maintained\n(minor flip rate, no change)': '#fee08b',
        'Worse Consistency\n(flip rate increased)': '#d73027',
        'Edge Case\n(complex pattern)': '#9467bd'
    }
    
    y_pos = np.arange(len(categories))
    colors_list = [colors_map.get(cat, '#808080') for cat in categories]
    
    # Horizontal bar chart
    bars = ax.barh(y_pos, percentages, color=colors_list, alpha=0.85, 
                   edgecolor='black', linewidth=1.8, height=0.7)
    
    # Add labels
    for i, (cat, count, pct) in enumerate(zip(categories, counts, percentages)):
        ax.text(pct + max(percentages)*0.03, i, f'n={count} ({pct:.1f}%)', 
               va='center', fontsize=12, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, fontweight='normal')
    ax.set_xlabel('Percentage of Total Responses', fontsize=14, fontweight='bold')
    ax.set_title('How QD Refinement Affected Each Response\nN=136 total unique student responses', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(percentages) * 1.35)
    ax.grid(axis='x', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('/Users/paniz/Documents/GitHub/Research-Doc/4_response_types.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 4_response_types.png")
    
    # By question breakdown
    type_by_question = pd.crosstab(df_processed['question_id'], df_processed['response_type'])
    print("\nResponse Types by Question:")
    print(type_by_question)

    # ============================================================================
    # FIGURE 5: Instructor Agreement
    # ============================================================================
    print("\n" + "=" * 80)
    print("FIGURE 5: INSTRUCTOR AGREEMENT")
    print("=" * 80)
    
    df_with_instructor = df_processed[df_processed['expected_majority'].notna()].copy()
    
    print(f"Instructor-graded responses: {len(df_with_instructor)}")
    
    # Calculate agreement
    df_with_instructor['pre_qd_agrees'] = (
        df_with_instructor['baseline_majority'] == df_with_instructor['expected_majority']
    ).astype(int)
    df_with_instructor['post_qd_agrees'] = (
        df_with_instructor['qd_majority'] == df_with_instructor['expected_majority']
    ).astype(int)
    
    pre_agreement = df_with_instructor['pre_qd_agrees'].mean() * 100
    post_agreement = df_with_instructor['post_qd_agrees'].mean() * 100
    
    print(f"\nAgreement with Instructor:")
    print(f"  Pre-QD: {pre_agreement:.1f}%")
    print(f"  Post-QD: {post_agreement:.1f}%")
    print(f"  Change: {post_agreement - pre_agreement:+.1f} pp")
    
    # Agreement patterns
    both_agree = ((df_with_instructor['pre_qd_agrees'] == 1) & 
                 (df_with_instructor['post_qd_agrees'] == 1)).sum()
    both_disagree = ((df_with_instructor['pre_qd_agrees'] == 0) & 
                    (df_with_instructor['post_qd_agrees'] == 0)).sum()
    qd_fixed = ((df_with_instructor['pre_qd_agrees'] == 0) & 
               (df_with_instructor['post_qd_agrees'] == 1)).sum()
    qd_broke = ((df_with_instructor['pre_qd_agrees'] == 1) & 
               (df_with_instructor['post_qd_agrees'] == 0)).sum()
    
    print(f"\nAgreement Patterns:")
    print(f"  Both agree with instructor: {both_agree}")
    print(f"  QD fixed (wrong → correct): {qd_fixed}")
    print(f"  QD broke (correct → wrong): {qd_broke}")
    print(f"  Both disagree with instructor: {both_disagree}")
    
    # Create figure
    fig = plt.figure(figsize=(18, 6), facecolor='white')
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.3, 1.3])
    
    # LEFT: Overall agreement
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('white')
    
    categories = ['Pre-QD', 'Post-QD']
    values = [pre_agreement, post_agreement]
    colors_agree = ['#e57373', '#66bb6a']
    bars = ax1.bar(categories, values, color=colors_agree, edgecolor='black', 
                   linewidth=2, width=0.6)
    
    ax1.set_ylabel('Agreement Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Overall Agreement\nwith Instructor', fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()
    ax1.set_xticklabels(categories, fontsize=13)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # MIDDLE: Agreement by instructor confidence
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('white')
    
    df_with_instructor['instructor_confidence'] = df_with_instructor['likert_score'].apply(
        lambda x: abs(x - 5) if pd.notna(x) else np.nan
    )
    df_with_instructor['conf_level'] = pd.cut(
        df_with_instructor['instructor_confidence'],
        bins=[0, 1, 2, 3, 5],
        labels=['Low\n(Likert~5)', 'Medium\n(±2 from 5)', 'High\n(±3 from 5)', 'Very High\n(±4-5 from 5)']
    )
    
    agreement_by_conf = df_with_instructor.groupby('conf_level').agg({
        'pre_qd_agrees': 'mean',
        'post_qd_agrees': 'mean'
    }) * 100
    
    x = np.arange(len(agreement_by_conf))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, agreement_by_conf['pre_qd_agrees'], width,
                    label='Pre-QD', color='#e57373', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, agreement_by_conf['post_qd_agrees'], width,
                    label='Post-QD', color='#66bb6a', edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Agreement Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Instructor Confidence Level', fontsize=14, fontweight='bold')
    ax2.set_title('Agreement by Instructor Confidence\nHow certain was the instructor?', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(agreement_by_conf.index, fontsize=11)
    ax2.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', 
              framealpha=0.95, fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax2.minorticks_on()
    
    # RIGHT: Agreement pattern distribution
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('white')
    
    pattern_data = pd.DataFrame({
        'Pattern': ['Both Agree\n(stable correct)', 
                   'QD Fixed\n(wrong→correct)', 
                   'QD Broke\n(correct→wrong)', 
                   'Both Disagree\n(stable wrong)'],
        'Count': [both_agree, qd_fixed, qd_broke, both_disagree]
    })
    pattern_data['Percentage'] = (pattern_data['Count'] / len(df_with_instructor) * 100).round(1)
    pattern_data = pattern_data.sort_values('Count', ascending=True)
    
    colors_pattern = ['#1a9850', '#4575b4', '#fc8d59', '#d73027']
    
    y_pos = np.arange(len(pattern_data))
    bars = ax3.barh(y_pos, pattern_data['Percentage'], 
                    color=colors_pattern, alpha=0.85, edgecolor='black', linewidth=1.8)
    
    # Labels
    for i, row in pattern_data.iterrows():
        idx = list(pattern_data.index).index(i)
        ax3.text(row['Percentage'] + 2, idx, 
                f"n={row['Count']} ({row['Percentage']:.1f}%)", 
                va='center', fontsize=11, fontweight='bold')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(pattern_data['Pattern'], fontsize=11)
    ax3.set_xlabel('Percentage of Instructor-Graded Responses', fontsize=14, fontweight='bold')
    ax3.set_title('Agreement Pattern Distribution\nHow did QD affect correctness?', 
                 fontsize=15, fontweight='bold', pad=15)
    ax3.set_xlim(0, max(pattern_data['Percentage']) * 1.4)
    ax3.grid(axis='x', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax3.grid(axis='x', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax3.minorticks_on()
    ax3.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('/Users/paniz/Documents/GitHub/Research-Doc/5_instructor_agreement.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 5_instructor_agreement.png")

    # ============================================================================
    # FIGURE 6: Flip Rate vs Instructor Confidence
    # ============================================================================
    print("\n" + "=" * 80)
    print("FIGURE 6: FLIP RATE vs INSTRUCTOR CONFIDENCE")
    print("=" * 80)
    
    flip_by_conf = df_with_instructor.groupby('conf_level').agg({
        'baseline_flip_rate': 'mean',
        'qd_flip_rate': 'mean',
        'likert_score': 'count'
    }).round(2)
    flip_by_conf.columns = ['Pre_QD_Flip', 'Post_QD_Flip', 'Count']
    
    print("\nMean Flip Rate by Instructor Confidence:")
    print(flip_by_conf)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')
    
    # LEFT: Bar chart
    ax1.set_facecolor('white')
    x = np.arange(len(flip_by_conf))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, flip_by_conf['Pre_QD_Flip'], width,
                    label='Pre-QD', color='#e57373', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, flip_by_conf['Post_QD_Flip'], width,
                    label='Post-QD', color='#66bb6a', edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Mean Flip Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Instructor Confidence Level', fontsize=14, fontweight='bold')
    ax1.set_title('Mean Flip Rate by Instructor Confidence\nLower flip rate = more reliable grading', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(flip_by_conf.index, fontsize=11)
    ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', 
              framealpha=0.95, fontsize=12)
    ax1.set_ylim(0, max(flip_by_conf['Pre_QD_Flip'].max(), flip_by_conf['Post_QD_Flip'].max()) * 1.2)
    ax1.grid(axis='y', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()
    
    # Add sample size labels
    for i, count in enumerate(flip_by_conf['Count']):
        ax1.text(i, -max(flip_by_conf['Pre_QD_Flip'].max(), flip_by_conf['Post_QD_Flip'].max())*0.08, 
                f'n={int(count)}', ha='center', fontsize=10, fontweight='bold')
    
    # RIGHT: Scatter with error bars
    ax2.set_facecolor('white')
    
    # Calculate stats per bin
    bin_stats = df_with_instructor.groupby('conf_level').agg({
        'qd_flip_rate': ['mean', 'std', 'count']
    }).reset_index()
    bin_stats.columns = ['conf_level', 'mean', 'std', 'count']
    
    x_pos = np.arange(len(bin_stats))
    
    # Error bars
    ax2.errorbar(x_pos, bin_stats['mean'], yerr=bin_stats['std'], 
                fmt='o', markersize=12, capsize=10, capthick=2.5,
                color='#4575b4', ecolor='#4575b4', linewidth=2.5, alpha=0.8,
                label='Mean ± SD')
    
    # Add scatter of actual points
    np.random.seed(42)
    for i, conf_level in enumerate(bin_stats['conf_level']):
        subset = df_with_instructor[df_with_instructor['conf_level'] == conf_level]
        jitter = np.random.normal(0, 0.06, len(subset))
        ax2.scatter(i + jitter, subset['qd_flip_rate'], 
                   alpha=0.4, s=40, color='gray', edgecolor='black', linewidth=0.5,
                   label='Individual responses' if i == 0 else '')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_stats['conf_level'], fontsize=11)
    ax2.set_xlabel('Instructor Confidence Level', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Post-QD Flip Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Post-QD Flip Rate Distribution\nEach point = one response', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', 
              framealpha=0.95, fontsize=11)
    ax2.grid(alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax2.grid(which='minor', alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
    ax2.minorticks_on()
    ax2.set_ylim(-3, max(df_with_instructor['qd_flip_rate'].max(), 50) + 5)
    
    plt.tight_layout()
    plt.savefig('/Users/paniz/Documents/GitHub/Research-Doc/6_flip_rate_vs_confidence.png', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print("✓ Saved: 6_flip_rate_vs_confidence.png")

    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)
    print("\nGenerated figures:")
    print("  1. 1_baseline_instability.png")
    print("  2. 2_flip_rate_improvements.png")
    print("  3. 3_flip_rate_change_scatter.png")
    print("  4. 4_response_types.png")
    print("  5. 5_instructor_agreement.png")
    print("  6. 6_flip_rate_vs_confidence.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
