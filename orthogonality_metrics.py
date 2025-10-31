"""
Mathematical Orthogonality Metrics
Quantitative measures of QD independence beyond LLM judgment
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class OrthogonalityAnalyzer:
    """
    Compute multiple mathematical measures of QD orthogonality
    """
    
    def __init__(self, df_annotations: pd.DataFrame, qd_names: List[str]):
        """
        df_annotations: DataFrame with columns for each QD (binary 0/1)
        qd_names: List of QD names (column names)
        """
        self.df = df_annotations
        self.qd_names = qd_names
        self.n_samples = len(df_annotations)
        
    # ========================================================================
    # METRIC 1: MUTUAL INFORMATION (already computed, but normalized)
    # ========================================================================
    
    def mutual_information_matrix(self) -> pd.DataFrame:
        """
        Mutual Information: I(X; Y) = H(X) + H(Y) - H(X, Y)
        Measures reduction in uncertainty of X when Y is known
        
        Range: [0, min(H(X), H(Y))]
        Higher = more dependence
        """
        n = len(self.qd_names)
        mi_matrix = np.zeros((n, n))
        
        for i, qd1 in enumerate(self.qd_names):
            for j, qd2 in enumerate(self.qd_names):
                if i == j:
                    mi_matrix[i, j] = 1.0
                else:
                    mi = mutual_info_score(self.df[qd1], self.df[qd2])
                    mi_matrix[i, j] = mi
        
        return pd.DataFrame(mi_matrix, index=self.qd_names, columns=self.qd_names)
    
    def normalized_mutual_information_matrix(self) -> pd.DataFrame:
        """
        Normalized MI: NMI(X; Y) = I(X; Y) / sqrt(H(X) * H(Y))
        
        Range: [0, 1]
        0 = independent
        1 = perfectly dependent
        """
        n = len(self.qd_names)
        nmi_matrix = np.zeros((n, n))
        
        for i, qd1 in enumerate(self.qd_names):
            for j, qd2 in enumerate(self.qd_names):
                if i == j:
                    nmi_matrix[i, j] = 1.0
                else:
                    # Compute entropies
                    h_x = entropy(self.df[qd1].value_counts(normalize=True))
                    h_y = entropy(self.df[qd2].value_counts(normalize=True))
                    
                    if h_x == 0 or h_y == 0:
                        nmi_matrix[i, j] = 0
                    else:
                        mi = mutual_info_score(self.df[qd1], self.df[qd2])
                        nmi = mi / np.sqrt(h_x * h_y)
                        nmi_matrix[i, j] = nmi
        
        return pd.DataFrame(nmi_matrix, index=self.qd_names, columns=self.qd_names)
    
    # ========================================================================
    # METRIC 2: CHI-SQUARED TEST OF INDEPENDENCE
    # ========================================================================
    
    def chi_squared_independence_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chi-squared test: H0: QD1 and QD2 are independent
        
        Returns: (chi2_statistic_matrix, p_value_matrix)
        
        Low p-value (<0.05) = reject independence = coupled
        High chi2 statistic = strong dependence
        """
        n = len(self.qd_names)
        chi2_matrix = np.zeros((n, n))
        p_value_matrix = np.ones((n, n))
        
        for i, qd1 in enumerate(self.qd_names):
            for j, qd2 in enumerate(self.qd_names):
                if i == j:
                    chi2_matrix[i, j] = 0
                    p_value_matrix[i, j] = 1.0
                else:
                    # Create contingency table
                    contingency = pd.crosstab(self.df[qd1], self.df[qd2])
                    
                    if contingency.shape == (2, 2):
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        chi2_matrix[i, j] = chi2
                        p_value_matrix[i, j] = p_value
        
        chi2_df = pd.DataFrame(chi2_matrix, index=self.qd_names, columns=self.qd_names)
        p_value_df = pd.DataFrame(p_value_matrix, index=self.qd_names, columns=self.qd_names)
        
        return chi2_df, p_value_df
    
    # ========================================================================
    # METRIC 3: CRAMÉR'S V (normalized chi-squared)
    # ========================================================================
    
    def cramers_v_matrix(self) -> pd.DataFrame:
        """
        Cramér's V: V = sqrt(chi2 / (n * min(r-1, c-1)))
        
        Range: [0, 1]
        0 = no association
        1 = perfect association
        
        Advantages:
        - Normalized version of chi-squared
        - Comparable across different sample sizes
        - Symmetric measure
        """
        n = len(self.qd_names)
        cramers_matrix = np.zeros((n, n))
        
        for i, qd1 in enumerate(self.qd_names):
            for j, qd2 in enumerate(self.qd_names):
                if i == j:
                    cramers_matrix[i, j] = 1.0
                else:
                    contingency = pd.crosstab(self.df[qd1], self.df[qd2])
                    
                    if contingency.shape == (2, 2):
                        chi2, _, _, _ = chi2_contingency(contingency)
                        n_samples = contingency.sum().sum()
                        
                        # Cramér's V
                        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
                        if min_dim > 0:
                            cramers_v = np.sqrt(chi2 / (n_samples * min_dim))
                            cramers_matrix[i, j] = cramers_v
        
        return pd.DataFrame(cramers_matrix, index=self.qd_names, columns=self.qd_names)
    
    # ========================================================================
    # METRIC 4: CONDITIONAL ENTROPY
    # ========================================================================
    
    def conditional_entropy_matrix(self) -> pd.DataFrame:
        """
        Conditional Entropy: H(X|Y) = H(X, Y) - H(Y)
        Measures uncertainty in X given Y
        
        Range: [0, H(X)]
        Low H(X|Y) = knowing Y reduces uncertainty in X = coupled
        High H(X|Y) = knowing Y doesn't help predict X = independent
        
        Perfect coupling: H(X|Y) = 0
        Perfect independence: H(X|Y) = H(X)
        """
        n = len(self.qd_names)
        cond_entropy_matrix = np.zeros((n, n))
        
        for i, qd1 in enumerate(self.qd_names):
            h_x = entropy(self.df[qd1].value_counts(normalize=True))
            
            for j, qd2 in enumerate(self.qd_names):
                if i == j:
                    cond_entropy_matrix[i, j] = 0
                else:
                    # H(X|Y) = H(X) - I(X;Y)
                    mi = mutual_info_score(self.df[qd1], self.df[qd2])
                    h_x_given_y = h_x - mi
                    cond_entropy_matrix[i, j] = max(0, h_x_given_y)
        
        return pd.DataFrame(cond_entropy_matrix, index=self.qd_names, columns=self.qd_names)
    
    # ========================================================================
    # METRIC 5: POINTWISE MUTUAL INFORMATION (PMI)
    # ========================================================================
    
    def pmi_both_present(self) -> pd.DataFrame:
        """
        Pointwise Mutual Information: PMI(x=1, y=1) = log(P(x=1, y=1) / (P(x=1) * P(y=1)))
        
        Measures how much more likely x=1 and y=1 occur together than by chance
        
        PMI > 0: Co-occur more than expected (positive association)
        PMI = 0: Independent
        PMI < 0: Co-occur less than expected (negative association)
        
        Range: [-∞, +∞] but typically [-10, +10]
        """
        n = len(self.qd_names)
        pmi_matrix = np.zeros((n, n))
        
        for i, qd1 in enumerate(self.qd_names):
            for j, qd2 in enumerate(self.qd_names):
                if i == j:
                    pmi_matrix[i, j] = 0
                else:
                    # P(X=1, Y=1)
                    p_both = ((self.df[qd1] == 1) & (self.df[qd2] == 1)).sum() / self.n_samples
                    
                    # P(X=1), P(Y=1)
                    p_x = (self.df[qd1] == 1).sum() / self.n_samples
                    p_y = (self.df[qd2] == 1).sum() / self.n_samples
                    
                    if p_both > 0 and p_x > 0 and p_y > 0:
                        pmi = np.log2(p_both / (p_x * p_y))
                        pmi_matrix[i, j] = pmi
        
        return pd.DataFrame(pmi_matrix, index=self.qd_names, columns=self.qd_names)
    
    # ========================================================================
    # METRIC 6: JACCARD SIMILARITY (set-theoretic)
    # ========================================================================
    
    def jaccard_similarity_matrix(self) -> pd.DataFrame:
        """
        Jaccard Similarity: J(A, B) = |A ∩ B| / |A ∪ B|
        
        Treats QDs as sets of texts where they are present
        
        Range: [0, 1]
        0 = no overlap
        1 = perfect overlap
        """
        n = len(self.qd_names)
        jaccard_matrix = np.zeros((n, n))
        
        for i, qd1 in enumerate(self.qd_names):
            for j, qd2 in enumerate(self.qd_names):
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                else:
                    # Texts where QD is present
                    set_x = set(self.df[self.df[qd1] == 1].index)
                    set_y = set(self.df[self.df[qd2] == 1].index)
                    
                    intersection = len(set_x & set_y)
                    union = len(set_x | set_y)
                    
                    if union > 0:
                        jaccard = intersection / union
                        jaccard_matrix[i, j] = jaccard
        
        return pd.DataFrame(jaccard_matrix, index=self.qd_names, columns=self.qd_names)
    
    # ========================================================================
    # METRIC 7: CORRELATION (Pearson, Spearman, Kendall)
    # ========================================================================
    
    def correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Correlation coefficients
        
        Methods:
        - pearson: Linear correlation
        - spearman: Rank correlation
        - kendall: Ordinal association
        
        Range: [-1, 1]
        -1 = perfect negative correlation
        0 = no correlation
        +1 = perfect positive correlation
        """
        corr_matrix = self.df[self.qd_names].corr(method=method)
        return corr_matrix
    
    # ========================================================================
    # METRIC 8: ORTHOGONALITY SCORE (Composite)
    # ========================================================================
    
    def orthogonality_score(self) -> float:
        """
        Composite orthogonality score for entire QD set
        
        Higher score = more orthogonal (better)
        
        Score = 1 - mean(pairwise_coupling_scores)
        
        Range: [0, 1]
        0 = completely coupled
        1 = perfectly orthogonal
        """
        # Get multiple coupling measures
        nmi = self.normalized_mutual_information_matrix()
        cramers = self.cramers_v_matrix()
        jaccard = self.jaccard_similarity_matrix()
        
        # Average coupling for each pair (exclude diagonal)
        n = len(self.qd_names)
        coupling_scores = []
        
        for i in range(n):
            for j in range(i+1, n):
                # Average of normalized coupling metrics
                avg_coupling = (nmi.iloc[i, j] + cramers.iloc[i, j] + jaccard.iloc[i, j]) / 3
                coupling_scores.append(avg_coupling)
        
        if len(coupling_scores) == 0:
            return 1.0
        
        mean_coupling = np.mean(coupling_scores)
        orthogonality = 1 - mean_coupling
        
        return orthogonality
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive orthogonality report
        """
        print("\n" + "="*80)
        print("ORTHOGONALITY ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nQuality Dimensions: {len(self.qd_names)}")
        print(f"Corpus Size: {self.n_samples}")
        
        # Overall orthogonality score
        orth_score = self.orthogonality_score()
        print(f"\n📊 OVERALL ORTHOGONALITY SCORE: {orth_score:.3f}")
        print(f"   (0 = fully coupled, 1 = perfectly orthogonal)")
        
        if orth_score > 0.8:
            print("   ✓ Excellent orthogonality")
        elif orth_score > 0.6:
            print("   → Moderate orthogonality (room for improvement)")
        else:
            print("   ⚠ Poor orthogonality (needs refinement)")
        
        # Compute all metrics
        nmi = self.normalized_mutual_information_matrix()
        cramers = self.cramers_v_matrix()
        chi2_stat, p_values = self.chi_squared_independence_matrix()
        jaccard = self.jaccard_similarity_matrix()
        pmi = self.pmi_both_present()
        
        # Find highly coupled pairs
        print("\n" + "="*80)
        print("HIGHLY COUPLED PAIRS (need merging)")
        print("="*80)
        
        coupling_threshold = 0.5  # High coupling
        coupled_pairs = []
        
        n = len(self.qd_names)
        for i in range(n):
            for j in range(i+1, n):
                qd1 = self.qd_names[i]
                qd2 = self.qd_names[j]
                
                # Multiple indicators of coupling
                nmi_val = nmi.iloc[i, j]
                cramers_val = cramers.iloc[i, j]
                jaccard_val = jaccard.iloc[i, j]
                p_val = p_values.iloc[i, j]
                
                # Consider coupled if multiple metrics agree
                indicators = 0
                if nmi_val > coupling_threshold:
                    indicators += 1
                if cramers_val > coupling_threshold:
                    indicators += 1
                if jaccard_val > coupling_threshold:
                    indicators += 1
                if p_val < 0.01:  # Chi-squared significant
                    indicators += 1
                
                if indicators >= 3:  # Majority vote
                    coupled_pairs.append({
                        'qd1': qd1,
                        'qd2': qd2,
                        'nmi': nmi_val,
                        'cramers_v': cramers_val,
                        'jaccard': jaccard_val,
                        'p_value': p_val,
                        'pmi': pmi.iloc[i, j],
                        'indicators': indicators
                    })
        
        coupled_pairs.sort(key=lambda x: x['indicators'], reverse=True)
        
        if len(coupled_pairs) == 0:
            print("✓ No highly coupled pairs detected")
        else:
            print(f"\nFound {len(coupled_pairs)} highly coupled pairs:\n")
            for idx, pair in enumerate(coupled_pairs, 1):
                print(f"{idx}. {pair['qd1']} ↔ {pair['qd2']}")
                print(f"   NMI: {pair['nmi']:.3f} | Cramér's V: {pair['cramers_v']:.3f} | " +
                      f"Jaccard: {pair['jaccard']:.3f}")
                print(f"   Chi² p-value: {pair['p_value']:.4f} | PMI: {pair['pmi']:.3f}")
                print(f"   Coupling indicators: {pair['indicators']}/4 ⚠")
                print()
        
        # QD frequency analysis
        print("="*80)
        print("QD FREQUENCY DISTRIBUTION")
        print("="*80)
        
        for qd in self.qd_names:
            freq = self.df[qd].mean()
            status = ""
            if freq < 0.05:
                status = "⚠ Too rare (consider dropping)"
            elif freq > 0.95:
                status = "⚠ Too common (consider dropping)"
            else:
                status = "✓ Good variance"
            
            print(f"{qd:40s} {freq:6.1%}  {status}")
        
        return {
            'orthogonality_score': orth_score,
            'num_qds': len(self.qd_names),
            'corpus_size': self.n_samples,
            'coupled_pairs': coupled_pairs,
            'nmi_matrix': nmi,
            'cramers_v_matrix': cramers,
            'jaccard_matrix': jaccard,
            'chi2_p_values': p_values
        }


# ============================================================================
# IMPROVEMENT QUANTIFICATION
# ============================================================================

def quantify_orthogonality_improvement(initial_qds: List[str], 
                                      refined_qds: List[str],
                                      initial_annotations: pd.DataFrame,
                                      refined_annotations: pd.DataFrame) -> Dict:
    """
    Mathematically quantify improvement in orthogonality after refinement
    """
    print("\n" + "="*80)
    print("ORTHOGONALITY IMPROVEMENT ANALYSIS")
    print("="*80)
    
    # Analyze initial QD set
    print("\n--- INITIAL QD SET ---")
    analyzer_initial = OrthogonalityAnalyzer(initial_annotations, initial_qds)
    initial_score = analyzer_initial.orthogonality_score()
    
    # Analyze refined QD set
    print("\n--- REFINED QD SET ---")
    analyzer_refined = OrthogonalityAnalyzer(refined_annotations, refined_qds)
    refined_score = analyzer_refined.orthogonality_score()
    
    # Compute improvement
    improvement = refined_score - initial_score
    
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"\nInitial Orthogonality:  {initial_score:.3f}")
    print(f"Refined Orthogonality:  {refined_score:.3f}")
    print(f"Improvement:            {improvement:+.3f}")
    
    if improvement > 0.1:
        print(f"\n✓ SIGNIFICANT IMPROVEMENT in orthogonality")
    elif improvement > 0:
        print(f"\n→ Modest improvement in orthogonality")
    else:
        print(f"\n⚠ No improvement (or degradation)")
    
    # Additional metrics
    print(f"\nQD Count:  {len(initial_qds)} → {len(refined_qds)} " +
          f"(reduction: {len(initial_qds) - len(refined_qds)})")
    
    # Average pairwise coupling
    nmi_initial = analyzer_initial.normalized_mutual_information_matrix()
    nmi_refined = analyzer_refined.normalized_mutual_information_matrix()
    
    # Get off-diagonal elements (pairwise couplings)
    n_init = len(initial_qds)
    n_ref = len(refined_qds)
    
    if n_init > 1:
        initial_couplings = []
        for i in range(n_init):
            for j in range(i+1, n_init):
                initial_couplings.append(nmi_initial.iloc[i, j])
        avg_coupling_initial = np.mean(initial_couplings)
    else:
        avg_coupling_initial = 0
    
    if n_ref > 1:
        refined_couplings = []
        for i in range(n_ref):
            for j in range(i+1, n_ref):
                refined_couplings.append(nmi_refined.iloc[i, j])
        avg_coupling_refined = np.mean(refined_couplings)
    else:
        avg_coupling_refined = 0
    
    coupling_reduction = avg_coupling_initial - avg_coupling_refined
    
    print(f"\nAverage Pairwise Coupling:")
    print(f"  Initial:  {avg_coupling_initial:.3f}")
    print(f"  Refined:  {avg_coupling_refined:.3f}")
    print(f"  Reduction: {coupling_reduction:+.3f}")
    
    return {
        'initial_score': initial_score,
        'refined_score': refined_score,
        'improvement': improvement,
        'qd_count_initial': len(initial_qds),
        'qd_count_refined': len(refined_qds),
        'qd_reduction': len(initial_qds) - len(refined_qds),
        'avg_coupling_initial': avg_coupling_initial,
        'avg_coupling_refined': avg_coupling_refined,
        'coupling_reduction': coupling_reduction
    }


if __name__ == "__main__":
    # Test with synthetic data
    import pandas as pd
    
    # Create sample annotations
    np.random.seed(42)
    n_samples = 100
    
    # Coupled dimensions
    qd1 = np.random.binomial(1, 0.5, n_samples)
    qd2 = np.where(np.random.random(n_samples) < 0.8, qd1, 1 - qd1)  # 80% same as qd1
    
    # Independent dimensions
    qd3 = np.random.binomial(1, 0.5, n_samples)
    qd4 = np.random.binomial(1, 0.5, n_samples)
    
    df = pd.DataFrame({
        'coupled_a': qd1,
        'coupled_b': qd2,
        'independent_a': qd3,
        'independent_b': qd4
    })
    
    analyzer = OrthogonalityAnalyzer(df, ['coupled_a', 'coupled_b', 'independent_a', 'independent_b'])
    report = analyzer.generate_report()
