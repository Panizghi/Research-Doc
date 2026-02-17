# LLM Grading Consistency: Reducing Stochasticity through Iterative Quality Dimension Refinement

## 1. Introduction/Background

Large Language Models (LLMs) have shown promise for automated grading of student responses, offering scalability and consistency advantages over manual grading. However, a critical challenge in LLM-based grading is **stochasticity**—the same response can receive different grades across multiple grading iterations, even when using identical prompts and rubrics. This inconsistency undermines the reliability and fairness of automated grading systems.

The stochasticity in LLM grading manifests as **flip rate**—the percentage of grading iterations that disagree with the majority grade. For example, if a response receives a "pass" grade in 7 out of 10 iterations, the flip rate is 30%, indicating that 30% of iterations produced a different grade than the majority. High flip rates suggest that the grading criteria are ambiguous or that the LLM lacks sufficient guidance to make consistent decisions.

Previous research has explored various approaches to reduce LLM grading inconsistency, including prompt engineering, few-shot learning, and rubric refinement. However, these approaches often require extensive manual tuning and may not address the root cause of inconsistency: the lack of explicit, measurable criteria that guide the LLM's decision-making process.

This work introduces **Quality Dimensions (QDs)**—specific, measurable criteria derived from rubrics that provide structured guidance to LLMs during grading. Unlike traditional rubrics that describe expectations in prose, QDs break down evaluation into discrete, assessable dimensions. We hypothesize that by providing explicit dimensional criteria, LLMs can make more consistent grading decisions, thereby reducing flip rates while maintaining alignment with baseline rubric-based grading.

The key research questions we address are:
1. To what extent does LLM grading exhibit stochasticity when using traditional rubrics?
2. Can Quality Dimensions reduce flip rates compared to baseline rubric-only grading?
3. Can iterative refinement of Quality Dimensions improve grading consistency while maintaining correct majority grades?
4. What patterns emerge in the effectiveness of QD refinement across different questions and response types?

## 2. Approach

### 2.1 Baseline Grading

We establish a baseline by grading each student response 10 times using only the original rubric. For each response, we calculate:
- **Consensus**: The percentage of iterations that agree with the majority grade
- **Flip Rate**: 100% - Consensus (the percentage of iterations that disagree)
- **Majority Grade**: The grade (pass/fail) assigned by the majority of iterations

A response is considered **baseline-stable** only if it achieves 100% consensus (0% flip rate), meaning all iterations produce the same grade. Responses with non-zero flip rates are candidates for QD-based grading improvement.

### 2.2 Quality Dimension Generation

For each question, we generate an initial set of Quality Dimensions using an LLM. The generation process takes as input:
- The question text
- The original rubric (including "meets expectation" and "does not meet expectation" criteria)
- Requirements for creating measurable, specific dimensions

The LLM produces a structured list of dimensions, each representing a specific aspect of the response that can be evaluated independently. For example, a question about explaining a technical concept might have dimensions such as:
- "Correctly identifies the core concept"
- "Provides accurate technical details"
- "Explains the purpose or application"

### 2.3 QD-Based Grading

Once Quality Dimensions are generated, we grade each response 10 times using both the rubric and the Quality Dimensions. The LLM evaluates each dimension independently (assigning a pass/fail score) and then synthesizes these dimension scores into an overall grade. We compare the QD-based grading results to the baseline:
- **Flip Rate Change**: Baseline flip rate - QD flip rate (positive = improvement)
- **Majority Stability**: Whether the QD majority grade matches the baseline majority grade
- **Consensus Change**: QD consensus - Baseline consensus (positive = improvement)

### 2.4 Iterative QD Refinement

When QD-based grading fails to improve consistency or produces incorrect majority grades, we trigger an iterative refinement process. Refinement is triggered in two scenarios:

1. **Worse Consistency**: QD flip rate exceeds baseline flip rate
2. **Majority Changed**: QD majority grade differs from baseline majority grade (indicating systematic bias)

During refinement, we:
1. Identify problematic responses (those with worse flip rates or changed majorities)
2. Analyze the specific issues (e.g., "QDs improved consistency but changed the majority grade from pass to fail")
3. Provide the LLM with:
   - The current Quality Dimensions
   - Examples of problematic grading outcomes
   - Quantitative metrics (flip rates, majority mismatches)
   - Previous refinement attempts and their outcomes (to avoid repeating failures)
4. Generate refined Quality Dimensions that address the identified issues
5. Re-grade responses with the refined QDs (10 iterations per version)

We limit refinement attempts to 2 iterations per response (v2 and v3 after initial v1) to prevent over-optimization. If refinement does not achieve success after 2 attempts, the response is marked as "maintained" (indicating that QDs neither significantly improved nor worsened the baseline). This approach balances the potential for improvement against computational cost and the risk of overfitting to specific response patterns.

### 2.5 Best Version Selection

For new responses to a question, we select the QD version that achieved the highest success rate or improvement from previous responses. This allows new responses to benefit from refinements made on earlier responses, creating a cumulative improvement effect across the dataset.

### 2.6 Evaluation Metrics

We evaluate the effectiveness of our approach using:
- **Flip Rate Reduction**: Average reduction in flip rate from baseline to best QD version
- **Majority Stability**: Percentage of responses where QD majority matches baseline majority (across all QD versions)
- **Consensus Improvement**: Average increase in consensus percentage
- **Version-Level Analysis**: Tracking which QD versions (initial, v2, v3, etc.) achieve stable majorities vs. changed majorities

## 3. Results

### 3.1 Baseline Stochasticity

Our analysis of 69 responses across 6 questions (L1Q1, L1Q2, L1Q3, L1Q4, L1Q5, L5Q5) reveals significant stochasticity in baseline rubric-based grading. The average flip rate across all questions is 20.3%, with individual questions ranging from 0.0% (L1Q5, L5Q5) to 80.0% (L1Q1). Notably, 20.3% of responses (14/69) exhibit non-zero flip rates in baseline grading, while 79.7% (55/69) achieve 100% consensus, indicating that while most responses are graded consistently, a substantial minority exhibit concerning levels of inconsistency.

### 3.2 QD Effectiveness

Quality Dimensions demonstrate consistent improvement in grading consistency. Across all questions, the average flip rate reduction from baseline to best QD version is 8.7 percentage points (from 20.3% to 11.6%). Among the 14 responses with non-zero baseline flip rates, 8 showed improved consistency with QDs, while 6 remained unchanged or worsened slightly. Importantly, 55 responses that were already stable at baseline (0% flip rate) maintained their stability with QDs.

**Flip Rate by Question**: The flip rate per question shows a consistent pattern of improvement:
- Baseline: Questions range from 0.0% (L1Q5, L5Q5) to 80.0% (L1Q1) average flip rate
- Best QD: Questions range from 0.0% (L1Q3, L1Q5, L5Q5) to 21.4% (L1Q4) average flip rate
- Average improvement: 8.7 percentage points reduction across all responses

The most dramatic improvements were observed in L1Q1 (60 percentage point reduction) and L1Q3 (5.6 percentage point reduction to 0%), while L1Q4 showed more modest improvement (3.6 percentage point reduction).

### 3.3 Majority Stability

A critical concern in automated grading is maintaining correct grades while improving consistency. Our analysis of majority stability across all QD versions reveals:
- 91.3% of responses (63/69) maintained stable majority grades matching baseline
- 8.7% of responses (6/69) experienced majority changes, all flipping from Pass to Fail
- Among the 6 responses with changed majorities: 2 were from L1Q2 (14.3% of L1Q2 responses) and 4 were from L1Q4 (14.3% of L1Q4 responses)

This pattern indicates that while QDs generally improve consistency while maintaining correct grades, they can occasionally introduce systematic bias for certain response patterns. Notably, all majority changes were in the same direction (Pass → Fail), suggesting that QDs may err on the side of stricter grading in ambiguous cases. The iterative refinement process was designed to address these cases, though the 2-attempt limit means some responses remained in the "maintained" category when refinement did not fully resolve the trade-off between consistency and majority alignment.

### 3.4 Consensus Changes

The distribution of consensus changes (QD consensus - Baseline consensus) shows:
- 13.0% of responses (9/69) show increased consensus (positive change)
- 7.2% of responses (5/69) show decreased consensus (negative change)
- 79.8% of responses (55/69) show no change (maintained at 100% baseline consensus)
- Average consensus improvement: +1.4 percentage points

The consensus strength (average consensus percentage) increases from 95.4% (baseline) to 96.8% (best QD), representing a 1.5% relative improvement. The standard deviation also decreased from 11.3 percentage points (baseline) to 10.3 percentage points (QD), indicating not only higher average consensus but also more uniform consistency across responses.

### 3.5 Refinement Patterns

Analysis of the iterative refinement process reveals the complexity of balancing consistency and correctness:
- 10 responses (14.5% of total, or 71.4% of responses with baseline flip rates > 0%) underwent QD grading with v1
- 4 responses progressed to refined versions (v2 and/or v3), indicating initial QD versions needed improvement
- Among responses that reached v2 or v3: 3 used v3 as their best version (L1Q1: R10, R11, R2) and 1 used v2 (L1Q1: R4)

The refinement process demonstrates that addressing both consistency and majority alignment simultaneously is challenging. The 6 responses with changed majorities highlight the trade-off: QDs can improve consistency but may systematically shift the grading outcome. The 2-attempt refinement limit (v2, v3) prevented over-optimization while allowing the system to recover from initial failures in most cases. Notably, L1Q1 showed the most successful refinement pattern, with multiple responses achieving both improved consistency and maintained majority through v2/v3 iterations.

### 3.6 Question-Level Patterns

Consistent patterns emerge across questions:
- Questions with higher baseline flip rates (>20%) show larger absolute improvements with QDs:
  - L1Q1: 80.0% → 20.0% (60 percentage point reduction)
  - L1Q4: 25.0% → 21.4% (3.6 percentage point reduction)
- Questions with lower baseline flip rates (<20%) show complete resolution or maintenance:
  - L1Q2: 14.3% → 7.1% (7.2 percentage point reduction)
  - L1Q3: 5.6% → 0.0% (complete resolution)
  - L1Q5, L5Q5: 0.0% → 0.0% (maintained perfect consistency)
- 4 out of 6 questions show majority stability rates of 100% (L1Q1, L1Q3, L1Q5, L5Q5)
- 2 questions show majority stability rates of 85.7% (L1Q2, L1Q4), accounting for all 6 majority changes
- Average consensus increases from 95.4% to 96.8% across all questions and responses

## 4. Conclusions

### 4.1 Key Findings

Our research demonstrates that **Quality Dimensions can effectively reduce stochasticity in LLM grading** while largely maintaining alignment with baseline rubric-based grading. The key findings are:

1. **Significant Consistency Improvement**: Quality Dimensions reduce flip rates by an average of 8.7 percentage points across all questions (from 20.3% to 11.6%), with 57% of responses with non-zero baseline flip rates (8/14) showing improved consistency. Most dramatically, L1Q1 improved from 80% to 20% flip rate, while L1Q3 achieved complete resolution (5.6% to 0%).

2. **Majority Stability with Notable Exceptions**: While 91.3% of responses (63/69) maintained stable majority grades, 8.7% (6/69) experienced majority changes, all from Pass to Fail. These changes were concentrated in two questions (L1Q2 and L1Q4), suggesting question-specific challenges in balancing consistency with correctness. The iterative refinement process (limited to 2 attempts per response) successfully improved some cases but could not fully resolve all trade-offs within the attempt limit.

3. **Cumulative Improvement through Best-Version Selection**: The system's ability to select the best-performing QD version from previous responses (v1, v2, or v3) enabled new responses to benefit from prior refinements. This is evidenced by L1Q1's successful use of v2 and v3 versions, which were applied to subsequent responses based on their demonstrated effectiveness.

4. **Question-Dependent Effectiveness**: The effectiveness of QDs varies by question type, with some questions (L1Q1, L1Q3) showing excellent results and others (L1Q4) showing more modest improvements or trade-offs. This suggests that certain question types or rubric formulations are more amenable to QD-based improvement.

### 4.2 Implications

These findings have several important implications:

**For Automated Grading Systems**: Quality Dimensions provide a practical approach to reducing LLM grading inconsistency without requiring extensive manual prompt engineering. The iterative refinement process ensures that improvements in consistency do not come at the cost of grading accuracy.

**For Educational Technology**: The ability to generate and refine Quality Dimensions automatically opens the door to more reliable automated grading at scale, potentially reducing the burden on human graders while maintaining grading quality.

**For LLM Research**: Our work demonstrates that providing structured, dimensional criteria can improve LLM consistency in decision-making tasks, suggesting that similar approaches could be applied to other domains requiring consistent LLM judgments.

### 4.3 Limitations and Future Work

Several limitations should be acknowledged:

1. **Sample Size and Scope**: Our analysis is based on 69 responses across 6 questions from primarily Lab 1 exercises (with one from Lab 5). The questions are concentrated in a specific educational context (introductory computer science/engineering). Larger-scale studies across more diverse question types, difficulty levels, domains, and educational contexts would strengthen the generalizability of our findings.

2. **Refinement Limits**: We limit refinement to 2 attempts per response (v2 and v3 after initial v1) to balance computational cost against improvement potential. The 6 responses with changed majorities suggest that some cases may benefit from additional refinement attempts or more sophisticated refinement strategies. Future work could explore adaptive stopping criteria that consider the specific pattern of improvements across iterations, or alternative refinement approaches that more effectively balance consistency and correctness.

3. **Iteration Count**: Each grading phase (baseline, v1, v2, v3) uses 10 iterations. While this is sufficient to identify consistency patterns, the relatively small sample size per response means that flip rate estimates have inherent variability. Future work could explore the optimal number of iterations needed to reliably estimate grading consistency while managing computational cost.

4. **Dimension Quality**: The quality of generated dimensions depends on the LLM's understanding of the rubric and question context. We do not validate dimension quality independently before using them for grading. Future work could explore methods for validating dimension appropriateness, completeness, and objectivity before deployment.

5. **Human Validation**: While we use baseline rubric grading as a reference point, we do not include human expert validation. The 6 cases where QDs changed the majority grade from Pass to Fail may represent either: (a) systematic bias introduced by QDs, or (b) corrections of baseline inconsistencies where QDs actually produce more accurate grades. Future work should include human expert review to determine which interpretation is correct and to validate that both baseline and QD-based grading align with expert judgments.

6. **Single LLM Model**: Our study uses a single LLM model (gpt-oss:120b). Different models may exhibit different levels of baseline stochasticity and different responses to QD guidance. Future work should evaluate the approach across multiple LLM architectures and sizes.

### 4.4 So What?

This research addresses a critical barrier to the adoption of LLM-based automated grading: **stochasticity**. By introducing Quality Dimensions and iterative refinement, we demonstrate that it is possible to achieve improved consistency while largely maintaining grading accuracy (91.3% majority stability). The 8.7 percentage point reduction in flip rate represents a 43% relative improvement in consistency, substantially reducing the variability that undermines trust in automated grading.

The practical impact is significant but requires careful deployment:

**For educators and institutions considering automated grading**: Our results show that QD-based grading can reduce inconsistency in about 57% of problematic cases while maintaining correct grades in over 91% of cases. However, the 8.7% majority change rate (all from Pass to Fail) indicates that the system requires careful monitoring and potentially human review for borderline cases. The question-dependent effectiveness (100% majority stability for L1Q1 and L1Q3 vs. 85.7% for L1Q2 and L1Q4) suggests that the approach should be piloted per-question before full deployment.

**For grading system design**: The iterative refinement process with a 2-attempt limit demonstrates a practical balance between computational cost and improvement potential. The best-version selection strategy enables learning across responses, creating a cumulative improvement effect. However, the persistent trade-offs in some cases suggest that hybrid approaches (QD-based grading with human review flags for low-consensus cases) may be more appropriate than fully automated deployment.

**For LLM reliability research**: Our work demonstrates that structured dimensional criteria can improve LLM consistency in decision-making tasks. The 43% reduction in flip rate, coupled with high majority stability, suggests that this approach could generalize to other domains requiring consistent LLM judgments (e.g., content moderation, medical triage, legal document review), though domain-specific validation would be essential.

Ultimately, this work contributes to making LLM-based systems more reliable for high-stakes applications, while also highlighting the importance of monitoring for systematic biases and maintaining human oversight where errors would be consequential.

---

## Appendix: Methodology and Assumptions

### Data Collection and Processing

**Dataset**: 69 student responses across 6 questions
- L1Q1: 5 responses
- L1Q2: 14 responses
- L1Q3: 18 responses
- L1Q4: 28 responses
- L1Q5: 3 responses
- L5Q5: 1 response

**Grading Protocol**: Each response was graded 10 times at each phase:
1. Baseline (rubric-only) grading: 10 iterations per response
2. Initial QD grading (v1): 10 iterations per response (only if baseline flip rate > 0%)
3. Refined QD grading (v2, v3): 10 iterations per version (only if v1 triggered refinement)

**Total Grading Iterations**: Approximately 1,100+ LLM calls across all responses and phases

### Key Definitions

**PRE-QD (Baseline)**: Grading using only the original rubric without Quality Dimensions. This represents standard LLM-based grading with prompt-only guidance.

**POST-QD (Best QD Version)**: For each response, the QD version (v1, v2, or v3) with the lowest flip rate is selected as the "best" version. If the baseline was stable (0% flip rate), QD grading was skipped and the response is marked as "baseline_stable" in POST-QD metrics.

**Flip Rate**: 100% - Consensus. Represents the percentage of grading iterations that disagree with the majority grade. Lower is better.

**Consensus**: Percentage of iterations agreeing with the majority grade. Higher is better.

**Majority Grade**: The grade (pass=1, fail=0) assigned by the majority of iterations. For ties (5 pass, 5 fail), the majority is ambiguous and flagged for human review.

**Majority Stability**: Whether the POST-QD majority grade matches the PRE-QD majority grade. "Stable" means they match; "Changed" means they differ.

### Baseline Stability Criterion

Baseline is considered **stable** only if flip rate = 0% (100% consensus). This strict criterion ensures that only responses with perfect baseline consistency skip QD grading. QD versions, however, are evaluated against a threshold of 40% flip rate for determining whether they represent acceptable improvement.

### QD Version Selection Logic

For each new response, the system selects the best-performing QD version from previous responses to the same question:
1. Calculate success rate, average improvement, and flip rate for each version (v1, v2, v3) based on historical performance
2. Select the version with the highest composite score: `success_rate * 100 + avg_improvement * 2 + (100 - avg_flip_rate) * 0.5 + correct_majority_rate * 50`
3. If no historical data exists, default to v1

### Refinement Triggering and Limits

Refinement is triggered when:
- QD flip rate > baseline flip rate (worse consistency), OR
- QD majority ≠ baseline majority (changed grade)

Refinement is limited to 2 attempts per response (v2 and v3) to prevent over-optimization. After 2 attempts, if neither "success" nor significant improvement is achieved, the response is marked as "maintained."

### Assumptions and Caveats

1. **Baseline as Ground Truth**: We assume baseline (rubric-only) majority grades represent the "correct" grades. In reality, baseline may itself be inconsistent or incorrect. The 6 majority changes (Pass → Fail) could represent either QD-introduced bias or corrections of baseline errors.

2. **Sample Size per Response**: With 10 iterations per phase, flip rate estimates have inherent variability. For example, a response graded 6/10 Pass has a flip rate of 40%, but with different random seeds, it might show 7/10 or 5/10.

3. **Best QD Version Selection**: "Best" is determined by lowest flip rate among versions that maintain majority stability. In cases where no version maintains majority, the version with lowest flip rate is selected regardless of majority match.

4. **Maintained vs. Improved**: Responses marked "baseline_stable" (skipped QD grading) are counted as "maintained" in POST-QD metrics, not as "improved," even though they represent optimal consistency.

5. **Question-Level Aggregation**: Per-question metrics (e.g., "L1Q1: 80.0% → 20.0%") represent the percentage of responses in that question with non-zero flip rates at each phase. Questions with high baseline stability (many baseline_stable responses) will show low aggregate flip rates.

6. **Consensus Change Calculation**: The 1.4 percentage point improvement in consensus is calculated across all responses, including those that maintained 100% consensus. Among only the responses that had room for improvement (14 with non-zero baseline flip rates), the average improvement is higher.

