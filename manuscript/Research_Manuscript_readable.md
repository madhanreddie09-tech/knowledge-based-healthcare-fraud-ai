# Research manuscript — readable text mirror

> This is a text mirror of the author-updated Word manuscript, not the original DOCX/PDF. Figures are in the existing `results` SVG files. Please consult the official Word/PDF file for formatted tables and journal submission.


Knowledge-Based AI and Data-Driven Digital Transformation of Healthcare
Audits: A Capacity-Constrained Decision Policy Benchmark

Short Communication — target: Knowledge-Based Systems | Research draft for DBA 802
Madhankumar Reddy Bathula | Midwest University | Corresponding author: madhanreddie09@gmail.com
Abstract

Healthcare fraud screening is often evaluated by ranking metrics, even though organizations operate under limited

investigation capacity and may wish to incorporate interpretable business rules. This Short Communication evaluates

capacity-aware decision policies using 5,410 labeled providers aggregated from public healthcare insurance claims

and beneficiary data. Two supervised models and an explicit, train-fold-derived rule trigger are assessed with three

repeats of five-fold stratified provider-disjoint cross-validation. At capacities of 5%, 10%, and 20%, logistic

regression captures mean 42.7%, 64.1%, and 85.2% of positive dataset labels, respectively; corresponding random-

forest recall is 41.8%, 63.7%, and 85.1%. We then ablate the minimum quota reserved for providers satisfying at least

one transparent rule. At 10% capacity, quotas through 40% leave selection and recall unchanged, whereas reserving

every slot for rule-positive providers reduces mean positive-label recall to 43.3%. These findings show why rule-

aware governance should report selection trade-offs rather than assume that rule inclusion improves detection.

Because the dataset label is PotentialFraud rather than adjudicated fraud, and neither the rules nor the results have

external or temporal validation, the contribution is an auditable retrospective benchmarking protocol, not a deployed

fraud-detection method. In an additional exploratory state-grouped internal stress test, 10% capacity recall was

64.08%, but shared beneficiaries prevent treating that result as externally independent.

Keywords: knowledge-based systems; artificial intelligence; data analytics; digital transformation; decision

intelligence; audit capacity; healthcare fraud

1. Introduction

Digital transformation of health-insurance audit operations requires connecting data-driven risk rankings to

constrained human decisions. Prior healthcare-fraud studies span reviews, probabilistic models, claims attention

networks, graph learning and explainable methods [1–7]. These works motivate predictive screening but do not

establish that adding a separate business-rule quota necessarily improves a fixed-capacity provider review queue.

The resource-allocation issue has important precedents. Capacity-aware learning and cost-sensitive classification

explicitly distinguish decision utility from generic prediction metrics [8–10], while explainable healthcare decision

support emphasizes transparent processes and human oversight [11–16]. This study therefore tests an operational

policy proposition, not a claim to have invented cost-constrained classification or expert-validated insurance rules.



                                           DBA 802 • Research draft • 23 October 2026

Contributions are threefold: (i) an executable, provider-level capacity-constrained comparison of logistic regression

and random forest; (ii) a systematic quota ablation showing when a simple rule reservation is nonbinding and when it

displaces high-scoring cases; and (iii) an exploratory grouped-state stress test with quantified cross-provider

beneficiary overlap. Contemporary graph-based transaction-fraud methods illustrate richer alternatives [17–23], which

this compact, interpretable benchmark does not claim to outperform.

2. Data and methodology

2.1 Dataset and leakage controls

We use four publicly supplied training files from the Healthcare Provider Fraud Detection Analysis release [24]:

provider labels, inpatient claims, outpatient claims, and beneficiary attributes. Claims are linked by BeneID and

aggregated within Provider; the resulting 5,410-provider design matrix contains nine numeric descriptors including

claim counts, unique beneficiaries, mean and total reimbursement, reimbursement dispersion, deductible averages,

inpatient share, an illustrative chronic-condition count, and claims per beneficiary. Provider labels are mapped from

Yes/No to 1/0 (positive prevalence approximately 9.35%). The outcome is named PotentialFraud by the source dataset

and must not be described as legally established fraud.

For each of 15 evaluation folds (5 stratified folds × 3 repeats, fixed seed 71), the providers in the held-out fold are

excluded from model fitting. Numeric missing-value imputation is fit using training providers; logistic regression

additionally uses training-fitted standardization and class weighting. A class-weighted random forest uses 160 trees

and a minimum leaf size of five. Provider-level aggregation does not itself learn global parameters across providers,

but a beneficiary may occur under multiple providers, so entity-level dependency remains a limitation. Temporal and

independent institutional splits are unavailable. Because the class is imbalanced, capacity-based precision and recall

are emphasized over raw accuracy [5].

2.2 Explicit policy and knowledge predicates

For provider i, let p_i be a model-based positive-class score and r_i∈{0,1} an interpretable flag. The flag equals one

when *any* of three training-derived 90th-percentile thresholds is exceeded: claims per unique beneficiary, mean

reimbursement, or inpatient fraction. These are reproducible statistical heuristics, not expert-validated compliance

rules or causal indicators.

For review capacity b∈{0.05,0.10,0.20}, we select k_b = ⌈bN_test⌉ providers. The model-only policy M_b selects the

k_b highest scores p_i. For a minimum rule-positive reservation fraction q∈{0,0.1,…,1.0}, policy Q(b,q) first reserves

up to round(qk_b) places for highest-scored r_i=1 providers, then fills the remaining budget with the highest p_i

among unselected providers. Thus |Q(b,q)|=k_b. A separate rule-first comparator sorts by the count of threshold

exceedances, with model scores breaking ties.
                                            DBA 802 • Research draft • 23 October 2026

Capacity recall is R_b = (Σ_{i∈Q(b,q)} y_i)/(Σ_{i∈test} y_i), while capacity precision is P_b = (Σ_{i∈Q(b,q)}

y_i)/k_b. Rule coverage is the fraction of all held-out rule-positive providers selected. All capacity values, rule

thresholds, and quota levels are predeclared; we do not choose the best quota using a held-out fold.

3. Results and policy ablations

Table 1. Measured capacity-aware model comparison (mean across 15 overlapping validation folds).
 Model                      Review capacity                Label recall                 Label precision          Lift
 Logistic                   5%                             42.7%                        78.5%                    8.40×
 Logistic                   10%                            64.1%                        59.5%                    6.36×
 Logistic                   20%                            85.2%                        39.8%                    4.25×
 Random forest              5%                             41.8%                        77.0%                    8.23×
 Random forest              10%                            63.7%                        59.1%                    6.32×
 Random forest              20%                            85.1%                        39.7%                    4.24×
Table 1 shows that both tested models retrieve a substantial share of positive *dataset labels* at fixed capacities. For

logistic regression, reviewing one in ten providers retrieves approximately 64.1% of positives with 59.5% precision

and 6.36× lift over label prevalence. The random forest retrieves 63.7% at the same review capacity. Neither figure is

an estimate of actual detected fraud or monetary savings.




Figure 1. Empirical positive-label capture as review capacity rises. Shaded envelopes show minimum to maximum fold observations, not
confidence intervals.
Table 2. Actual rule-reservation ablation at 10% review capacity; mean of 15 correlated folds.
 Minimum rule quota                Positive-label recall                  Flagged queue share             Providers replaced
 0%                                64.1%                                  51.6%                           0.0
 40%                               64.1%                                  51.6%                           0.0
 50%                               64.2%                                  52.3%                           0.8
 60%                               63.5%                                  59.6%                           8.8
 80%                               58.8%                                  79.8%                           30.8
 100%                              43.3%                                  100.0%                          52.8
 Rule-first                        31.5%                                  100.0%                          66.4
The quota experiment isolates whether imposing a transparent selection constraint changes the decision queue. The

model-only queue already contains on average 51.6% rule-positive cases at 10% audit capacity. Consequently,

minimum reservations of 0–40% have no effect on selections or label recall; at 50%, the mean recall change is

                                                  DBA 802 • Research draft • 23 October 2026

negligible. As reservation rises further, positive-label recall declines (e.g., approximately 58.8% at 80%, 43.3% at

100%). Rule-first sorting gives 31.5% recall. The result is an empirically observed *saturation-to-trade-off* pattern:

symbolic rules can be audited transparently, but forcing them into all cases can be expensive in ranking performance.




Figure 2. Held-out positive-label recall versus minimum allocation to rule-positive providers. Every plotted point is derived from saved
fold-level experiment results; no generative image tools were used.

3.3 Exploratory cross-state transfer stress test

To probe geographic shift without using any additional outcome dataset, an exploratory 5-fold GroupKFold analysis

withheld all providers whose majority beneficiary State belonged to each held-out group. Each of the 52 recorded state

codes was assigned entirely to one test fold. Training-only imputation, scaling and model fitting were repeated

separately. At 10% review capacity, logistic regression achieved mean recall 64.08% (fold range 61.17%–67.00%),

mean precision 59.56%, mean AP 0.6892 and ROC AUC 0.9370. Random forest attained mean recall 64.77%, mean

precision 60.30%, mean AP 0.6967 and AUC 0.9385. These results are descriptive cross-state partition checks, not

out-of-study validation. All folds contain positive labels, with test prevalences from 8.31% to 10.53%.

Crucially, majority beneficiary State is not a verified provider practice location, and held-out providers can share

beneficiaries with training providers. The five folds have respectively 14,025; 13,136; 20,666; 20,001; and 15,539

shared beneficiary identifiers. Therefore this stress test does not establish independence across individuals or

institutions. It was added after the initial benchmark, making it exploratory; we do not use it to assert performance on

a new population. The full fold-level CSV, split construction and reproducibility instructions are supplied separately.

4. Discussion and decision implications

For organizations modernizing manual audits, capacity is a governance parameter, not simply a threshold on model

predictions. Work on limited-resource classification and cost-sensitive review [8–10] shows that evaluation should

                                                 DBA 802 • Research draft • 23 October 2026

reflect actual workload and decision costs. In our data, the model-only ranking at a 10% review budget recovered

64.10% of the dataset’s positive proxy labels on average; no statement about confirmed fraud or savings follows from

that result.

The transparent rule quota functions as an auditable selection policy rather than an expert-certified knowledge base. At

10% capacity, a minimum 40% rule-positive quota does not change the model queue, while an 80% minimum

decreases mean positive-label recall to 58.76%. Thus, symbolic constraints can be redundant or harmful under the

chosen proxy objective. Richer graph, explainable and transaction-history approaches [3–7,17–23] are important

untested comparators; the result is an analytical benchmark, not demonstrated algorithmic superiority.

5. Threats to validity

First, PotentialFraud is a provided binary label with unknown adjudication, sampling protocol, and potential label bias.

Second, all experiments use one public dataset; cross-validation partitions are correlated and may share beneficiary

identities across provider splits. We therefore report descriptive means without treating 15 folds as independent

subjects in a p-value calculation. Third, reimbursement thresholds are statistical percentiles and are not approved

compliance policies. Fourth, the benchmark lacks reviewer costs, varying provider case complexity, investigator

feedback, temporal drift and external evaluation. Fifth, using fixed budget fractions approximates but does not

measure real operational resource constraints. Additional grouped-state transfer results do not resolve independent

validation because cross-fold beneficiaries overlap and the underlying population and labeling source remain the

same. State is only the majority beneficiary state recorded for a provider, not a verified provider address.

Future evaluation should elicit rules from audit professionals, pre-register decision utilities and staffing costs, validate

on chronological cohorts or other insurance organizations, and include adjudicated outcomes. None of those

validations are claimed to have been completed.

6. Conclusion

A resource-constrained evaluation of healthcare claims screening found that a logistic ranking recovered

approximately 64.1% of positive proxy labels when 10% of providers were hypothetically reviewed. A transparent

knowledge-rule reservation up to 40% was redundant with the existing ranked queue; stronger mandatory reservations

reduced positive-label recall. The work provides reproducible evidence for explicitly examining rule/score interactions

during digital transformation of audit workflows, while establishing an important negative result about unvalidated

rule enforcement. Strong external, temporal, operational and expert validation remains required before deployment or

journal-level generalization. An exploratory geographic-grouped split reproduced approximately 64.1% recall at 10%

capacity but cannot substitute for true external population validation.



                                            DBA 802 • Research draft • 23 October 2026

Data availability and reproducibility

Provider-level source: Healthcare Provider Fraud Detection Analysis, Mendeley Data, DOI 10.17632/gsn2hyty37.1

[24]. The public GitHub repository https://github.com/madhanreddie09-tech/knowledge-based-healthcare-fraud-ai

contains Python scripts and summary outputs; this submission package also contains fold-level data and two

programmatically generated vector figures. The underlying raw claims files are not redistributed. The seed and exact

commands appear in the README. The author should independently check execution before formal journal

submission. Additional geographic_stress.py generates geographic_state_heldout_5fold.csv and

geographic_stress_summary.json with five grouped-state validation folds and quantified beneficiary overlap.

Declaration of generative AI and AI-assisted technologies

OpenAI ChatGPT was used to support research design suggestions, executable code drafting, result checking,

visualization, manuscript drafting and language editing. Numerical outcomes reported here were calculated by

executing Python code on the stated dataset. Any author submitting this work must independently inspect and validate

all source data, claims, citations and code, edit the manuscript, and accept full accountability.

Author declarations (require human confirmation)

Author affiliation and contact provided by the author: Midwest University; madhanreddie09@gmail.com. Funding,

competing interests, author CRediT roles and final approval: require truthful author confirmation. No declaration of no

funding or no conflicts is presumed.

References

[1] Fraud detection in healthcare claims using machine learning: A systematic review. Artificial Intelligence in Medicine 160 (2025) 103061.
https://doi.org/10.1016/j.artmed.2024.103061
[2] Using a Bayesian Belief Network to detect healthcare fraud. Expert Systems with Applications 238 (2024) 122241.
https://doi.org/10.1016/j.eswa.2023.122241
[3] Predictive analysis for healthcare fraud detection: Integration of probabilistic model and interpretable machine learning. Information Sciences 719
(2025) 122499. https://doi.org/10.1016/j.ins.2025.122499
[4] Research on medical insurance anti-gang fraud model based on the knowledge graph. Engineering Applications of Artificial Intelligence 134 (2024)
108627. https://doi.org/10.1016/j.engappai.2024.108627
[5] Health insurance fraud detection based on multi-channel heterogeneous graph structure learning. Heliyon 10 (2024) e30045.
https://doi.org/10.1016/j.heliyon.2024.e30045
[6] HeFDN: Heterogeneous graph adversarial generation model for Chinese medical insurance fraud detection. Expert Systems with Applications 299
(2026) 130159. https://doi.org/10.1016/j.eswa.2025.130159
[7] An explainable attention network for fraud detection in claims management. Journal of Econometrics 228 (2022) 244–258.
https://doi.org/10.1016/j.jeconom.2020.05.021
[8] A new perspective on classification: Optimally allocating limited resources to uncertain tasks. Decision Support Systems 179 (2024) 114151.
https://doi.org/10.1016/j.dss.2023.114151
[9] Cost-sensitive thresholding over a two-dimensional decision region for fraud detection. Information Sciences 657 (2024) 119956.
https://doi.org/10.1016/j.ins.2023.119956
[10] Auto insurance fraud detection: Leveraging cost sensitive and insensitive algorithms for comprehensive analysis. Insurance: Mathematics and
Economics 122 (2025) 44–60. https://doi.org/10.1016/j.insmatheco.2025.02.001
[11] Enhancing healthcare decision support through explainable AI models for risk prediction. Decision Support Systems 181 (2024) 114228.
https://doi.org/10.1016/j.dss.2024.114228
[12] Explainable AI for enhanced decision-making. Decision Support Systems 184 (2024) 114276. https://doi.org/10.1016/j.dss.2024.114276
[13] Explainable Artificial Intelligence-Based Decision Support Systems: A Recent Review. Electronics 13 (2024) 2842.
https://doi.org/10.3390/electronics13142842
[14] XAI-Based Clinical Decision Support Systems: A Systematic Review. Applied Sciences 14 (2024) 6638. https://doi.org/10.3390/app14156638

                                                       DBA 802 • Research draft • 23 October 2026

[15] Methodology for Detecting Suspicious Claims in Health Insurance Using Supervised Machine Learning. Future Internet 17 (2025) 584.
https://doi.org/10.3390/fi17120584
[16] Explainable Machine Learning in Critical Decision Systems: Ensuring Safe Application and Correctness. AI 5 (2024) 2864–2896.
https://doi.org/10.3390/ai5040138
[17] Spade+: A Generic Real-Time Fraud Detection Framework on Dynamic Graphs. IEEE Transactions on Knowledge and Data Engineering 36 (2024)
7058–7073. https://doi.org/10.1109/TKDE.2024.3394155
[18] Towards Fraud Detection via Fine-Grained Classification of User Behavior. IEEE Transactions on Big Data 11 (2025) 1994–2007.
https://doi.org/10.1109/TBDATA.2024.3517313
[19] ASA-GNN: Adaptive Sampling and Aggregation-Based Graph Neural Network for Transaction Fraud Detection. IEEE Transactions on Computational
Social Systems 11 (2024) 3536–3549. https://doi.org/10.1109/TCSS.2023.3335485
[20] Parallel Graph Learning With Temporal Stamp Encoding for Fraudulent Transactions Detections. IEEE Transactions on Big Data 11 (2025) 1945–
1958. https://doi.org/10.1109/TBDATA.2024.3499338
[21] XAI for Fraud Detection: An Attention-Based Ensemble of CNNs, GNNs, and Confidence-Driven Gating for Reliable Decision-Making. IEEE
Transactions on Information Forensics and Security 21 (2026) 347–356. https://doi.org/10.1109/TIFS.2025.3646480
[22] A GNN-based fraud detector with dual resistance to graph disassortativity and imbalance. Information Sciences 669 (2024) 120580.
https://doi.org/10.1016/j.ins.2024.120580
[23] BalancerGNN: Balancer Graph Neural Networks for imbalanced datasets: A case study on fraud detection. Neural Networks (2025) article 106926.
https://doi.org/10.1016/j.neunet.2024.106926
[24] Healthcare Provider Fraud Detection Analysis. Mendeley Data, V1 (2024). https://doi.org/10.17632/gsn2hyty37.1




                                                   DBA 802 • Research draft • 23 October 2026