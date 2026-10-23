# External Validation — Feasibility and Protocol

> This report documents a proposed evaluation design, **not a completed independent external validation**.


Independent External Validation:
Evidence and Feasibility Audit
DBA 802 | Target: Knowledge-Based Systems | 23 October 2026


Evidence verdict
INDEPENDENT EXTERNAL PREDICTIVE VALIDATION NOT ESTABLISHED. This report
documents the public data sources, compatibility obstacles and a prospective
experimental protocol; it is not an executed external test or proof of generalization.



Existing completed evidence
The supplied Mendeley Data collection contains 5,410 provider-level PotentialFraud labels
with linked inpatient claims, outpatient claims and beneficiary characteristics. Internal
repeated provider-level validation, rule-quota policy ablation and an exploratory state-
grouped stress test have been performed. All still use the same source dataset, and a state-
grouped split permits cross-fold beneficiary overlap. They are not independent external
validation.



Independent external-source candidates

CMS Medicare Physician & Other Practitioners — by Provider
https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-
practitioners

Independent annual provider-level payment/utilization aggregates. Public feature
definitions do not supply claim-level inpatient fractions, patient-level chronic-condition
summaries or the entire original input schema.

HHS OIG List of Excluded Individuals/Entities (LEIE)
https://oig.hhs.gov/exclusions/leie-database-supplement-downloads/

Government-maintained exclusion history; match may be possible by NPI for some
records. Exclusion grounds vary and are not equivalent to adjudicated fraud. Dates, NPI
matching coverage, re-instatement, and false matches must be audited.

Original study source (not an external cohort)
https://doi.org/10.17632/gsn2hyty37.1

This supplies only the existing test population; redistributed copies and Kaggle mirrored
files are not independent evidence.



Feature and label compatibility
Variable / requirement         Existing model                  CMS + LEIE independent
                                                               source

Provider identifier            Synthetic-style Provider        NPI (not directly linkable)
                               code
Claims per beneficiary         Required; aggregated from       Not directly available as
                               linked claims                   same definition
Mean claim                     Claim-level reimbursement       Provider-level aggregate
reimbursement                  aggregate                       with different definition
Inpatient claim fraction       Required                        Not directly available in
                                                               physician Part B summary
Chronic condition mean         Required beneficiary join       Not available in provider
                                                               aggregate
Fraud target                   PotentialFraud proxy            Exclusion status; different
                                                               target and time
                                                               dependence



Minimum valid external protocol
1. Freeze model, fitted preprocessing, features and review policy before accessing
independent outcomes. Record code commit, software versions, split dates and all model
hyperparameters.

2. Secure a separate healthcare organization or cohort supplying matching claims,
beneficiary attributes, stable provider identifiers and documented sampling. Do not
relabel or engineer an artificial substitute for missing features.

3. Obtain independent chart-/case-review adjudications or a precisely documented
external outcome; define whether the target is potential fraud, confirmed fraud,
administrative exclusion or another proxy.

4. Establish temporal alignment and disjointness at provider and beneficiary levels, report
inclusion/exclusion counts, missing features, outcome prevalence, and any
overlap/leakage.

5. Execute unchanged model on the external population; measure average precision, ROC-
AUC, precision and positive-label recall at 5%, 10%, 20% capacity, calibration and
uncertainty intervals, by relevant subgroups.

6. Recruit qualified fraud-audit/domain experts to review rule interpretation, expected
harms, appeal processes, quota feasibility and usefulness; log independent review and
dated approval.

7. Report all deviations, failed transport cases and confidence limitations before claiming
external transportability or suitability for clinical/financial deployment.



Scope and integrity statement
A model retrained on CMS/LEIE with different input features would constitute a NEW
external-domain experiment, not validation of the original frozen model. Matching a
public dataset replica, a geographic split of the original claims, or an exclusion label with
unverified temporal provenance would not close the current external-validity gap.



Author record and pending declarations
Author: Madhankumar Reddy Bathula | Affiliation: Midwest University | Corresponding
email: madhanreddie09@gmail.com. Funding, competing interests, and authorship
contribution statements need author confirmation. The present report does not assert
their values.



Final recommendation
Suitable to present as an internally validated DBA 802 reproducibility and policy-ablation
study with clearly stated limitations, following author review. For formal Knowledge-
Based Systems submission, MAJOR REVISIONS remain: independent labeled transfer
validation, expert-validated knowledge rules, and novelty relative to established
literature.