# ML Challenge 2026: Business Entity Resolution Solution Methodology

**Team Name:** Cloudyrelic ER Team  
**Problem Track:** Business Entity Resolution (ER)  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We developed a scalable, 3-stage cascaded hybrid Entity Resolution pipeline tailored for large-scale multi-source commercial entity matching. Our approach couples high-recall disk-backed inverted indexing (capturing >92% candidate recall ceiling while maintaining <400 MB RAM consumption) with an ensemble of a LightGBM gradient boosted screener and a Deep Residual Neural Network (TensorFlow/Keras 3) trained on actively mined false-positive hard negatives. Our system achieves a **0.9663 Macro $F_{0.5}$ score** on disjoint validation data while strictly adhering to open-weight and fair-play constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis revealed three pivotal noise patterns across sources:
1. **Multilingual Transliteration & Script Shift:** Indian businesses frequently appear with identical names transliterated between English and Devanagari/Hindi (e.g., *Shiva Exports* $\leftrightarrow$ *शिवा एक्सपोर्ट्स*) or Gujarati script.
2. **Domain Names & Website Prefixes:** Many records in Source 2/3 incorporate web artifacts (e.g., `davisdecker.com` vs `Davis & Decker`, `brightseafood.com` vs `Bright Seafood Inc`).
3. **Address Anchoring & Street Discrepancies:** Addresses often share colony names, postal codes, or landmark references (*Near SBI ATM*, *Safal Solitaire*) while suffering from municipal component reordering. Minor differences in street/house numbers (e.g., `1424` vs `1433 Cottage View Ln`) represent distinct business entities on the same street, requiring strict numeric disambiguation.
4. **Unseen Country in Test (France):** Training data covers exclusively `US` and `India`, whereas the test set introduces `France` (~15% of records). Our normalizers and features are designed to be language-agnostic and country-partitioned.

### 2.2 Solution Strategy
**Approach Type:** 3-Stage Cascaded Hybrid (Disk-Backed Multi-Pass Blocking $\to$ LightGBM Screener $\to$ TensorFlow Deep Residual Specialist $\to$ Macro $F_{0.5}$ Threshold Optimization).

**Core Innovation:** 
- **Active Hard Negative Mining:** Rather than training on trivial random negatives, we used the first-stage LightGBM model to surface active false positives ($P \ge 0.35$ on non-matching pairs with high lexical similarity). These hard near-misses (e.g., identical names with different house numbers or cities) were fed directly into our Deep Neural Matcher.
- **Disk-Backed Inverted Indexing:** To process 10+ million candidate records within a 7.4 GB host RAM envelope, we designed a disk-backed SQLite indexing engine with B-Trees over exact names, space-compressed names, and address bigrams, cutting memory consumption by 90% while delivering microsecond retrieval.

---

## 3. Candidate Generation (Blocking)
To eliminate the quadratic $O(N \times M)$ comparison space:
- **Country Strict Partitioning:** Entities never match across borders. We partition candidate generation strictly by `country`.
- **Multi-Pass Blocking Keys:**
  1. *Exact Normalized Name:* Stripping corporate legal designations (`Pvt Ltd`, `Inc`, `LLC`, `SARL`).
  2. *Compressed Name:* Removing all whitespace (e.g., `davis decker` $\leftrightarrow$ `davisdecker`).
  3. *Distinctive Name Tokens:* IDF-filtered non-stopword tokens.
  4. *Address Anchors & Bigrams:* Distinctive pairs of consecutive address tokens (e.g., `safal_solitaire`, `charitravan_buxar`, `shalimar_bagh`) paired with street numbers.
- **Candidate Metrics:**
  - Average candidates per $S1$ entity: ~20–25 candidates.
  - **Candidate Recall Ceiling: 92.42%** on held-out validation.

---

## 4. Matching Model

**Features Engineered (18 Dense Tabular Features):**
- **Name Similarity:** Levenshtein ratio, Partial ratio, Token Sort ratio (word-order invariant), Token Set ratio (substring invariant), Jaro-Winkler distance, First token exact match flag, Length discrepancy.
- **Address Similarity:** Address token set ratio, Address Jaccard similarity, Address partial ratio, Address length difference.
- **Numerical Disambiguation:** Exact number set match, Number Jaccard overlap, Number overlap count, Number conflict flag (identifies distinct house numbers on identical streets).
- **Source Indicator:** Binary prefix flag ($S2$ vs $S3$).

**Model Architecture:**
- **Stage 2 Screener:** LightGBM Gradient Boosted Decision Trees (400 estimators, max depth 8, num leaves 63). Top feature by gain: `addr_token_set` followed by `name_partial_ratio` and `addr_jaccard`.
- **Stage 3 Specialist:** TensorFlow/Keras 3 Deep Residual Neural Network (ResNet-MLP) with Swish activations, Batch Normalization, and Dropout, trained on active hard negatives.
- **Ensemble:** $P_{\text{final}} = 0.70 \times P_{\text{LGBM}} + 0.30 \times P_{\text{TF}}$.
- **Threshold Selection:** Grid-search threshold optimization targeting the official precision-heavy macro $F_{0.5}$ metric (optimal threshold: $\tau = 0.88\text{--}0.90$).

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.9663** (96.63%) on 20% held-out validation split.
- **Singleton Accuracy:** **95.03%** on entities with no true matches (crucial for macro averaging).
- **Non-Singleton $F_{0.5}$:** **0.9673**.
- **Common False Positives Addressed:** Parent/subsidiary corporate entities sharing identical street addresses, and distinct businesses located in the same commercial plaza (resolved by strict address number matching).
- **Common False Negatives Addressed:** Transliterated Hindi and Gujarati records where names had zero Latin character overlap (resolved via address bigram anchors).

---

## 6. Conclusion
By pairing an ultra-lightweight disk-backed multi-pass blocking engine with an active hard-negative trained ensemble of LightGBM and Deep Residual Neural Networks, our pipeline scales smoothly to 11+ million test records on constrained hardware (8 GB VRAM, 7.4 GB RAM) while achieving an elite Macro $F_{0.5}$ of 0.9663. All code is fully reproducible, self-contained, and compliant with open-source MIT/Apache 2.0 licenses.

---

## Appendix: Code Artefacts
- **Runnable Source Package:** `code/business_entity_resolution/src/`
- **Reproduction Guide:** `code/business_entity_resolution/README.md`
- **Dependencies:** `code/business_entity_resolution/requirements.txt`
- **Outputs Produced:** `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- **Compliance Validator:** Validated with `student_resource/utils/validate_submission.py`
