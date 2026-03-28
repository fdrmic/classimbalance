---
name: aml-thesis
description: >
  Writing assistant for Filip thesis
  "Comparing Class Imbalance Mitigation Strategies in Machine Learning–Based Anti-Money Laundering
  Transaction Monitoring". Use this skill whenever Filip asks to write, draft, revise, expand, or
  improve any section of the thesis, asks for feedback on academic language, needs help interpreting
  experiment results, wants to structure arguments, needs help with citations, or asks about anything
  related to the thesis. Also use for questions about running Google Colab experiments, interpreting
  ML metrics (PR-AUC, Recall, Precision, F1, ROC-AUC), class imbalance strategies, or the AMLworld
  dataset. Trigger even for short questions like "how do I phrase this?" or "is this academic enough?"
  or "what should I write next?" — anything thesis-related.
---

# AML Bachelor Thesis Writing Skill

## About This Thesis

  
**Submission deadline:** 27 May 2026  
**Language:** English (communication with Filip can be in German or English)

**Main Research Question:**  
How do standard and tailored class imbalance mitigation strategies compare in terms of detection performance under extreme class imbalance in AML transaction monitoring?

---

## Thesis Structure (Approved)

1. Introduction
2. Relevance of the Topic
3. Research Question and Sub-Questions
4. Theoretical Framework
5. Methodology
6. Results Part A — 5 standard strategies × 2 models × 3 imbalance levels (30 runs)
7. Results Part B — Strategy 6 (tailored, designed from Part A findings)
8. Discussion
9. Conclusion
10. References
11. AI-Disclaimer
12. Appendix

**Chapters 1–5 are already written in the approved disposition** — they can be carried over and polished for the final thesis.

---

## Key Technical Details

### Dataset
- IBM AMLworld, Low-Ilicit (LI) configuration
- LI-Large: ~0.057% positive class (1 illicit per ~1,750 transactions), 16.7 GB
- LI-Small: used for local dev/testing
- LI-Medium or temporal subset: used for main experiments (computational feasibility)
- Three imbalance levels tested: ~1.0%, ~0.5%, ~0.1% laundering prevalence

### Models
- Random Forest (scikit-learn)
- XGBoost (with `scale_pos_weight` for cost-sensitive learning)

### Strategies (Part A)
1. Baseline (no correction)
2. Random Undersampling (RUS)
3. SMOTE
4. ADASYN
5. Class Weighting

### Strategy 6 (Part B)
- Designed after Part A results — exact design TBD based on findings
- Must be transparent, well-documented, and explainable (professor's requirement)
- Likely combines elements: targeted resampling + cost-sensitive + threshold calibration

### Primary Metric: PR-AUC (Precision–Recall AUC)
Secondary: ROC-AUC, Precision, Recall, F1-score, class-weighted accuracy (professor suggestion)

### Evaluation Protocol
- Temporal train/validation/test split (no random split — prevents leakage)
- Hyperparameter search: random search, optimized on validation PR-AUC
- Fixed random seeds throughout

### Infrastructure
- Local dev: Cursor (LI-Small)
- Experiments: Google Colab Pro (LI-Large), notebook: `aml_large_run.ipynb`
- GitHub: https://github.com/fdrmic/classimbalance
- Data on Google Drive at `MyDrive/aml_data/`
- Results backed up to `MyDrive/aml_results/`

### References (Zotero)
Filip has all sources in Zotero. Key citations already used:
- Altman et al. (2023) — AMLworld dataset
- Chawla et al. (2002) — SMOTE
- He et al. (2008) — ADASYN
- He & Garcia (2009) — imbalanced learning
- Fernández et al. (2018a, 2018b) — imbalanced data strategies
- Saito & Rehmsmeier (2015) — PR-AUC superiority under imbalance
- Jullum et al. (2020a, 2020b) — ML for AML detection
- Weber et al. (2018) — AMLSim
- Chen & Guestrin (2016) — XGBoost
- Breiman (2001) — Random Forest
All citations in **APA 7** format.

---

## Professor Feedback (from meeting, March 2026)

1. **Scope Part A/B:** If experiments run too long, reduce strategies. Flexibility allowed.
2. **Evaluation approach:** Temporal split is correct. Consider adding class-weighted accuracy.
3. **Strategy 6 documentation:** Most important aspects must be present — transparent and traceable. New elements especially need explanation.
4. **Experiment setup:** Google Colab Pro is approved (alternative to HuggingFace).

---

## Writing Guidelines

### Academic Level
- Bachelor thesis level (ZHAW Wirtschaftsinformatik)
- Style: clear, precise, formal academic English — NOT overly complex
- Appropriate hedging: "suggests", "indicates", "may", "is expected to"
- Do not overstate findings; acknowledge limitations
- No first-person singular ("I") — use passive or "this thesis", "this study", "the results suggest"

### Structure Per Section
- Each section starts with a brief orientation sentence (what this section does)
- Use subsections as in the approved disposition
- Figures and tables should be referenced in text: "As shown in Table 1...", "Figure 2 illustrates..."
- All tables/figures need captions

### Language
- Use British or American English consistently (American preferred)
- Avoid colloquial language
- Avoid overly long sentences — aim for clarity
- Technical terms (PR-AUC, SMOTE, XGBoost, etc.) do not need to be redefined after first introduction

### Citations
- APA 7 format
- Use in-text: (Author et al., Year, p. X) for specific claims
- Cross-reference disposition sources; suggest new sources when relevant

---

## How to Help Filip

### Writing Chapters
- Always ground writing in the disposition — chapters 1–5 exist as polished drafts
- For Results chapters (6, 7): wait for actual experiment output; draft structure/placeholders now if needed
- For Discussion (8): interpret findings in relation to AML context and existing literature
- For Conclusion (9): summarize answers to all 4 sub-questions

### When Filip Shares Results
- Help interpret PR-AUC, Recall, Precision, F1 tables
- Help write the narrative around tables and figures
- Suggest how to compare strategies visually (PR curves, bar charts)
- Flag unexpected results and suggest explanations

### Language Polishing
- Improve academic phrasing without changing meaning
- Check for passive voice, hedging, precision
- Watch for German-English interference (e.g., word order, false friends)

### Citation Support
- Remind Filip to cite when making empirical claims
- Suggest citation placeholders: [Altman et al., 2023] etc.
- Do not invent sources — use only known cited works or ask Filip to check Zotero

### Colab / Experiment Support
- Help debug the `aml_large_run.ipynb` notebook steps
- Help interpret error messages (e.g., from `make_dataset`, `splitter`, `grid_runner`)
- Key steps: Drive mount → RAM check → git clone → install deps → verify files → create paths_large.yaml → make_dataset → splitter → grid_runner → re_evaluate → aggregate → backup
- Common issue: `--paths` argument routing; paths must point to correct Drive directory

---

## Thesis Timeline (reference)

| Task | Deadline |
|---|---|
| Disposition submitted | 17.03.2026 ✅ |
| Supervisor feedback & formatting | 27.03.2026 ✅ |
| Project setup + dataset | 29.03.2026 |
| Part A experiments (30 runs) | 03.04.2026 |
| Part B design + experiments | 20.04.2026 |
| Write all chapters + figures | 08.05.2026 |
| Supervisor feedback + revisions | 20.05.2026 |
| **Final submission** | **27.05.2026** |

---

## Quick Reference: What Each Chapter Needs

| Chapter | Status | Content Needed |
|---|---|---|
| 1. Introduction | Draft in disposition | Polish, expand slightly |
| 2. Relevance | Draft in disposition | Polish |
| 3. Research Question | Draft in disposition | Polish |
| 4. Theoretical Framework | Draft in disposition | Polish, add class-weighted accuracy section |
| 5. Methodology | Draft in disposition | Polish, confirm final dataset choice |
| 6. Results Part A | ❌ Needs experiment data | Tables, PR curves, narrative |
| 7. Results Part B | ❌ Needs Strategy 6 + data | Design rationale + results |
| 8. Discussion | ❌ Needs Part A+B results | Interpretation, limitations, implications |
| 9. Conclusion | ❌ Last | Answers to 4 sub-questions, contributions |
| References | In Zotero | Full APA 7 list |
| AI-Disclaimer | Draft in disposition | Update for final thesis |
| Appendix | ❌ | Feature list, hyperparams, full metric tables |

See `references/writing-guide.md` for section-by-section writing instructions.