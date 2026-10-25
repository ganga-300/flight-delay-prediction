# Literature Review

Fourteen sources, each verified against a publisher, DOI (Crossref/OpenAlex), arXiv, PMLR, PLOS,
institutional repository or BTS page on 25 October 2026. BibTeX entries are in
`report/references.bib`. Where only the abstract could be read, the summary says so. "Not
stated" means the abstract does not say, not that the paper omits it.

## Summary table

| # | Source | Data | Model | Metrics | Time / leakage handling | Takeaway for this project |
|---|---|---|---|---|---|---|
| 1 | Carvalho et al. 2021, *Transport Reviews* (systematic review) | Review | Review | — | Not stated (abstract) | Places day-ahead delay classification within data-science flight-delay research |
| 2 | Rebollo & Balakrishnan 2014, *TR-C* | US 2007–2008, 100 most-delayed links | Random Forest | Classification error 19% (2 h horizon, 60 min threshold); median error 21 min | Explicit 2–24 h forecast horizons | Network and historical delay states predict delays; supports past-only history features |
| 3 | Gui et al. 2020, *IEEE T-VT* | ADS-B, weather, schedule, airport | LSTM, Random Forest | Accuracy (RF 90.2%, binary) | Not stated (abstract) | Strong results rely on weather and ADS-B, which our dataset does not have |
| 4 | Yi et al. 2021, *J. Adv. Transp.* | Boston Logan, 2019 | Stacking (KNN, RF, LR, DT, NB; LR meta-learner), SMOTE | Accuracy, precision, recall, F1, ROC-AUC | Not stated (abstract) | Common practice reports accuracy/ROC; we use PR-AUC and calibration instead |
| 5 | Choi et al. 2016, *IEEE/AIAA DASC* | US domestic flights and weather, 2005–2015 | DT, RF, AdaBoost, kNN, with sampling | Accuracy, ROC | Not stated (abstract) | Weather forecasts are central to delay prediction; we lack them |
| 6 | Pyrgiotis et al. 2013, *TR-C* | 34 busiest US airports | Analytical queuing and propagation model | — | Not applicable | Delays propagate and accumulate through the day; motivates time-of-day, congestion and history features |
| 7 | Chen & Guestrin 2016, *KDD* | Benchmarks | XGBoost | — | Not applicable | Method source for XGBoost |
| 8 | Niculescu-Mizil & Caruana 2005, *ICML* | Benchmarks | Many learners; Platt and isotonic calibration | Probability quality | Not applicable | Boosted trees give distorted probabilities; Platt scaling and isotonic regression fix them |
| 9 | Guo et al. 2017, *ICML (PMLR 70)* | Image and text datasets | Neural networks; temperature, Platt, isotonic, histogram binning | ECE, reliability diagrams | Not applicable | Defines ECE with equal-width bins and reliability diagrams |
| 10 | Saito & Rehmsmeier 2015, *PLoS ONE* | Simulations and real data | — | ROC vs precision–recall | Not applicable | Precision–recall is more informative than ROC under class imbalance; justifies PR-AUC as primary |
| 11 | Brier 1950, *Monthly Weather Review* | — | — | Brier score | Not applicable | Origin of the Brier score (metadata only verified) |
| 12 | Kaufman et al. 2012, *ACM TKDD* | Case studies | — | — | Core topic: leakage and learn–predict separation | Supports using only information available at D−1 and past-only features |
| 13 | Bergmeir & Benítez 2012, *Information Sciences* | Synthetic and real time series | Four ML/regression models, six selection procedures | Forecast error | Core topic: CV vs last-block hold-out for time series | Temporal dependence matters for evaluation. Note: it recommends blocked CV rather than rejecting CV outright |
| 14 | Bureau of Transportation Statistics, TranStats data profile | US on-time data | — | — | — | A flight is late if it arrives 15 minutes or more after schedule; defines our target |

## Notes per theme

**Flight delay prediction (1–5).** ML work on flight delays spans regression and classification,
tree ensembles and deep models, and varied horizons and thresholds (systematic review: Carvalho et
al.). The best-performing studies use information this project does not have at decision time:
real-time network delay states (Rebollo & Balakrishnan, 2–24 h ahead), ADS-B trajectories and
weather (Gui et al.), and weather forecasts (Choi et al.). Several report accuracy or ROC-AUC on
imbalanced data, without probability calibration (Gui et al., Yi et al., Choi et al., from their
abstracts). Their split methods could not be checked, so no claim is made about whether their
evaluation was chronological.

**Delay propagation (6).** Delays ripple through airport networks and push demand into the
evening (Pyrgiotis et al.). This matches the steady rise in delay rate through the day in our
training data (10.8% at 05:00 to 40.5% at 21:00) and the role of previous-day delays.

**Methods (7–11).** XGBoost (Chen & Guestrin) is the gradient-boosting model compared here.
Probability calibration with Platt scaling or isotonic regression (Niculescu-Mizil & Caruana; Guo
et al.) addresses distorted probabilities. ECE and reliability diagrams follow Guo et al., and the
Brier score follows Brier. PR-AUC is primary because delayed flights are the minority class (Saito
& Rehmsmeier).

**Leakage and time-aware evaluation (12–13).** Leakage means using information that would not be
legitimately available when predicting (Kaufman et al.). Temporal dependence affects how forecasts
should be evaluated (Bergmeir & Benítez). We use a chronological hold-out because of the D−1
deployment setting and month-to-month changes in delay rates, not because cross-validation is
invalid in general.

## Identified gap

Day-ahead prediction from the published schedule and past-only outcomes, with explicit
leakage controls, a chronological one-decision-per-period protocol, calibrated probabilities and
PR-AUC reporting, is not the focus of the surveyed empirical work. Those studies rely on real-time
or weather information and report ranking or accuracy metrics without calibration. This project
quantifies what a schedule-only, day-ahead model can achieve under those stricter conditions.

## Considered but not cited
- Platt (1999/2000), *Probabilistic outputs for SVMs*. The year could not be confirmed (sources
  disagree), so Platt scaling is cited through Niculescu-Mizil & Caruana and Guo et al.
- Davis & Goadrich (2006), ROC vs PR curves. Not verified; Saito & Rehmsmeier covers the point.
- Kafle & Zou (2016), delay propagation. Metadata confirmed, but the abstract could not be read.
