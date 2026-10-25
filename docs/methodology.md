# Methodology (Theoretical Rigor)

The theory behind each modelling choice, tied to what this project did. Decision IDs refer to
`docs/decision_log.md`.

## 1. Problem and decision time

- **Target:** $y = 1$ if a flight arrives at least 15 minutes late (US BTS on-time definition),
  else $0$ (D-010). Cancelled flights and flights without an arrival record are excluded, so the
  model estimates $P(\text{delay} \geq 15 \text{ min} \mid \text{flight operates and arrives})$
  (D-011).
- **Decision time:** end of day $D-1$ (D-012). A feature for a flight on date $D$ may use the
  published schedule (including all of $D$'s flights) and outcomes of earlier flights known before
  $D$. Nothing from day $D$ itself.

## 2. Why a chronological split, and why random K-fold leaks

Flights on nearby days share weather, congestion and delay waves (daily delay rate lag-1
autocorrelation 0.356 in Jan–Aug, D-101). With a random split, a model trains on flights from the
same day as the flights it is tested on, and history features can include outcomes from after the
test flight. Both make offline scores look better than any real deployment.

The split is therefore chronological (D-013): train January–August, validation
September–October, test November–December. Each period has one job, so no period scores a
decision it helped make:

| Period | Job |
|---|---|
| January–July | Fit candidate models |
| August | Tune hyperparameters; XGBoost early stopping (D-201) |
| January–August | Retrain the chosen model (D-201) |
| September | Fit probability calibration (D-203) |
| October | Choose the decision threshold (D-203) |
| November–December | Test, once (D-205) |

## 3. Logistic Regression

**Model.** For features $\mathbf{x}$, the predicted delay probability is
$$p(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$
Each coefficient $w_j$ is the change in log-odds of delay for a one-unit change in $x_j$, holding
the others fixed.

**Training objective** (scikit-learn's parameterisation, L2 penalty):
$$\min_{\mathbf{w}, b} \; \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C \sum_{i} c_{y_i}\,
\Big[-y_i \log p_i - (1 - y_i)\log(1 - p_i)\Big].$$
- **$C$** is the inverse regularisation strength: a small $C$ shrinks coefficients more. The chosen
  model uses $C = 0.01$ (D-202).
- **$c_{y}$** are class weights. With `class_weight="balanced"`, $c_y = n / (2\,n_y)$, so the
  minority (delayed) class counts more. This changes the decision boundary and **inflates
  predicted probabilities**, which is why calibration is required afterwards (D-204).

**Preprocessing.** Numeric features are standardised (zero mean, unit variance), so the L2
penalty treats them on the same scale. Categorical features are one-hot encoded, with unseen
levels ignored. Both are fitted on training data only.

## 4. XGBoost (gradient-boosted trees)

**Additive model.** The score is a sum of $K$ regression trees, $\hat{y}_i = \sum_{k=1}^{K}
f_k(\mathbf{x}_i)$, turned into a probability with $\sigma$.

**Objective.** Each new tree $f_t$ minimises
$$\mathcal{L}^{(t)} = \sum_i \ell\big(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\big) +
\Omega(f_t), \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert \mathbf{w} \rVert^2,$$
where $T$ is the number of leaves and $\mathbf{w}$ the leaf weights. XGBoost uses a second-order
Taylor expansion of the log-loss $\ell$ (gradients $g_i$ and Hessians $h_i$), which gives the
optimal leaf weight $w_j^* = -\sum_{i \in j} g_i / (\sum_{i \in j} h_i + \lambda)$ and a closed-form
split gain.

**Controls used here (D-201):**
- `max_depth` (3 or 5): limits interactions per tree.
- `learning_rate` $\eta$ (0.03 or 0.1): shrinks each tree's contribution, trading more trees for
  less overfitting.
- `subsample` and `colsample_bytree` (0.8): random rows and columns per tree, for variance
  reduction.
- **Early stopping** on August: stop adding trees when August PR-AUC has not improved for 50
  rounds.
- `scale_pos_weight` fixed at 1.0, not tuned.

## 5. Metrics

**PR-AUC (average precision), primary.** With precision $P = \frac{TP}{TP+FP}$ and recall
$R = \frac{TP}{TP+FN}$ across thresholds,
$$\text{AP} = \sum_n (R_n - R_{n-1})\, P_n.$$
A random ranking scores the delay rate (e.g. 0.258 on the test set), so PR-AUC is read against that
baseline. It focuses on the delayed (positive) class, which is the minority (about one flight in
four), so it is more informative than ROC-AUC under imbalance.

**ROC-AUC.** The probability that a randomly chosen delayed flight gets a higher score than a
randomly chosen on-time flight. Random = 0.5. It is insensitive to the class balance.

**Brier score.** Mean squared error of probabilities, $\frac{1}{N}\sum_i (p_i - y_i)^2$. Lower is
better; it rewards both ranking and calibration.

**Expected Calibration Error (ECE).** Split predictions into $M = 10$ equal-width bins $B_m$:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\big|\,\overline{y}_{B_m} - \overline{p}_{B_m}\big|.$$
It measures how far predicted probabilities are from observed delay rates. A reliability diagram
plots $\overline{y}_{B_m}$ against $\overline{p}_{B_m}$; perfect calibration lies on the diagonal.

## 6. Probability calibration

A model can rank well but give probabilities that are systematically too high or too low. The
chosen model is trained with balanced class weights, so its raw probabilities are too high
(October mean 0.40 vs a 0.18 delay rate, D-204).

- **Platt scaling (sigmoid):** fit $p_{\text{cal}} = \sigma(a\,s + b)$ on the model's score $s$,
  with two parameters learned on held-out data. Used here, fitted on September only (D-203).
- **Isotonic regression:** a non-decreasing step function. More flexible but can overfit;
  reported only for comparison.

Calibration is a monotone transformation, so it leaves PR-AUC and ROC-AUC essentially unchanged
and only fixes the probability scale.

**Distribution shift.** Calibration learns the delay rate of the month it is fitted on. September
(13.7% delayed) was calmer than the test period (25.8%), so calibrated probabilities run low on the
test set (0.18 predicted vs 0.26 observed, D-205).

## 7. Choosing the decision threshold

Operations planners act on every flagged flight, so false alarms cost money (D-012). The rule
(D-203): on October only, pick the threshold with the **highest recall subject to precision at
least 0.60** (an assumption, D-015). A flight is flagged when its calibrated probability is at or
above the threshold.

On October no threshold reached 0.60 precision (best about 0.51, D-204). The target was not
lowered, because changing a target after seeing results turns validation into a tuning set for the
target itself.

## 8. Leakage prevention, summarised

- Only features computable at the end of $D-1$ (D-012, `docs/leakage_policy.md`).
- History features count only outcomes known before $D$, using actual arrival time plus a 30-minute
  reporting buffer (D-105).
- Learned transformations (category collapsing, imputation, scaling, encoding) are fitted on
  training data only.
- The test period was physically separated before any work (D-002) and evaluated once, after every
  choice was frozen (D-205).
