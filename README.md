# Can a Transformer Learn Economic Relationships?

**Replication of Gupta & Imas (2025) — "Revisiting the Lucas Critique in the Age of Transformers"**

This project tests whether a causal Transformer, trained purely on simulated macroeconomic data, can internalise the dynamics of a structural New Keynesian (NK) model and predict the effects of out-of-sample policy counterfactuals. The experiment directly engages with the Lucas (1976) Critique: the claim that reduced-form econometric models cannot predict the consequences of policy regime changes because economic agents modify their behaviour in response to policy shifts.

---

## Table of Contents

1. [Background](#background)
2. [The New Keynesian Model](#the-new-keynesian-model)
3. [Parameter Prior](#parameter-prior)
4. [Transformer Architecture](#transformer-architecture)
5. [Benchmark Models](#benchmark-models)
6. [Data Generation](#data-generation)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Key Assumptions](#key-assumptions)
9. [Theoretical Notes & Limitations](#theoretical-notes--limitations)
10. [Code Structure](#code-structure)
11. [Usage](#usage)
12. [Results](#results)

---

## Background

Robert Lucas (1976) argued that correlational relationships estimated from historical data are not *structural* — they are not invariant to policy changes. Agents form expectations about future policy and adjust their behaviour, so the observed relationship between variables shifts when policy shifts. His proposed solution was to build structural models from microfoundations: specify individuals' preferences and firms' objective functions, derive equilibrium conditions, and estimate the resulting parameters.

The structural approach requires a *model*. If the model is misspecified, the predictions are wrong. Transformers, by contrast, do not assume a parametric form for the data-generating process. Recent evidence suggests that Transformer models can learn causal structure from data, raising the question: can a Transformer trained on simulated data from a *known* structural model approximate its dynamics well enough to make useful policy counterfactual predictions?

Gupta & Imas (2025) test this by training a Transformer on data simulated from the NK model and evaluating:
1. **Out-of-sample tracking** of the three key observables.
2. **Impulse-response** prediction under a holdout policy regime.
3. **Comparison against reduced-form benchmarks** (OLS VAR and Bayesian VAR with Minnesota prior).

This repository contains a self-contained implementation of that experiment.

---

## The New Keynesian Model

The linearized 3-equation New Keynesian model consists of:

### Equations

**IS Curve (output gap):**

$$x_t = \mathbb{E}_t[x_{t+1}] - \frac{1}{\sigma}(i_t - \mathbb{E}_t[\pi_{t+1}] - r_t^n)$$

**Phillips Curve (inflation):**

$$\pi_t = \beta \mathbb{E}_t[\pi_{t+1}] + \kappa x_t + u_t$$

**Taylor Rule (interest rate):**

$$i_t = \phi_\pi \pi_t + \phi_x x_t + v_t$$

### Shock Processes (AR(1))

$$r_t^n = \rho_r r_{t-1}^n + \varepsilon_t^r, \quad \varepsilon_t^r \sim \mathcal{N}(0, \sigma_r^2)$$

$$u_t = \rho_u u_{t-1} + \varepsilon_t^u, \quad \varepsilon_t^u \sim \mathcal{N}(0, \sigma_u^2)$$

$$v_t = \rho_v v_{t-1} + \varepsilon_t^v, \quad \varepsilon_t^v \sim \mathcal{N}(0, \sigma_v^2)$$

### Observable Variables

| Variable | Symbol | Description |
|----------|--------|-------------|
| Output gap | $x_t$ | Deviation of output from flexible-price equilibrium |
| Inflation | $\pi_t$ | Quarterly inflation rate |
| Interest rate | $i_t$ | Nominal interest rate set by central bank |

### Shocks

| Shock | Symbol | Interpretation |
|-------|--------|----------------|
| Natural rate | $r_t^n$ | Exogenous shift in the equilibrium real interest rate |
| Cost-push | $u_t$ | Exogenous markup disturbance |
| Monetary policy | $v_t$ | Deviation from Taylor rule |

### Solution Method

The model is solved using **Sims' (2002) method** for linear rational expectations models. The policy function takes the form:

$$y_t = P \cdot s_t$$

where $y_t = (x_t, \pi_t)$ are the endogenous variables, $s_t = (r_t^n, u_t, v_t)$ are the structural shocks, and $P \in \mathbb{R}^{2 \times 3}$ is the policy matrix obtained via the generalized Schur (QZ) decomposition of the system:

$$K = I_3 \otimes A - \rho^\top \otimes B$$

$$A = \begin{bmatrix} 1 + \phi_x / \sigma & \phi_\pi / \sigma \\ -\kappa & 1 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 & 1/\sigma \\ 0 & \beta \end{bmatrix}, \quad
\rho = \text{diag}(\rho_r, \rho_u, \rho_v)$$

The solution is valid only if the **Taylor principle** holds ($\phi_\pi > 1$).

---

## Parameter Prior

Each simulation draw samples parameters independently from a uniform prior. The parameter vector $\theta = (\sigma, \beta, \kappa, \phi_\pi, \phi_x, \rho_r, \rho_u, \rho_v, \sigma_r, \sigma_u, \sigma_v)$ is drawn as:

| Parameter | Symbol | Distribution | Range | Notes |
|-----------|--------|-------------|-------|-------|
| Inverse IES | $\sigma$ | $\mathcal{U}(1.0, 3.0)$ | $[1.0, 3.0]$ | Risk aversion |
| Discount factor | $\beta$ | Fixed | $0.99$ | Quarterly |
| Phillips curve slope | $\kappa$ | $\mathcal{U}(0.05, 0.50)$ | $[0.05, 0.50]$ | Price stickiness |
| Taylor rule — inflation | $\phi_\pi$ | $\mathcal{U}(1.1, 3.0)$ | $[1.1, 3.0]$ | Must exceed 1 (Taylor principle) |
| Taylor rule — output | $\phi_x$ | $\mathcal{U}(0.0, 1.0)$ | $[0.0, 1.0]$ | Output gap response |
| Natural rate persistence | $\rho_r$ | $\mathcal{U}(0.50, 0.95)$ | $[0.50, 0.95]$ | Shock autocorrelation |
| Cost-push persistence | $\rho_u$ | $\mathcal{U}(0.30, 0.90)$ | $[0.30, 0.90]$ | Shock autocorrelation |
| Policy shock persistence | $\rho_v$ | $\mathcal{U}(0.30, 0.90)$ | $[0.30, 0.90]$ | Shock autocorrelation |
| Natural rate volatility | $\sigma_r$ | $\mathcal{U}(0.005, 0.030)$ | $[0.005, 0.030]$ | Shock std. dev. |
| Cost-push volatility | $\sigma_u$ | $\mathcal{U}(0.001, 0.015)$ | $[0.001, 0.015]$ | Shock std. dev. |
| Policy shock volatility | $\sigma_v$ | $\mathcal{U}(0.001, 0.015)$ | $[0.001, 0.015]$ | Shock std. dev. |

**Justification:** These ranges span empirically plausible values estimated from US data. $\beta = 0.99$ is fixed because it is tightly identified and has negligible prior uncertainty at quarterly frequency. The prior is chosen to generate diverse economic dynamics — from aggressive inflation targeting to mild responses — so the model encounters a wide range of regimes during training.

---

## Transformer Architecture

### Input Representation

The model receives a time series of length $T$ where each token at time $t$ concatenates:

$$\text{input}_t = [\underbrace{\theta}_{11 \text{ dims}}, \underbrace{(\varepsilon_t^r, \varepsilon_t^u, \varepsilon_t^v)}_{3 \text{ dims}}, \underbrace{(x_{t-1}, \pi_{t-1}, i_{t-1})}_{3 \text{ dims}}] \in \mathbb{R}^{17}$$

The parameters $\theta$ are time-invariant (repeated at every position). The model must learn to map this 17-dimensional input to the current observable triplet $(x_t, \pi_t, i_t)$.

### Model Specification

| Component | Value |
|-----------|-------|
| Input dimension | 17 |
| Output dimension | 3 |
| Model dimension ($d_{model}$) | 64 |
| Attention heads | 4 |
| Transformer layers | 3 |
| Feedforward dimension | 256 |
| Activation | GELU |
| Dropout rate | 0.1 |
| Positional encoding | Sinusoidal (max length 200) |
| Attention mask | Causal (lower-triangular) |
| Total parameters | ~151,000 |

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-3}$ |
| Weight decay | $1 \times 10^{-4}$ |
| Batch size | 128 |
| Epochs | 100 (with early stopping after 10 no-improve) |
| LR schedule | Cosine annealing to $1 \times 10^{-5}$ |
| Loss function | Mean squared error (MSE) |
| Gradient clipping | max norm $= 1.0$ |
| Mixed precision | Automatic (AMP) on CUDA |
| Compilation | `torch.compile` on CUDA (PyTorch ≥ 2.0) |

### Autoregressive Forecasting

At inference time, multi-step forecasts are generated autoregressively. The model first processes the full context sequence of length $T_{context}$ to obtain a prediction of the last context observable $\hat{y}_{T_{context}-1}$. This prediction is then fed as the lagged observable for the next forecast step:

1. Compute $\hat{y}_{T_{context}-1}$ from context.
2. For $h = 0, \dots, H-1$:
   - Construct input token: $[\theta, \text{shocks}_{T_{context}+h}, \hat{y}_{T_{context}+h-1}]$
   - Append to sequence
   - Run full forward pass (causal attention over growing sequence)
   - Extract last position output as $\hat{y}_{T_{context}+h}$
   - Use prediction as lag for next step

Future shocks are provided during evaluation (drawn from true innovations in the test data) and default to zeros for pure counterfactual exercises.

---

## Benchmark Models

Two reduced-form benchmarks are implemented for comparison. Both are estimated *per simulation* on the 150-period test trajectory.

### OLS Vector Autoregression (VAR)

$$\mathbf{y}_t = \mathbf{c} + \sum_{\ell=1}^{p} \mathbf{B}_\ell \mathbf{y}_{t-\ell} + \mathbf{u}_t, \quad \mathbf{u}_t \sim \mathcal{N}(0, \Sigma)$$

- Lag order $p$ selected by **AIC** from $\{1, \dots, 8\}$.
- Estimated equation-by-equation via OLS.
- Impulse response functions computed via **Cholesky decomposition** of $\Sigma$.

### Bayesian VAR (Minnesota Prior)

$$\text{vec}(\mathbf{B}) \mid \Sigma \sim \mathcal{N}(\text{vec}(\mathbf{B}_0), \Sigma \otimes \mathbf{V}_0)$$
$$\Sigma \sim \mathcal{IW}(S_0, \nu_0)$$

- Lag order fixed at $p = 4$.
- **Minnesota prior**: coefficients on own first lag centred at 1 (random walk), all others at 0.
- Hyperparameters: $\lambda_1 = 0.2$ (overall tightness), $\lambda_2 = 0.5$ (cross-variable shrinkage), $\lambda_3 = 1.0$ (lag decay).
- Posterior via conjugate Normal-inverse-Wishart update.
- **The prior variance on own first lag**: $\lambda_1^2 / \hat{\sigma}_i^2$, where $\hat{\sigma}_i^2$ is the residual variance from a univariate AR(p) fit.
- **The prior variance on cross-variable lags**: $(\lambda_1 \lambda_2 / \ell^{\lambda_3})^2 \cdot (\hat{\sigma}_i^2 / \hat{\sigma}_j^2)$, where $\ell$ is the lag index.
- Point forecasts use posterior mean coefficients.
- When the posterior covariance matrix is ill-conditioned (condition number $> 10^{12}$), posterior draws are suppressed and the posterior mean is used directly.

---

## Data Generation

**Simulation pipeline:**

1. Draw $N_{total} = 60,000$ parameter vectors from the prior.
2. For each valid draw, simulate $T_{sim} = 200$ periods (burn-in $= 50$).
3. Retain the last 150 periods as the usable trajectory.
4. Partition: $N_{train} = 50,000$, $N_{val} = 5,000$, $N_{test} = 5,000$.

**Feature construction:** For each simulation $i$ and time step $t$:

$$X_{i,t} = [\theta_i, \text{shocks}_{i,t}, y_{i,t-1}] \in \mathbb{R}^{17}$$

where $y_{i,-1} := \mathbf{0}$ (zero lag for first step).

**Normalisation:** All features and targets are standardised to zero mean and unit variance using training-set statistics:

$$\tilde{X} = \frac{X - \mu_X}{\sigma_X}, \quad \tilde{Y} = \frac{Y - \mu_Y}{\sigma_Y}$$

with $\sigma$ clipped to a minimum of $10^{-10}$ to avoid division by zero.

**Caching:** Raw data, normalised data, and normalisation statistics are cached to disk (pickle) so subsequent runs skip the generation step.

---

## Evaluation Metrics

### 1. One-Step-Ahead MSE

For each test simulation, the model predicts $y_t$ given the full information set at time $t$ (parameters, shock at $t$, and true lagged observable $y_{t-1}$). MSE is computed over the last 100 steps (after a 50-step warmup) and averaged across all 5,000 test simulations:

$$\text{MSE}_{1\text{-step}} = \frac{1}{N \cdot (T - 50) \cdot k} \sum_{i=1}^{N} \sum_{t=50}^{T-1} \sum_{j=1}^{k} (\hat{y}_{i,t,j} - y_{i,t,j})^2$$

This is an **ex-post** evaluation (the model has access to the true contemporaneous shock and the true lagged observable).

### 2. Multi-Step-Ahead MSE

For each test simulation, use the first 50 time steps as *context* and forecast $h \in \{1, 4, 8, 12, 20\}$ steps autoregressively. The model receives the **true future shocks** at each forecast step. MSE is computed in the original (unnormalised) scale:

$$\text{MSE}_h = \frac{1}{N \cdot h \cdot k} \sum_{i=1}^{N} \sum_{\tau=0}^{h-1} \sum_{j=1}^{k} (\hat{y}_{i,T_{ctx}+\tau,j} - y_{i,T_{ctx}+\tau,j})^2$$

The VAR and BVAR benchmarks use their own multi-step forecast procedures (iterating forward using the estimated companion form for VAR, posterior draws for BVAR).

### 3. IRF Accuracy

Impulse-response functions are computed for all three shock types, comparing Transformer predictions against the true DSGE IRF (computed analytically from the solved NK model) and against VAR/BVAR IRFs (Cholesky-identified).

For each simulation $i$ and shock type $s$:

$$\text{IRF-MSE}_{i,s,v} = \frac{1}{H+1} \sum_{h=0}^{H} (\hat{\text{IRF}}_{i,s,v}(h) - \text{IRF}^\text{true}_{i,s,v}(h))^2$$

where $H = 20$ quarters is the IRF horizon, $v$ indexes the observable variable, and $\hat{\text{IRF}}$ is the predicted impulse response.

**Sign accuracy** is the fraction of periods (excluding zero-crossings of the true IRF) where the sign of the predicted response matches the true sign:

$$\text{SignAcc}_{i,s,v} = \frac{\sum_{h=0}^{H} \mathbb{I}[(\text{sign}(\hat{\text{IRF}}_h) = \text{sign}(\text{IRF}^\text{true}_h)) \land (\text{IRF}^\text{true}_h \neq 0)]}{\sum_{h=0}^{H} \mathbb{I}[\text{IRF}^\text{true}_h \neq 0]}$$

### 4. Learning Curve (Sample Size Experiment)

The Transformer is re-trained on subsets of the training data of size $N \in \{100, 300, 1000, 3000, 10000, 30000, 50000\}$ and one-step MSE is evaluated on the same test set. VAR and BVAR are *per-simulation* estimators (each test simulation receives its own fit using only its 150 time periods), so their performance is independent of the number of training simulations — they are shown as flat baselines.

---

## Key Assumptions

1. **The true data-generating process is the NK model.** This is the central assumption of the experiment: we know the ground-truth structure and evaluate whether the Transformer can learn it.

2. **The linearized NK model is a valid approximation.** The NK model solved here is the first-order perturbation around a zero-inflation steady state. Nonlinearities (e.g., occasionally-binding zero lower bound) are absent.

3. **$\beta = 0.99$ is fixed across all draws.** The discount factor is tightly identified and its prior uncertainty is negligible at quarterly frequency.

4. **The prior ranges are representative of real economies.** Parameter ranges span empirically plausible values from macroeconometric estimates for the US. Draws that violate the Taylor principle ($\phi_\pi \leq 1$) or yield an unsolvable model (rank-deficient system matrix) are rejected and re-drawn.

5. **The simulation uses a constant burn-in of 50 periods.** This is sufficient to wash out the effect of initial conditions (zero shocks at $t=0$) given the moderate persistence parameters in the prior.

6. **Training, validation, and test draws are independent.** Each simulation is an independent draw from the same prior distribution. The test set represents a *nearby* but unseen parameter regime.

7. **The Transformer has access to structural parameters and shocks during training.** The input token includes $\theta$ (the 11-dimensional parameter vector) and $\varepsilon_t$ (the 3-dimensional innovation vector at time $t$). This is a **more informative** input than what VAR/BVAR receive (lagged observables only), deliberately reflecting the design of Gupta & Imas (2025). A separate experiment with *y-only* inputs (not implemented here) compares against a Kalman filter.

8. **Autoregressive forecasting uses the model's own predictions as lags.** After the first forecast step, the lagged observable is the model's own output — not the true observable. This reflects realistic forecasting conditions where future observables are unknown.

9. **VAR IRFs use Cholesky identification.** The ordering $(i_t, \pi_t, x_t)$ imposes a recursive structure: monetary policy shocks affect all variables contemporaneously, inflation affects output with a lag, and output affects neither contemporaneously. This ordering is arbitrary and has **no economic basis** relative to the true structural shocks in the NK model. The comparison between Cholesky IRFs and true structural IRFs is therefore **apples-to-oranges at the theoretical level**.

10. **BVAR posterior draws are suppressed when the covariance is ill-conditioned.** If the condition number of the posterior covariance matrix $V_{post}$ exceeds $10^{12}$, the Kronecker product $\Sigma_{draw} \otimes V_{post}$ may be singular, causing sampling failures. In this case, posterior draws are skipped and the posterior mean is used directly for point forecasts.

---

## Theoretical Notes & Limitations

### Information Asymmetry Between Transformer and Benchmarks

The Transformer receives $\theta$ (structural parameters) and $\varepsilon_t$ (innovations) as inputs. VAR/BVAR only see lagged observables $\{y_{t-1}, \dots, y_{t-p}\}$. This information advantage may explain part of the Transformer's performance gap. However, note:
- The Transformer does not assume linearity or know the model equations.
- Mapping from $(\theta, \varepsilon_t, y_{t-1})$ to $y_t$ requires learning the policy function $P(\theta)$.
- The reference paper includes an additional experiment (not implemented here) where a *y-only* Transformer competes against a Kalman filter, isolating the contribution of structural information.

### Cholesky IRFs vs. Structural IRFs

The VAR/BVAR IRFs are computed via Cholesky decomposition of the residual covariance matrix, which imposes a recursive contemporaneous ordering. This is a **data-driven** identification scheme with no economic justification. The true NK IRFs are **structurally identified** — each shock corresponds to a well-defined economic disturbance. Comparing these two types of IRFs mixes identification approaches and is not a clean comparison.

### Linear NK Is a Strong VAR Baseline

The linearized NK model implies that observables follow a VARMA process. For empirically calibrated parameters, the MA component decays quickly, so a VAR(p) with $p$ sufficiently large approximates the true dynamics well. This means **VAR is a strong baseline** — the Transformer's task is to match or exceed a model that is already well-suited to the data-generating process.

### Regime Transfer (Lucas Critique)

The test set parameters are drawn from the same prior as the training set. This tests *interpolation* within the prior support — not *extrapolation* to entirely new parameter regimes. The experiment evaluates whether the Transformer generalises to unseen parameter vectors, but always within the same model *class* (same 3-equation NK structure).

### Interpretability

The Transformer is a black box. We can evaluate its predictive accuracy but cannot extract the economic mechanisms it has learned. We cannot say whether it has recovered the NK model equations, learned a different but equally predictive structure, or simply memorised patterns.

---

## Code Structure

```
nk-transformers/
├── run.py              # Main entry point — orchestrates full pipeline
├── src/
│   ├── simulator.py    # NK model solution, data generation, caching
│   ├── model.py        # Causal Transformer architecture
│   ├── train.py        # Training loop, dataset, AMP & compile
│   ├── evaluate.py     # All evaluation metrics (one-step, multi-step, IRF)
│   ├── benchmarks.py   # VAR and BVAR (Minnesota prior) implementations
│   ├── plots.py        # Figures and result tables
│   └── __init__.py     # Package exports
├── results/
│   ├── cache/          # Pre-processed data (pickle)
│   ├── checkpoints/    # Saved model weights
│   └── figures/        # Generated figures (.png)
├── readings/
│   └── Litterman1986.pdf  # Original Minnesota prior paper
├── REFERENCE.md        # Full text of Gupta & Imas (2025)
├── requirements.txt    # Python dependencies
└── README.md           # This document
```

### Key Functions

| File | Function | Purpose |
|------|----------|---------|
| `simulator.py` | `solve_nk_model` | Solve NK model via Sims' method, return policy matrix $P$ |
| `simulator.py` | `generate_datasets` | Sample parameters, simulate, build features, normalize |
| `simulator.py` | `load_and_prepare` | Load cached data or generate fresh |
| `model.py` | `NKTransformer` | Causal Transformer with 17-dim input, 3-dim output |
| `model.py` | `autoregressive_forecast` | Multi-step forecast using model's own predictions |
| `benchmarks.py` | `select_var_order` | Select VAR lag by AIC |
| `benchmarks.py` | `bvar_minnesota_fit` | Fit BVAR with conjugate Normal-IW posterior |
| `evaluate.py` | `evaluate_one_step_mse` | One-step MSE for Transformer |
| `evaluate.py` | `evaluate_multistep_mse` | Multi-step MSE for Transformer (with true future shocks) |
| `evaluate.py` | `evaluate_irf_accuracy` | IRF-MSE and sign accuracy for all models |
| `evaluate.py` | `compute_dsge_irf` | True structural IRF from solved NK model |

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (generate data → train → evaluate → figures & tables)
python run.py

# Use existing checkpoint, skip training
python run.py --skip-train

# Skip slow VAR/BVAR benchmarks (Transformer-only evaluation)
python run.py --skip-benchmarks

# Custom training settings
python run.py --epochs 200 --batch-size 256 --device cpu

# Adjust sample sizes for learning curve experiment
python run.py --sample-sizes "100,500,1000,5000,10000"

# All options
python run.py --help
```

---

## Results

*(Results will be populated after a full pipeline run with benchmarks enabled.)*

