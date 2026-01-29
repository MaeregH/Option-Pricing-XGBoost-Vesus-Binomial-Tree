# Pricing American NVDA Options with Binomial Trees and XGBoost

This project compares a **model‑based**, American‑aware options pricer (binomial / trinomial lattice) with a **data‑driven** model (XGBoost) on listed options for NVIDIA (NVDA). The goal is to understand how a structured, risk‑neutral pricing framework stacks up against a flexible machine‑learning approach when both are applied to real American options.

---

## 1. Project Overview

**Objective**

Compare NVDA equity option prices from:

1. **Lattice models** (binomial, optionally trinomial) that explicitly model early exercise for American options.
2. **XGBoost** models trained on observed option quotes and engineered features.

**Key Questions**

- How close do lattice model prices come to market prices across moneyness and maturities?
- Can XGBoost systematically reduce pricing errors relative to the lattice model?
- In which regions (deep ITM/OTM, short/long maturity) does each approach perform best?

**Data**

Historical NVDA option chains (American‑style U.S. equity options), containing at minimum:

- Strike, expiration, option type (call/put)
- Quote date and bid/ask prices

These are aligned with NVDA’s underlying price to construct features and labels.

---

## 2. Methods

### 2.1 Lattice (Binomial / Trinomial) Pricer

**Inputs per option**

- Underlying spot price \(S_0\) at quote time
- Strike \(K\)
- Time to maturity \(T\) (in years)
- Risk‑free rate \(r\) (simple proxy or short‑rate)
- Volatility \(\sigma\) (historical or implied proxy)
- Option type (call/put)

**Binomial Algorithm (American option)**

1. Choose number of steps \(N\), compute \(\Delta t = T/N\).
2. Define up/down factors \(u, d\) and risk‑neutral probability \(p\).
3. Build a recombining tree of stock prices \(S_{n,j}\) for \(n = 0, \dots, N\).
4. At maturity, set values to payoff:
   - Call: \(\max(S_{N,j} - K, 0)\)
   - Put: \(\max(K - S_{N,j}, 0)\)
5. Work backward for \(n = N-1, \dots, 0\):
   - Continuation value:
     \[
     V_{\text{cont}} = e^{-r \Delta t} \left( p V_{n+1,j+1} + (1-p) V_{n+1,j} \right)
     \]
   - Intrinsic value:
     - Call: \(\max(S_{n,j} - K, 0)\)
     - Put: \(\max(K - S_{n,j}, 0)\)
   - American node value: \(V_{n,j} = \max(V_{\text{cont}}, V_{\text{intrinsic}})\)
6. The root node value \(V_{0,0}\) is the model price.

A trinomial tree can be added later by introducing an additional “middle” branch and corresponding probabilities.

### 2.2 XGBoost Model

**Target**

Observed option mid‑price:
\[
\text{mid} = \frac{\text{bid} + \text{ask}}{2}
\]

**Example Feature Set**

- Underlying spot \(S_0\)
- Strike \(K\)
- Moneyness (e.g., \(S_0 / K\) or \(\ln(S_0 / K)\))
- Time to maturity \(T\)
- Risk‑free rate \(r\)
- Volatility proxy (e.g., rolling historical volatility)
- Option type (call/put encoded as 0/1)
- Liquidity features if available: volume, open interest, bid‑ask spread, etc.

**Training / Evaluation**

- Split by date into **train** and **test** sets to avoid look‑ahead bias.
- Train an XGBoost regressor on the training set to predict mid‑prices.
- Evaluate on the test set with:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Optional: Mean Absolute Percentage Error (MAPE)

---

## 3. Workflow

1. **Data Preparation**
   - Load NVDA options dataset.
   - Align option quotes with underlying NVDA prices.
   - Compute:
     - Time to maturity \(T\)
     - Mid‑price target
     - Volatility proxy (e.g., rolling historical volatility)
   - Filter out bad/illiquid quotes (e.g., zero volume with huge spreads).

2. **Lattice Pricing**
   - For each option:
     - Build a binomial tree using chosen \(N\), \(r\), \(\sigma\).
     - Compute the American option price via backward induction.
   - Save lattice model prices alongside market mid‑prices.

3. **XGBoost Modeling**
   - Build feature matrix and train/test split.
   - Train XGBoost on the training set.
   - Generate predictions on the test set.

4. **Comparison & Analysis**
   - Compute error metrics for:
     - Market vs lattice price
     - Market vs XGBoost prediction
   - Slice errors by:
     - Moneyness (deep ITM, ATM, deep OTM)
     - Time to maturity (short vs long)
     - Option type (call vs put)
   - Plot:
     - Error vs moneyness
     - Error vs maturity
     - Error distributions per model

5. **Reporting**
   - Summarize:
     - Overall test errors for each model
     - Where lattice performs well or poorly
     - Where XGBoost improves (or worsens) pricing
   - Discuss:
     - Trade‑offs between theoretical structure (lattice) and flexibility (ML)
     - How American early exercise and dividends may show up in error patterns
     - Limitations: volatility estimates, simple rate assumptions, data quality

---

## 4. Repository Structure

Suggested layout:

```text
.
├── data/
│   ├── nvda_raw.csv
│   ├── nvda_clean.csv
│   └── nvda_with_lattice.csv
├── src/
│   ├── lattice_pricer.py
│   ├── features.py
│   ├── xgb_model.py
│   └── evaluation.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_lattice_pricing.ipynb
│   └── 03_xgboost_model.ipynb
├── figures/
│   ├── error_vs_moneyness.png
│   ├── error_vs_maturity.png
│   └── error_histograms.png
└── README.md
