# Weekly Price of Corn Predictor

Regime detection and one-step forecasting for **weekly corn futures prices** using Hidden Markov Models and complementary techniques.

This repository started as a from-scratch educational HMM notebook (price discretization → discrete emissions → heuristic EM). It is now a **modular research package** with proper EM inference, classical baselines, OHLC features, statistical tests, walk-forward validation, and a full figure/report suite.

Data source: [Weekly Corn Prices (Kaggle)](https://www.kaggle.com/nickwong64/corn2015-2017) / Quantopian corn futures.

---

## Why HMMs for corn prices?

Weekly agricultural futures are not a single stationary process. They alternate between quieter mean-reverting stretches and higher-volatility moves driven by inventory, weather, and macro shocks. An HMM treats those episodes as **latent regimes** $z_t$ that emit observable prices or returns $x_t$:

$$
z_t \sim P(z_t \mid z_{t-1}), \qquad x_t \sim P(x_t \mid z_t)
$$

Learning those distributions gives both a **regime timeline** and a **probabilistic one-step forecast**.

---

## Architecture

```text
data/                        # raw weekly series (price + OHLC)
data/cache/                  # live Yahoo corn + exogenous weekly caches
notebooks/                   # original educational HMM notebook
app/streamlit_app.py         # interactive regime explorer
corn_predictor/
  data/                      # loaders, live fetch, exogenous features
  preprocess/                # K-Means / quantile / uniform discretization
  models/
    discrete_hmm.py          # Forward–Backward + Baum–Welch + Viterbi
    hybrid.py                # regime-drift & discrete-return HMMs
    gaussian_hmm.py          # continuous Gaussian / MV Gaussian HMMs
    sticky_hmm.py            # sticky Dirichlet-regularized Gaussian HMM
    soft_regime.py           # soft-EM regime-switching AR(1)
    regime_switching.py      # hard-EM Markov-switching AR(1)
    classical.py             # ARIMA, GARCH, OHLC HMM, ensemble
    exogenous.py             # exo regression + exo Gaussian HMM
    baselines.py             # persistence, MA, drift
  evaluation/                # metrics, BIC, DM, calibration, hedge sim
  visualization/             # figures used in this README
  pipeline.py                # core end-to-end experiment
  advanced_pipeline.py       # sticky / exo / intervals / simulator
scripts/
  run_experiment.py
  run_advanced.py
  refresh_live_data.py
tests/
figures/
reports/
.github/workflows/ci.yml
```

```mermaid
flowchart LR
  A[Weekly prices / OHLC / live exo] --> B[Features + time split]
  B --> S[BIC state selection]
  S --> C1[Discretizer]
  S --> C2[Log-returns / OHLC / exo feats]
  C1 --> D1[Discrete HMM]
  D1 --> E1[Regime drift / return HMM]
  C2 --> D2[Gaussian / MV / OHLC / Sticky HMM]
  C2 --> D3[Hard + Soft Regime-Switching AR]
  C2 --> D4[ARIMA / GARCH / Exo models]
  B --> D5[Baselines + Ensemble]
  E1 --> F[Holdout + walk-forward + DM + PIT]
  D2 --> F
  D3 --> F
  D4 --> F
  D5 --> F
  F --> G[Figures + CSV reports + Streamlit]
```

---

## Model zoo

| Family | Model | Role |
| --- | --- | --- |
| Original idea (fixed) | Discrete HMM + Baum–Welch | Regime detection on quantized prices |
| Enhanced original | Discrete HMM + Drift | Regime-conditioned return forecast |
| Complementary HMM | Discrete Return HMM | Quantized log-returns |
| Complementary HMM | Gaussian / MV Gaussian HMM | Continuous emissions on returns |
| Complementary HMM | OHLC Gaussian HMM | Uses open/high/low/close features |
| Complementary HMM | Sticky Gaussian HMM | Dirichlet priors + sticky self-transitions |
| Complementary RS | Regime-switching AR(1) (hard EM) | Interpretable AR dynamics per regime |
| Complementary RS | Soft Regime-Switching AR(1) | Forward–Backward responsibilities + WLS |
| Exogenous | Exo regression / Exo Gaussian HMM | Lagged oil, DXY, wheat, soy returns |
| Classical | ARIMA(1,1,1), GARCH(1,1) | Standard time-series baselines |
| Meta | Equal-weight ensemble | Blend Gaussian HMM + discrete drift + drift |
| Naive | Persistence / MA / Drift | Sanity checks |

**Important:** forecasting the next **cluster center** (legacy notebook idea) is kept only as an educational baseline — quantization error makes it unusable for price RMSE.

---

## Algorithms (core ideas)

### Discrete HMM (enhanced original)
Quantize prices with K-Means into `M` symbols, then fit a categorical HMM with `N` hidden states using log-domain Forward–Backward / Baum–Welch / Viterbi. Preferred forecast path is **regime + drift**, not cluster-center decoding.

### Gaussian / Sticky HMMs
Model continuous log-returns with state-dependent Gaussians. The sticky variant adds Dirichlet row priors and extra self-transition pseudo-counts `κ` so regimes persist longer — a practical stand-in for full HDP-HMM on short weekly samples.

### Soft regime-switching AR(1)
Each regime `k` has its own AR(1) on returns `r_t`:

```text
r_t = c_{z_t} + φ_{z_t} * r_{t-1} + ε_t ,    ε_t ~ N(0, σ_{z_t}^2)
```

Soft EM updates `(c_k, φ_k, σ_k)` with Forward–Backward responsibilities instead of hard Viterbi labels.

### Exogenous models
Lagged macro/agri proxies `x_{t-1}` enter either a linear return regression

```text
r_t = β_0 + βᵀ x_{t-1} + u_t
```

or a multivariate Gaussian HMM emission vector `[r_t, x_{t-1}]`.

---

## Quick start

```bash
pip install -r requirements.txt
# or: pip install -e .

python scripts/run_experiment.py
python scripts/run_advanced.py
python -m pytest -q

# optional interactive demo
streamlit run app/streamlit_app.py

# optional live refresh (Yahoo Finance)
python scripts/refresh_live_data.py --no-cache-read
```

Skip the slower walk-forward block:

```bash
python scripts/run_experiment.py --skip-walk-forward
```

Programmatic API:

```python
from corn_predictor.data import load_prices, train_test_split_time
from corn_predictor.models import DiscreteHMMRegimeDrift, GaussianReturnHMM, OHLCGaussianHMM
from corn_predictor.evaluation import select_gaussian_states, summarize_regimes

df = load_prices("corn2013-2017.txt")
train, test = train_test_split_time(df, test_ratio=0.25)
prices = train["price"].to_numpy()

print(select_gaussian_states(prices))

model = GaussianReturnHMM(n_states=2).fit(prices)
print(model.predict_next(prices))
```

---

## Validation stack

1. **BIC state selection** on the training window (discrete & Gaussian HMMs)
2. **Chronological holdout** (last 25%) with MAE / RMSE / MAPE / directional accuracy
3. **Diebold–Mariano** tests vs persistence (squared-error loss)
4. **Walk-forward** expanding-window backtest with periodic refits
5. **Regime analytics** — occupancy, mean/max duration, return moments
6. **Predictive intervals + PIT calibration** (advanced pipeline)
7. **CSV reports** under `reports/` for every table above

### Reports inventory

| File | Contents |
| --- | --- |
| `reports/holdout_metrics.csv` | Core model holdout metrics |
| `reports/holdout_forecasts.csv` | Per-week predictions |
| `reports/diebold_mariano.csv` | DM stats vs persistence |
| `reports/walk_forward_metrics.csv` | Expanding-window metrics |
| `reports/*_state_selection.csv` | BIC grids |
| `reports/regime_*.csv` | Regime occupancy / runs |
| `reports/advanced_holdout_metrics.csv` | Sticky / soft RS metrics |
| `reports/exogenous_metrics.csv` | Exo model holdout metrics |
| `reports/calibration_summary.csv` | Interval coverage + PIT KS |
| `reports/hedge_*.csv` | Simulator equity + stats |
| `reports/sticky_regime_summary.csv` | Sticky HMM regime table |

---

## Results (reproducible)

From `python scripts/run_experiment.py` on `corn2013-2017.txt`:

**BIC-selected states:** discrete HMM → **3**, Gaussian HMM → **2**

| Model | MAE | RMSE | MAPE % | Dir. acc. |
| --- | ---: | ---: | ---: | ---: |
| Gaussian HMM | 0.058 | 0.073 | 1.51 | 0.58 |
| Persistence | 0.059 | 0.073 | 1.53 | 0.00* |
| OHLC Gaussian HMM | 0.058 | 0.073 | 1.51 | **0.60** |
| Drift | 0.059 | 0.073 | 1.52 | 0.58 |
| Ensemble | 0.059 | 0.073 | 1.52 | 0.58 |
| GARCH | 0.059 | 0.074 | 1.53 | 0.58 |
| Discrete HMM + Drift | 0.059 | 0.074 | 1.54 | 0.58 |
| Regime-Switching AR | 0.060 | 0.076 | 1.56 | 0.50 |
| ARIMA | 0.062 | 0.076 | 1.61 | 0.45 |
| Discrete HMM (centers) | 0.576 | 0.592 | 15.13 | 0.42 |

\*Persistence predicts flat prices, so directional accuracy is 0 by definition.

**Takeaways**

- Weekly corn is close to a random walk: RMSE gains vs persistence are small; **directional accuracy** and **regime interpretation** are the interesting outputs.
- **OHLC Gaussian HMM** posts the best directional accuracy on this sample.
- DM tests do **not** find significant RMSE improvement vs persistence for the competitive models (as expected on a near-martingale series). The legacy center decoder is significantly *worse*.
- Discrete HMM regimes separate a long low-price occupancy state from shorter elevated / volatile spells (see regime summary CSVs).

---

## Visualizations

### Price series & regimes

![Weekly corn close price](figures/01_price_series.png)

![Discrete HMM regimes](figures/02_regimes_on_price.png)

### Learned discrete HMM

![Transition matrix](figures/03_transition_matrix.png)

![Emission matrix](figures/04_emission_matrix.png)

![Baum-Welch convergence](figures/05_loglik_convergence.png)

### Gaussian regimes

![Return distributions by regime](figures/06_returns_by_regime.png)

![Gaussian regimes on training prices](figures/07_gaussian_regimes.png)

### Forecasts & metrics

![Holdout forecasts](figures/08_forecast_overlay.png)

![RMSE comparison](figures/09_rmse_comparison.png)

![Directional accuracy](figures/10_directional_accuracy.png)

### Model selection, durations, DM, walk-forward

![Discrete BIC selection](figures/11_discrete_state_selection.png)

![Gaussian BIC selection](figures/12_gaussian_state_selection.png)

![Regime durations](figures/13_regime_durations.png)

![Diebold-Mariano](figures/14_diebold_mariano.png)

![Walk-forward errors](figures/15_walk_forward_errors.png)

---

## Data files

| File | Contents |
| --- | --- |
| `data/corn2013-2017.txt` | Weekly close prices (2013–2017) |
| `data/corn2015-2017.txt` | Shorter close-price subset |
| `data/corn_OHLC2013-2017.txt` | Weekly open / high / low / close |
| `data/cache/corn_weekly_live.csv` | Live Yahoo `ZC=F` weekly closes (refreshable) |
| `data/cache/exogenous_weekly.csv` | Live weekly oil / DXY / wheat / soy proxies |

---

## Original notebook

Educational prototype (dictionary parameters, manual EM):

`notebooks/Hidden Markov Models.ipynb`

Prefer `corn_predictor` + `scripts/run_experiment.py` for experiments.

---

## Tests & CI

```bash
python -m pytest -q
```

GitHub Actions runs pytest and smoke experiments on every push/PR (`.github/workflows/ci.yml`).

---

## Advanced enhancements

All of the former “next upgrades” are now implemented.

### Exogenous drivers
Lagged weekly returns of **WTI oil**, **DXY**, **WEAT**, and **SOYB** (Yahoo Finance proxies — not official USDA prints) feed:
- OLS exogenous return regression
- Gaussian HMM on `[corn_return, exo_lags…]`

```bash
python scripts/refresh_live_data.py --no-cache-read   # refresh caches
python scripts/run_advanced.py
```

Cached series live under `data/cache/` (currently through 2026).

### Predictive intervals & PIT calibration
Sticky / Gaussian HMMs emit mixture return moments → lognormal price intervals. Reports include empirical coverage and a Kolmogorov–Smirnov PIT uniformity check (`reports/calibration_summary.csv`).

### Sticky Bayesian-flavoured HMM
`StickyGaussianHMM` adds Dirichlet priors and a **sticky self-transition bias (κ)**, plus occupancy-based pruning of unused states — a practical alternative to full HDP-HMM sampling on short weekly series.

### Soft EM regime-switching AR
`SoftRegimeSwitchingAR` replaces hard Viterbi assignment with Forward–Backward responsibilities and weighted least squares.

### Regime long/flat simulator
Research-only long/flat policy using train-selected regimes (`reports/hedge_sim_stats.csv`). **Not investment advice.**

### Streamlit explorer
```bash
streamlit run app/streamlit_app.py
```
Interactive regime decode, predictive interval, and equity overlay.

### Advanced figures
![Sticky regimes](figures/16_sticky_regimes.png)

![Predictive intervals](figures/17_predictive_intervals.png)

![PIT histogram](figures/18_pit_histogram.png)

![Hedge equity](figures/19_hedge_equity.png)

![Advanced RMSE](figures/20_advanced_rmse.png)

![Exogenous RMSE](figures/21_exogenous_rmse.png)

### Advanced holdout snapshot
| Model | RMSE | Dir. acc. |
| --- | ---: | ---: |
| Sticky Gaussian HMM | 0.074 | 0.55 |
| Soft Regime-Switching AR | 0.075 | 0.52 |
| Exo Gaussian HMM | 0.074 | 0.57 |
| Exo Regression | 0.090 | 0.34 |

Sticky HMM 90% intervals: empirical coverage ≈ **0.97** (slightly wide), PIT KS p ≈ **0.25** (not reject uniformity).

---

## License / disclaimer

Personal research code. Futures markets are risky; nothing here is investment advice. The hedge simulator is a pedagogical backtest only.
