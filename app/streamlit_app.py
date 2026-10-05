"""Streamlit regime explorer for the corn price predictor.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn_predictor.data import add_features, load_prices, train_test_split_time
from corn_predictor.evaluation.calibration import hmm_predictive_interval
from corn_predictor.evaluation.regimes import summarize_regimes
from corn_predictor.evaluation.simulator import (
    choose_long_regimes_by_mean_return,
    simulate_regime_long_flat,
)
from corn_predictor.models import GaussianReturnHMM, SoftRegimeSwitchingAR, StickyGaussianHMM


st.set_page_config(page_title="Corn Regime Explorer", layout="wide")
st.title("Weekly Corn Regime Explorer")
st.caption(
    "Research demo — not investment advice. Decodes latent market regimes and "
    "compares simple long/flat policies."
)


@st.cache_data(show_spinner=False)
def _load_df():
    return add_features(load_prices("corn2013-2017.txt"))


df = _load_df()
prices = df["price"].to_numpy()

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox(
        "Regime model",
        ["Sticky Gaussian HMM", "Gaussian HMM", "Soft Regime-Switching AR"],
    )
    n_states = st.slider("States", 2, 4, 3)
    test_ratio = st.slider("Holdout fraction", 0.1, 0.4, 0.25, 0.05)
    cost_bps = st.slider("Simulated cost (bps)", 0.0, 10.0, 1.0, 0.5)
    run = st.button("Fit & decode", type="primary")


def _fit_decode(name: str, n_states: int, train_prices: np.ndarray, all_prices: np.ndarray):
    if name == "Sticky Gaussian HMM":
        model = StickyGaussianHMM(n_states=n_states, random_state=24).fit(train_prices)
        rets = np.diff(np.log(all_prices))
        states = model.decode_returns(rets)
        states_price = np.concatenate([[states[0]], states])
        return model, states_price
    if name == "Gaussian HMM":
        model = GaussianReturnHMM(n_states=n_states, random_state=24).fit(train_prices)
        rets = np.diff(np.log(all_prices))
        states = model.decode_returns(rets)
        states_price = np.concatenate([[states[0]], states])
        return model, states_price
    model = SoftRegimeSwitchingAR(n_states=n_states, random_state=24).fit(train_prices)
    # use hard labels from argmax responsibilities on full series by refit decode approx:
    # Soft model stores responsibilities on training only; refit on all for viz
    model.fit(all_prices)
    states_price = np.concatenate(
        [[int(np.argmax(model.responsibilities_[0]))], np.argmax(model.responsibilities_, axis=1)]
    )
    # responsibilities align with returns length = n-1; prepend first
    if len(states_price) != len(all_prices):
        # responsibilities for y[1:], length n-1; build length n
        st_path = np.argmax(model.responsibilities_, axis=1)
        states_price = np.concatenate([[st_path[0]], st_path])
    return model, states_price


if run:
    train_df, test_df = train_test_split_time(df, test_ratio=test_ratio)
    train_prices = train_df["price"].to_numpy()
    with st.spinner("Fitting..."):
        model, states = _fit_decode(model_name, n_states, train_prices, prices)

    summary = summarize_regimes(
        states, prices=prices, returns=np.diff(np.log(prices))
    )
    long_regs = choose_long_regimes_by_mean_return(states, prices)
    sim = simulate_regime_long_flat(
        prices, states, long_regs, weeks=pd.to_datetime(df["week"]), cost_bps=cost_bps
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy return", f"{sim.stats['total_return']:.1%}")
    c2.metric("Buy & hold", f"{sim.stats['buy_hold_return']:.1%}")
    c3.metric("Sharpe (ann.)", f"{sim.stats['sharpe']:.2f}")
    c4.metric("Max drawdown", f"{sim.stats['max_drawdown']:.1%}")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price & regimes", "Position", "Equity vs buy&hold"),
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(x=df["week"], y=prices, mode="lines", name="Price", line=dict(color="#1f2a24")),
        row=1,
        col=1,
    )
    palette = ["#2f6b4f", "#c45c26", "#3a6ea5", "#b08900"]
    for s in sorted(np.unique(states)):
        mask = states == s
        fig.add_trace(
            go.Scatter(
                x=df["week"][mask],
                y=prices[mask],
                mode="markers",
                name=f"Regime {s}",
                marker=dict(size=7, color=palette[int(s) % len(palette)]),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=sim.positions.index,
            y=sim.positions.values,
            mode="lines",
            name="Position",
            line=dict(color="#2f6b4f", shape="hv"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=sim.equity_curve.index, y=sim.equity_curve.values, name="Strategy"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=sim.buy_hold.index, y=sim.buy_hold.values, name="Buy & hold"),
        row=3,
        col=1,
    )
    fig.update_layout(height=820, legend=dict(orientation="h"), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regime summary")
    st.dataframe(summary.round(4), use_container_width=True)
    st.write(f"Long regimes used in simulator: `{long_regs}`")

    # One-step predictive interval from sticky/gaussian if available
    if hasattr(model, "predictive_return_moments"):
        rets = np.diff(np.log(train_prices))
        mu, sd = model.predictive_return_moments(rets)
        interval = hmm_predictive_interval(
            train_prices[-1],
            model.next_state_probs(rets),
            model.means_,
            model.vars_,
            level=0.9,
        )
        st.subheader("Next-week predictive interval (90%)")
        st.write(
            {
                "last_price": float(train_prices[-1]),
                "mean": interval.mean,
                "lower": interval.lower,
                "upper": interval.upper,
            }
        )
else:
    st.info("Configure the sidebar and click **Fit & decode**.")
    st.line_chart(df.set_index("week")["price"])
