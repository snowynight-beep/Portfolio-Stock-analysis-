from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import math
import altair as alt
import riskfolio as rp
from io import BytesIO 
import zipfile
from streamlit_lightweight_charts import renderLightweightCharts

DARTS_AVAILABLE = True
try:
    from darts import TimeSeries
    from darts.models import ExponentialSmoothing, AutoARIMA, Prophet, XGBModel, NBEATSModel
except Exception:
    DARTS_AVAILABLE = False



# Set up Streamlit app
st.set_page_config(
    page_title="Stock Portfolio Analysis Dashboard",
    layout="wide"
)

st.write("""# Stock Portfolio Analysis Dashboard
This app allows you to input stock tickers and their corresponding weights to analyze the portfolio's cumulative returns over the past year.
""")

st.markdown("""
<style>
:root {
  --bg: #010b13;
  --bg2:#0b1620;
  --txt:#ffffff;
  --muted:#6f8aa8;
}
html, body, [data-testid="stAppViewContainer"] { background-color: var(--bg); color: var(--txt); }
[data-testid="stHeader"] { background-color: var(--bg); }
[data-testid="stSidebar"] { background-color: var(--bg2) !important; color: var(--txt); }
[data-testid="stSidebar"] * { color: var(--txt) !important; }

/* Keep headings/labels light-blue; DO NOT paint every span/div (lets metric deltas keep green/red) */
h1, h2, h3, h4, h5, h6, label { color: var(--txt); }

/* Leave deltas alone so Streamlit uses its default up/down colors */
.stMetric label { color: var(--txt) !important; }

.css-1dp5vir, .stDataFrame, .stTable { color: var(--txt); }
</style>
""", unsafe_allow_html=True)

# Helper: pills fallback if not available (older Streamlit)
def pills(label, options, default):
    if hasattr(st, "pills"):
        return st.pills(label, options=options, default=default)
    # fallback to radio; keep horizontal if available
    idx = options.index(default) if default in options else 0
    return st.radio(label, options, index=idx, horizontal=True)

def _to_float(x):
    if isinstance(x, pd.Series):
        x = x.iloc[0] if len(x) else np.nan
    elif isinstance(x, (np.ndarray, list, tuple)):
        arr = np.asarray(x).ravel()
        x = arr[0] if arr.size else np.nan
    try:
        return float(x)
    except Exception:
        return np.nan
    
def download_ohlcv(ticker: str, period_str: str, interval: str = "1d") -> pd.DataFrame | None:
    start = _period_to_start(period_str)
    df = yf.download(ticker, period="max", interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    if start is not None:
        df = df.loc[df.index >= start]
    df = df.dropna()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in needed):
        return None
    return df[needed].copy()

def _sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(int(w)).mean()

def _ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=int(w), adjust=False).mean()

def _rsi(s: pd.Series, w: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    rs = _ema(up, w) / _ema(dn, w)
    return 100 - 100 / (1 + rs)

def _macd(s: pd.Series, fast=12, slow=26, signal=9):
    macd_line = _ema(s, fast) - _ema(s, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _bbands(s: pd.Series, w=20, k=2.0):
    m = _sma(s, w)
    sd = s.rolling(int(w)).std()
    return m, m + k * sd, m - k * sd

# VWAP (session reset for intraday, anchored for daily/weekly)
def _vwap(df: pd.DataFrame, intraday: bool) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df.get("Volume", pd.Series(index=df.index, dtype=float)).fillna(0.0)
    if intraday:
        dates = pd.to_datetime(df.index).date
        pv = (tp * vol).groupby(dates).cumsum()
        vv = vol.groupby(dates).cumsum().replace(0, np.nan)
        return pv / vv
    pv = (tp * vol).cumsum()
    vv = vol.cumsum().replace(0, np.nan)
    return pv / vv

def _to_lwc_time(idx: pd.DatetimeIndex, intraday: bool) -> list:
    idx = idx.tz_localize(None)
    if intraday:
        return [int(ts.timestamp()) for ts in idx]
    return [ts.date().isoformat() for ts in idx]

# --- NEW: signal helpers ---
def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.astype(float), b.astype(float)
    return (a.shift(1) <= b.shift(1)) & (a > b)

def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.astype(float), b.astype(float)
    return (a.shift(1) >= b.shift(1)) & (a < b)

def _markers_from_signals(index: pd.DatetimeIndex, buys: pd.Series, sells: pd.Series, intraday: bool):
    times = _to_lwc_time(index, intraday)
    markers = []
    for i, t in enumerate(times):
        if bool(buys.iloc[i]):
            markers.append({"time": t, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "BUY"})
        if bool(sells.iloc[i]):
            markers.append({"time": t, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "SELL"})
    return markers

# input section
columns = st.columns([1, 3])
left_col_columns = columns[0].container(border=True, height="stretch", vertical_alignment="center")
right_col_columns = columns[1].container(border=True, height="stretch", vertical_alignment="center")

with left_col_columns:
    tickers_input = st.text_input("Enter tickers (comma-separated)", value="AAPL, MSFT, GOOG")
    weights_input = st.text_input("Enter weights (comma-separated). Use fractions (0.5)", value="0.5,0.3,0.2")
    initial_capital = st.number_input("Initial investment (USD)", value=100000, step=1000, format="%d")

# ticker for time horizon selection
horizon_map = {
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "Max": "max",
}
with left_col_columns:
    time = horizon_map[pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="5 Years",
    )]

# tabs for output
port_tab, dcf_tab, Tech_tab = st.tabs(["Portfolio", "DCF Analysis", "Technical Analysis"])

# --- Common-overlap downloader (handles 6mo/1y/… and trims to the shortest stock) ---
def _period_to_start(period_str: str) -> pd.Timestamp | None:
    s = str(period_str).lower().strip()
    today = pd.Timestamp.today().normalize()
    if s == "max":
        return None
    if s.endswith("y"):
        yrs = int(float(s[:-1]))
        return today - pd.DateOffset(years=yrs)
    if s.endswith("mo"):
        mos = int(float(s[:-2]))
        return today - pd.DateOffset(months=mos)
    return None

def download_adj_close_common(tickers, period_str, interval="1d"):
    raw = yf.download(tickers, period="max", interval=interval, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        return None, None
    # Adjusted Close
    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw['Adj Close']
        if isinstance(adj, pd.Series):
            adj = adj.to_frame()
    else:
        if 'Adj Close' not in raw.columns:
            return None, None
        adj = raw[['Adj Close']]
        if len(tickers) == 1:
            adj.columns = pd.Index([tickers[0]])
    adj = adj.reindex(columns=tickers, copy=False).dropna(how="all")

    # shortest history (latest first valid)
    first_valid_each = [adj[c].first_valid_index() for c in adj.columns]
    first_valid_each = [d for d in first_valid_each if d is not None]
    if not first_valid_each:
        return None, None
    first_common = max(first_valid_each)

    # user horizon cap
    user_start = _period_to_start(period_str)
    start = max(first_common, user_start) if user_start is not None else first_common

    # trim and drop any remaining gaps
    adj = adj.loc[adj.index >= start].dropna(how="any")
    return adj, start

# parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(tickers) == 0:
    st.error("Please enter at least one ticker.")
    st.stop()


with dcf_tab:
    adjusted_close, eff_start = download_adj_close_common(tickers, time, interval="1d")
    if adjusted_close is None or adjusted_close.empty:
        st.error("No data returned. Check tickers or network.")
        st.stop()

    # Daily returns -> cumulative
    daily_returns = adjusted_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    asset_cum = (1 + daily_returns).cumprod() - 1

    # wide-to-long for chart
    plot_df = asset_cum.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Cumulative Return')
    with st.container():  
        chart = alt.Chart(plot_df).mark_line().encode(
            x='Date:T',
            y=alt.Y('Cumulative Return:Q', axis=alt.Axis(format='.0%')),
            color='Ticker:N',
            tooltip=['Date:T', 'Ticker:N', alt.Tooltip('Cumulative Return:Q', format='.2%')]
        ).interactive()
        st.altair_chart(chart, width='stretch') 


    st.subheader("DCF Analysis")
    tab_inputs, tab_analysis = st.tabs(["Simple", "Advanced"])

    # ---- Inputs (Simple) ----
    with tab_inputs:
        years_to_forecast = st.number_input(
            "Years to forecast (0 = use current year only)",
            min_value=0, max_value=15, value=5, step=1, key="dcf_years"
        )
        roll_window = 3

        # WACC constants
        rf = 0.03
        credit_spread = 0.01
        tax_rate = 0.21

        # Market return: ^GSPC 5y daily average annualized
        try:
            _spx = yf.download("^GSPC", period="5y", auto_adjust=False, progress=False)["Adj Close"].pct_change(fill_method=None).dropna()
            market_return_annualized = float(_spx.mean() * 252.0)
        except Exception:
            market_return_annualized = 0.08  # fallback

        rows = []
        projections = []  # <-- collect per-ticker projection details for export
        for tk in tickers:
            try:
                t = yf.Ticker(tk)

                # ---- Balance sheet (for D/E and net debt) ----
                de = None
                net_debt = None
                bs = t.balance_sheet
                if isinstance(bs, pd.DataFrame) and not bs.empty:
                    bs = bs.loc[:, sorted(bs.columns)] if len(bs.columns) > 1 else bs

                    # D/E
                    total_liab = None
                    total_eq = None
                    for k in ["Total Liab", "Total Liabilities Net Minority Interest", "Total Liabilities"]:
                        if k in bs.index:
                            total_liab = pd.to_numeric(bs.loc[k], errors="coerce").iloc[-1]
                            break
                    for k in ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Equity"]:
                        if k in bs.index:
                            total_eq = pd.to_numeric(bs.loc[k], errors="coerce").iloc[-1]
                            break
                    if total_liab is not None and total_eq not in (None, 0):
                        de = max(float(total_liab) / float(total_eq), 0.0)

                    # Net debt
                    if "Net Debt" in bs.index:
                        net_debt = float(pd.to_numeric(bs.loc["Net Debt"], errors="coerce").iloc[-1])
                    else:
                        debt = None
                        if "Total Debt" in bs.index:
                            debt = float(pd.to_numeric(bs.loc["Total Debt"], errors="coerce").iloc[-1])
                        else:
                            sd = float(pd.to_numeric(bs.loc.get("Short Long Term Debt", pd.Series([0])), errors="coerce").iloc[-1]) if "Short Long Term Debt" in bs.index else 0.0
                            ld = float(pd.to_numeric(bs.loc.get("Long Term Debt", pd.Series([0])), errors="coerce").iloc[-1]) if "Long Term Debt" in bs.index else 0.0
                            debt = sd + ld
                        cash = None
                        for k in [
                            "Cash And Cash Equivalents",
                            "Cash And Cash Equivalents Including Marketable Securities",
                            "Cash",
                        ]:
                            if k in bs.index:
                                cash = float(pd.to_numeric(bs.loc[k], errors="coerce").iloc[-1])
                                break
                        if debt is not None:
                            net_debt = (debt or 0.0) - (cash or 0.0)

                # ---- Cash flow -> FCF (CFO + CapEx) ----
                cf = t.cashflow
                fcf = None
                if isinstance(cf, pd.DataFrame) and not cf.empty:
                    cf = cf.loc[:, sorted(cf.columns)] if len(cf.columns) > 1 else cf
                    op = None
                    capex = None
                    for k in ["Total Cash From Operating Activities", "Operating Cash Flow", "Cash Flow From Operating Activities"]:
                        if k in cf.index:
                            op = pd.to_numeric(cf.loc[k], errors="coerce")
                            break
                    for k in ["Capital Expenditures", "Capital Expenditure"]:
                        if k in cf.index:
                            capex = pd.to_numeric(cf.loc[k], errors="coerce")
                            break
                    if op is not None and capex is not None:
                        fcf = (op + capex).dropna()
                    elif "Free Cash Flow" in cf.index:
                        fcf = pd.to_numeric(cf.loc["Free Cash Flow"], errors="coerce").dropna()

                if fcf is None or len(fcf) < 2:
                    st.warning(f"{tk}: Not enough FCF history for DCF.")
                    continue

                fcf = fcf.sort_index()
                # Fix unit outliers (sometimes values are in thousands)
                if fcf.abs().median() > 1e12:
                    fcf = fcf / 1_000.0
                fcf_series = pd.Series(fcf.values.astype(float), index=pd.to_datetime(fcf.index))

                # ---- Rolling average YoY growth ----
                yoy = fcf_series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                avg_g = float(yoy.rolling(window=int(roll_window), min_periods=1).mean().iloc[-1])
                avg_g = float(np.clip(avg_g, -0.30, 0.30))

                # ---- WACC ----
                # Beta from yahoo
                beta = None
                try:
                    beta = t.fast_info.get("beta", None)
                except Exception:
                    beta = None
                if beta in (None, 0, np.nan) and hasattr(t, "info"):
                    beta = t.info.get("beta", None)
                beta = _to_float(beta)
                if np.isnan(beta) or beta == 0:
                    beta = 1.0

                Re = rf + float(beta) * (market_return_annualized - rf)
                Rd = rf + credit_spread

                if de is None:
                    wD, wE = 0.30, 0.70
                else:
                    wD = float(de) / (1.0 + float(de))
                    wE = 1.0 - wD

                wacc = float(np.clip(wE * Re + wD * Rd * (1.0 - tax_rate), 0.03, 0.25))

                # ---- Forecast + Terminal ----
                last_fcf = float(fcf_series.iloc[-1])
                proj_years = list(range(1, int(years_to_forecast) + 1))
                proj_fcfs = [last_fcf * (1.0 + avg_g) ** i for i in proj_years]

                g_terminal = float(np.clip(avg_g, -0.02, 0.04))
                if g_terminal >= wacc:
                    g_terminal = max(wacc - 0.01, -0.02)

                denom = max(wacc - g_terminal, 1e-9)

                if years_to_forecast == 0:
                    # Use only this year's cash flow + terminal (no discounting)
                    pv_proj = float(last_fcf)
                    terminal_value = float(last_fcf) * (1.0 + float(g_terminal)) / denom
                    pv_terminal_value = terminal_value
                else:
                    pv_proj = sum(fcf_y / ((1.0 + wacc) ** i) for i, fcf_y in zip(proj_years, proj_fcfs))
                    terminal_value = proj_fcfs[-1] * (1.0 + g_terminal) / (wacc - g_terminal)
                    pv_terminal_value = terminal_value / ((1.0 + wacc) ** proj_years[-1])

                enterprise_value = pv_proj + pv_terminal_value
                equity_value = enterprise_value - net_debt if net_debt is not None else enterprise_value

                # Shares + price
                shares = _to_float(t.fast_info.get("sharesOutstanding", np.nan))
                if (np.isnan(shares) or shares == 0) and hasattr(t, "info"):
                    shares = _to_float(t.info.get("sharesOutstanding", np.nan))
                price = _to_float(t.fast_info.get("lastPrice", np.nan))
                if np.isnan(price) or price == 0:
                    try:
                        price = _to_float(t.history(period="1d")["Close"].iloc[-1])
                    except Exception:
                        price = np.nan

                per_share = equity_value / float(shares) if shares not in (None, 0) else np.nan

                # --- Save summary row ---
                rows.append({
                    "Ticker": tk,
                    "WACC": wacc,
                    "Rolling Growth": avg_g,
                    "Terminal Value ($)": terminal_value,
                    "PV Terminal ($)": pv_terminal_value,
                    "EV ($)": enterprise_value,
                    "Equity Value ($)": equity_value,
                    "Calc Price/Share ($)": per_share,
                    "Actual Price/Share ($)": price,
                    "Shares Outstanding": shares  # <- add shares for Market Cap
                })

                # --- Build detailed projection table for export ---
                # Year-by-year details
                base_date = pd.to_datetime(fcf.index[-1]) if len(fcf.index) else pd.Timestamp.today()
                if years_to_forecast == 0:
                   # Build a minimal two-row table: current year + terminal
                    proj_df = pd.DataFrame({
                        "Year #": [0],
                        "Fiscal Year": [f"FY {int(base_date.year)}"],
                        "Starting FCF ($)": [last_fcf],
                        "Growth Assumption": [avg_g],
                        "Implied YoY Growth": [np.nan],
                        "FCF Forecast ($)": [last_fcf],
                        "Discount Factor": [1.0],
                        "PV of FCF ($)": [last_fcf],
                        "Cum PV of FCF ($)": [last_fcf],
                        "Terminal Value ($)": [np.nan],
                        "PV Terminal ($)": [np.nan],
                        "Row": ["Projection"]
                    })
                    terminal_row = pd.DataFrame({
                        "Year #": [0],
                        "Fiscal Year": [f"Terminal after FY {int(base_date.year)}"],
                        "Starting FCF ($)": [last_fcf],
                        "Growth Assumption": [g_terminal],
                        "Implied YoY Growth": [g_terminal],
                        "FCF Forecast ($)": [last_fcf * (1.0 + g_terminal)],
                        "Discount Factor": [1.0],
                        "PV of FCF ($)": [np.nan],
                        "Cum PV of FCF ($)": [last_fcf],
                        "Terminal Value ($)": [terminal_value],
                        "PV Terminal ($)": [pv_terminal_value],
                        "Row": ["Terminal"]
                    })
                    proj_df = pd.concat([proj_df, terminal_row], ignore_index=True)
                else:
                    # Year-by-year details
                    fiscal_years = [f"FY {int(base_date.year) + i}" for i in proj_years]
                    starting_fcfs = [last_fcf] + proj_fcfs[:-1]
                    implied_growth = []
                    for i, f in enumerate(proj_fcfs):
                        base0 = starting_fcfs[i]
                        implied_growth.append((f / base0 - 1.0) if base0 not in (0.0, None) else np.nan)
                    disc_factors = [1.0 / ((1.0 + wacc) ** i) for i in proj_years]
                    pv_fcfs = [fcf * d for fcf, d in zip(proj_fcfs, disc_factors)]
                    cum_pv_fcfs = np.cumsum(pv_fcfs)
                    proj_df = pd.DataFrame({
                        "Year #": proj_years,
                        "Fiscal Year": fiscal_years,
                        "Starting FCF ($)": starting_fcfs,
                        "Growth Assumption": [avg_g] * len(proj_years),
                        "Implied YoY Growth": implied_growth,
                        "FCF Forecast ($)": proj_fcfs,
                        "Discount Factor": disc_factors,
                        "PV of FCF ($)": pv_fcfs,
                        "Cum PV of FCF ($)": cum_pv_fcfs,
                        "Terminal Value ($)": np.nan,
                        "PV Terminal ($)": np.nan,
                        "Row": "Projection"
                    })
                    terminal_row = pd.DataFrame({
                        "Year #": [proj_years[-1]],
                        "Fiscal Year": [f"Terminal after FY {int(base_date.year) + proj_years[-1]}"],
                        "Starting FCF ($)": [proj_fcfs[-1]],
                        "Growth Assumption": [g_terminal],
                        "Implied YoY Growth": [g_terminal],
                        "FCF Forecast ($)": [proj_fcfs[-1] * (1.0 + g_terminal)],
                        "Discount Factor": [disc_factors[-1]],
                        "PV of FCF ($)": [np.nan],
                        "Cum PV of FCF ($)": [cum_pv_fcfs[-1]],
                        "Terminal Value ($)": [terminal_value],
                        "PV Terminal ($)": [pv_terminal_value],
                        "Row": ["Terminal"]
                    })
                    proj_df = pd.concat([proj_df, terminal_row], ignore_index=True)

                # Per-ticker assumptions and summary for a quick sheet
                ticker_summary = pd.DataFrame({
                    "Metric": [
                        "Risk-free rate", "Market return (annualized)", "Beta",
                        "Credit spread", "Tax rate", "D/E", "WACC",
                        "Rolling average growth", "Terminal growth",
                        "Last FCF ($)", "PV of projections ($)",
                        "Terminal value ($)", "PV terminal ($)",
                        "Enterprise value ($)", "Net debt ($)",
                        "Equity value ($)", "Shares outstanding",
                        "Calculated price/share ($)", "Actual price/share ($)",
                        "Years to forecast", "Rolling window"
                    ],
                    "Value": [
                        rf, market_return_annualized, beta,
                        credit_spread, tax_rate, de, wacc,
                        avg_g, g_terminal,
                        last_fcf, pv_proj,
                        terminal_value, pv_terminal_value,
                        enterprise_value, net_debt,
                        equity_value, shares,
                        per_share, price,
                        years_to_forecast, roll_window
                    ]
                })

                projections.append({
                    "Ticker": tk,
                    "projections": proj_df, 
                    "summary": ticker_summary
                })

            except Exception as e:
                st.warning(f"{tk}: DCF failed: {e}")

        if rows:
            dcf_df = pd.DataFrame(rows)

            st.subheader("Valuation summary")
            st.dataframe(
                dcf_df.set_index("Ticker").style.format({
                    "WACC": "{:.2%}",
                    "Rolling Growth": "{:.2%}",
                    "Terminal Value ($)": "${:,.0f}",
                    "PV Terminal ($)": "${:,.0f}",
                    "EV ($)": "${:,.0f}",
                    "Equity Value ($)": "${:,.0f}",
                    "Calc Price/Share ($)": "${:,.2f}",
                    "Actual Price/Share ($)": "${:,.2f}"
                }),
                width='stretch'
            )

            # Price per share comparison (grouped bars, not stacked)
            price_comp = dcf_df.melt(
                id_vars="Ticker",
                value_vars=["Calc Price/Share ($)", "Actual Price/Share ($)"],
                var_name="Type",
                value_name="Price"
            )
            st.subheader("Calculated vs Actual Price per Share")
            bar1 = alt.Chart(price_comp).mark_bar().encode(
                x=alt.X("Ticker:N", title=""),
                xOffset=alt.XOffset("Type:N"),
                y=alt.Y("Price:Q", stack=None, axis=alt.Axis(format="$,.2f")),
                color=alt.Color("Type:N", legend=alt.Legend(title="")),
                tooltip=["Ticker:N", "Type:N", alt.Tooltip("Price:Q", format="$,.2f")]
            )
            st.altair_chart(bar1, width='stretch')

            # NEW: Market Cap vs Enterprise Value when Years to forecast == 0
            if years_to_forecast == 0:
                mc_df = dcf_df.copy()
                # Require valid price and shares
                mc_df["Market Cap ($)"] = mc_df["Actual Price/Share ($)"] * mc_df["Shares Outstanding"]
                mc_df = mc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Market Cap ($)", "EV ($)"])
                if mc_df.empty:
                    st.info("No data to plot Market Cap vs EV (missing price/shares).")
                else:
                    st.subheader("Market Cap vs Enterprise Value")
                    plot_mc_ev = mc_df.melt(
                        id_vars="Ticker",
                        value_vars=["Market Cap ($)", "EV ($)"],
                        var_name="Metric",
                        value_name="Value"
                    )
                    mc_chart = alt.Chart(plot_mc_ev).mark_bar().encode(
                        x=alt.X("Ticker:N", title=""),
                        xOffset="Metric",
                        y=alt.Y("Value:Q", axis=alt.Axis(title="Value ($)", format="~s")),
                        color=alt.Color("Metric:N", legend=alt.Legend(title="")),
                        tooltip=["Ticker:N", "Metric:N", alt.Tooltip("Value:Q", format="$,.0f")]
                    ).properties(height=350)
                    st.altair_chart(mc_chart, width='stretch')
        if rows:
            dcf_df = pd.DataFrame(rows)
            # ---------------- Export section ----------------
            st.subheader("Export DCF results")
            export_fmt = st.selectbox(
                "Choose export format",
                options=["Excel (.xlsx)", "CSV (.zip of CSV files)"],
                index=0,
                key="dcf_export_fmt"
            )

            # Build common assumptions table
            assumptions = pd.DataFrame({
                "Assumption": [
                    "Years to forecast", "Rolling window (years)",
                    "Risk-free rate", "Market return (annualized)",
                    "Credit spread", "Tax rate"
                ],
                "Value": [
                    years_to_forecast, roll_window,
                    rf, market_return_annualized,
                    credit_spread, tax_rate
                ]
            })

            if export_fmt.startswith("Excel"):
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    # Summary and assumptions
                    dcf_df.to_excel(writer, sheet_name="Summary", index=False)
                    assumptions.to_excel(writer, sheet_name="Assumptions", index=False)

                    # Per-ticker sheets
                    for item in projections:
                        sheet_base = item["Ticker"]
                        item["projections"].to_excel(writer, sheet_name=f"{sheet_base}_FCF", index=False)
                        item["summary"].to_excel(writer, sheet_name=f"{sheet_base}_Summary", index=False)

                st.download_button(
                    label="Download DCF workbook",
                    data=buffer.getvalue(),
                    file_name="dcf_valuation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                # Zip of CSV files: Summary, Assumptions, and per-ticker details
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("Summary.csv", dcf_df.to_csv(index=False))
                    zf.writestr("Assumptions.csv", assumptions.to_csv(index=False))
                    for item in projections:
                        zf.writestr(f"{item['Ticker']}_FCF.csv", item["projections"].to_csv(index=False))
                        zf.writestr(f"{item['Ticker']}_Summary.csv", item["summary"].to_csv(index=False))

                st.download_button(
                    label="Download DCF pack (CSV .zip)",
                    data=zip_buf.getvalue(),
                    file_name="dcf_valuation_csv.zip",
                    mime="application/zip"
            )

    with tab_analysis:
        st.subheader("DCF Analysis (Advanced)")

        years_to_forecast_adv = st.number_input(
            "Years to forecast (Advanced) — 0 = use current year only",
            min_value=0, max_value=20, value=5, step=1, key="dcf_years_adv"
        )

        # Per-ticker override inputs (comma-separated like weights). Empty -> defaults.
        credit_spread_str = st.text_input(
            "Credit spread(s) (comma-separated; blank = 1%)", value="", key="adv_cs"
        )
        tax_rate_str = st.text_input(
            "Tax rate(s) (comma-separated; blank = 21%)", value="", key="adv_tax"
        )

        risk_free_rate = st.text_input(
            "Risk-free rate (value in decimal; blank = 3%)", value="", key="adv_rf"
        )

        # Growth: either rolling period or a custom schedule
        roll_period_str = st.text_input(
            "Rolling window (years) for growth (leave blank to use custom growth schedule)",
            value="",
            key="adv_rollwin"
        )
        if roll_period_str.strip() == "":
            growth_sched_str = st.text_input(
                "Custom growth schedule:(enter 1 value for growth rate or 3 values growth rate,reduction,step)",
                value="0.05",
                key="adv_growth_sched"
            )
        else:
            growth_sched_str = None

        # Helpers to parse vectors per ticker
        def _parse_vector(s: str, default_val: float):
            if not s or not s.strip():
                return [default_val] * len(tickers)
            parts = [p.strip() for p in s.split(",") if p.strip()]
            try:
                vals = [float(p) for p in parts]
            except Exception:
                st.warning("Could not parse list; using defaults.")
                return [default_val] * len(tickers)
            if len(vals) == 1:
                return [vals[0]] * len(tickers)
            if len(vals) == len(tickers):
                return vals
            st.warning("List length mismatch; broadcasting the first value.")
            return [vals[0]] * len(tickers)

        credit_spreads = _parse_vector(credit_spread_str, 0.01)
        tax_rates = _parse_vector(tax_rate_str, 0.21)
        risk_free_rates = _parse_vector(risk_free_rate, 0.03)

        rows_adv = []
        projections_adv = []

        for idx, tk in enumerate(tickers):
            try:
                t = yf.Ticker(tk)

                # ---- Balance sheet (D/E and net debt) ----
                de = None
                net_debt = None
                bs = t.balance_sheet
                if isinstance(bs, pd.DataFrame) and not bs.empty:
                    bs = bs.loc[:, sorted(bs.columns)] if len(bs.columns) > 1 else bs
                    total_liab = None
                    total_eq = None
                    for k in ["Total Liab", "Total Liabilities Net Minority Interest", "Total Liabilities"]:
                        if k in bs.index:
                            total_liab = pd.to_numeric(bs.loc[k], errors="coerce").iloc[-1]
                            break
                    for k in ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Equity"]:
                        if k in bs.index:
                            total_eq = pd.to_numeric(bs.loc[k], errors="coerce").iloc[-1]
                            break
                    if total_liab is not None and total_eq not in (None, 0):
                        de = max(float(total_liab) / float(total_eq), 0.0)

                    # Net debt
                    if "Net Debt" in bs.index:
                        net_debt = float(pd.to_numeric(bs.loc["Net Debt"], errors="coerce").iloc[-1])
                    else:
                        debt = None
                        if "Total Debt" in bs.index:
                            debt = float(pd.to_numeric(bs.loc["Total Debt"], errors="coerce").iloc[-1])
                        else:
                            sd = float(pd.to_numeric(bs.loc.get("Short Long Term Debt", pd.Series([0])), errors="coerce").iloc[-1]) if "Short Long Term Debt" in bs.index else 0.0
                            ld = float(pd.to_numeric(bs.loc.get("Long Term Debt", pd.Series([0])), errors="coerce").iloc[-1]) if "Long Term Debt" in bs.index else 0.0
                            debt = sd + ld
                        cash = None
                        for k in ["Cash And Cash Equivalents", "Cash And Cash Equivalents Including Marketable Securities", "Cash"]:
                            if k in bs.index:
                                cash = float(pd.to_numeric(bs.loc[k], errors="coerce").iloc[-1])
                                break
                        if debt is not None:
                            net_debt = (debt or 0.0) - (cash or 0.0)

                # ---- Cash flow -> FCF (CFO + CapEx) ----
                cf = t.cashflow
                fcf = None
                if isinstance(cf, pd.DataFrame) and not cf.empty:
                    cf = cf.loc[:, sorted(cf.columns)] if len(cf.columns) > 1 else cf
                    op = None
                    capex = None
                    for k in ["Total Cash From Operating Activities", "Operating Cash Flow", "Cash Flow From Operating Activities"]:
                        if k in cf.index:
                            op = pd.to_numeric(cf.loc[k], errors="coerce")
                            break
                    for k in ["Capital Expenditures", "Capital Expenditure"]:
                        if k in cf.index:
                            capex = pd.to_numeric(cf.loc[k], errors="coerce")
                            break
                    if op is not None and capex is not None:
                        fcf = (op + capex).dropna()
                    elif "Free Cash Flow" in cf.index:
                        fcf = pd.to_numeric(cf.loc["Free Cash Flow"], errors="coerce").dropna()

                if fcf is None or len(fcf) < 2:
                    st.warning(f"{tk}: Not enough FCF history for DCF.")
                    continue

                fcf = fcf.sort_index()
                # Fix unit outliers (sometimes values are in thousands)
                if fcf.abs().median() > 1e12:
                    fcf = fcf / 1_000.0
                fcf_series = pd.Series(fcf.values.astype(float), index=pd.to_datetime(fcf.index))
                last_fcf = float(fcf_series.iloc[-1])

                # ---- Growth: rolling or custom schedule ----
                growths = []
                if roll_period_str.strip():
                    roll_n = max(2, int(float(roll_period_str)))
                    yoy = fcf_series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                    avg_g = float(yoy.rolling(window=roll_n, min_periods=1).mean().iloc[-1])
                    avg_g = float(np.clip(avg_g, -0.30, 0.30))
                    growths = [avg_g] * years_to_forecast_adv
                else:
                    parts = [p.strip() for p in (growth_sched_str or "").split(",") if p.strip()]
                    vals = [float(p) for p in parts] if parts else [0.05]
                    if len(vals) == 1:
                        g0 = float(vals[0])
                        growths = [g0] * years_to_forecast_adv
                        avg_g = g0
                    elif len(vals) >= 3:
                        g0, red, step = float(vals[0]), float(vals[1]), max(1, int(vals[2]))
                        growths = []
                        for i in range(1, years_to_forecast_adv + 1):
                            k = (i - 1) // step
                            gi = g0 - k * red
                            growths.append(float(np.clip(gi, -0.30, 0.30)))
                        avg_g = np.mean(growths[-min(3, len(growths)):])
                    else:
                        growths = [0.05] * years_to_forecast_adv
                        avg_g = 0.05

                # ---- WACC ----
                rf = _to_float(risk_free_rates[idx])

                try:
                    _spx = yf.download("^GSPC", period="5y", auto_adjust=False, progress=False)["Adj Close"].pct_change(fill_method=None).dropna()
                    mrp = _to_float(_spx.mean() * 252.0) - rf
                except Exception:
                    mrp = 0.05

                # Beta yahoo
                beta = t.fast_info.get("beta", None)
                if beta in (None, 0, np.nan) and hasattr(t, "info"):
                    beta = t.info.get("beta", None)
                beta = _to_float(beta)
                if np.isnan(beta) or beta == 0:
                    beta = 1.0

                Re = rf + float(beta) * mrp
                Rd = rf + float(credit_spreads[idx])
                tr = float(tax_rates[idx])

                if de is None:
                    wD, wE = 0.30, 0.70
                else:
                    wD = float(de) / (1.0 + float(de))
                    wE = 1.0 - wD
                wacc = float(np.clip(wE * Re + wD * Rd * (1.0 - tr), 0.03, 0.25))

                # ---- Forecast with per-year growths ----
                proj_years = list(range(1, int(years_to_forecast_adv) + 1))
                proj_fcfs = []
                prev = last_fcf
                for gi in growths:
                    nxt = prev * (1.0 + gi)
                    proj_fcfs.append(nxt)
                    prev = nxt

                # Terminal growth: if 0 years, fall back to avg_g
                if years_to_forecast_adv == 0:
                    g_terminal = float(np.clip(avg_g, -0.02, 0.04))
                else:
                    g_terminal = float(np.clip(growths[-1], -0.02, 0.04))
                if g_terminal >= wacc:
                     g_terminal = max(wacc - 0.01, -0.02)

                denom = max(wacc - g_terminal, 1e-9)

                if years_to_forecast_adv == 0:
                    pv_proj = float(last_fcf)
                    terminal_value = float(last_fcf) * (1.0 + float(g_terminal)) / denom
                    pv_terminal_value = terminal_value
                else:
                    pv_proj = sum(f / ((1.0 + wacc) ** i) for i, f in zip(proj_years, proj_fcfs))
                    terminal_value = proj_fcfs[-1] * (1.0 + g_terminal) / (wacc - g_terminal)
                    pv_terminal_value = terminal_value / ((1.0 + wacc) ** proj_years[-1])

                enterprise_value = pv_proj + pv_terminal_value
                equity_value = enterprise_value - net_debt if net_debt is not None else enterprise_value

                shares = t.fast_info.get("sharesOutstanding", None) or t.info.get("sharesOutstanding", None)
                price = t.fast_info.get("lastPrice", None)
                if price in (None, 0):
                    try:
                        price = float(t.history(period="1d")["Close"].iloc[-1])
                    except Exception:
                        price = None
                per_share = equity_value / float(shares) if shares not in (None, 0) else np.nan

                rows_adv.append({
                    "Ticker": tk,
                    "WACC": wacc,
                    "Beta": beta,
                    "Credit Spread": credit_spreads[idx],
                    "Rolling/Avg Growth": avg_g,
                    "Terminal g": g_terminal,
                    "PV Projections ($)": pv_proj,
                    "PV Terminal ($)": pv_terminal_value,
                    "EV ($)": enterprise_value,
                    "Equity Value ($)": equity_value,
                    "Calc Price/Share ($)": per_share,
                    "Actual Price/Share ($)": price,
                    "Shares Outstanding": float(shares) if shares not in (None, 0) else np.nan
                })

                # ---- Per-year export table ----
                base_date = pd.to_datetime(fcf.index[-1]) if len(fcf.index) else pd.Timestamp.today()
                if years_to_forecast_adv == 0:
                    proj_df = pd.DataFrame({
                        "Year #": [0],
                        "Fiscal Year": [f"FY {int(base_date.year)}"],
                        "Starting FCF ($)": [last_fcf],
                        "Growth Used": [avg_g],
                        "FCF Forecast ($)": [last_fcf],
                        "Discount Factor": [1.0],
                        "PV of FCF ($)": [last_fcf],
                        "Cum PV of FCF ($)": [last_fcf],
                        "Terminal Value ($)": [np.nan],
                        "PV Terminal ($)": [np.nan],
                        "Row": "Projection"
                    })
                    terminal_row = pd.DataFrame({
                        "Year #": [0],
                        "Fiscal Year": [f"Terminal after FY {int(base_date.year)}"],
                        "Starting FCF ($)": [last_fcf],
                        "Growth Used": [g_terminal],
                        "FCF Forecast ($)": [last_fcf * (1.0 + g_terminal)],
                        "Discount Factor": [1.0],
                        "PV of FCF ($)": [np.nan],
                        "Cum PV of FCF ($)": [last_fcf],
                        "Terminal Value ($)": [terminal_value],
                        "PV Terminal ($)": [pv_terminal_value],
                        "Row": ["Terminal"]
                    })
                    proj_df = pd.concat([proj_df, terminal_row], ignore_index=True)
                else:
                    fiscal_years = [f"FY {int(base_date.year) + i}" for i in proj_years]
                    disc_factors = [1.0 / ((1.0 + wacc) ** i) for i in proj_years]
                    pv_fcfs = [f * d for f, d in zip(proj_fcfs, disc_factors)]
                    cum_pv_fcfs = np.cumsum(pv_fcfs)
                    proj_df = pd.DataFrame({
                        "Year #": proj_years,
                        "Fiscal Year": fiscal_years,
                        "Starting FCF ($)": [last_fcf] + proj_fcfs[:-1],
                        "Growth Used": growths,
                        "FCF Forecast ($)": proj_fcfs,
                        "Discount Factor": disc_factors,
                        "PV of FCF ($)": pv_fcfs,
                        "Cum PV of FCF ($)": cum_pv_fcfs,
                        "Terminal Value ($)": [np.nan] * len(proj_years),
                        "PV Terminal ($)": [np.nan] * len(proj_years),
                        "Row": "Projection"
                    })
                    terminal_row = pd.DataFrame({
                        "Year #": [proj_years[-1]],
                        "Fiscal Year": [f"Terminal after FY {int(base_date.year) + proj_years[-1]}"],
                        "Starting FCF ($)": [proj_fcfs[-1]],
                        "Growth Used": [g_terminal],
                        "FCF Forecast ($)": [proj_fcfs[-1] * (1.0 + g_terminal)],
                        "Discount Factor": [disc_factors[-1]],
                        "PV of FCF ($)": [np.nan],
                        "Cum PV of FCF ($)": [cum_pv_fcfs[-1]],
                        "Terminal Value ($)": [terminal_value],
                        "PV Terminal ($)": [pv_terminal_value],
                        "Row": ["Terminal"]
                    })
                    proj_df = pd.concat([proj_df, terminal_row], ignore_index=True)

                ticker_summary = pd.DataFrame({
                    "Metric": [
                        "rf", "Market Risk Premium", "Beta", "Credit Spread", "Tax Rate", "D/E", "WACC",
                        "Avg/Rolling Growth", "Terminal g", "PV Projections ($)", "PV Terminal ($)",
                        "EV ($)", "Net Debt ($)", "Equity Value ($)", "Shares",
                        "Calc Price/Share ($)", "Actual Price/Share ($)", "Years"
                    ],
                    "Value": [
                        0.03, mrp, beta, credit_spreads[idx], tr, de, wacc,
                        avg_g, g_terminal, pv_proj, pv_terminal_value,
                        enterprise_value, net_debt, equity_value, shares,
                        per_share, price, years_to_forecast_adv
                    ]
                })

                projections_adv.append({"Ticker": tk, "projections": proj_df, "summary": ticker_summary})

            except Exception as e:
                st.warning(f"{tk}: Advanced DCF failed: {e}")

        if rows_adv:
            df_adv = pd.DataFrame(rows_adv)
            st.dataframe(
                df_adv.set_index("Ticker").style.format({
                    "WACC": "{:.2%}",
                    "Credit Spread": "{:.2%}",
                    "Tax Rate": "{:.2%}",
                    "Rolling/Avg Growth": "{:.2%}",
                    "Terminal g": "{:.2%}",
                    "PV Projections ($)": "${:,.0f}",
                    "PV Terminal ($)": "${:,.0f}",
                    "EV ($)": "${:,.0f}",
                    "Equity Value ($)": "${:,.0f}",
                    "Calc Price/Share ($)": "${:,.2f}",
                    "Actual Price/Share ($)": "${:,.2f}"
                }),
                width='stretch'
            )

            # Grouped bar: calculated vs actual price
            price_comp = df_adv.melt(
                id_vars="Ticker",
                value_vars=["Calc Price/Share ($)", "Actual Price/Share ($)"],
                var_name="Type",
                value_name="Price"
            )
            bar_adv = alt.Chart(price_comp).mark_bar().encode(
                x=alt.X("Ticker:N", title=""),
                xOffset=alt.XOffset("Type:N"),
                y=alt.Y("Price:Q", stack=None, axis=alt.Axis(format="$,.2f")),
                color=alt.Color("Type:N", legend=alt.Legend(title="")),
                tooltip=["Ticker:N", "Type:N", alt.Tooltip("Price:Q", format="$,.2f")]
            )
            st.altair_chart(bar_adv, width='stretch')

            if years_to_forecast_adv == 0:
                mc_df = df_adv.copy()
                if "Shares Outstanding" in mc_df.columns:
                    mc_df["Market Cap ($)"] = mc_df["Actual Price/Share ($)"] * mc_df["Shares Outstanding"]
                    mc_df = mc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Market Cap ($)", "EV ($)"])
                    if not mc_df.empty:
                        st.subheader("Market Cap vs Enterprise Value")
                        plot_mc_ev_adv = mc_df.melt(
                            id_vars="Ticker",
                            value_vars=["Market Cap ($)", "EV ($)"],
                            var_name="Metric",
                            value_name="Value"
                        )
                        mc_chart_adv = alt.Chart(plot_mc_ev_adv).mark_bar().encode(
                            x=alt.X("Ticker:N", title=""),
                            xOffset="Metric",
                            y=alt.Y("Value:Q", axis=alt.Axis(title="Value ($)", format="~s")),
                            color=alt.Color("Metric:N", legend=alt.Legend(title="")),
                            tooltip=["Ticker:N", "Metric:N", alt.Tooltip("Value:Q", format="$,.0f")]
                        ).properties(height=350)
                        st.altair_chart(mc_chart_adv, width='stretch')
                    else:
                        st.info("No data to plot Market Cap vs EV (Advanced).")
                else:
                    st.info("Shares Outstanding missing; cannot compute Market Cap (Advanced).")

# if weights are provided, parse and validate them
if weights_input.strip() == "":
    weights = [1.0 / len(tickers)] * len(tickers)
else:
    weights = []
    err = None
    try:
        raw = [w.strip() for w in weights_input.split(",") if w.strip()]
        weights = [float(w) for w in raw]
    except Exception:
        err = "Weights must be numeric and comma-separated."
    if len(tickers) != len(weights):
        err = "Number of tickers must match number of weights."
    if err is None and not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-8):
        err = "Weights must sum to 1."
    if err:
        st.error(err)
        st.stop()

# Download historical data (CLEAN, common-overlap)
adjusted_close, eff_start = download_adj_close_common(tickers, time, interval="1d")
if adjusted_close is None or adjusted_close.empty:
    st.error("No data returned. Check tickers or network.")
    st.stop()

# Build monthly returns from clean daily returns (no NaNs/Infs)
daily_ret = adjusted_close.pct_change(fill_method=None)
daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan).dropna(how="any")  # drop any row with a missing ticker
monthly_return = (1.0 + daily_ret).resample("ME").prod() - 1.0
monthly_return = monthly_return.replace([np.inf, -np.inf], np.nan).dropna(how="any")

if monthly_return.empty or not np.isfinite(monthly_return.to_numpy()).all():
    st.error("Not enough clean data after aligning tickers (NaNs/Infs present). Try a shorter horizon.")
    st.stop()



with port_tab:
# --- UI columns for comparison ---
    bottom_col = st.columns([1, 3])
    left_col = bottom_col[0].container()
    right_col = bottom_col[1].container()
# optimization method selection
method_mu = "hist"
method_cov = "hist"
rm_method = {
    'MSV': "MSV",
    "CVaR": "CVaR",
    "MAD": "MAD",
    "WR": "WR",
    "EVaR": "EVaR",
    "MDD": "MDD"
}
with left_col:
    rm = pills(
        "Risk Measure Method",
        options=list(rm_method.keys()),
        default="MSV"
    )
    rm_selected = rm_method[rm]

# optimizer settings
with left_col:
    st.markdown("##### Optimizer settings")
    obj_selected = st.selectbox(
        "Objective",
        options=["Sharpe", "MinRisk", "MaxRet", "MaxDiv", "Utility"],
        index=0,
        help="Objective function used by Riskfolio optimization."
    )
    alpha_selected = None
    if rm_selected in {"CVaR", "EVaR", "CDaR"}:
        alpha_selected = st.slider(
            "Alpha (tail probability)",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Used by CVaR/EVaR/CDaR risk measures."
        )

    # NEW: Lambda for Utility objective
    lambda_selected = None
    if obj_selected == "Utility":
        lambda_selected = st.number_input(
            "Lambda (risk aversion)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Only used for Utility objective. Maximizes E[R] - λ * Risk."
        )

# Optimize portfolio using Riskfolio
wt_opt = None
opt_w = None
try:
    portfolio_returns = rp.Portfolio(returns=monthly_return)
    portfolio_returns.assets_stats(method_mu=method_mu, method_cov=method_cov)

    # NEW: build kwargs from UI
    opt_kwargs = dict(model='Classic', rm=rm_selected, obj=obj_selected, rf=0, l=0)
    if alpha_selected is not None:
        portfolio_returns.alpha = float(alpha_selected)
    if lambda_selected is not None:
        portfolio_returns.l = float(lambda_selected)

    wt_opt = portfolio_returns.optimization(**opt_kwargs)

    # Align optimized weights with return columns
    if wt_opt is not None:
        if isinstance(wt_opt, pd.DataFrame):
            if 'weights' in wt_opt.columns:
                opt_w = wt_opt['weights'].reindex(monthly_return.columns).fillna(0.0).values
            else:
                opt_w = wt_opt.reindex(monthly_return.columns).fillna(0.0).values.ravel()
        elif isinstance(wt_opt, pd.Series):
            opt_w = wt_opt.reindex(monthly_return.columns).fillna(0.0).values
        else:
            arr = np.asarray(wt_opt).ravel()
            if arr.shape[0] == monthly_return.shape[1]:
                opt_w = arr
            else:
                opt_w = None
except Exception as e:
    st.warning(f"Optimization failed: {e}")
    wt_opt = None
    opt_w = None

# NEW: show optimized weights under the risk measure selector
with left_col:
    st.markdown("##### Optimized portfolio weights")
    if opt_w is not None:
        _weights_df = pd.DataFrame({
            "Ticker": list(monthly_return.columns),
            "Weight": np.asarray(opt_w, dtype=float)
        })
        _weights_df["Allocation ($)"] = _weights_df["Weight"] * float(initial_capital)
        st.dataframe(
            _weights_df.set_index("Ticker").style.format({
                "Weight": "{:.2%}",
                "Allocation ($)": "${:,.0f}"
            }),
            width='stretch'
        )
    else:
        st.info("No optimized weights available for this selection.")

# Compute user and optimized cumulative returns
user_w = np.array(weights, dtype=float)
if monthly_return.shape[1] != len(user_w):
    st.error("Monthly returns columns do not match the number of weights.")
    st.stop()

user_monthly_ret = monthly_return.dot(user_w)
user_cum = (1.0 + user_monthly_ret).cumprod() - 1.0

opt_cum = None
opt_monthly_ret = None
if opt_w is not None:
    opt_monthly_ret = monthly_return.dot(np.asarray(opt_w, dtype=float))
    opt_cum = (1.0 + opt_monthly_ret).cumprod() - 1.0

# Plot cumulative returns (user vs optimized)
plot_parts = []
plot_parts.append(pd.DataFrame({
    "Date": user_cum.index,
    "Portfolio": "User Portfolio",
    "Cumulative Return": user_cum.values
}))
if opt_cum is not None and len(opt_cum) > 0:
    plot_parts.append(pd.DataFrame({
        "Date": opt_cum.index,
        "Portfolio": "Optimized Portfolio",
        "Cumulative Return": opt_cum.values
    }))
plot_df = pd.concat(plot_parts, ignore_index=True)

with right_col:
    st.subheader("Cumulative Returns")
    chart = alt.Chart(plot_df).mark_line().encode(
        x='Date:T',
        y=alt.Y('Cumulative Return:Q', axis=alt.Axis(format='.0%')),
        color='Portfolio:N',
        tooltip=['Date:T', 'Portfolio:N', alt.Tooltip('Cumulative Return:Q', format='.2%')]
    ).interactive()
    st.altair_chart(chart, width='stretch')

# --- Current returns/value for User and Optimized portfolios ---
with right_col_columns:
    parent_cols = st.columns(2)

    # User portfolio metric (centered)
    inner = parent_cols[0].columns([1, 6, 1])
    with inner[1]:
        if user_cum is not None and len(user_cum) > 0:
            curr_user_ret = float(user_cum.iloc[-1])
            curr_user_val = initial_capital * (1.0 + curr_user_ret)
            last_period_ret = None
            try:
                last_period_ret = float(user_monthly_ret.iloc[-1])
            except Exception:
                last_period_ret = None
            delta_display = f"{last_period_ret:.2%}" if last_period_ret is not None else None
            st.metric(label="User portfolio Return",
                      value=f"{curr_user_ret:.2%}",
                      delta=delta_display)
            st.write(f"Value: ${curr_user_val:,.0f}")
        else:
            st.write("User portfolio: N/A")

    # Optimized portfolio metric (centered)
    inner = parent_cols[1].columns([1, 6, 1])
    with inner[1]:
        if opt_cum is not None and len(opt_cum) > 0:
            curr_opt_ret = float(opt_cum.iloc[-1])
            curr_opt_val = initial_capital * (1.0 + curr_opt_ret)
            last_period_ret = None
            try:
                last_period_ret = float(opt_monthly_ret.iloc[-1])
            except Exception:
                last_period_ret = None
            delta_display = f"{last_period_ret:.2%}" if last_period_ret is not None else None
            st.metric(label="Optimized portfolio Return",
                      value=f"{curr_opt_ret:.2%}",
                      delta=delta_display)
            st.write(f"Value: ${curr_opt_val:,.0f}")
        else:
            st.write("Optimized portfolio: N/A")

# ------------- Forecasts -------------
with port_tab:
    st.header("Future forecast for portfolio")

    col = st.columns([1, 1])
    left_col = col[0].container()
    right_col = col[1].container()

    # Only proceed if Darts is available
    if not DARTS_AVAILABLE:
        st.info("Forecasting models unavailable (Darts not installed). Skipping forecast sections.")
    else:
        # Helper: Darts TimeSeries from Series
        def to_ts(series: pd.Series):
            s = series.copy()
            s.index = pd.to_datetime(s.index)
            return TimeSeries.from_series(s)

        def _ts_to_df(ts, col_name="Forecast"):
            if ts is None:
                return None
            try:
                df = ts.pd_dataframe() if hasattr(ts, "pd_dataframe") else ts.to_dataframe()
            except Exception:
                df = pd.DataFrame(ts)
            df = df.reset_index().rename(columns={df.columns[0]: "Date", df.columns[-1]: col_name})
            return df

        series_user = to_ts(user_cum)
        series_opt = to_ts(opt_cum) if (opt_cum is not None and len(opt_cum) > 0) else None

        # Left: ETS/ARIMA/Prophet
        with left_col:
            model_choosen1 = {'ExponentialSmoothing': ExponentialSmoothing, 'AutoARIMA': AutoARIMA, 'Prophet': Prophet}
            model_name = pills("Forecasting Model", options=list(model_choosen1.keys()), default="ExponentialSmoothing")
            forecast_periods = st.number_input("Enter number of months to forecast", min_value=1, max_value=24, value=12, step=1, key="forecast_periods_main")

            def fit_predict(model_cls, ts, periods):
                try:
                    m = model_cls()
                    m.fit(ts)
                    return m.predict(periods)
                except Exception as e:
                    st.warning(f"{model_cls.__name__} failed: {e}")
                    return None

            # --- minimum history check for ETS (24 months) ---
            required_months_ets = 24
            len_months = len(series_user)
            if model_name == "ExponentialSmoothing" and len_months < required_months_ets:
                st.warning(f"Not enough monthly history ({len_months} < {required_months_ets}). ETS forecast disabled.")
                forecast_user_ts = None
                forecast_opt_ts = None
            else:
                forecast_user_ts = fit_predict(model_choosen1[model_name], series_user, forecast_periods)
                forecast_opt_ts = fit_predict(model_choosen1[model_name], series_opt, forecast_periods) if series_opt is not None else None
            forecast_df_user = _ts_to_df(forecast_user_ts, "Forecast") if forecast_user_ts is not None else None
            forecast_df_opt = _ts_to_df(forecast_opt_ts, "Forecast") if forecast_opt_ts is not None else None

            if forecast_df_user is None:
                st.info("No forecast available.")
            else:
                # Historical dataframes
                hist_user_df = pd.DataFrame({
                    "Date": pd.to_datetime(user_cum.index),
                    "User Portfolio": user_cum.values
                })
                hist_opt_df = None
                if opt_cum is not None and len(opt_cum) > 0:
                    hist_opt_df = pd.DataFrame({
                        "Date": pd.to_datetime(opt_cum.index),
                        "Optimized Portfolio": opt_cum.values
                    })

                # Forecast frames
                forecast_user_df = forecast_df_user[['Date', 'Forecast']].rename(columns={'Forecast': 'User Portfolio'})
                forecast_opt_df = forecast_df_opt[['Date', 'Forecast']].rename(columns={'Forecast': 'Optimized Portfolio'}) if forecast_df_opt is not None else None

                # Melt with Type flag
                hist_user_df['Type'] = 'Historical'
                forecast_user_df['Type'] = 'Forecast'

                dfs = []
                dfs.append(hist_user_df.melt(id_vars=['Date', 'Type'], var_name='Portfolio', value_name='Cumulative Return'))
                dfs.append(forecast_user_df.melt(id_vars=['Date', 'Type'], var_name='Portfolio', value_name='Cumulative Return'))
                if hist_opt_df is not None:
                    hist_opt_df['Type'] = 'Historical'
                    dfs.append(hist_opt_df.melt(id_vars=['Date', 'Type'], var_name='Portfolio', value_name='Cumulative Return'))
                if forecast_opt_df is not None:
                    forecast_opt_df['Type'] = 'Forecast'
                    dfs.append(forecast_opt_df.melt(id_vars=['Date', 'Type'], var_name='Portfolio', value_name='Cumulative Return'))

                plot_df_combined = pd.concat(dfs, ignore_index=True)

                st.subheader(f"Forecast ({model_name})")
                chart = alt.Chart(plot_df_combined).mark_line().encode(
                    x='Date:T',
                    y=alt.Y('Cumulative Return:Q', axis=alt.Axis(format='.0%')),
                    color='Portfolio:N',
                    strokeDash=alt.condition(alt.datum.Type == 'Forecast', alt.value([5, 5]), alt.value([])),
                    tooltip=['Date:T', 'Portfolio:N', alt.Tooltip('Cumulative Return:Q', format='.2%'), 'Type:N']
                ).interactive()
                st.altair_chart(chart, width='stretch')

        # Right: XGBModel / NBEATSModel
        with right_col:
            if not DARTS_AVAILABLE:
                st.info("ML forecast models unavailable.")
            else:
                model_choosen2 = {'XGBModel': 'XGBModel', 'NBEATSModel': 'NBEATSModel'}
                model_name2 = pills(
                    "Forecasting Model",
                    options=list(model_choosen2.keys()),
                    default="XGBModel"
                )
                forecast_periods2 = st.number_input("Enter number of months to forecast", min_value=1, max_value=24, value=12, step=1, key="forecast_periods_ml")
                simulation_amount = st.number_input("Enter number of simulations", min_value=10, max_value=100, value=25, step=10, key="simulation_amount_ml")
                lag_amount = st.number_input("Enter lag amount", min_value=1, max_value=24, value=6, step=6, key="lag_amount_ml")

                series_user_ml = series_user
                series_opt_ml = series_opt
                lag_amount = int(lag_amount)


                fut_index_user = None
                fut_vals_user = None
                fut_index_opt = None
                fut_vals_opt = None

                try:
                    # derive chunks safely from data and UI
                    min_len = len(series_user_ml)
                    if series_opt_ml is not None:
                        min_len = min(min_len, len(series_opt_ml))

                    # If too short, skip ML safely
                    if min_len < 2:
                        st.info("Not enough history (< 2 observations). Skipping ML forecast.")
                    else:
                        # Clamp lags/chunks to be trainable on the shortest series
                        safe_ic = max(1, min(int(lag_amount), min_len - 1))           # lookback
                        safe_oc = max(1, min(int(forecast_periods2), min_len - safe_ic))  # horizon per step
                        xgb_lags = safe_ic  # keep XGB consistent with lookback

                        if model_name2 == 'XGBModel':
                            xgb_user = XGBModel(lags=int(xgb_lags))
                            xgb_user.fit(series_user_ml)
                            xgb_forecast_user = xgb_user.predict(forecast_periods2)
                            fut_index_user = xgb_forecast_user.time_index
                            fut_vals_user = xgb_forecast_user.values().ravel()

                            if series_opt_ml is not None:
                                xgb_opt = XGBModel(lags=int(xgb_lags))
                                xgb_opt.fit(series_opt_ml)
                                xgb_forecast_opt = xgb_opt.predict(forecast_periods2)
                                fut_index_opt = xgb_forecast_opt.time_index
                                fut_vals_opt = xgb_forecast_opt.values().ravel()

                        elif model_name2 == 'NBEATSModel':
                            nbeats_user = NBEATSModel(input_chunk_length=int(safe_ic),
                                                      output_chunk_length=int(safe_oc),
                                                      n_epochs=50, random_state=0)
                            nbeats_user.fit(series_user_ml)
                            nbeats_forecast_user = nbeats_user.predict(forecast_periods2)
                            fut_index_user = nbeats_forecast_user.time_index
                            fut_vals_user = nbeats_forecast_user.values().ravel()

                            if series_opt_ml is not None:
                                nbeats_opt = NBEATSModel(input_chunk_length=int(safe_ic),
                                                         output_chunk_length=int(safe_oc),
                                                         n_epochs=50, random_state=0)
                                nbeats_opt.fit(series_opt_ml)
                                nbeats_forecast_opt = nbeats_opt.predict(forecast_periods2)
                                fut_index_opt = nbeats_forecast_opt.time_index
                                fut_vals_opt = nbeats_forecast_opt.values().ravel()
                except Exception as e:
                    st.error(f"Could not build {model_name2} forecast: {e}")

                try:
                    if fut_index_user is None or fut_vals_user is None:
                        st.info("No ML forecast available.")
                    else:
                        fut_dates_user = pd.to_datetime(list(fut_index_user))
                        forecast_df_ml_user = pd.DataFrame({
                            "Date": fut_dates_user,
                            "Forecast": np.asarray(fut_vals_user)
                        })

                        forecast_df_ml_opt = None
                        if fut_index_opt is not None and fut_vals_opt is not None:
                            fut_dates_opt = pd.to_datetime(list(fut_index_opt))
                            forecast_df_ml_opt = pd.DataFrame({
                                "Date": fut_dates_opt,
                                "Forecast": np.asarray(fut_vals_opt)
                            })

                        hist_user_df = pd.DataFrame({
                            "Date": pd.to_datetime(user_cum.index),
                            "User Portfolio": user_cum.values
                        })
                        hist_opt_df = None
                        if opt_cum is not None and len(opt_cum) > 0:
                            hist_opt_df = pd.DataFrame({
                                "Date": pd.to_datetime(opt_cum.index),
                                "Optimized Portfolio": opt_cum.values
                            })

                        forecast_user_df = forecast_df_ml_user.rename(columns={"Forecast": "User Portfolio"})
                        forecast_opt_df = forecast_df_ml_opt.rename(columns={"Forecast": "Optimized Portfolio"}) if forecast_df_ml_opt is not None else None

                        hist_user_df['Type'] = 'Historical'
                        forecast_user_df['Type'] = 'Forecast'

                        dfs_ml = []
                        dfs_ml.append(hist_user_df.melt(id_vars=['Date', 'Type'], var_name='Portfolio', value_name='Cumulative Return'))
                        dfs_ml.append(forecast_user_df.melt(id_vars=['Date', 'Type'], var_name='Portfolio', value_name='Cumulative Return'))
                        if hist_opt_df is not None:
                            hist_opt_df['Type'] = 'Historical'
                            dfs_ml.append(hist_opt_df.melt(id_vars=['Date', 'Type'], var_name='Portfolio', value_name='Cumulative Return'))
                        if forecast_opt_df is not None:
                            forecast_opt_df['Type'] = 'Forecast'
                            dfs_ml.append(forecast_opt_df.melt(id_vars=['Date', 'Type'], var_name='Portfolio', value_name='Cumulative Return'))

                        plot_df_combined_ml = pd.concat(dfs_ml, ignore_index=True)

                        chart_ml = alt.Chart(plot_df_combined_ml).mark_line().encode(
                            x='Date:T',
                            y=alt.Y('Cumulative Return:Q', axis=alt.Axis(format='.0%')),
                            color='Portfolio:N',
                            strokeDash=alt.condition(alt.datum.Type == 'Forecast', alt.value([5, 5]), alt.value([])),
                            tooltip=['Date:T', 'Portfolio:N', alt.Tooltip('Cumulative Return:Q', format='.2%'), 'Type:N']
                        ).interactive()

                        st.subheader(f"Forecast ({model_name2})")
                        st.altair_chart(chart_ml, width='stretch')
                except Exception as e:
                    st.error(f"Could not build forecast chart: {e}")

# ---- Technical Analysis (refactored) ----
with Tech_tab:
    st.write("## Technical Analysis")
    ta_interval = st.selectbox("Interval", ["1d", "1wk"], index=0, key="ta_interval")
    # Simplified names
    ind_options = ["SMA", "EMA", "BBands", "VWAP", "RSI", "MACD", "Volume", "Signals"]
    default_inds = []

    ta_tabs = st.tabs(tickers)
    for tk, ttab in zip(tickers, ta_tabs):
        with ttab:
            raw = yf.download(tk, period="max", interval=ta_interval, auto_adjust=False, progress=False)
            if raw is None or raw.empty:
                st.warning(f"No data for {tk} (interval={ta_interval}).")
                continue

            # MultiIndex safety
            if isinstance(raw.columns, pd.MultiIndex):
                try:
                    ohlc = raw.xs(tk, level=1, axis=1)
                except Exception:
                    ohlc = pd.DataFrame({
                        "Open": raw["Open"].iloc[:, 0],
                        "High": raw["High"].iloc[:, 0],
                        "Low": raw["Low"].iloc[:, 0],
                        "Close": raw["Close"].iloc[:, 0],
                        "Volume": raw["Volume"].iloc[:, 0],
                    })
            else:
                ohlc = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

            # Trim to horizon
            start_h = _period_to_start(time)
            if start_h is not None:
                ohlc = ohlc.loc[ohlc.index >= start_h]

            ohlc = ohlc.apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
            if ohlc.empty:
                st.warning(f"No numeric OHLC after cleaning for {tk}.")
                continue

            intraday = ta_interval not in ("1d", "1wk")
            tvals = _to_lwc_time(ohlc.index, intraday=intraday)

            # UI
            left_ui, right_ui = st.columns([2, 1])
            with left_ui:
                indicators = st.multiselect(f"Indicators — {tk}", ind_options, default=default_inds, key=f"inds2_{tk}")
            with right_ui:
                if "SMA" in indicators:
                    sma_short = st.number_input("SMA Short", 2, 200, 20, key=f"sma_short_{tk}")
                    sma_long  = st.number_input("SMA Long",  2, 400, 50, key=f"sma_long_{tk}")
                else:
                    sma_short = sma_long = None
                if "EMA" in indicators:
                    ema_short = st.number_input("EMA Short", 2, 200, 20, key=f"ema_short_{tk}")
                    ema_long  = st.number_input("EMA Long",  2, 400, 50, key=f"ema_long_{tk}")
                else:
                    ema_short = ema_long = None
                if "RSI" in indicators or "Signals" in indicators:
                    rsi_len = st.number_input("RSI Length", 2, 100, 14, key=f"rsi_len_{tk}")
                else:
                    rsi_len = None
                # Signal rule controls
                if "Signals" in indicators:
                    sig_rule = st.selectbox(
                        "Signal rule",
                        [
                            "EMA Cross",
                            "EMA Cross + RSI filter",
                            "SMA Cross",
                            "SMA Cross + RSI filter",
                            "SMA+EMA Alignment",
                            "Price vs VWAP",
                            "Bollinger Breakout",
                            "MACD Signal Cross",
                            "MACD+RSI Confirmation",
                            "RSI 30/70 Cross"
                        ],
                        key=f"sig_rule_{tk}"
                    )
                    rsi_buy_th = st.number_input("RSI buy ≤", 2, 98, 30, key=f"sig_rsi_buy_{tk}")
                    rsi_sell_th = st.number_input("RSI sell ≥", 2, 98, 70, key=f"sig_rsi_sell_{tk}")
                else:
                    sig_rule, rsi_buy_th, rsi_sell_th = None, None, None

            # Build a key that changes when indicator set changes
            key_suffix = "-".join(sorted(indicators + ([sig_rule] if sig_rule else []))) if indicators else "none"

            # Candles always
            candles = [
                {"time": tvals[i], "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
                for i, (o, h, l, c) in enumerate(zip(ohlc.Open, ohlc.High, ohlc.Low, ohlc.Close))
            ]
            series_price = [{"type": "Candlestick", "data": candles}]

            close = ohlc.Close

            # SMA overlays
            if "SMA" in indicators:
                for w, col in [(sma_short, "#ffc107"), (sma_long, "#ff9800")]:
                    if w and w > 1:
                        s = _sma(close, w).dropna()
                        series_price.append({
                            "type": "Line",
                            "data": [{"time": (t.date().isoformat() if not intraday else int(t.timestamp())), "value": float(v)} for t, v in s.items()],
                            "options": {"color": col, "lineWidth": 2}
                        })

            # EMA overlays
            ema_short_eff = ema_short or 20
            ema_long_eff = ema_long or 50
            if "EMA" in indicators:
                for w, col in [(ema_short_eff, "#42a5f5"), (ema_long_eff, "#1e88e5")]:
                    if w and w > 1:
                        e = _ema(close, w).dropna()
                        series_price.append({
                            "type": "Line",
                            "data": [{"time": (t.date().isoformat() if not intraday else int(t.timestamp())), "value": float(v)} for t, v in e.items()],
                            "options": {"color": col, "lineWidth": 2}
                        })

            # VWAP overlay
            if "VWAP" in indicators:
                vwap_line = _vwap(ohlc, intraday).dropna()
                series_price.append({
                    "type": "Line",
                    "data": [{"time": (t.date().isoformat() if not intraday else int(t.timestamp())), "value": float(v)}
                             for t, v in vwap_line.items()],
                    "options": {"color": "#e91e63", "lineWidth": 2}
                })

            # Signals (buy/sell markers on price pane)
            if "Signals" in indicators:
                markers = []
                # Precompute commonly used series
                e_fast = _ema(close, ema_short_eff)
                e_slow = _ema(close, ema_long_eff)
                s_fast = _sma(close, sma_short or 20)
                s_slow = _sma(close, sma_long or 50)
                r_series = _rsi(close, rsi_len or 14)
                macd_line, signal_line, macd_hist = _macd(close)
                vwap_series = _vwap(ohlc, intraday)
                bb_mid, bb_up, bb_lo = _bbands(close, 20, 2.0)

                if sig_rule in ("EMA Cross", "EMA Cross + RSI filter"):
                    buy = _cross_up(e_fast, e_slow)
                    sell = _cross_down(e_fast, e_slow)
                    if sig_rule.endswith("RSI filter"):
                        buy = buy & (r_series <= (rsi_buy_th or 30))
                        sell = sell & (r_series >= (rsi_sell_th or 70))
                    markers = _markers_from_signals(ohlc.index, buy.fillna(False), sell.fillna(False), intraday)

                elif sig_rule in ("SMA Cross", "SMA Cross + RSI filter"):
                    buy = _cross_up(s_fast, s_slow)
                    sell = _cross_down(s_fast, s_slow)
                    if sig_rule.endswith("RSI filter"):
                        buy = buy & (r_series <= (rsi_buy_th or 30))
                        sell = sell & (r_series >= (rsi_sell_th or 70))
                    markers = _markers_from_signals(ohlc.index, buy.fillna(False), sell.fillna(False), intraday)

                elif sig_rule == "SMA+EMA Alignment":
                    align_up = (e_fast > e_slow) & (s_fast > s_slow) & (close > e_fast) & (close > s_fast)
                    align_dn = (e_fast < e_slow) & (s_fast < s_slow) & (close < e_fast) & (close < s_fast)
                    buy = align_up & (~align_up.shift(1).fillna(False))
                    sell = align_dn & (~align_dn.shift(1).fillna(False))
                    markers = _markers_from_signals(ohlc.index, buy.fillna(False), sell.fillna(False), intraday)

                elif sig_rule == "Price vs VWAP":
                    buy = _cross_up(close, vwap_series)
                    sell = _cross_down(close, vwap_series)
                    markers = _markers_from_signals(ohlc.index, buy.fillna(False), sell.fillna(False), intraday)

                elif sig_rule == "Bollinger Breakout":
                    buy = _cross_up(close, bb_up)
                    sell = _cross_down(close, bb_lo)
                    markers = _markers_from_signals(ohlc.index, buy.fillna(False), sell.fillna(False), intraday)

                elif sig_rule == "MACD Signal Cross":
                    buy = _cross_up(macd_line, signal_line)
                    sell = _cross_down(macd_line, signal_line)
                    markers = _markers_from_signals(ohlc.index, buy.fillna(False), sell.fillna(False), intraday)

                elif sig_rule == "MACD+RSI Confirmation":
                    buy = _cross_up(macd_line, signal_line) & (r_series > 50)
                    sell = _cross_down(macd_line, signal_line) & (r_series < 50)
                    markers = _markers_from_signals(ohlc.index, buy.fillna(False), sell.fillna(False), intraday)

                elif sig_rule == "RSI 30/70 Cross":
                    buy = _cross_up(r_series, pd.Series(rsi_buy_th or 30, index=r_series.index))
                    sell = _cross_down(r_series, pd.Series(rsi_sell_th or 70, index=r_series.index))
                    markers = _markers_from_signals(ohlc.index, buy.fillna(False), sell.fillna(False), intraday)

                if markers:
                    series_price[0]["markers"] = markers

            # Bollinger Bands
            if "BBands" in indicators:
                m, up, lo = _bbands(close, 20, 2.0)
                for ser, color in ((m, "#bdbdbd"), (up, "#9e9e9e"), (lo, "#9e9e9e")):
                    s = ser.dropna()
                    series_price.append({
                        "type": "Line",
                        "data": [{"time": (t.date().isoformat() if not intraday else int(t.timestamp())), "value": float(v)} for t, v in s.items()],
                        "options": {"color": color, "lineWidth": 1}
                    })

            # Volume histogram
            if "Volume" in indicators and "Volume" in ohlc.columns:
                vol_colors = np.where(close >= ohlc.Open, "#26a69a", "#ef5350").tolist()
                vols = [{"time": tvals[i], "value": float(v), "color": vol_colors[i]} for i, v in enumerate(ohlc.Volume.fillna(0))]
                series_price.append({
                    "type": "Histogram",
                    "data": vols,
                    "options": {
                        "priceFormat": {"type": "volume"},
                        "priceScaleId": "",
                        "scaleMargins": {"top": 0.80, "bottom": 0.0}
                    }
                })

            charts = [{
                "chart": {
                    "layout": {"background": {"type": "solid", "color": "#010b13"}, "textColor": "#ffffff"},
                    "grid": {"vertLines": {"color": "rgba(255,255,255,0.05)"}, "horzLines": {"color": "rgba(255,255,255,0.05)"}},
                    "timeScale": {"borderColor": "rgba(255,255,255,0.15)"},
                    "rightPriceScale": {"borderColor": "rgba(255,255,255,0.15)"},
                    "height": 520
                },
                "series": series_price
            }]

            # RSI pane
            if "RSI" in indicators and rsi_len:
                r = _rsi(close, rsi_len).dropna()
                # constant 70/30 guides
                t_series = [(t.date().isoformat() if not intraday else int(t.timestamp())) for t in r.index]
                line70 = [{"time": t, "value": 70.0} for t in t_series]
                line30 = [{"time": t, "value": 30.0} for t in t_series]

                rsi_series = [
                    {
                        "type": "Line",
                        "data": [{"time": (t.date().isoformat() if not intraday else int(t.timestamp())), "value": float(v)} for t, v in r.items()],
                        "options": {"color": "#ab47bc", "lineWidth": 2}
                    },
                    {
                        "type": "Line",
                        "data": line70,
                        "options": {"color": "#9e9e9e", "lineWidth": 1, "lineStyle": 1}  # dashed
                    },
                    {
                        "type": "Line",
                        "data": line30,
                        "options": {"color": "#9e9e9e", "lineWidth": 1, "lineStyle": 1}  # dashed
                    }
                ]
                charts.append({
                    "chart": {
                        "layout": {"background": {"type": "solid", "color": "#010b13"}, "textColor": "#ffffff"},
                        "grid": {"vertLines": {"color": "rgba(255,255,255,0.05)"}, "horzLines": {"color": "rgba(255,255,255,0.05)"}},
                        "timeScale": {"visible": False},
                        "rightPriceScale": {"borderColor": "rgba(255,255,255,0.15)"},
                        "height": 160
                    },
                    "series": rsi_series
                })

            # MACD pane
            if "MACD" in indicators:
                m_line, s_line, hist = _macd(close)
                md = m_line.dropna()
                sd = s_line.reindex(md.index).dropna()
                hd = hist.reindex(md.index).dropna()
                macd_series = [
                    {"type": "Line",
                     "data": [{"time": (t.date().isoformat() if not intraday else int(t.timestamp())), "value": float(v)} for t, v in md.items()],
                     "options": {"color": "#26c6da", "lineWidth": 2}},
                    {"type": "Line",
                     "data": [{"time": (t.date().isoformat() if not intraday else int(t.timestamp())), "value": float(v)} for t, v in sd.items()],
                     "options": {"color": "#ef5350", "lineWidth": 2}},
                    {"type": "Histogram",
                     "data": [{"time": (t.date().isoformat() if not intraday else int(t.timestamp())), "value": float(v),
                               "color": "#26a69a" if v >= 0 else "#ef5350"} for t, v in hd.items()]}
                ]
                charts.append({
                    "chart": {
                        "layout": {"background": {"type": "solid", "color": "#010b13"}, "textColor": "#ffffff"},
                        "grid": {"vertLines": {"color": "rgba(255,255,255,0.05)"}, "horzLines": {"color": "rgba(255,255,255,0.05)"}},
                        "timeScale": {"visible": False},
                        "rightPriceScale": {"borderColor": "rgba(255,255,255,0.15)"},
                        "height": 180
                    },
                    "series": macd_series
                })

            # --- Legend (HTML badges matching series colors) ---
            def _badge(txt, color):
                return f'<span style="display:inline-flex;align-items:center;margin-right:12px;margin-bottom:6px;">' \
                       f'<span style="display:inline-block;width:12px;height:12px;background:{color};border-radius:2px;margin-right:6px;"></span>{txt}</span>'

            legend_items = []
            legend_items.append(_badge("Price (Candles)", "#26a69a"))
            if "SMA" in indicators:
                if sma_short and sma_short > 1:
                    legend_items.append(_badge(f"SMA Short ({int(sma_short)})", "#ffc107"))
                if sma_long and sma_long > 1:
                    legend_items.append(_badge(f"SMA Long ({int(sma_long)})", "#ff9800"))
            if "EMA" in indicators:
                if ema_short and ema_short > 1:
                    legend_items.append(_badge(f"EMA Short ({int(ema_short)})", "#42a5f5"))
                if ema_long and ema_long > 1:
                    legend_items.append(_badge(f"EMA Long ({int(ema_long)})", "#1e88e5"))
            if "BBands" in indicators:
                legend_items.append(_badge("BB Upper", "#9e9e9e"))
                legend_items.append(_badge("BB Mid", "#bdbdbd"))
                legend_items.append(_badge("BB Lower", "#9e9e9e"))
            if "Volume" in indicators and "Volume" in ohlc.columns:
                legend_items.append(_badge("Volume (up/down)", "linear-gradient(90deg,#26a69a 50%,#ef5350 50%)"))
            if "RSI" in indicators and rsi_len:
                legend_items.append(_badge(f"RSI ({int(rsi_len)})", "#ab47bc"))
                legend_items.append(_badge("RSI 70/30", "#9e9e9e"))
            if "MACD" in indicators:
                legend_items.append(_badge("MACD", "#26c6da"))
                legend_items.append(_badge("Signal", "#ef5350"))
                legend_items.append(_badge("MACD Hist", "linear-gradient(90deg,#26a69a 50%,#ef5350 50%)"))
            if "Signals" in indicators and sig_rule:
                legend_items.append(_badge(f"Signals ({sig_rule})", "linear-gradient(90deg,#26a69a 50%,#ef5350 50%)"))
            if "VWAP" in indicators:
                legend_items.append(_badge("VWAP", "#e91e63"))

            st.markdown(
                "<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:6px 0;'>"
                + "".join(legend_items) +
                "</div>",
                unsafe_allow_html=True
            )

            # Render (always at least price chart)
            renderLightweightCharts(charts, key=f"lwc_{tk}_{key_suffix}")
