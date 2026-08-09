"""
Stock Analyzer V2
Application Streamlit d'analyse technique et de backtest.
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta_classic as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Analyzer V2",
    page_icon="📈",
    layout="wide",
)

TRADING_DAYS = 252

DEFAULT_TICKERS = (
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,"
    "AVGO,AMD,JPM,AIR.PA,MC.PA,TTE.PA,OR.PA"
)


# ============================================================
# OUTILS
# ============================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def fmt(value, decimals=2):
    value = safe_float(value)
    return "N/A" if pd.isna(value) else f"{value:.{decimals}f}"


# ============================================================
# TELECHARGEMENT
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def download_data(ticker, period, interval):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    data = data[required].copy()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data.index = pd.to_datetime(data.index)

    return data


# ============================================================
# INDICATEURS
# ============================================================

def calculate_indicators(data):
    df = data.copy()

    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)

    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(
        df["Close"],
        fast=12,
        slow=26,
        signal=9,
    )

    if macd is not None and not macd.empty:
        macd_main = [c for c in macd.columns if str(c).startswith("MACD_")]
        macd_hist = [c for c in macd.columns if str(c).startswith("MACDh_")]
        macd_signal = [c for c in macd.columns if str(c).startswith("MACDs_")]

        if macd_main:
            df["MACD"] = macd[macd_main[0]]
        if macd_hist:
            df["MACD_HIST"] = macd[macd_hist[0]]
        if macd_signal:
            df["MACD_SIGNAL"] = macd[macd_signal[0]]

    bb = ta.bbands(
        df["Close"],
        length=20,
        std=2,
    )

    if bb is not None and not bb.empty:
        lower = [c for c in bb.columns if str(c).startswith("BBL")]
        middle = [c for c in bb.columns if str(c).startswith("BBM")]
        upper = [c for c in bb.columns if str(c).startswith("BBU")]

        if lower:
            df["BB_LOWER"] = bb[lower[0]]
        if middle:
            df["BB_MIDDLE"] = bb[middle[0]]
        if upper:
            df["BB_UPPER"] = bb[upper[0]]

    df["ATR"] = ta.atr(
        df["High"],
        df["Low"],
        df["Close"],
        length=14,
    )

    df["VOLUME_SMA20"] = ta.sma(df["Volume"], length=20)
    df["VOLUME_RATIO"] = df["Volume"] / df["VOLUME_SMA20"]

    df["RETURN"] = df["Close"].pct_change()

    df["VOLATILITY_20"] = (
        df["RETURN"].rolling(20).std()
        * np.sqrt(TRADING_DAYS)
        * 100
    )

    return df


# ============================================================
# SCORE
# ============================================================

def calculate_score(df):
    if df.empty:
        return 0, "⚪ N/A", {}

    row = df.iloc[-1]
    close = safe_float(row["Close"])

    score = 0
    details = {}

    # Tendance : 40
    points = 0
    sma20 = safe_float(row.get("SMA_20"))
    sma50 = safe_float(row.get("SMA_50"))
    sma200 = safe_float(row.get("SMA_200"))

    if pd.notna(sma20) and close > sma20:
        points += 10
    if pd.notna(sma50) and close > sma50:
        points += 15
    if pd.notna(sma200) and close > sma200:
        points += 10
    if pd.notna(sma50) and pd.notna(sma200) and sma50 > sma200:
        points += 5

    score += points
    details["Tendance"] = points

    # RSI : 20
    rsi = safe_float(row.get("RSI"))
    points = 0

    if pd.notna(rsi):
        if 45 <= rsi <= 60:
            points = 20
        elif 35 <= rsi < 45:
            points = 15
        elif 60 < rsi <= 70:
            points = 15
        elif 30 <= rsi < 35:
            points = 10
        elif rsi < 30:
            points = 12
        elif rsi > 70:
            points = 5

    score += points
    details["RSI"] = points

    # MACD : 15
    macd = safe_float(row.get("MACD"))
    macd_signal = safe_float(row.get("MACD_SIGNAL"))
    points = 0

    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal and macd > 0:
            points = 15
        elif macd > macd_signal:
            points = 10
        else:
            points = 3

    score += points
    details["MACD"] = points

    # Bollinger : 10
    lower = safe_float(row.get("BB_LOWER"))
    upper = safe_float(row.get("BB_UPPER"))
    points = 0

    if pd.notna(lower) and pd.notna(upper) and upper > lower:
        position = (close - lower) / (upper - lower)
        if 0.30 <= position <= 0.70:
            points = 10
        elif 0.15 <= position < 0.30:
            points = 8
        elif 0.70 < position <= 0.85:
            points = 7
        elif position < 0.15:
            points = 6
        else:
            points = 3

    score += points
    details["Bollinger"] = points

    # Volume : 10
    volume_ratio = safe_float(row.get("VOLUME_RATIO"))
    points = 0

    if pd.notna(volume_ratio):
        if volume_ratio >= 1.5:
            points = 10
        elif volume_ratio >= 1.0:
            points = 8
        elif volume_ratio >= 0.7:
            points = 5
        else:
            points = 2

    score += points
    details["Volume"] = points

    # Momentum : 5
    points = 0
    if len(df) >= 21:
        old_price = safe_float(df["Close"].iloc[-21])
        if pd.notna(old_price) and old_price > 0:
            momentum = (close / old_price - 1) * 100
            if momentum > 5:
                points = 5
            elif momentum > 0:
                points = 3
            else:
                points = 1

    score += points
    details["Momentum"] = points

    score = int(max(0, min(100, round(score))))

    if score >= 80:
        signal = "🟢 ACHAT FORT"
    elif score >= 65:
        signal = "🟢 ACHAT"
    elif score >= 50:
        signal = "🟡 NEUTRE"
    elif score >= 35:
        signal = "🟠 VENTE PRUDENTE"
    else:
        signal = "🔴 VENTE"

    return score, signal, details


# ============================================================
# TENDANCE
# ============================================================

def detect_trend(df):
    row = df.iloc[-1]
    close = safe_float(row["Close"])
    sma20 = safe_float(row.get("SMA_20"))
    sma50 = safe_float(row.get("SMA_50"))
    sma200 = safe_float(row.get("SMA_200"))

    if any(pd.isna(x) for x in [close, sma20, sma50, sma200]):
        return "⚪ Données insuffisantes"

    if close > sma20 > sma50 > sma200:
        return "🟢 Tendance fortement haussière"
    if close > sma50 > sma200:
        return "🟢 Tendance haussière"
    if close > sma200 and close < sma50:
        return "🟡 Correction dans tendance haussière"
    if close < sma200 and close > sma50:
        return "🟠 Reprise potentielle"
    if close < sma50 < sma200:
        return "🔴 Tendance baissière"

    return "🟡 Tendance neutre"


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_performance(equity):
    equity = equity.dropna()
    if len(equity) < 2:
        return {}

    start = safe_float(equity.iloc[0])
    end = safe_float(equity.iloc[-1])

    total_return = (end / start - 1) * 100
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25

    cagr = ((end / start) ** (1 / years) - 1) * 100
    returns = equity.pct_change().dropna()

    if len(returns) > 1 and returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)
    else:
        sharpe = np.nan

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
    }


# ============================================================
# BACKTEST (Optimisé pour éviter les boucles lentes)
# ============================================================

def backtest_strategy(df, initial_capital, transaction_cost, threshold):
    data = df.copy()
    if data.empty:
        return pd.DataFrame(), {}

    # Calcul vectorisé / itératif optimisé des scores historiques
    scores = []
    for i in range(len(data)):
        subset = data.iloc[:i + 1]
        score, _, _ = calculate_score(subset)
        scores.append(score)

    data["SCORE"] = scores
    data["POSITION"] = (data["SCORE"] >= threshold).astype(int)

    # Décalage pour éviter d'utiliser le signal du même jour
    data["POSITION_LAG"] = data["POSITION"].shift(1).fillna(0)
    data["MARKET_RETURN"] = data["Close"].pct_change()

    data["STRATEGY_RETURN"] = data["POSITION_LAG"] * data["MARKET_RETURN"]

    position_change = data["POSITION_LAG"].diff().abs().fillna(0)
    data["COST"] = position_change * transaction_cost

    data["STRATEGY_RETURN_NET"] = data["STRATEGY_RETURN"] - data["COST"]

    data["EQUITY"] = (
        initial_capital * (1 + data["STRATEGY_RETURN_NET"].fillna(0)).cumprod()
    )
    data["BUY_HOLD"] = (
        initial_capital * (1 + data["MARKET_RETURN"].fillna(0)).cumprod()
    )

    metrics = {
        "strategy": calculate_performance(data["EQUITY"]),
        "buy_hold": calculate_performance(data["BUY_HOLD"]),
    }

    return data, metrics


# ============================================================
# GRAPHIQUES
# ============================================================

def price_chart(df, ticker):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Cours",
        ),
        row=1,
        col=1,
    )

    for column, name, width in [
        ("SMA_20", "SMA 20", 1),
        ("SMA_50", "SMA 50", 2),
        ("SMA_200", "SMA 200", 2),
        ("BB_UPPER", "BB Haut", 1),
        ("BB_LOWER", "BB Bas", 1),
    ]:
        if column in df:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    name=name,
                    line=dict(
                        width=width,
                        dash="dot" if "BB_" in column else "solid",
                    ),
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VOLUME_SMA20"],
            name="Volume SMA20",
            line=dict(width=2),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} — Cours, moyennes mobiles et volume",
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    return fig


def rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI 14",
            line=dict(width=2),
        )
    )

    for level, text, dash in [
        (70, "Surachat", "dash"),
        (50, "Neutre", "dot"),
        (30, "Survente", "dash"),
    ]:
        fig.add_hline(
            y=level,
            line_dash=dash,
            annotation_text=text,
        )

    fig.update_layout(
        title="RSI 14",
        height=350,
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
    )

    return fig


def macd_chart(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_SIGNAL"],
            name="Signal",
            line=dict(width=2),
        )
    )
    fig.add_bar(
        x=df.index,
        y=df["MACD_HIST"],
        name="Histogramme",
    )
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(
        title="MACD 12 / 26 / 9",
        height=350,
        hovermode="x unified",
    )
    return fig


def bollinger_chart(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Cours",
            line=dict(width=2),
        )
    )

    for column, name in [
        ("BB_UPPER", "BB Haut"),
        ("BB_MIDDLE", "BB Moyenne"),
        ("BB_LOWER", "BB Bas"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                name=name,
                line=dict(
                    width=1,
                    dash="dot" if column != "BB_MIDDLE" else "solid",
                ),
            )
        )

    fig.update_layout(
        title="Bandes de Bollinger",
        height=450,
        hovermode="x unified",
    )
    return fig


def backtest_chart(bt, ticker):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bt["EQUITY"],
            name="Stratégie",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bt["BUY_HOLD"],
            name="Buy & Hold",
            line=dict(width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{ticker} — Stratégie vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Capital",
        height=500,
        hovermode="x unified",
    )
    return fig


# ============================================================
# EXPORT
# ============================================================

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Classement")
    output.seek(0)
    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Paramètres")

ticker_input = st.sidebar.text_area(
    "Actions à analyser",
    value=DEFAULT_TICKERS,
    height=130,
)

tickers = [
    ticker.strip().upper()
    for ticker in ticker_input.replace("\n", ",").split(",")
    if ticker.strip()
]

period = st.sidebar.selectbox(
    "Historique",
    ["6mo", "1y", "2y", "5y", "10y", "max"],
    index=3,
)

interval = st.sidebar.selectbox(
    "Intervalle",
    ["1d", "1wk", "1mo"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")

threshold = st.sidebar.slider(
    "Seuil d'entrée",
    40,
    90,
    65,
    5,
)

transaction_cost = st.sidebar.number_input(
    "Frais par transaction (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.05,
)

initial_capital = st.sidebar.number_input(
    "Capital initial",
    min_value=100.0,
    max_value=10_000_000.0,
    value=10_000.0,
    step=1_000.0,
)


# ============================================================
# TITRE
# ============================================================

st.title("📈 Stock Analyzer V2")
st.markdown("Analyse technique • Scanner • Score • Backtest • Performance")


# ============================================================
# ANALYSE
# ============================================================

if not tickers:
    st.warning("Saisissez au moins un ticker.")
    st.stop()

all_data = {}
results = []

with st.spinner("Téléchargement et calcul des indicateurs..."):
    for ticker in tickers:
        raw = download_data(ticker, period, interval)
        if raw.empty:
            continue

        df = calculate_indicators(raw)
        if len(df) < 50:
            continue

        all_data[ticker] = df

        score, signal, details = calculate_score(df)
        trend = detect_trend(df)
        row = df.iloc[-1]
        close = safe_float(row["Close"])
        one_month_return = np.nan

        if len(df) >= 21:
            old = safe_float(df["Close"].iloc[-21])
            if pd.notna(old) and old != 0:
                one_month_return = (close / old - 1) * 100

        results.append(
            {
                "Ticker": ticker,
                "Prix": close,
                "RSI": safe_float(row.get("RSI")),
                "SMA 50": safe_float(row.get("SMA_50")),
                "SMA 200": safe_float(row.get("SMA_200")),
                "MACD": safe_float(row.get("MACD")),
                "Volatilité 20j": safe_float(row.get("VOLATILITY_20")),
                "Volume / Moy.20": safe_float(row.get("VOLUME_RATIO")),
                "Perf. 1 mois": one_month_return,
                "Score": score,
                "Signal": signal,
                "Tendance": trend,
            }
        )

if not results:
    st.error("Aucune donnée exploitable. Vérifiez les tickers et les dépendances.")
    st.stop()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Score", ascending=False).reset_index(drop=True)


# ============================================================
# VUE GENERALE
# ============================================================

st.header("🏆 Vue d'ensemble")
best = results_df.iloc[0]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Actions analysées", len(results_df))
with c2:
    st.metric("Meilleure action", f"{best['Ticker']} — {int(best['Score'])}/100")
with c3:
    st.metric("Signaux ≥ 65", int((results_df["Score"] >= 65).sum()))
with c4:
    st.metric("Signaux < 50", int((results_df["Score"] < 50).sum()))


# ============================================================
# CLASSEMENT
# ============================================================

st.subheader("🏆 Classement technique")

numeric_cols = results_df.select_dtypes(include=[np.number]).columns
display_df = results_df.copy()
display_df[numeric_cols] = display_df[numeric_cols].round(2)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

fig = go.Figure()
fig.add_bar(
    x=results_df["Ticker"],
    y=results_df["Score"],
    text=results_df["Score"],
    textposition="auto",
)
fig.update_layout(
    title="Score technique",
    yaxis=dict(range=[0, 100]),
    height=400,
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ANALYSE DETAILLEE
# ============================================================

st.markdown("---")
st.header("🔎 Analyse détaillée")

selected = st.selectbox(
    "Action",
    list(all_data.keys()),
)

df = all_data[selected]

score, signal, details = calculate_score(df)
trend = detect_trend(df)
row = df.iloc[-1]

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Prix", fmt(row["Close"]))
with c2:
    st.metric("RSI 14", fmt(row["RSI"], 1))
with c3:
    st.metric("SMA 50", fmt(row["SMA_50"]))
with c4:
    st.metric("SMA 200", fmt(row["SMA_200"]))
with c5:
    st.metric("Score", f"{score}/100")
with c6:
    st.metric("Signal", signal)

st.info(f"**Tendance :** {trend}")

st.progress(
    score / 100,
    text=f"Score technique : {score}/100",
)

st.subheader("Décomposition du score")

detail_df = pd.DataFrame(
    {
        "Indicateur": list(details.keys()),
        "Points": list(details.values()),
    }
)

st.dataframe(
    detail_df,
    use_container_width=True,
    hide_index=True,
)

st.subheader("📈 Cours")
st.plotly_chart(
    price_chart(df, selected),
    use_container_width=True,
)


# ============================================================
# INDICATEURS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["RSI", "MACD", "Bollinger", "Données"])

with tab1:
    st.plotly_chart(rsi_chart(df), use_container_width=True)

with tab2:
    st.plotly_chart(macd_chart(df), use_container_width=True)

with tab3:
    st.plotly_chart(bollinger_chart(df), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Volatilité annualisée 20j",
            fmt(row.get("VOLATILITY_20"), 2) + "%"
            if pd.notna(row.get("VOLATILITY_20"))
            else "N/A",
        )
    with c2:
        st.metric("ATR 14", fmt(row.get("ATR"), 2))

with tab4:
    st.dataframe(df.tail(250), use_container_width=True)


# ============================================================
# BACKTEST
# ============================================================

st.markdown("---")
st.header("🧪 Backtest")

st.write(
    f"""
    La stratégie prend une position lorsque le score atteint
    **{threshold}/100**. Le signal est décalé d'une période
    pour éviter d'utiliser directement l'information du jour.
    Frais simulés : **{transaction_cost:.2f}%** par changement de position.
    """
)

if st.button("▶️ Lancer le backtest", type="primary"):
    with st.spinner("Calcul du backtest..."):
        bt, metrics = backtest_strategy(
            df,
            initial_capital,
            transaction_cost / 100,
            threshold,
        )

    if bt.empty:
        st.error("Backtest impossible.")
    else:
        strategy = metrics["strategy"]
        buy_hold = metrics["buy_hold"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Stratégie", fmt(strategy.get("Total Return")) + "%")
        with c2:
            st.metric("Buy & Hold", fmt(buy_hold.get("Total Return")) + "%")
        with c3:
            st.metric("CAGR", fmt(strategy.get("CAGR")) + "%")
        with c4:
            st.metric("Sharpe", fmt(strategy.get("Sharpe")))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Drawdown max", fmt(strategy.get("Max Drawdown")) + "%")
        with c2:
            outperformance = (
                strategy.get("Total Return", np.nan)
                - buy_hold.get("Total Return", np.nan)
            )
            st.metric("Surperformance", fmt(outperformance) + "%")
        with c3:
            trades = (
                bt["POSITION"].diff().abs().sum() / 2
            )
            st.metric("Trades", int(trades))

        st.plotly_chart(
            backtest_chart(bt, selected),
            use_container_width=True,
        )

        running_max = bt["EQUITY"].cummax()
        drawdown = (bt["EQUITY"] / running_max - 1) * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=bt.index,
                y=drawdown,
                name="Drawdown",
                fill="tozeroy",
                line=dict(width=2),
            )
        )
        fig_dd.update_layout(
            title="Drawdown",
            yaxis_title="%",
            height=350,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        with st.expander("Données du backtest"):
            st.dataframe(bt.tail(300), use_container_width=True)


# ============================================================
# EXPORT
# ============================================================

st.markdown("---")
st.header("📥 Export")

c1, c2 = st.columns(2)

with c1:
    st.download_button(
        "📄 Télécharger CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="stock_analysis.csv",
        mime="text/csv",
    )

with c2:
    st.download_button(
        "📊 Télécharger Excel",
        data=to_excel(results_df),
        file_name="stock_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Stock Analyzer V2 — outil d'analyse et de simulation. "
    "Les résultats historiques ne garantissent pas les résultats futurs."
)
