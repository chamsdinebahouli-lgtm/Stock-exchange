```python
"""
Stock Analyzer V2
=================

Application Streamlit d'analyse technique et de backtest.

Fonctionnalités
---------------
- Yahoo Finance
- Analyse d'une ou plusieurs actions
- RSI 14
- SMA 20 / 50 / 200
- MACD
- Bandes de Bollinger
- ATR
- Volume
- Score technique 0-100
- Signal ACHAT / NEUTRE / VENTE
- Scanner multi-actions
- Classement
- Backtest
- Buy & Hold
- Rendement cumulé
- CAGR
- Sharpe Ratio
- Drawdown maximum
- Graphiques Plotly
- Export CSV
- Export Excel
- Interface Streamlit

IMPORTANT
---------
Cette application est un outil d'analyse technique et de simulation.
Elle ne constitue pas une recommandation financière.
Les résultats historiques ne garantissent pas les résultats futurs.
"""

import io
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Analyzer V2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 2.5rem;
        font-weight: 700;
    }

    .subtitle {
        font-size: 1.1rem;
        opacity: 0.75;
        margin-bottom: 2rem;
    }

    .signal-box {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .small-text {
        font-size: 0.85rem;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTES
# ============================================================

TRADING_DAYS = 252

DEFAULT_TICKERS = (
    "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, "
    "AVGO, AMD, JPM, AIR.PA, MC.PA, TTE.PA, OR.PA"
)


# ============================================================
# OUTILS
# ============================================================

def safe_float(value):
    """Convertit proprement une valeur en float."""

    try:
        if pd.isna(value):
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def format_price(value):
    """Format prix."""

    value = safe_float(value)

    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


def format_percent(value):
    """Format pourcentage."""

    value = safe_float(value)

    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}%"


# ============================================================
# TELECHARGEMENT
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def download_data(ticker, period="5y", interval="1d"):
    """
    Télécharge les données depuis Yahoo Finance.
    """

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

    # Gestion des MultiIndex yfinance
    if isinstance(data.columns, pd.MultiIndex):

        # Dans certains cas le ticker est le deuxième niveau
        if len(data.columns.levels) >= 2:

            try:
                data.columns = data.columns.get_level_values(0)
            except Exception:
                pass

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        return pd.DataFrame()

    data = data[required].copy()

    data = data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    data.index = pd.to_datetime(data.index)

    return data


# ============================================================
# INDICATEURS
# ============================================================

def calculate_indicators(data):
    """
    Calcule l'ensemble des indicateurs techniques.
    """

    df = data.copy()

    # --------------------------------------------------------
    # MOYENNES MOBILES
    # --------------------------------------------------------

    df["SMA_20"] = ta.sma(
        df["Close"],
        length=20,
    )

    df["SMA_50"] = ta.sma(
        df["Close"],
        length=50,
    )

    df["SMA_200"] = ta.sma(
        df["Close"],
        length=200,
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = ta.rsi(
        df["Close"],
        length=14,
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = ta.macd(
        df["Close"],
        fast=12,
        slow=26,
        signal=9,
    )

    if macd is not None and not macd.empty:

        # pandas-ta produit normalement :
        # MACD_12_26_9
        # MACDh_12_26_9
        # MACDs_12_26_9

        macd_col = [
            c for c in macd.columns
            if str(c).startswith("MACD_")
            and not str(c).startswith("MACDh")
            and not str(c).startswith("MACDs")
        ]

        hist_col = [
            c for c in macd.columns
            if str(c).startswith("MACDh")
        ]

        signal_col = [
            c for c in macd.columns
            if str(c).startswith("MACDs")
        ]

        if macd_col:
            df["MACD"] = macd[macd_col[0]]

        if hist_col:
            df["MACD_HIST"] = macd[hist_col[0]]

        if signal_col:
            df["MACD_SIGNAL"] = macd[signal_col[0]]

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    bb = ta.bbands(
        df["Close"],
        length=20,
        std=2,
    )

    if bb is not None and not bb.empty:

        lower = [
            c for c in bb.columns
            if str(c).startswith("BBL")
        ]

        middle = [
            c for c in bb.columns
            if str(c).startswith("BBM")
        ]

        upper = [
            c for c in bb.columns
            if str(c).startswith("BBU")
        ]

        bandwidth = [
            c for c in bb.columns
            if str(c).startswith("BBB")
        ]

        percent = [
            c for c in bb.columns
            if str(c).startswith("BBP")
        ]

        if lower:
            df["BB_LOWER"] = bb[lower[0]]

        if middle:
            df["BB_MIDDLE"] = bb[middle[0]]

        if upper:
            df["BB_UPPER"] = bb[upper[0]]

        if bandwidth:
            df["BB_WIDTH"] = bb[bandwidth[0]]

        if percent:
            df["BB_PERCENT"] = bb[percent[0]]

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["ATR"] = ta.atr(
        df["High"],
        df["Low"],
        df["Close"],
        length=14,
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["VOLUME_SMA20"] = ta.sma(
        df["Volume"],
        length=20,
    )

    df["VOLUME_RATIO"] = (
        df["Volume"] /
        df["VOLUME_SMA20"]
    )

    # --------------------------------------------------------
    # RENDEMENTS
    # --------------------------------------------------------

    df["RETURN"] = df["Close"].pct_change()

    # --------------------------------------------------------
    # VOLATILITE
    # --------------------------------------------------------

    df["VOLATILITY_20"] = (
        df["RETURN"]
        .rolling(20)
        .std()
        * np.sqrt(TRADING_DAYS)
        * 100
    )

    # --------------------------------------------------------
    # TENDANCE
    # --------------------------------------------------------

    df["ABOVE_SMA50"] = (
        df["Close"] > df["SMA_50"]
    )

    df["ABOVE_SMA200"] = (
        df["Close"] > df["SMA_200"]
    )

    df["SMA50_ABOVE_SMA200"] = (
        df["SMA_50"] > df["SMA_200"]
    )

    # --------------------------------------------------------
    # SIGNAL DE CROSSOVER
    # --------------------------------------------------------

    df["SMA50_CROSS"] = (
        df["SMA_50"] > df["SMA_200"]
    ).astype(int)

    df["MACD_BULLISH"] = (
        df["MACD"] > df["MACD_SIGNAL"]
    )

    return df


# ============================================================
# SCORE TECHNIQUE
# ============================================================

def calculate_score(df):
    """
    Score technique de 0 à 100.

    Tendance : 40 points
    RSI      : 20 points
    MACD     : 15 points
    Bollinger: 10 points
    Volume   : 10 points
    Momentum : 5 points
    """

    if df.empty:
        return 0, "⚪ N/A", {}

    row = df.iloc[-1]

    score = 0

    details = {}

    close = safe_float(row["Close"])

    # ========================================================
    # TENDANCE — 40
    # ========================================================

    trend_score = 0

    sma20 = safe_float(row.get("SMA_20"))
    sma50 = safe_float(row.get("SMA_50"))
    sma200 = safe_float(row.get("SMA_200"))

    if pd.notna(sma20) and close > sma20:
        trend_score += 10

    if pd.notna(sma50) and close > sma50:
        trend_score += 15

    if pd.notna(sma200) and close > sma200:
        trend_score += 10

    if (
        pd.notna(sma50)
        and pd.notna(sma200)
        and sma50 > sma200
    ):
        trend_score += 5

    score += trend_score

    details["Tendance"] = trend_score

    # ========================================================
    # RSI — 20
    # ========================================================

    rsi = safe_float(row.get("RSI"))

    rsi_score = 0

    if pd.notna(rsi):

        if 45 <= rsi <= 60:
            rsi_score = 20

        elif 35 <= rsi < 45:
            rsi_score = 15

        elif 60 < rsi <= 70:
            rsi_score = 15

        elif 30 <= rsi < 35:
            rsi_score = 10

        elif rsi < 30:
            rsi_score = 12

        elif rsi > 70:
            rsi_score = 5

    score += rsi_score

    details["RSI"] = rsi_score

    # ========================================================
    # MACD — 15
    # ========================================================

    macd = safe_float(row.get("MACD"))
    macd_signal = safe_float(row.get("MACD_SIGNAL"))

    macd_score = 0

    if pd.notna(macd) and pd.notna(macd_signal):

        if macd > macd_signal and macd > 0:
            macd_score = 15

        elif macd > macd_signal:
            macd_score = 10

        else:
            macd_score = 3

    score += macd_score

    details["MACD"] = macd_score

    # ========================================================
    # BOLLINGER — 10
    # ========================================================

    bb_score = 0

    lower = safe_float(row.get("BB_LOWER"))
    upper = safe_float(row.get("BB_UPPER"))

    if (
        pd.notna(lower)
        and pd.notna(upper)
        and upper > lower
    ):

        position = (
            (close - lower)
            / (upper - lower)
        )

        if 0.30 <= position <= 0.70:
            bb_score = 10

        elif 0.15 <= position < 0.30:
            bb_score = 8

        elif 0.70 < position <= 0.85:
            bb_score = 7

        elif position < 0.15:
            bb_score = 6

        else:
            bb_score = 3

    score += bb_score

    details["Bollinger"] = bb_score

    # ========================================================
    # VOLUME — 10
    # ========================================================

    volume_ratio = safe_float(
        row.get("VOLUME_RATIO")
    )

    volume_score = 0

    if pd.notna(volume_ratio):

        if volume_ratio >= 1.5:
            volume_score = 10

        elif volume_ratio >= 1.0:
            volume_score = 8

        elif volume_ratio >= 0.7:
            volume_score = 5

        else:
            volume_score = 2

    score += volume_score

    details["Volume"] = volume_score

    # ========================================================
    # MOMENTUM — 5
    # ========================================================

    momentum_score = 0

    if len(df) >= 21:

        old_price = safe_float(
            df["Close"].iloc[-21]
        )

        if (
            pd.notna(old_price)
            and old_price > 0
        ):

            momentum = (
                close / old_price - 1
            ) * 100

            if momentum > 5:
                momentum_score = 5

            elif momentum > 0:
                momentum_score = 3

            else:
                momentum_score = 1

    score += momentum_score

    details["Momentum"] = momentum_score

    score = int(round(
        min(max(score, 0), 100)
    ))

    # ========================================================
    # SIGNAL
    # ========================================================

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

    if df.empty:
        return "⚪ Données insuffisantes"

    row = df.iloc[-1]

    close = safe_float(row["Close"])
    sma20 = safe_float(row.get("SMA_20"))
    sma50 = safe_float(row.get("SMA_50"))
    sma200 = safe_float(row.get("SMA_200"))

    if any(
        pd.isna(x)
        for x in [close, sma20, sma50, sma200]
    ):
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
# METRIQUES DE PERFORMANCE
# ============================================================

def calculate_performance(equity):
    """
    Calcule les principales métriques de performance.
    """

    if equity.empty:
        return {}

    equity = equity.dropna()

    if len(equity) < 2:
        return {}

    start_value = safe_float(
        equity.iloc[0]
    )

    end_value = safe_float(
        equity.iloc[-1]
    )

    total_return = (
        end_value / start_value - 1
    ) * 100

    # --------------------------------------------------------
    # Durée
    # --------------------------------------------------------

    days = (
        equity.index[-1]
        - equity.index[0]
    ).days

    years = max(
        days / 365.25,
        1 / 365.25,
    )

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    if start_value > 0:

        cagr = (
            (end_value / start_value)
            ** (1 / years)
            - 1
        ) * 100

    else:

        cagr = np.nan

    # --------------------------------------------------------
    # Rendements journaliers
    # --------------------------------------------------------

    returns = equity.pct_change().dropna()

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    if (
        len(returns) > 1
        and returns.std() != 0
    ):

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(TRADING_DAYS)
        )

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    ) * 100

    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
    }


# ============================================================
# BACKTEST
# ============================================================

def backtest_strategy(
    df,
    initial_capital=10000,
    transaction_cost=0.001,
):
    """
    Backtest simple.

    Règle :

    Position = 1 si score >= seuil.
    Position = 0 sinon.

    Pour éviter le look-ahead bias, la position
    du jour suivant est utilisée pour le rendement.
    """

    data = df.copy()

    if data.empty:
        return pd.DataFrame(), {}

    # --------------------------------------------------------
    # Calcul du score historique
    # --------------------------------------------------------

    scores = []

    for i in range(len(data)):

        subset = data.iloc[:i + 1]

        score, _, _ = calculate_score(
            subset
        )

        scores.append(score)

    data["SCORE"] = scores

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    threshold = st.session_state.get(
        "backtest_threshold",
        65,
    )

    data["POSITION"] = (
        data["SCORE"] >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Position décalée
    # --------------------------------------------------------

    data["POSITION_LAG"] = (
        data["POSITION"]
        .shift(1)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Rendement marché
    # --------------------------------------------------------

    data["MARKET_RETURN"] = (
        data["Close"].pct_change()
    )

    # --------------------------------------------------------
    # Rendement stratégie
    # --------------------------------------------------------

    data["STRATEGY_RETURN"] = (
        data["POSITION_LAG"]
        * data["MARKET_RETURN"]
    )

    # --------------------------------------------------------
    # Frais de transaction
    # --------------------------------------------------------

    position_change = (
        data["POSITION_LAG"]
        .diff()
        .abs()
        .fillna(0)
    )

    data["COST"] = (
        position_change
        * transaction_cost
    )

    data["STRATEGY_RETURN_NET"] = (
        data["STRATEGY_RETURN"]
        - data["COST"]
    )

    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    data["EQUITY"] = (
        initial_capital
        * (
            1
            + data["STRATEGY_RETURN_NET"]
        ).cumprod()
    )

    # --------------------------------------------------------
    # Buy & Hold
    # --------------------------------------------------------

    data["BUY_HOLD"] = (
        initial_capital
        * (
            1
            + data["MARKET_RETURN"].fillna(0)
        ).cumprod()
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    strategy_metrics = calculate_performance(
        data["EQUITY"]
    )

    buy_hold_metrics = calculate_performance(
        data["BUY_HOLD"]
    )

    metrics = {
        "strategy": strategy_metrics,
        "buy_hold": buy_hold_metrics,
    }

    return data, metrics


# ============================================================
# GRAPHIQUE PRINCIPAL
# ============================================================

def create_price_chart(df, ticker):

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    # --------------------------------------------------------
    # Chandeliers
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SMA20
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_20"],
            name="SMA 20",
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # SMA50
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_50"],
            name="SMA 50",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # SMA200
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_200"],
            name="SMA 200",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Bollinger haut
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_UPPER"],
            name="BB Haut",
            line=dict(
                width=1,
                dash="dot",
            ),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Bollinger bas
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_LOWER"],
            name="BB Bas",
            line=dict(
                width=1,
                dash="dot",
            ),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

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
        legend=dict(
            orientation="h",
            y=1.02,
            x=0,
        ),
    )

    return fig


# ============================================================
# RSI
# ============================================================

def create_rsi_chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI 14",
            line=dict(width=2),
        )
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="70",
    )

    fig.add_hline(
        y=50,
        line_dash="dot",
        annotation_text="50",
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="30",
    )

    fig.update_layout(
        title="RSI 14",
        height=350,
        yaxis=dict(
            range=[0, 100]
        ),
        hovermode="x unified",
    )

    return fig


# ============================================================
# MACD
# ============================================================

def create_macd_chart(df):

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

    fig.add_hline(
        y=0,
        line_dash="dot",
    )

    fig.update_layout(
        title="MACD 12 / 26 / 9",
        height=350,
        hovermode="x unified",
    )

    return fig


# ============================================================
# GRAPHIQUE BACKTEST
# ============================================================

def create_backtest_chart(
    backtest,
    ticker,
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["EQUITY"],
            name="Stratégie",
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["BUY_HOLD"],
            name="Buy & Hold",
            line=dict(
                width=2,
                dash="dash",
            ),
        )
    )

    fig.update_layout(
        title=f"{ticker} — Backtest",
        xaxis_title="Date",
        yaxis_title="Capital",
        height=500,
        hovermode="x unified",
    )

    return fig


# ============================================================
# EXPORT EXCEL
# ============================================================

def dataframe_to_excel(df):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            index=True,
            sheet_name="Analyse",
        )

    output.seek(0)

    return output


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Stock Analyzer V2")

st.sidebar.markdown(
    "### Actions à analyser"
)

ticker_input = st.sidebar.text_area(
    "Tickers",
    value=DEFAULT_TICKERS,
    height=120,
    help=(
        "Séparez les tickers par des virgules."
    ),
)

tickers = [
    ticker.strip().upper()
    for ticker in ticker_input.replace(
        "\n",
        ","
    ).split(",")
    if ticker.strip()
]

st.sidebar.markdown("---")

period = st.sidebar.selectbox(
    "Historique",
    [
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "max",
    ],
    index=3,
)

interval = st.sidebar.selectbox(
    "Intervalle",
    [
        "1d",
        "1wk",
        "1mo",
    ],
    index=0,
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Paramètres backtest"
)

backtest_threshold = st.sidebar.slider(
    "Seuil d'entrée",
    min_value=40,
    max_value=90,
    value=65,
    step=5,
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

st.session_state[
    "backtest_threshold"
] = backtest_threshold

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Score technique**

    Tendance : 40 points

    RSI : 20 points

    MACD : 15 points

    Bollinger : 10 points

    Volume : 10 points

    Momentum : 5 points
    """
)


# ============================================================
# TITRE
# ============================================================

st.markdown(
    '<div class="title">📈 Stock Analyzer V2</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Analyse technique • Scanner • Score • Backtest • Performance
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VERIFICATION
# ============================================================

if not tickers:

    st.warning(
        "Saisissez au moins un ticker."
    )

    st.stop()


# ============================================================
# STOCKAGE
# ============================================================

all_results = []

all_data = {}

backtest_results = {}


# ============================================================
# TELECHARGEMENT ET ANALYSE
# ============================================================

with st.spinner(
    "Téléchargement et analyse des actions..."
):

    for ticker in tickers:

        raw = download_data(
            ticker,
            period,
            interval,
        )

        if raw.empty:
            continue

        df = calculate_indicators(
            raw
        )

        if len(df) < 50:
            continue

        all_data[ticker] = df

        score, signal, details = (
            calculate_score(df)
        )

        trend = detect_trend(df)

        row = df.iloc[-1]

        close = safe_float(
            row["Close"]
        )

        rsi = safe_float(
            row.get("RSI")
        )

        sma50 = safe_float(
            row.get("SMA_50")
        )

        sma200 = safe_float(
            row.get("SMA_200")
        )

        macd = safe_float(
            row.get("MACD")
        )

        volatility = safe_float(
            row.get("VOLATILITY_20")
        )

        volume_ratio = safe_float(
            row.get("VOLUME_RATIO")
        )

        if len(df) >= 21:

            price_1m = safe_float(
                df["Close"].iloc[-21]
            )

            if (
                pd.notna(price_1m)
                and price_1m != 0
            ):

                return_1m = (
                    close / price_1m
                    - 1
                ) * 100

            else:

                return_1m = np.nan

        else:

            return_1m = np.nan

        all_results.append(
            {
                "Ticker": ticker,
                "Prix": close,
                "RSI": rsi,
                "SMA 50": sma50,
                "SMA 200": sma200,
                "MACD": macd,
                "Volatilité 20j": volatility,
                "Volume / Moy.20": volume_ratio,
                "Perf. 1 mois": return_1m,
                "Score": score,
                "Signal": signal,
                "Tendance": trend,
            }
        )


# ============================================================
# VERIFICATION RESULTATS
# ============================================================

if not all_results:

    st.error(
        "Aucune donnée exploitable n'a été récupérée."
    )

    st.stop()


results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    "Score",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# KPI GLOBALS
# ============================================================

st.subheader(
    "🏆 Vue d'ensemble"
)

best = results_df.iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Actions analysées",
        len(results_df),
    )

with col2:

    st.metric(
        "Meilleur score",
        f"{best['Ticker']} — {int(best['Score'])}/100",
    )

with col3:

    buy_count = results_df[
        results_df["Score"] >= 65
    ].shape[0]

    st.metric(
        "Signaux favorables",
        buy_count,
    )

with col4:

    neutral_count = results_df[
        (
            results_df["Score"] >= 50
        )
        &
        (
            results_df["Score"] < 65
        )
    ].shape[0]

    st.metric(
        "Neutres",
        neutral_count,
    )

with col5:

    sell_count = results_df[
        results_df["Score"] < 50
    ].shape[0]

    st.metric(
        "Signaux faibles",
        sell_count,
    )


# ============================================================
# CLASSEMENT
# ============================================================

st.markdown("---")

st.header(
    "🏆 Classement technique"
)

display_ranking = results_df.copy()

numeric_columns = [
    "Prix",
    "RSI",
    "SMA 50",
    "SMA 200",
    "MACD",
    "Volatilité 20j",
    "Volume / Moy.20",
    "Perf. 1 mois",
    "Score",
]

for col in numeric_columns:

    if col in display_ranking.columns:

        display_ranking[col] = (
            display_ranking[col]
            .round(2)
        )

st.dataframe(
    display_ranking,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# GRAPHIQUE CLASSEMENT
# ============================================================

fig_ranking = go.Figure()

fig_ranking.add_bar(
    x=results_df["Ticker"],
    y=results_df["Score"],
    text=results_df["Score"],
    textposition="auto",
)

fig_ranking.update_layout(
    title="Score technique par action",
    yaxis_title="Score / 100",
    xaxis_title="Action",
    yaxis=dict(
        range=[0, 100]
    ),
    height=450,
)

st.plotly_chart(
    fig_ranking,
    use_container_width=True,
)


# ============================================================
# SELECTION ACTION
# ============================================================

st.markdown("---")

st.header(
    "🔎 Analyse détaillée"
)

selected_ticker = st.selectbox(
    "Choisir une action",
    list(all_data.keys()),
)


df = all_data[
    selected_ticker
]

score, signal, details = (
    calculate_score(df)
)

trend = detect_trend(df)

row = df.iloc[-1]


# ============================================================
# KPI ACTION
# ============================================================

st.subheader(
    f"{selected_ticker}"
)

col1, col2, col3, col4, col5, col6 = (
    st.columns(6)
)

with col1:

    st.metric(
        "Prix",
        format_price(
            row["Close"]
        ),
    )

with col2:

    st.metric(
        "RSI 14",
        format_price(
            row["RSI"]
        ),
    )

with col3:

    st.metric(
        "SMA 50",
        format_price(
            row["SMA_50"]
        ),
    )

with col4:

    st.metric(
        "SMA 200",
        format_price(
            row["SMA_200"]
        ),
    )

with col5:

    st.metric(
        "Score",
        f"{score}/100",
    )

with col6:

    st.metric(
        "Signal",
        signal,
    )


# ============================================================
# TENDANCE
# ============================================================

st.info(
    f"**Tendance détectée :** {trend}"
)

st.progress(
    score / 100,
    text=f"Score technique : {score}/100",
)


# ============================================================
# SCORE DETAIL
# ============================================================

st.subheader(
    "🎯 Décomposition du score"
)

score_detail = pd.DataFrame(
    {
        "Indicateur": list(
            details.keys()
        ),
        "Points": list(
            details.values()
        ),
    }
)

st.dataframe(
    score_detail,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# GRAPHIQUE PRIX
# ============================================================

st.subheader(
    "📈 Cours et indicateurs"
)

st.plotly_chart(
    create_price_chart(
        df,
        selected_ticker,
    ),
    use_container_width=True,
)


# ============================================================
# ONGLETS
# ============================================================

tab_rsi, tab_macd, tab_bollinger, tab_data = (
    st.tabs(
        [
            "RSI",
            "MACD",
            "Bollinger / Volatilité",
            "Données",
        ]
    )
)


# ============================================================
# RSI
# ============================================================

with tab_rsi:

    st.plotly_chart(
        create_rsi_chart(df),
        use_container_width=True,
    )

    current_rsi = safe_float(
        row["RSI"]
    )

    if pd.notna(current_rsi):

        if current_rsi < 30:

            st.success(
                f"RSI = {current_rsi:.1f} : "
                "zone de survente."
            )

        elif current_rsi > 70:

            st.warning(
                f"RSI = {current_rsi:.1f} : "
                "zone de surachat."
            )

        else:

            st.info(
                f"RSI = {current_rsi:.1f} : "
                "zone intermédiaire."
            )


# ============================================================
# MACD
# ============================================================

with tab_macd:

    st.plotly_chart(
        create_macd_chart(df),
        use_container_width=True,
    )

    macd_value = safe_float(
        row.get("MACD")
    )

    signal_value = safe_float(
        row.get("MACD_SIGNAL")
    )

    if (
        pd.notna(macd_value)
        and pd.notna(signal_value)
    ):

        if macd_value > signal_value:

            st.success(
                "MACD au-dessus de sa ligne de signal."
            )

        else:

            st.warning(
                "MACD sous sa ligne de signal."
            )


# ============================================================
# BOLLINGER
# ============================================================

with tab_bollinger:

    fig_bb = go.Figure()

    fig_bb.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Cours",
            line=dict(width=2),
        )
    )

    fig_bb.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_UPPER"],
            name="Bande supérieure",
            line=dict(
                width=1,
                dash="dot",
            ),
        )
    )

    fig_bb.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_MIDDLE"],
            name="Moyenne",
            line=dict(width=1),
        )
    )

    fig_bb.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_LOWER"],
            name="Bande inférieure",
            line=dict(
                width=1,
                dash="dot",
            ),
        )
    )

    fig_bb.update_layout(
        title="Bandes de Bollinger",
        height=450,
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_bb,
        use_container_width=True,
    )

    vol = safe_float(
        row.get("VOLATILITY_20")
    )

    atr = safe_float(
        row.get("ATR")
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Volatilité annualisée 20j",
            (
                f"{vol:.2f}%"
                if pd.notna(vol)
                else "N/A"
            ),
        )

    with col2:

        st.metric(
            "ATR 14",
            (
                f"{atr:.2f}"
                if pd.notna(atr)
                else "N/A"
            ),
        )


# ============================================================
# DONNEES
# ============================================================

with tab_data:

    st.dataframe(
        df.tail(200),
        use_container_width=True,
    )


# ============================================================
# BACKTEST
# ============================================================

st.markdown("---")

st.header(
    "🧪 Backtest de la stratégie"
)

st.markdown(
    f"""
    **Règle de simulation :**

    Une position est prise lorsque le score technique
    atteint au moins **{backtest_threshold}/100**.

    Le signal est décalé d'une période afin de limiter
    le biais d'anticipation.

    Les frais de transaction simulés sont de
    **{transaction_cost:.2f}%**.
    """
)

run_backtest = st.button(
    "▶️ Lancer le backtest",
    type="primary",
)


if run_backtest:

    with st.spinner(
        "Calcul du backtest..."
    ):

        bt, metrics = backtest_strategy(
            df,
            initial_capital,
            transaction_cost / 100,
        )

    if bt.empty:

        st.error(
            "Impossible d'effectuer le backtest."
        )

    else:

        backtest_results[
            selected_ticker
        ] = bt

        strategy = metrics.get(
            "strategy",
            {}
        )

        buy_hold = metrics.get(
            "buy_hold",
            {}
        )

        st.subheader(
            "📊 Résultats"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            value = strategy.get(
                "Total Return",
                np.nan,
            )

            st.metric(
                "Stratégie",
                (
                    f"{value:.2f}%"
                    if pd.notna(value)
                    else "N/A"
                ),
            )

        with col2:

            value = buy_hold.get(
                "Total Return",
                np.nan,
            )

            st.metric(
                "Buy & Hold",
                (
                    f"{value:.2f}%"
                    if pd.notna(value)
                    else "N/A"
                ),
            )

        with col3:

            value = strategy.get(
                "CAGR",
                np.nan,
            )

            st.metric(
                "CAGR",
                (
                    f"{value:.2f}%"
                    if pd.notna(value)
                    else "N/A"
                ),
            )

        with col4:

            value = strategy.get(
                "Sharpe",
                np.nan,
            )

            st.metric(
                "Sharpe",
                (
                    f"{value:.2f}"
                    if pd.notna(value)
                    else "N/A"
                ),
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            value = strategy.get(
                "Max Drawdown",
                np.nan,
            )

            st.metric(
                "Drawdown maximum",
                (
                    f"{value:.2f}%"
                    if pd.notna(value)
                    else "N/A"
                ),
            )

        with col2:

            value = (
                strategy.get(
                    "Total Return",
                    np.nan,
                )
                -
                buy_hold.get(
                    "Total Return",
                    np.nan,
                )
            )

            st.metric(
                "Surperformance",
                (
                    f"{value:.2f}%"
                    if pd.notna(value)
                    else "N/A"
                ),
            )

        with col3:

            trades = (
                bt["POSITION"]
                .diff()
                .abs()
                .sum()
                / 2
            )

            st.metric(
                "Nombre approximatif de trades",
                f"{int(trades)}",
            )

        # ----------------------------------------------------
        # Graphique
        # ----------------------------------------------------

        st.plotly_chart(
            create_backtest_chart(
                bt,
                selected_ticker,
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Courbe du drawdown
        # ----------------------------------------------------

        running_max = (
            bt["EQUITY"]
            .cummax()
        )

        drawdown = (
            bt["EQUITY"]
            / running_max
            - 1
        ) * 100

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
            title="Drawdown de la stratégie",
            yaxis_title="Drawdown (%)",
            height=350,
        )

        st.plotly_chart(
            fig_dd,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Tableau
        # ----------------------------------------------------

        with st.expander(
            "Voir les données du backtest"
        ):

            st.dataframe(
                bt[
                    [
                        "Close",
                        "SCORE",
                        "POSITION",
                        "MARKET_RETURN",
                        "STRATEGY_RETURN_NET",
                        "EQUITY",
                        "BUY_HOLD",
                    ]
                ].tail(300),
                use_container_width=True,
            )


# ============================================================
# EXPORT
# ============================================================

st.markdown("---")

st.header(
    "📥 Export"
)

col1, col2 = st.columns(2)

with col1:

    csv_data = results_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="📄 Télécharger CSV",
        data=csv_data,
        file_name="stock_analysis.csv",
        mime="text/csv",
    )

with col2:

    excel_data = dataframe_to_excel(
        results_df
    )

    st.download_button(
        label="📊 Télécharger Excel",
        data=excel_data,
        file_name="stock_analysis.xlsx",
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown("---")

st.header(
    "ℹ️ Interprétation"
)

st.markdown(
    """
    ### Score

    Le score est un indicateur synthétique permettant de
    comparer les actions entre elles.

    **80–100 :** configuration technique très favorable

    **65–79 :** configuration favorable

    **50–64 :** configuration neutre

    **35–49 :** configuration défavorable

    **0–34 :** configuration très défavorable

    ### Backtest

    Le backtest permet de vérifier comment une règle donnée
    aurait fonctionné historiquement.

    Il faut notamment regarder :

    - le rendement total ;
    - le CAGR ;
    - le Sharpe Ratio ;
    - le drawdown maximum ;
    - le nombre de trades ;
    - la comparaison avec Buy & Hold.

    Un rendement historique élevé ne signifie pas que la
    stratégie fonctionnera nécessairement dans le futur.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    """
    Stock Analyzer V2 • Python / Streamlit / Plotly / Yahoo Finance

    Outil d'analyse et de simulation uniquement.
    Les informations affichées ne constituent pas un conseil
    en investissement.
    """
)
```
