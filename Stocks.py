"""
Stock Analyzer V4
Application Streamlit d'analyse technique et de backtest.

Nouveautés V4 — aide à la décision sur la tendance à venir :
- Module DESCRIPTIF (pas de prédiction) : ADX (force de tendance) + canal de régression
  linéaire projeté, avec bandes d'écart-type. Purement une extrapolation géométrique du
  passé récent, explicitement labellisée comme non prédictive.
- Module PROBABILISTE (ML) : modèle de classification (probabilité de hausse à horizon
  N jours) entraîné sur les indicateurs déjà calculés, validé en walk-forward strict
  (TimeSeriesSplit, jamais de fuite du futur vers le passé), avec diagramme de calibration
  et comparaison à une base de référence naïve (fréquence historique de hausse).

Améliorations héritées de la V3 :
- Backtest performant (O(n) au lieu de O(n²)) et vectorisation du scoring historique
- Annualisation dynamique selon l'intervalle (jour / semaine / mois)
- Gestion du risque dans le backtest : stop-loss, take-profit, sizing basé sur la volatilité
- Téléchargement des données en parallèle
- auto_adjust=True pour éviter les artefacts liés aux splits
- Backtest de portefeuille (agrégé, comparé à un panier équipondéré)
- Heatmap de corrélation entre actifs
- Alertes sur les changements de signaux récents
- Historique du score dans le temps
- Tickers ignorés listés explicitement
- Section "Limites de l'outil"

⚠️ Cet outil est pédagogique. Il ne constitue pas un conseil en investissement.
Dépendance additionnelle pour le module ML : scikit-learn (pip install scikit-learn).
"""

import io
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta_classic as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Analyzer V3",
    page_icon="📈",
    layout="wide",
)

DEFAULT_TICKERS = (
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,"
    "AVGO,AMD,JPM,AIR.PA,MC.PA,TTE.PA,OR.PA"
)

ANNUALIZATION_FACTORS = {
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}


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


def fmt(value, decimals=2, suffix=""):
    value = safe_float(value)
    return "N/A" if pd.isna(value) else f"{value:.{decimals}f}{suffix}"


def get_annualization_factor(interval):
    return ANNUALIZATION_FACTORS.get(interval, 252)


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
            auto_adjust=True,   # évite les artefacts de prix liés aux splits/dividendes
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


def download_all(tickers, period, interval, max_workers=8):
    """Télécharge plusieurs tickers en parallèle (I/O bound -> threads)."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_data, ticker, period, interval): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception:
                results[ticker] = pd.DataFrame()
    # Restaure l'ordre initial des tickers
    return {ticker: results[ticker] for ticker in tickers if ticker in results}


def _get_fast_info_value(fast_info, keys):
    """yfinance expose fast_info tantôt en accès dict, tantôt en attribut selon la version."""
    for key in keys:
        try:
            value = fast_info[key]
            if value is not None:
                return value
        except Exception:
            pass
        try:
            value = getattr(fast_info, key)
            if value is not None:
                return value
        except Exception:
            pass
    return np.nan


@st.cache_data(ttl=60, show_spinner=False)
def fetch_last_price(ticker):
    """
    Récupère uniquement le dernier prix connu (pas tout l'historique), avec un cache
    court (60s) pour permettre une actualisation plus fréquente sans retélécharger le
    dataset complet à chaque fois. Reste soumis au même délai (~15-20 min) que le reste
    des données Yahoo Finance : ce n'est PAS un flux temps réel garanti.
    """
    try:
        fast = yf.Ticker(ticker).fast_info
    except Exception:
        return None

    price = safe_float(_get_fast_info_value(fast, ["last_price", "lastPrice"]))
    prev_close = safe_float(
        _get_fast_info_value(fast, ["previous_close", "previousClose", "regular_market_previous_close"])
    )

    if pd.isna(price):
        return None

    return {
        "price": price,
        "prev_close": prev_close,
        "fetched_at": pd.Timestamp.now(),
    }


# ============================================================
# INDICATEURS
# ============================================================

def calculate_indicators(data):
    df = data.copy()

    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)

    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)

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

    bb = ta.bbands(df["Close"], length=20, std=2)

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

    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is not None and not adx.empty:
        adx_col = [c for c in adx.columns if str(c).startswith("ADX_")]
        dip_col = [c for c in adx.columns if str(c).startswith("DMP_")]
        dim_col = [c for c in adx.columns if str(c).startswith("DMN_")]
        if adx_col:
            df["ADX"] = adx[adx_col[0]]
        if dip_col:
            df["DI_PLUS"] = adx[dip_col[0]]
        if dim_col:
            df["DI_MINUS"] = adx[dim_col[0]]

    df["VOLUME_SMA20"] = ta.sma(df["Volume"], length=20)
    df["VOLUME_RATIO"] = df["Volume"] / df["VOLUME_SMA20"]

    df["RETURN"] = df["Close"].pct_change()

    annual_factor = st.session_state.get("annual_factor", 252)
    df["VOLATILITY_20"] = (
        df["RETURN"].rolling(20).std() * np.sqrt(annual_factor) * 100
    )

    return df


# ============================================================
# SCORE (vectorisé sur tout l'historique en une passe)
# ============================================================

def calculate_score_series(df):
    """
    Calcule le score (0-100) et sa décomposition pour CHAQUE ligne du DataFrame,
    de façon vectorisée. Comme tous les indicateurs sont calculés de façon causale
    (rolling windows classiques, aucune fuite du futur), le score à la date i ne
    dépend que des données jusqu'à i inclus -> pas besoin de re-slicer le DataFrame
    ligne par ligne comme dans une version naïve (ce qui serait O(n²)).
    """
    n = len(df)
    idx = df.index

    def col(name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=idx)

    close = col("Close")
    sma20, sma50, sma200 = col("SMA_20"), col("SMA_50"), col("SMA_200")
    rsi = col("RSI")
    macd, macd_signal = col("MACD"), col("MACD_SIGNAL")
    bb_lower, bb_upper = col("BB_LOWER"), col("BB_UPPER")
    volume_ratio = col("VOLUME_RATIO")

    # --- Tendance : 40 pts ---
    trend_points = (
        np.where(close > sma20, 10, 0)
        + np.where(close > sma50, 15, 0)
        + np.where(close > sma200, 10, 0)
        + np.where(sma50 > sma200, 5, 0)
    ).astype(float)

    # --- RSI : 20 pts ---
    rsi_points = np.select(
        [
            (rsi >= 45) & (rsi <= 60),
            (rsi >= 35) & (rsi < 45),
            (rsi > 60) & (rsi <= 70),
            (rsi >= 30) & (rsi < 35),
            (rsi < 30),
            (rsi > 70),
        ],
        [20, 15, 15, 10, 12, 5],
        default=0,
    ).astype(float)

    # --- MACD : 15 pts ---
    macd_valid = macd.notna() & macd_signal.notna()
    macd_points = np.select(
        [
            macd_valid & (macd > macd_signal) & (macd > 0),
            macd_valid & (macd > macd_signal),
            macd_valid,
        ],
        [15, 10, 3],
        default=0,
    ).astype(float)

    # --- Bollinger : 10 pts ---
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    bb_valid = bb_lower.notna() & bb_upper.notna() & (bb_upper > bb_lower)
    position = (close - bb_lower) / bb_range
    bb_points = np.select(
        [
            bb_valid & (position >= 0.30) & (position <= 0.70),
            bb_valid & (position >= 0.15) & (position < 0.30),
            bb_valid & (position > 0.70) & (position <= 0.85),
            bb_valid & (position < 0.15),
            bb_valid,
        ],
        [10, 8, 7, 6, 3],
        default=0,
    ).astype(float)

    # --- Volume : 10 pts ---
    vol_points = np.select(
        [
            volume_ratio >= 1.5,
            volume_ratio >= 1.0,
            volume_ratio >= 0.7,
            volume_ratio.notna(),
        ],
        [10, 8, 5, 2],
        default=0,
    ).astype(float)

    # --- Momentum : 5 pts ---
    old_price = close.shift(20)
    momentum = (close / old_price - 1) * 100
    momentum_valid = old_price.notna() & (old_price > 0)
    momentum_points = np.select(
        [
            momentum_valid & (momentum > 5),
            momentum_valid & (momentum > 0),
            momentum_valid,
        ],
        [5, 3, 1],
        default=0,
    ).astype(float)

    total = (
        trend_points + rsi_points + macd_points
        + bb_points + vol_points + momentum_points
    )
    total = pd.Series(total, index=idx).clip(0, 100).round().astype(int)

    signal = pd.Series(np.select(
        [total >= 80, total >= 65, total >= 50, total >= 35],
        ["🟢 ACHAT FORT", "🟢 ACHAT", "🟡 NEUTRE", "🟠 VENTE PRUDENTE"],
        default="🔴 VENTE",
    ), index=idx)

    components = pd.DataFrame(
        {
            "SCORE": total,
            "SIGNAL": signal,
            "Tendance": trend_points,
            "RSI_pts": rsi_points,
            "MACD_pts": macd_points,
            "Bollinger_pts": bb_points,
            "Volume_pts": vol_points,
            "Momentum_pts": momentum_points,
        },
        index=idx,
    )

    return components


def get_latest_score_info(components):
    """Extrait le score, le signal et le détail pour la dernière ligne."""
    if components.empty:
        return 0, "⚪ N/A", {}
    row = components.iloc[-1]
    details = {
        "Tendance": int(row["Tendance"]),
        "RSI": int(row["RSI_pts"]),
        "MACD": int(row["MACD_pts"]),
        "Bollinger": int(row["Bollinger_pts"]),
        "Volume": int(row["Volume_pts"]),
        "Momentum": int(row["Momentum_pts"]),
    }
    return int(row["SCORE"]), row["SIGNAL"], details


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
# ALERTES / SIGNAUX RECENTS
# ============================================================

def detect_recent_signals(df, lookback=10):
    """Repère les croisements/événements techniques survenus dans les `lookback` dernières séances."""
    events = []
    recent = df.tail(lookback + 1)  # +1 pour pouvoir calculer les diff
    if len(recent) < 2:
        return events

    macd = recent.get("MACD")
    macd_signal = recent.get("MACD_SIGNAL")
    close = recent["Close"]
    sma50 = recent.get("SMA_50")
    rsi = recent.get("RSI")
    bb_upper = recent.get("BB_UPPER")
    bb_lower = recent.get("BB_LOWER")

    for i in range(1, len(recent)):
        date = recent.index[i]

        if macd is not None and macd_signal is not None:
            prev_diff = safe_float(macd.iloc[i - 1]) - safe_float(macd_signal.iloc[i - 1])
            curr_diff = safe_float(macd.iloc[i]) - safe_float(macd_signal.iloc[i])
            if pd.notna(prev_diff) and pd.notna(curr_diff):
                if prev_diff <= 0 < curr_diff:
                    events.append((date, "🟢 MACD croise au-dessus du signal"))
                elif prev_diff >= 0 > curr_diff:
                    events.append((date, "🔴 MACD croise en-dessous du signal"))

        if sma50 is not None:
            prev_c, curr_c = safe_float(close.iloc[i - 1]), safe_float(close.iloc[i])
            prev_s, curr_s = safe_float(sma50.iloc[i - 1]), safe_float(sma50.iloc[i])
            if all(pd.notna(x) for x in [prev_c, curr_c, prev_s, curr_s]):
                if prev_c <= prev_s < curr_c and curr_c > curr_s:
                    events.append((date, "🟢 Cours croise au-dessus de la SMA 50"))
                elif prev_c >= prev_s > curr_c and curr_c < curr_s:
                    events.append((date, "🔴 Cours croise en-dessous de la SMA 50"))

        if rsi is not None:
            prev_r, curr_r = safe_float(rsi.iloc[i - 1]), safe_float(rsi.iloc[i])
            if pd.notna(prev_r) and pd.notna(curr_r):
                if prev_r < 30 <= curr_r:
                    events.append((date, "🟢 RSI sort de la zone de survente (<30)"))
                elif prev_r > 70 >= curr_r:
                    events.append((date, "🔴 RSI sort de la zone de surachat (>70)"))

        if bb_upper is not None and bb_lower is not None:
            curr_c = safe_float(close.iloc[i])
            curr_up = safe_float(bb_upper.iloc[i])
            curr_low = safe_float(bb_lower.iloc[i])
            if pd.notna(curr_c) and pd.notna(curr_up) and curr_c > curr_up:
                events.append((date, "🟠 Cassure au-dessus de la bande de Bollinger haute"))
            elif pd.notna(curr_c) and pd.notna(curr_low) and curr_c < curr_low:
                events.append((date, "🟠 Cassure en-dessous de la bande de Bollinger basse"))

    return sorted(events, key=lambda x: x[0], reverse=True)


# ============================================================
# MODULE DESCRIPTIF : ADX + CANAL DE REGRESSION
# (extrapolation géométrique du passé récent — PAS une prédiction)
# ============================================================

def interpret_adx(adx_value, di_plus, di_minus):
    if pd.isna(adx_value):
        return "⚪ Données insuffisantes"

    if adx_value < 20:
        strength = "tendance faible / marché sans direction claire"
        icon = "⚪"
    elif adx_value < 40:
        strength = "tendance modérée"
        icon = "🟡"
    else:
        strength = "tendance forte"
        icon = "🟢"

    direction = ""
    if pd.notna(di_plus) and pd.notna(di_minus):
        direction = " (orientation haussière)" if di_plus > di_minus else " (orientation baissière)"

    return f"{icon} ADX {adx_value:.1f} — {strength}{direction}"


def regression_channel(df, window=50, projection=10, interval="1d"):
    """
    Régression linéaire sur les `window` dernières clôtures + bandes d'écart-type,
    projetées sur `projection` périodes futures. C'est une simple extrapolation
    géométrique de la tendance récente — elle ne modélise aucun mécanisme de marché
    et n'a aucune valeur prédictive démontrée. À interpréter comme une visualisation
    descriptive, pas comme une prévision.
    """
    close = df["Close"].dropna()
    if len(close) < window:
        return None

    recent = close.iloc[-window:]
    x = np.arange(window)
    slope, intercept = np.polyfit(x, recent.values, 1)
    fit = slope * x + intercept
    residuals = recent.values - fit
    std = residuals.std()

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((recent.values - recent.values.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    freq_map = {"1d": "B", "1wk": "W", "1mo": "ME"}
    freq = freq_map.get(interval, "B")
    future_dates = pd.date_range(
        start=recent.index[-1], periods=projection + 1, freq=freq
    )[1:]

    x_future = np.arange(window, window + projection)
    fit_future = slope * x_future + intercept

    return {
        "dates": recent.index,
        "fit": fit,
        "future_dates": future_dates,
        "fit_future": fit_future,
        "std": std,
        "slope": slope,
        "r_squared": r_squared,
        "close": recent,
    }


def regression_channel_chart(df, channel, context_window=100):
    context = df["Close"].dropna().iloc[-context_window:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=context.index, y=context.values, name="Cours", line=dict(width=2)))

    all_x = list(channel["dates"]) + list(channel["future_dates"])
    all_fit = list(channel["fit"]) + list(channel["fit_future"])

    fig.add_trace(
        go.Scatter(
            x=all_x, y=all_fit, name="Extrapolation linéaire",
            line=dict(width=2, dash="dash", color="orange"),
        )
    )

    for n_std, opacity in [(1, 0.18), (2, 0.08)]:
        upper = [v + n_std * channel["std"] for v in all_fit]
        lower = [v - n_std * channel["std"] for v in all_fit]
        fig.add_trace(go.Scatter(x=all_x, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=all_x, y=lower, line=dict(width=0), fill="tonexty",
                fillcolor=f"rgba(255,165,0,{opacity})",
                name=f"± {n_std} écart-type", hoverinfo="skip",
            )
        )

    fig.add_vline(x=channel["dates"][-1], line_dash="dot", line_color="gray")

    fig.update_layout(
        title="Canal de régression — extrapolation descriptive (NON prédictive)",
        height=450, hovermode="x unified",
    )
    return fig


def adx_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX", line=dict(width=2)))
    if "DI_PLUS" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["DI_PLUS"], name="DI+", line=dict(width=1)))
    if "DI_MINUS" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["DI_MINUS"], name="DI−", line=dict(width=1)))
    fig.add_hline(y=25, line_dash="dash", annotation_text="Tendance forte (>25)")
    fig.add_hline(y=20, line_dash="dot", annotation_text="Seuil tendance faible (<20)")
    fig.update_layout(title="ADX 14 — Force de la tendance (pas sa direction future)", height=350, hovermode="x unified")
    return fig


# ============================================================
# MODULE PROBABILISTE (ML) : classification à horizon N jours
# Validation en walk-forward strict — aucune fuite du futur.
# ============================================================

ML_FEATURE_COLUMNS = [
    "RSI", "MACD", "MACD_HIST", "ADX", "VOLUME_RATIO", "VOLATILITY_20",
]


def build_ml_dataset(df, horizon=5):
    """
    Construit X (features) / y (cible binaire : hausse à horizon `horizon` périodes)
    à partir des indicateurs déjà calculés de façon causale. Ajoute quelques features
    dérivées (distance relative aux moyennes mobiles, position dans les bandes de
    Bollinger, rendement récent) qui ne dépendent, comme le reste, que du passé.
    """
    data = df.copy()

    data["DIST_SMA50"] = (data["Close"] / data["SMA_50"] - 1) * 100
    data["DIST_SMA200"] = (data["Close"] / data["SMA_200"] - 1) * 100
    bb_range = (data["BB_UPPER"] - data["BB_LOWER"]).replace(0, np.nan)
    data["BB_POSITION"] = (data["Close"] - data["BB_LOWER"]) / bb_range
    data["RET_5"] = data["Close"].pct_change(5) * 100
    data["RET_10"] = data["Close"].pct_change(10) * 100

    feature_cols = ML_FEATURE_COLUMNS + [
        "DIST_SMA50", "DIST_SMA200", "BB_POSITION", "RET_5", "RET_10",
    ]
    feature_cols = [c for c in feature_cols if c in data.columns]

    # Cible : le cours sera-t-il plus haut dans `horizon` périodes ?
    data["TARGET"] = (data["Close"].shift(-horizon) > data["Close"]).astype(float)
    # Les dernières `horizon` lignes n'ont pas encore de cible connue (à prédire, pas à entraîner)
    data.loc[data.index[-horizon:], "TARGET"] = np.nan

    dataset = data[feature_cols + ["TARGET", "Close"]].dropna(subset=feature_cols)

    return dataset, feature_cols


def walk_forward_evaluation(dataset, feature_cols, n_splits=5):
    """
    TimeSeriesSplit = fenêtre d'entraînement en expansion, toujours suivie dans le temps
    par le pli de test (jamais l'inverse). C'est la condition minimale pour qu'une
    évaluation de modèle sur séries temporelles financières ait un sens.
    """
    labeled = dataset.dropna(subset=["TARGET"])
    if len(labeled) < 60:
        return None

    X = labeled[feature_cols].values
    y = labeled["TARGET"].values

    n_splits = min(n_splits, max(2, len(labeled) // 40))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    all_test_probs = []
    all_test_true = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        base_rate = y_train.mean()
        naive_brier = brier_score_loss(y_test, np.full_like(probs, base_rate))

        fold_results.append({
            "AUC": auc, "Brier": brier, "Brier_naif": naive_brier,
            "n_train": len(X_train), "n_test": len(X_test),
        })
        all_test_probs.extend(probs)
        all_test_true.extend(y_test)

    if not fold_results:
        return None

    return {
        "folds": pd.DataFrame(fold_results),
        "test_probs": np.array(all_test_probs),
        "test_true": np.array(all_test_true),
        "base_rate": labeled["TARGET"].mean(),
    }


def predict_latest_probability(dataset, feature_cols):
    """Entraîne sur tout l'historique labellisé, prédit sur la dernière ligne (non labellisée)."""
    labeled = dataset.dropna(subset=["TARGET"])
    unlabeled = dataset[dataset["TARGET"].isna()]

    if labeled.empty or unlabeled.empty:
        return np.nan

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    model.fit(labeled[feature_cols].values, labeled["TARGET"].values)

    latest_features = unlabeled[feature_cols].iloc[[-1]].values
    proba = model.predict_proba(latest_features)[0, 1]
    return proba


def reliability_diagram(test_probs, test_true, n_bins=8):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(test_probs, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    mean_pred, mean_actual, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(test_probs[mask].mean())
        mean_actual.append(test_true[mask].mean())
        counts.append(int(mask.sum()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite", line=dict(dash="dot", color="gray")))
    fig.add_trace(
        go.Scatter(
            x=mean_pred, y=mean_actual, mode="markers+lines", name="Modèle",
            marker=dict(size=[max(8, min(30, c)) for c in counts]),
            text=[f"n={c}" for c in counts], hovertemplate="Prédit=%{x:.2f}<br>Observé=%{y:.2f}<br>%{text}",
        )
    )
    fig.update_layout(
        title="Diagramme de calibration (plis de test walk-forward)",
        xaxis_title="Probabilité prédite", yaxis_title="Fréquence observée de hausse",
        height=400, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
    )
    return fig


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_performance(equity, annual_factor=252):
    equity = equity.dropna()
    if len(equity) < 2:
        return {}

    start = safe_float(equity.iloc[0])
    end = safe_float(equity.iloc[-1])

    if start <= 0:
        return {}

    total_return = (end / start - 1) * 100
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25

    cagr = ((end / start) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    returns = equity.pct_change().dropna()

    if len(returns) > 1 and returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(annual_factor)
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
# BACKTEST (O(n), avec gestion du risque)
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def backtest_strategy(
    df,
    initial_capital,
    transaction_cost,
    threshold,
    stop_loss_pct=0.0,
    take_profit_pct=0.0,
    sizing="fixed",
    annual_factor=252,
):
    """
    Backtest à une seule passe (O(n)).
    - Le signal d'entrée est décidé sur le score de la veille (pas de fuite d'info).
    - Le stop-loss / take-profit est vérifié intra-séance via le Low/High du jour.
    - Le sizing "volatility" réduit l'exposition quand l'ATR relatif est élevé.
    """
    data = df.copy()
    if data.empty or "SCORE" not in data.columns:
        return pd.DataFrame(), {}

    n = len(data)
    close = data["Close"].to_numpy()
    high = data["High"].to_numpy()
    low = data["Low"].to_numpy()
    score = data["SCORE"].to_numpy()
    atr = data["ATR"].to_numpy() if "ATR" in data.columns else np.full(n, np.nan)

    signal_on = score >= threshold

    equity = np.empty(n)
    position_flag = np.zeros(n)
    equity[0] = initial_capital

    in_position = False
    entry_price = np.nan
    position_size = 0.0
    trades = 0

    for i in range(1, n):
        yesterday_signal = bool(signal_on[i - 1]) if not np.isnan(score[i - 1]) else False
        net_return = 0.0

        if in_position:
            stop_hit = (
                stop_loss_pct > 0
                and not np.isnan(entry_price)
                and low[i] <= entry_price * (1 - stop_loss_pct / 100)
            )
            tp_hit = (
                take_profit_pct > 0
                and not np.isnan(entry_price)
                and high[i] >= entry_price * (1 + take_profit_pct / 100)
            )
            signal_exit = not yesterday_signal

            if stop_hit or tp_hit:
                exit_price = (
                    entry_price * (1 - stop_loss_pct / 100) if stop_hit
                    else entry_price * (1 + take_profit_pct / 100)
                )
                day_return = exit_price / close[i - 1] - 1
                net_return = position_size * day_return - transaction_cost
                in_position = False
                trades += 1
            elif signal_exit:
                day_return = close[i] / close[i - 1] - 1
                net_return = position_size * day_return - transaction_cost
                in_position = False
                trades += 1
            else:
                day_return = close[i] / close[i - 1] - 1
                net_return = position_size * day_return
                position_flag[i] = position_size
        else:
            if yesterday_signal:
                in_position = True
                entry_price = close[i]
                if sizing == "volatility" and not np.isnan(atr[i]) and close[i] > 0:
                    vol_pct = atr[i] / close[i]
                    position_size = float(np.clip(0.02 / vol_pct, 0.2, 1.0)) if vol_pct > 0 else 1.0
                else:
                    position_size = 1.0
                net_return = -transaction_cost  # coût d'entrée, pas de gain le jour même
                trades += 1
                position_flag[i] = position_size

        equity[i] = equity[i - 1] * (1 + net_return)

    data["SIGNAL_ON"] = signal_on
    data["POSITION"] = position_flag
    data["MARKET_RETURN"] = data["Close"].pct_change()
    data["EQUITY"] = equity
    data["BUY_HOLD"] = initial_capital * (1 + data["MARKET_RETURN"].fillna(0)).cumprod()

    metrics = {
        "strategy": calculate_performance(data["EQUITY"], annual_factor),
        "buy_hold": calculate_performance(data["BUY_HOLD"], annual_factor),
        "trades": trades,
    }

    return data, metrics


def portfolio_backtest(all_data_scored, initial_capital, transaction_cost, threshold,
                        stop_loss_pct, take_profit_pct, sizing, annual_factor):
    """
    Backtest agrégé (équipondéré, sans rebalancement quotidien explicite) : moyenne des
    courbes de capital normalisées de chaque actif. Approximation utile pour juger si le
    signal apporte quelque chose sur l'ensemble du panier plutôt que sur un seul titre
    choisi a posteriori.
    """
    normalized_curves = {}
    normalized_bh = {}

    for ticker, df in all_data_scored.items():
        bt, _ = backtest_strategy(
            df, initial_capital, transaction_cost, threshold,
            stop_loss_pct, take_profit_pct, sizing, annual_factor,
        )
        if bt.empty:
            continue
        normalized_curves[ticker] = bt["EQUITY"] / initial_capital
        normalized_bh[ticker] = bt["BUY_HOLD"] / initial_capital

    if not normalized_curves:
        return pd.DataFrame(), {}

    combined = pd.concat(normalized_curves, axis=1).sort_index().ffill().dropna(how="all")
    combined_bh = pd.concat(normalized_bh, axis=1).sort_index().ffill().dropna(how="all")

    portfolio_equity = combined.mean(axis=1) * initial_capital
    portfolio_bh = combined_bh.mean(axis=1) * initial_capital

    result = pd.DataFrame({"EQUITY": portfolio_equity, "BUY_HOLD": portfolio_bh})

    metrics = {
        "strategy": calculate_performance(result["EQUITY"], annual_factor),
        "buy_hold": calculate_performance(result["BUY_HOLD"], annual_factor),
        "n_assets": len(normalized_curves),
    }

    return result, metrics


# ============================================================
# GRAPHIQUES
# ============================================================

def price_chart(df, ticker):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Cours",
        ), row=1, col=1,
    )

    for column, name, width in [
        ("SMA_20", "SMA 20", 1), ("SMA_50", "SMA 50", 2), ("SMA_200", "SMA 200", 2),
        ("BB_UPPER", "BB Haut", 1), ("BB_LOWER", "BB Bas", 1),
    ]:
        if column in df:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[column], name=name,
                    line=dict(width=width, dash="dot" if "BB_" in column else "solid"),
                ), row=1, col=1,
            )

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["VOLUME_SMA20"], name="Volume SMA20", line=dict(width=2)),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{ticker} — Cours, moyennes mobiles et volume",
        height=700, xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    return fig


def rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI 14", line=dict(width=2)))
    for level, text, dash in [(70, "Surachat", "dash"), (50, "Neutre", "dot"), (30, "Survente", "dash")]:
        fig.add_hline(y=level, line_dash=dash, annotation_text=text)
    fig.update_layout(title="RSI 14", height=350, yaxis=dict(range=[0, 100]), hovermode="x unified")
    return fig


def macd_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(width=2)))
    fig.add_bar(x=df.index, y=df["MACD_HIST"], name="Histogramme")
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="MACD 12 / 26 / 9", height=350, hovermode="x unified")
    return fig


def bollinger_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Cours", line=dict(width=2)))
    for column, name in [("BB_UPPER", "BB Haut"), ("BB_MIDDLE", "BB Moyenne"), ("BB_LOWER", "BB Bas")]:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[column], name=name,
                line=dict(width=1, dash="dot" if column != "BB_MIDDLE" else "solid"),
            )
        )
    fig.update_layout(title="Bandes de Bollinger", height=450, hovermode="x unified")
    return fig


def score_history_chart(components, threshold):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=components.index, y=components["SCORE"], name="Score",
            line=dict(width=2), fill="tozeroy",
        )
    )
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="orange",
        annotation_text=f"Seuil d'entrée ({threshold})",
    )
    fig.update_layout(
        title="Évolution du score technique dans le temps",
        height=350, yaxis=dict(range=[0, 100]), hovermode="x unified",
    )
    return fig


def backtest_chart(bt, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["EQUITY"], name="Stratégie", line=dict(width=3)))
    fig.add_trace(
        go.Scatter(x=bt.index, y=bt["BUY_HOLD"], name="Buy & Hold", line=dict(width=2, dash="dash"))
    )
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Capital",
        height=500, hovermode="x unified",
    )
    return fig


def drawdown_chart(equity):
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=drawdown, name="Drawdown", fill="tozeroy", line=dict(width=2)))
    fig.update_layout(title="Drawdown", yaxis_title="%", height=350)
    return fig


def correlation_heatmap(all_data):
    closes = {ticker: df["Close"] for ticker, df in all_data.items()}
    price_df = pd.concat(closes, axis=1).sort_index().ffill()
    returns = price_df.pct_change().dropna(how="all")
    corr = returns.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Corrélation des rendements quotidiens", height=500)
    return fig


# ============================================================
# EXPORT
# ============================================================

def to_excel(sheets: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, index=True, sheet_name=name[:31])
    output.seek(0)
    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Paramètres")

ticker_input = st.sidebar.text_area("Actions à analyser", value=DEFAULT_TICKERS, height=130)

tickers = [
    ticker.strip().upper()
    for ticker in ticker_input.replace("\n", ",").split(",")
    if ticker.strip()
]

period = st.sidebar.selectbox("Historique", ["6mo", "1y", "2y", "5y", "10y", "max"], index=3)
interval = st.sidebar.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)

st.session_state["annual_factor"] = get_annualization_factor(interval)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Actualisation")

if st.sidebar.button("Actualiser maintenant"):
    download_data.clear()
    fetch_last_price.clear()
    st.session_state["last_manual_refresh"] = pd.Timestamp.now()
    st.rerun()

if st.session_state.get("last_manual_refresh") is not None:
    st.sidebar.caption(
        f"Dernière actualisation forcée : {st.session_state['last_manual_refresh'].strftime('%H:%M:%S')}"
    )

if AUTOREFRESH_AVAILABLE:
    auto_refresh = st.sidebar.checkbox("Auto-actualisation périodique", value=False)
    if auto_refresh:
        refresh_choice = st.sidebar.selectbox("Intervalle", ["1 min", "5 min", "15 min"], index=1)
        refresh_minutes = {"1 min": 1, "5 min": 5, "15 min": 15}[refresh_choice]
        st_autorefresh(interval=refresh_minutes * 60 * 1000, key="autorefresh_timer")
else:
    st.sidebar.caption(
        "Pour l'auto-actualisation périodique : `pip install streamlit-autorefresh`"
    )

st.sidebar.caption(
    "ℹ️ Les cours proviennent de Yahoo Finance et sont généralement différés de "
    "15 à 20 minutes, quel que soit le rythme d'actualisation choisi ici. "
    "L'historique complet est mis en cache 15 min ; le dernier cours (onglet détaillé) "
    "1 min."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")

threshold = st.sidebar.slider("Seuil d'entrée", 40, 90, 65, 5,
                               help="Position ouverte quand le score de la veille dépasse ce seuil.")

transaction_cost = st.sidebar.number_input(
    "Frais par transaction (%)", min_value=0.0, max_value=2.0, value=0.10, step=0.05,
)

initial_capital = st.sidebar.number_input(
    "Capital initial", min_value=100.0, max_value=10_000_000.0, value=10_000.0, step=1_000.0,
)

st.sidebar.markdown("**Gestion du risque**")

stop_loss_pct = st.sidebar.slider(
    "Stop-loss (%)", 0.0, 30.0, 8.0, 0.5,
    help="0 = désactivé. Sortie si le plus bas du jour touche ce niveau sous le prix d'entrée.",
)

take_profit_pct = st.sidebar.slider(
    "Take-profit (%)", 0.0, 60.0, 0.0, 1.0,
    help="0 = désactivé. Sortie si le plus haut du jour touche ce niveau au-dessus du prix d'entrée.",
)

sizing_label = st.sidebar.selectbox(
    "Sizing des positions", ["Fixe (100%)", "Basé sur la volatilité (ATR)"], index=0,
)
sizing = "volatility" if "volatilité" in sizing_label else "fixed"

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Tendance à venir")

regression_window = st.sidebar.slider(
    "Fenêtre du canal de régression", 20, 150, 50, 5,
    help="Nombre de séances récentes utilisées pour l'extrapolation linéaire.",
)
regression_projection = st.sidebar.slider(
    "Projection (périodes futures)", 5, 30, 10, 5,
)
ml_horizon = st.sidebar.slider(
    "Horizon du modèle ML (périodes)", 3, 20, 5, 1,
    help="Le modèle estime la probabilité que le cours soit plus haut dans N périodes.",
)


# ============================================================
# TITRE
# ============================================================

st.title("📈 Stock Analyzer V3")
st.markdown("Analyse technique • Scanner • Score • Backtest robuste • Risque • Portefeuille")


# ============================================================
# ANALYSE
# ============================================================

if not tickers:
    st.warning("Saisissez au moins un ticker.")
    st.stop()

annual_factor = st.session_state["annual_factor"]

with st.spinner("Téléchargement des données..."):
    raw_data = download_all(tickers, period, interval)

all_data = {}
results = []
skipped = []

with st.spinner("Calcul des indicateurs et des scores..."):
    for ticker in tickers:
        raw = raw_data.get(ticker, pd.DataFrame())
        if raw.empty:
            skipped.append((ticker, "Aucune donnée téléchargée (ticker invalide ?)"))
            continue

        df = calculate_indicators(raw)
        if len(df) < 50:
            skipped.append((ticker, f"Historique insuffisant ({len(df)} lignes < 50)"))
            continue

        components = calculate_score_series(df)
        df = pd.concat([df, components], axis=1)
        all_data[ticker] = df

        score, signal, details = get_latest_score_info(components)
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

if skipped:
    with st.expander(f"⚠️ {len(skipped)} ticker(s) ignoré(s)", expanded=False):
        for ticker, reason in skipped:
            st.write(f"- **{ticker}** : {reason}")

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

st.dataframe(display_df, use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_bar(x=results_df["Ticker"], y=results_df["Score"], text=results_df["Score"], textposition="auto")
fig.update_layout(title="Score technique", yaxis=dict(range=[0, 100]), height=400)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CORRELATIONS
# ============================================================

if len(all_data) >= 2:
    st.markdown("---")
    st.header("🔗 Corrélations entre actifs")
    st.caption(
        "Un panier de titres fortement corrélés diversifie moins que ce que leur nombre "
        "suggère. Une valeur proche de 1 indique des mouvements très similaires."
    )
    st.plotly_chart(correlation_heatmap(all_data), use_container_width=True)


# ============================================================
# ANALYSE DETAILLEE
# ============================================================

st.markdown("---")
st.header("🔎 Analyse détaillée")

selected = st.selectbox("Action", list(all_data.keys()))
df = all_data[selected]

live = fetch_last_price(selected)
if live:
    delta_pct = None
    if pd.notna(live["prev_close"]) and live["prev_close"] != 0:
        delta_pct = (live["price"] / live["prev_close"] - 1) * 100
    c_live, c_caption = st.columns([1, 3])
    with c_live:
        st.metric(
            f"🔴 {selected} — dernier cours connu",
            fmt(live["price"]),
            delta=f"{delta_pct:+.2f}%" if delta_pct is not None else None,
            help="Source Yahoo Finance (fast_info), généralement différée de 15 à 20 minutes.",
        )
    with c_caption:
        st.caption(
            f"Récupéré à {live['fetched_at'].strftime('%H:%M:%S')} (cache 60 s). "
            "Pas un flux temps réel garanti — voir la note sur l'actualisation dans la barre latérale."
        )

components = df[["SCORE", "SIGNAL", "Tendance", "RSI_pts", "MACD_pts", "Bollinger_pts", "Volume_pts", "Momentum_pts"]]

score, signal, details = get_latest_score_info(components)
trend = detect_trend(df)
row = df.iloc[-1]

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Prix", fmt(row["Close"]))
with c2:
    st.metric("RSI 14", fmt(row["RSI"], 1), help="< 30 : survente. > 70 : surachat.")
with c3:
    st.metric("SMA 50", fmt(row["SMA_50"]))
with c4:
    st.metric("SMA 200", fmt(row["SMA_200"]))
with c5:
    st.metric("Score", f"{score}/100", help="Heuristique pondérée (tendance, RSI, MACD, Bollinger, volume, momentum). Non calibrée statistiquement — voir la section Limites.")
with c6:
    st.metric("Signal", signal)

st.info(f"**Tendance :** {trend}")
st.progress(score / 100, text=f"Score technique : {score}/100")

st.subheader("Décomposition du score")
detail_df = pd.DataFrame({"Indicateur": list(details.keys()), "Points": list(details.values())})
st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.subheader("🔔 Signaux techniques récents")
recent_events = detect_recent_signals(df, lookback=10)
if recent_events:
    events_df = pd.DataFrame(recent_events, columns=["Date", "Événement"])
    events_df["Date"] = events_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(events_df, use_container_width=True, hide_index=True)
else:
    st.caption("Aucun croisement notable sur les 10 dernières séances.")

st.subheader("📈 Cours")
st.plotly_chart(price_chart(df, selected), use_container_width=True)


# ============================================================
# INDICATEURS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["RSI", "MACD", "Bollinger", "Score dans le temps", "Données"])

with tab1:
    st.plotly_chart(rsi_chart(df), use_container_width=True)

with tab2:
    st.plotly_chart(macd_chart(df), use_container_width=True)

with tab3:
    st.plotly_chart(bollinger_chart(df), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Volatilité annualisée 20j", fmt(row.get("VOLATILITY_20"), 2, "%"))
    with c2:
        st.metric("ATR 14", fmt(row.get("ATR"), 2))

with tab4:
    st.plotly_chart(score_history_chart(components, threshold), use_container_width=True)
    st.caption("Permet de voir si le score est stable dans le temps ou s'il oscille beaucoup autour du seuil.")

with tab5:
    st.dataframe(df.tail(250), use_container_width=True)


# ============================================================
# AIDE A LA DECISION — TENDANCE A VENIR
# ============================================================

st.markdown("---")
st.header("🔮 Aide à la décision — tendance à venir")

st.warning(
    "**À lire avant d'utiliser cette section.** Rien ci-dessous ne prédit le prix futur "
    "avec certitude. Le module *descriptif* prolonge géométriquement la tendance récente "
    "(aucun mécanisme de marché modélisé). Le module *probabiliste* donne une probabilité "
    "statistique issue d'un modèle simple, évaluée honnêtement en walk-forward — mais un "
    "AUC proche de 0,5 signifie une performance proche du hasard, et les marchés changent "
    "de régime dans le temps. Ce n'est pas un conseil en investissement."
)

desc_tab, ml_tab = st.tabs(["📐 Module descriptif (ADX + canal)", "🤖 Module probabiliste (ML)"])

with desc_tab:
    st.caption(
        "Extrapolation géométrique du passé récent — utile pour visualiser la pente et la "
        "force de la tendance actuelle, pas pour prédire un prix futur."
    )

    if "ADX" in df.columns:
        adx_row = df.iloc[-1]
        st.info(interpret_adx(
            safe_float(adx_row.get("ADX")),
            safe_float(adx_row.get("DI_PLUS")),
            safe_float(adx_row.get("DI_MINUS")),
        ))
        st.plotly_chart(adx_chart(df), use_container_width=True)
    else:
        st.caption("ADX indisponible pour ce titre.")

    channel = regression_channel(df, window=regression_window, projection=regression_projection, interval=interval)
    if channel is None:
        st.caption("Historique insuffisant pour calculer le canal de régression.")
    else:
        direction = "haussière" if channel["slope"] > 0 else "baissière"
        st.write(
            f"Pente de la régression sur les {regression_window} dernières séances : "
            f"orientation **{direction}**, ajustement R² = **{channel['r_squared']:.2f}** "
            f"(proche de 1 = tendance récente très linéaire, proche de 0 = bruitée)."
        )
        st.plotly_chart(regression_channel_chart(df, channel), use_container_width=True)

with ml_tab:
    if not SKLEARN_AVAILABLE:
        st.error(
            "scikit-learn n'est pas installé dans cet environnement. "
            "Installez-le avec `pip install scikit-learn` pour activer ce module."
        )
    else:
        st.caption(
            f"Estime la probabilité que {selected} soit plus haut dans {ml_horizon} "
            "période(s), à partir d'un modèle de régression logistique entraîné sur les "
            "indicateurs techniques déjà calculés."
        )

        if st.button("▶️ Entraîner et évaluer le modèle", type="primary"):
            with st.spinner("Construction du jeu de données et validation walk-forward..."):
                dataset, feature_cols = build_ml_dataset(df, horizon=ml_horizon)
                eval_result = walk_forward_evaluation(dataset, feature_cols)

            if eval_result is None:
                st.error(
                    "Historique insuffisant ou classes trop déséquilibrées pour une "
                    "évaluation walk-forward fiable sur ce titre."
                )
            else:
                folds = eval_result["folds"]
                mean_auc = folds["AUC"].mean()
                mean_brier = folds["Brier"].mean()
                mean_naive_brier = folds["Brier_naif"].mean()
                base_rate = eval_result["base_rate"]

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "AUC moyen (walk-forward)", fmt(mean_auc, 3),
                        help="0.50 = équivalent au hasard. > 0.55 déjà notable sur des prix d'actions. > 0.65 est rare et mérite d'être vérifié pour de la fuite de données.",
                    )
                with c2:
                    st.metric(
                        "Brier score du modèle", fmt(mean_brier, 3),
                        help="Plus bas = mieux calibré. À comparer au Brier naïf ci-contre.",
                    )
                with c3:
                    st.metric("Brier score naïf (fréquence de base)", fmt(mean_naive_brier, 3))
                with c4:
                    st.metric("Fréquence historique de hausse", fmt(base_rate * 100, 1, "%"))

                if mean_auc < 0.53:
                    st.warning(
                        "AUC proche de 0,50 : ce modèle n'apporte quasiment aucune information "
                        "directionnelle sur ce titre avec cet horizon. Le considérer comme non "
                        "informatif plutôt que d'en tirer un signal."
                    )
                elif mean_brier >= mean_naive_brier:
                    st.warning(
                        "Le modèle ne bat pas la base de référence naïve (fréquence historique "
                        "de hausse) en termes de calibration. Il n'apporte pas d'avantage "
                        "démontré ici."
                    )

                st.plotly_chart(
                    reliability_diagram(eval_result["test_probs"], eval_result["test_true"]),
                    use_container_width=True,
                )

                with st.expander("Détail par pli de validation (walk-forward)"):
                    st.dataframe(folds.round(3), use_container_width=True, hide_index=True)

                st.markdown("---")
                proba = predict_latest_probability(dataset, feature_cols)
                if pd.notna(proba):
                    st.metric(
                        f"Probabilité estimée de hausse à {ml_horizon} période(s)",
                        f"{proba * 100:.1f}%",
                        help="Calculée par un modèle entraîné sur tout l'historique labellisé disponible. Sa fiabilité est celle mesurée ci-dessus, pas une garantie.",
                    )
                    st.caption(
                        "Cette probabilité vient du même type de modèle que celui évalué "
                        "ci-dessus. Si l'AUC est proche de 0,50 ou si le modèle ne bat pas la "
                        "base naïve, ce chiffre ne doit pas être interprété comme un signal "
                        "fiable — il reflète surtout la fréquence historique de hausse."
                    )
                else:
                    st.caption("Impossible de calculer une probabilité pour la dernière séance (données manquantes).")


# ============================================================
# BACKTEST (titre sélectionné)
# ============================================================

st.markdown("---")
st.header("🧪 Backtest — action sélectionnée")

st.write(
    f"""
    Position ouverte lorsque le score de la veille atteint **{threshold}/100** (signal décalé
    d'une période pour éviter tout effet de bord). Frais simulés : **{transaction_cost:.2f}%**
    par changement de position. Stop-loss : **{stop_loss_pct:.1f}%** •
    Take-profit : **{take_profit_pct:.1f}%** (0 = désactivé) • Sizing : **{sizing_label}**.
    """
)

if st.button("▶️ Lancer le backtest", type="primary"):
    with st.spinner("Calcul du backtest..."):
        bt, metrics = backtest_strategy(
            df, initial_capital, transaction_cost / 100, threshold,
            stop_loss_pct, take_profit_pct, sizing, annual_factor,
        )

    if bt.empty:
        st.error("Backtest impossible.")
    else:
        strategy = metrics["strategy"]
        buy_hold = metrics["buy_hold"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Stratégie", fmt(strategy.get("Total Return"), suffix="%"))
        with c2:
            st.metric("Buy & Hold", fmt(buy_hold.get("Total Return"), suffix="%"))
        with c3:
            st.metric("CAGR", fmt(strategy.get("CAGR"), suffix="%"))
        with c4:
            st.metric("Sharpe", fmt(strategy.get("Sharpe")))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Drawdown max", fmt(strategy.get("Max Drawdown"), suffix="%"))
        with c2:
            outperformance = strategy.get("Total Return", np.nan) - buy_hold.get("Total Return", np.nan)
            st.metric("Surperformance", fmt(outperformance, suffix="%"))
        with c3:
            st.metric("Trades", int(metrics["trades"]))

        st.plotly_chart(backtest_chart(bt, f"{selected} — Stratégie vs Buy & Hold"), use_container_width=True)
        st.plotly_chart(drawdown_chart(bt["EQUITY"]), use_container_width=True)

        with st.expander("Données du backtest"):
            st.dataframe(bt.tail(300), use_container_width=True)

        st.session_state["last_backtest"] = bt
        st.session_state["last_backtest_ticker"] = selected


# ============================================================
# BACKTEST DE PORTEFEUILLE
# ============================================================

st.markdown("---")
st.header("📦 Backtest de portefeuille (tous les tickers)")

st.caption(
    "Applique la même stratégie à chaque actif du panier puis moyenne les courbes de "
    "capital normalisées (équipondéré, sans rebalancement quotidien explicite). "
    "Un edge qui ne survit que sur un seul titre choisi après coup est un signe classique "
    "de surapprentissage — tester sur l'ensemble du panier est plus honnête."
)

if st.button("▶️ Lancer le backtest de portefeuille"):
    with st.spinner("Calcul du backtest de portefeuille..."):
        pf_bt, pf_metrics = portfolio_backtest(
            all_data, initial_capital, transaction_cost / 100, threshold,
            stop_loss_pct, take_profit_pct, sizing, annual_factor,
        )

    if pf_bt.empty:
        st.error("Backtest de portefeuille impossible.")
    else:
        pf_strategy = pf_metrics["strategy"]
        pf_buy_hold = pf_metrics["buy_hold"]

        st.write(f"Portefeuille équipondéré sur **{pf_metrics['n_assets']}** actifs.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Stratégie", fmt(pf_strategy.get("Total Return"), suffix="%"))
        with c2:
            st.metric("Buy & Hold panier", fmt(pf_buy_hold.get("Total Return"), suffix="%"))
        with c3:
            st.metric("CAGR", fmt(pf_strategy.get("CAGR"), suffix="%"))
        with c4:
            st.metric("Sharpe", fmt(pf_strategy.get("Sharpe")))

        st.plotly_chart(backtest_chart(pf_bt, "Portefeuille — Stratégie vs Buy & Hold"), use_container_width=True)
        st.plotly_chart(drawdown_chart(pf_bt["EQUITY"]), use_container_width=True)

        st.session_state["last_portfolio_backtest"] = pf_bt


# ============================================================
# LIMITES DE L'OUTIL
# ============================================================

st.markdown("---")
with st.expander("⚠️ Limites de cet outil — à lire avant toute décision"):
    st.markdown(
        """
- **Le score est une heuristique, pas un modèle validé statistiquement.** Les poids
  (tendance 40, RSI 20, MACD 15, Bollinger 10, volume 10, momentum 5) et les seuils ont été
  choisis à la main, pas calibrés sur des données. Rien ne garantit qu'ils ont un pouvoir
  prédictif réel.
- **Le backtest n'est pas out-of-sample.** Le seuil d'entrée est réglable librement en
  observant les résultats passés, ce qui invite à l'ajuster jusqu'à trouver ce qui a "bien
  marché" — un biais de surapprentissage classique. Une validation sérieuse nécessiterait un
  découpage entraînement / test (walk-forward) sur plusieurs sous-périodes.
- **Coûts de transaction simplifiés.** Seuls des frais fixes par changement de position sont
  modélisés. Le slippage, le spread bid/ask et l'impact de marché ne le sont pas — la
  performance réelle serait probablement inférieure à celle affichée, surtout avec un nombre
  élevé de trades.
- **Le stop-loss/take-profit est une approximation.** L'exécution est simulée au niveau du
  stop/objectif via le plus bas/haut du jour, ce qui suppose une exécution parfaite — en
  réalité, un gap à l'ouverture peut faire exécuter l'ordre à un prix plus défavorable.
- **Le backtest de portefeuille est une moyenne de courbes normalisées**, pas une vraie
  simulation avec rebalancement, ce qui sous-estime certains effets de friction.
- **Ceci n'est pas un conseil en investissement.** Les performances passées, simulées ou
  réelles, ne garantissent pas les performances futures.
        """
    )


# ============================================================
# EXPORT
# ============================================================

st.markdown("---")
st.header("📥 Export")

c1, c2 = st.columns(2)

with c1:
    st.download_button(
        "📄 Télécharger CSV (classement)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="stock_analysis.csv",
        mime="text/csv",
    )

with c2:
    export_sheets = {"Classement": results_df}
    if "last_backtest" in st.session_state:
        export_sheets[f"Backtest_{st.session_state.get('last_backtest_ticker', 'ticker')}"] = st.session_state["last_backtest"]
    if "last_portfolio_backtest" in st.session_state:
        export_sheets["Backtest_Portefeuille"] = st.session_state["last_portfolio_backtest"]

    st.download_button(
        "📊 Télécharger Excel (classement + backtests lancés)",
        data=to_excel(export_sheets),
        file_name="stock_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Stock Analyzer V3 — outil d'analyse et de simulation à but pédagogique. "
    "Ce n'est pas un conseil en investissement. Les résultats historiques, simulés ou "
    "réels, ne garantissent pas les résultats futurs."
)
