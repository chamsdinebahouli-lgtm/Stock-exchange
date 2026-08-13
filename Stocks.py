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

DEFAULT_TICKERS = ["SOI.PA", "GNFT.PA", "DBV.PA"]

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
# SCORES (vectorisés sur tout l'historique en une passe)
# Trois scores séparés plutôt qu'un seul agrégat :
#   - DIRECTION : la structure actuelle favorise-t-elle une hausse ?
#   - QUALITE   : ce signal directionnel est-il propre ou bruité ?
#   - RISQUE    : quelle est la dangerosité de la position (0=faible, 100=élevé) ?
# Chaque composant n'apparaît que dans UN seul des trois scores, pour éviter de compter
# la même information plusieurs fois sous des habillages différents (ADX et la pente de
# régression, par exemple, se sont révélés largement redondants au test walk-forward —
# seul ADX est gardé, dans QUALITE). Comme l'ancien score, ce sont des poids/seuils
# choisis à la main (voir "Limites de l'outil"), pas calibrés statistiquement — sauf le
# module ML, qui reste la seule estimation calibrée de l'app.
# ============================================================

def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def calculate_direction_score_series(df):
    """
    DIRECTION (0-100) : la structure actuelle favorise-t-elle une hausse ?
    Régime de tendance (SMA) + RSI + MACD + momentum 20j. Ni ADX ni la pente de
    régression n'y figurent : ce sont des mesures de force/qualité de la tendance,
    pas de sa direction — elles vont dans le score QUALITE pour éviter les doublons.
    """
    idx = df.index
    close = _col(df, "Close")
    sma20, sma50, sma200 = _col(df, "SMA_20"), _col(df, "SMA_50"), _col(df, "SMA_200")
    rsi = _col(df, "RSI")
    macd, macd_signal = _col(df, "MACD"), _col(df, "MACD_SIGNAL")

    # --- Régime de tendance : 50 pts ---
    trend_points = (
        np.where(close > sma20, 12, 0)
        + np.where(close > sma50, 18, 0)
        + np.where(close > sma200, 12, 0)
        + np.where(sma50 > sma200, 8, 0)
    ).astype(float)

    # --- RSI : 25 pts ---
    rsi_points = np.select(
        [
            (rsi >= 45) & (rsi <= 60),
            (rsi >= 35) & (rsi < 45),
            (rsi > 60) & (rsi <= 70),
            (rsi >= 30) & (rsi < 35),
            (rsi < 30),
            (rsi > 70),
        ],
        [25, 18, 18, 12, 14, 6],
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

    # --- Momentum 20j : 10 pts ---
    old_price = close.shift(20)
    momentum = (close / old_price - 1) * 100
    momentum_valid = old_price.notna() & (old_price > 0)
    momentum_points = np.select(
        [
            momentum_valid & (momentum > 5),
            momentum_valid & (momentum > 0),
            momentum_valid,
        ],
        [10, 6, 2],
        default=0,
    ).astype(float)

    total = pd.Series(
        trend_points + rsi_points + macd_points + momentum_points, index=idx
    ).clip(0, 100).round().astype(int)

    signal = pd.Series(np.select(
        [total >= 80, total >= 65, total >= 50, total >= 35],
        ["🟢 ACHAT FORT", "🟢 ACHAT", "🟡 NEUTRE", "🟠 VENTE PRUDENTE"],
        default="🔴 VENTE",
    ), index=idx)

    return pd.DataFrame(
        {
            "DIRECTION_SCORE": total,
            "DIRECTION_SIGNAL": signal,
            "DIR_Tendance": trend_points,
            "DIR_RSI": rsi_points,
            "DIR_MACD": macd_points,
            "DIR_Momentum": momentum_points,
        },
        index=idx,
    )


def calculate_quality_score_series(df):
    """
    QUALITE (0-100) : ce signal directionnel est-il propre ou bruité ?
    ADX (force de tendance — seule mesure de force gardée, la pente/R² de régression
    étant redondante avec elle) + confirmation par le volume + cohérence entre 3 signaux
    indépendants (prix vs SMA50, MACD, RSI). Une tendance forte, confirmée par le volume
    et où les indicateurs sont d'accord entre eux, est un signal plus fiable qu'une
    tendance atteignant le même score DIRECTION mais sur fond de signaux contradictoires.
    """
    idx = df.index
    close = _col(df, "Close")
    sma50 = _col(df, "SMA_50")
    rsi = _col(df, "RSI")
    macd, macd_signal = _col(df, "MACD"), _col(df, "MACD_SIGNAL")
    adx = _col(df, "ADX")
    volume_ratio = _col(df, "VOLUME_RATIO")

    # --- ADX (force de tendance) : 50 pts ---
    adx_points = np.select(
        [adx >= 40, (adx >= 20) & (adx < 40), adx.notna()],
        [50, 30, 10],
        default=0,
    ).astype(float)

    # --- Confirmation volume : 25 pts ---
    vol_points = np.select(
        [
            volume_ratio >= 1.5,
            volume_ratio >= 1.0,
            volume_ratio >= 0.7,
            volume_ratio.notna(),
        ],
        [25, 20, 12, 5],
        default=0,
    ).astype(float)

    # --- Cohérence entre 3 signaux indépendants : 25 pts ---
    coherence_valid = close.notna() & sma50.notna() & macd.notna() & macd_signal.notna() & rsi.notna()
    bull_count = (
        (close > sma50).astype(int)
        + (macd > macd_signal).astype(int)
        + (rsi > 50).astype(int)
    )
    full_agreement = (bull_count == 3) | (bull_count == 0)
    coherence_points = np.where(coherence_valid, np.where(full_agreement, 25, 12), 0).astype(float)

    total = pd.Series(
        adx_points + vol_points + coherence_points, index=idx
    ).clip(0, 100).round().astype(int)

    signal = pd.Series(np.select(
        [total >= 70, total >= 40],
        ["🟢 Signal propre", "🟡 Signal moyen"],
        default="🔴 Signal bruité",
    ), index=idx)

    return pd.DataFrame(
        {
            "QUALITY_SCORE": total,
            "QUALITY_SIGNAL": signal,
            "QUAL_ADX": adx_points,
            "QUAL_Volume": vol_points,
            "QUAL_Coherence": coherence_points,
        },
        index=idx,
    )


def calculate_risk_score_series(df):
    """
    RISQUE (0-100, plus haut = plus risqué) : quelle est la dangerosité de la position ?
    Volatilité annualisée 20j + ATR en % du prix + distance à la SMA200 (un titre très
    étendu par rapport à sa moyenne longue est plus exposé à une correction/reversion).
    Reste un score descriptif à seuils fixes, pas une mesure de VaR ou de perte probable.
    """
    idx = df.index
    close = _col(df, "Close")
    sma200 = _col(df, "SMA_200")
    atr = _col(df, "ATR")
    volatility = _col(df, "VOLATILITY_20")

    # --- Volatilité annualisée 20j : 40 pts ---
    volat_points = np.select(
        [volatility >= 45, (volatility >= 30) & (volatility < 45), (volatility >= 15) & (volatility < 30), volatility.notna()],
        [40, 28, 14, 5],
        default=0,
    ).astype(float)

    # --- ATR en % du prix : 30 pts ---
    atr_pct = (atr / close.replace(0, np.nan)) * 100
    atr_points = np.select(
        [atr_pct >= 5, (atr_pct >= 3) & (atr_pct < 5), (atr_pct >= 1) & (atr_pct < 3), atr_pct.notna()],
        [30, 20, 10, 3],
        default=0,
    ).astype(float)

    # --- Distance (absolue) à la SMA200 : 30 pts ---
    dist_pct = ((close / sma200.replace(0, np.nan) - 1) * 100).abs()
    dist_points = np.select(
        [dist_pct >= 30, (dist_pct >= 15) & (dist_pct < 30), (dist_pct >= 5) & (dist_pct < 15), dist_pct.notna()],
        [30, 20, 10, 3],
        default=0,
    ).astype(float)

    total = pd.Series(
        volat_points + atr_points + dist_points, index=idx
    ).clip(0, 100).round().astype(int)

    signal = pd.Series(np.select(
        [total >= 70, total >= 40],
        ["🔴 Risque élevé", "🟡 Risque modéré"],
        default="🟢 Risque faible",
    ), index=idx)

    return pd.DataFrame(
        {
            "RISK_SCORE": total,
            "RISK_SIGNAL": signal,
            "RISK_Volatilite": volat_points,
            "RISK_ATR": atr_points,
            "RISK_Distance_SMA200": dist_points,
        },
        index=idx,
    )


def calculate_all_scores(df):
    """Assemble les trois scores (Direction / Qualité / Risque) en un seul DataFrame indexé comme `df`."""
    return pd.concat(
        [
            calculate_direction_score_series(df),
            calculate_quality_score_series(df),
            calculate_risk_score_series(df),
        ],
        axis=1,
    )


def get_latest_score(components, score_col, signal_col, detail_cols):
    """
    Extrait le score, le signal et le détail (dict libellé -> points) pour la dernière
    ligne. `detail_cols` est un dict {libellé affiché: nom de colonne}.
    """
    if components.empty:
        return 0, "⚪ N/A", {}
    row = components.iloc[-1]
    details = {label: int(row[col]) for label, col in detail_cols.items()}
    return int(row[score_col]), row[signal_col], details


def get_score_delta(components, score_col, lookback=1):
    """
    Variation d'un score entre la dernière séance et celle `lookback` période(s) avant.
    Retourne (delta, score_precedent) ou (np.nan, np.nan) si l'historique est trop court.
    """
    if len(components) <= lookback:
        return np.nan, np.nan
    latest = int(components[score_col].iloc[-1])
    previous = int(components[score_col].iloc[-1 - lookback])
    return latest - previous, previous


# ============================================================
# RECOMMANDATION DE POSITION (Garder / Renforcer / Vendre)
# ============================================================

def generate_position_recommendation(
    direction_score, quality_score, risk_score, pnl_pct, stop_loss_pct,
    direction_vente_seuil=50, qualite_fiable_seuil=40,
    direction_haussier_seuil=65, qualite_propre_seuil=70, risque_eleve_seuil=70,
):
    """
    Traduit les trois scores + le P&L latent en une recommandation actionnable, pour un
    titre DÉJÀ détenu. Volontairement un arbre de règles explicite (if/elif) plutôt qu'un
    nouveau score composite pondéré : chaque recommandation est justifiée par une règle
    identifiable et lisible, pas par une somme de poids invisibles. Reste une heuristique
    non calibrée, comme les scores dont elle dépend — voir "Limites de l'outil".

    Les 5 seuils sont réglables (sidebar) plutôt que figés dans le code :
    - direction_vente_seuil : sous ce score Direction, lecture baissière (règle 2)
    - qualite_fiable_seuil : sous ce score Qualité, signal jugé trop bruité pour agir (règles 2 et 5)
    - direction_haussier_seuil : à partir de ce score Direction, lecture haussière (règles 3 et 4)
    - qualite_propre_seuil : à partir de ce score Qualité, signal jugé propre (règle 4)
    - risque_eleve_seuil : à partir de ce score Risque, jugé élevé (règles 3 et 4)

    Retourne (label, couleur, explication).
    """
    # 1) Règle de gestion du risque : protège le capital, prioritaire sur toute lecture
    #    technique — un stop-loss touché reste un stop-loss touché.
    if stop_loss_pct > 0 and pd.notna(pnl_pct) and pnl_pct <= -stop_loss_pct:
        return (
            "🔴 VENDRE",
            "red",
            f"Perte latente ({pnl_pct:+.1f}%) a atteint le stop-loss configuré "
            f"(-{stop_loss_pct:.1f}%). Règle de gestion du risque, prioritaire sur la "
            f"lecture technique du moment.",
        )

    # 2) Signal baissier jugé fiable (pas juste du bruit)
    if direction_score < direction_vente_seuil and quality_score >= qualite_fiable_seuil:
        return (
            "🔴 VENDRE",
            "red",
            f"Score Direction en zone vente ({direction_score}/100 < {direction_vente_seuil}), "
            f"sur un signal jugé suffisamment propre pour ne pas l'ignorer "
            f"(Qualité {quality_score}/100 ≥ {qualite_fiable_seuil}).",
        )

    # 3) Combinaison dangereuse : risque élevé sans direction clairement favorable
    if risk_score >= risque_eleve_seuil and direction_score < direction_haussier_seuil:
        return (
            "🔴 VENDRE",
            "red",
            f"Risque élevé ({risk_score}/100 ≥ {risque_eleve_seuil}) combiné à une Direction "
            f"pas clairement haussière ({direction_score}/100 < {direction_haussier_seuil}) : "
            f"exposition disproportionnée par rapport au potentiel identifié.",
        )

    # 4) Configuration haussière forte, propre, et pas trop risquée
    if (
        direction_score >= direction_haussier_seuil
        and quality_score >= qualite_propre_seuil
        and risk_score < risque_eleve_seuil
    ):
        return (
            "🟢 RENFORCER",
            "green",
            f"Direction haussière ({direction_score}/100 ≥ {direction_haussier_seuil}) confirmée "
            f"par un signal propre (Qualité {quality_score}/100 ≥ {qualite_propre_seuil}) sans "
            f"risque excessif (Risque {risk_score}/100 < {risque_eleve_seuil}).",
        )

    # 5) Signal trop bruité pour agir dans un sens ou l'autre
    if quality_score < qualite_fiable_seuil:
        return (
            "🟡 GARDER",
            "orange",
            f"Signal bruité (Qualité {quality_score}/100 < {qualite_fiable_seuil}) : pas assez "
            f"fiable pour justifier un renfort ou une vente. Attendre une lecture plus claire.",
        )

    # 6) Cas par défaut : rien ne déclenche de décision nette
    return (
        "🟡 GARDER",
        "orange",
        f"Aucune règle déclenchée : Direction {direction_score}/100, "
        f"Qualité {quality_score}/100, Risque {risk_score}/100 ne justifient ni renfort "
        f"ni vente dans l'état actuel.",
    )


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


def rolling_regression_stats(df, window=50):
    """
    Calcule, pour CHAQUE date de l'historique, la pente et le R² de la régression
    linéaire sur les `window` séances précédentes (même logique que `regression_channel`,
    mais glissée sur tout l'historique au lieu d'un seul point). Calcul vectorisé en
    O(n) via des sommes cumulées pondérées, plutôt qu'un np.polyfit par fenêtre (ce qui
    serait O(n * window)).

    La pente est aussi exprimée en % du prix (SLOPE_PCT) pour rester comparable entre
    titres à des niveaux de prix différents. Comme le canal de régression, ceci reste une
    extrapolation géométrique descriptive du passé récent — pas une prédiction.
    """
    close = df["Close"]
    n = len(close)
    idx = df.index

    if n < window + 1:
        return pd.DataFrame(
            {"SLOPE": np.nan, "SLOPE_PCT": np.nan, "R2": np.nan}, index=idx
        )

    y = close.to_numpy(dtype=float)
    pos = np.arange(n, dtype=float)

    s1 = pd.Series(y, index=idx).rolling(window).sum()
    s2 = pd.Series(y ** 2, index=idx).rolling(window).sum()

    weighted_cumsum = pd.Series(np.cumsum(pos * y), index=idx)
    weighted_cumsum_shifted = weighted_cumsum.shift(window).fillna(0.0)
    sum_jy = weighted_cumsum - weighted_cumsum_shifted  # sum_{j=t-w+1}^{t} j*y_j

    t_pos = pd.Series(pos, index=idx)
    window_start = t_pos - window + 1
    sum_xy = sum_jy - window_start * s1  # ramène x à une base locale 0..window-1

    sum_x = window * (window - 1) / 2
    sum_x2 = (window - 1) * window * (2 * window - 1) / 6
    denom = window * sum_x2 - sum_x ** 2

    slope = (window * sum_xy - sum_x * s1) / denom

    ss_xx = sum_x2 - (sum_x ** 2) / window
    ss_yy = s2 - (s1 ** 2) / window

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (slope ** 2 * ss_xx) / ss_yy
    r2 = r2.replace([np.inf, -np.inf], np.nan)

    slope.iloc[: window - 1] = np.nan
    r2.iloc[: window - 1] = np.nan

    slope_pct = (slope / close.replace(0, np.nan)) * 100

    return pd.DataFrame({"SLOPE": slope, "SLOPE_PCT": slope_pct, "R2": r2}, index=idx)


def detect_regression_inflections(slope_stats, lookback=20):
    """
    Repère les changements de signe de la pente glissante sur les `lookback` dernières
    séances : la tendance récente s'inverse (extrapolation linéaire uniquement — pas de
    garantie que ça se poursuive).
    """
    events = []
    recent = slope_stats["SLOPE"].dropna().tail(lookback + 1)
    if len(recent) < 2:
        return events

    for i in range(1, len(recent)):
        prev_val, curr_val = recent.iloc[i - 1], recent.iloc[i]
        date = recent.index[i]
        if prev_val <= 0 < curr_val:
            events.append((date, "🟢 Pente de régression passe positive (inflexion potentielle)"))
        elif prev_val >= 0 > curr_val:
            events.append((date, "🔴 Pente de régression passe négative (inflexion potentielle)"))

    return sorted(events, key=lambda x: x[0], reverse=True)


def regression_slope_chart(slope_stats, window):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.65, 0.35],
        subplot_titles=(
            f"Pente de régression glissante ({window} séances) — en % du prix",
            "R² glissant — qualité de l'ajustement linéaire",
        ),
    )

    colors = np.where(slope_stats["SLOPE_PCT"] >= 0, "#2ca02c", "#d62728")
    fig.add_trace(
        go.Bar(x=slope_stats.index, y=slope_stats["SLOPE_PCT"], marker_color=colors, name="Pente (%)"),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=slope_stats.index, y=slope_stats["R2"], name="R²", line=dict(width=2, color="#1f77b4")),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="R² = 0.5", row=2, col=1)

    fig.update_yaxes(title_text="% du prix / période", row=1, col=1)
    fig.update_yaxes(title_text="R²", range=[0, 1], row=2, col=1)
    fig.update_layout(height=500, hovermode="x unified", showlegend=False)
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

ML_REGRESSION_FEATURE_COLUMNS = ["REG_SLOPE_PCT", "REG_R2"]


def build_ml_dataset(df, horizon=5, include_regression_features=False):
    """
    Construit X (features) / y (cible binaire : hausse à horizon `horizon` périodes)
    à partir des indicateurs déjà calculés de façon causale. Ajoute quelques features
    dérivées (distance relative aux moyennes mobiles, position dans les bandes de
    Bollinger, rendement récent) qui ne dépendent, comme le reste, que du passé.

    `include_regression_features` ajoute la pente de régression glissante (% du prix) et
    son R² — utile pour tester empiriquement si elles apportent une information marginale
    par rapport aux indicateurs déjà présents (RSI/MACD/ADX), plutôt que de le supposer.
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
    if include_regression_features:
        feature_cols = feature_cols + ML_REGRESSION_FEATURE_COLUMNS

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
    min_quality=0,
):
    """
    Backtest à une seule passe (O(n)).
    - Le signal d'entrée est décidé sur le score DIRECTION de la veille (pas de fuite
      d'info), optionnellement filtré par un score QUALITE minimal (si `min_quality` > 0
      et que la colonne QUALITY_SCORE est disponible) : un signal directionnel fort mais
      jugé bruité (ADX faible, volume absent, indicateurs contradictoires) est ignoré.
    - Le stop-loss / take-profit est vérifié intra-séance via le Low/High du jour.
    - Le sizing "volatility" réduit l'exposition quand l'ATR relatif est élevé.
    """
    data = df.copy()
    if data.empty or "DIRECTION_SCORE" not in data.columns:
        return pd.DataFrame(), {}

    n = len(data)
    close = data["Close"].to_numpy()
    high = data["High"].to_numpy()
    low = data["Low"].to_numpy()
    score = data["DIRECTION_SCORE"].to_numpy()
    atr = data["ATR"].to_numpy() if "ATR" in data.columns else np.full(n, np.nan)

    signal_on = score >= threshold
    if min_quality > 0 and "QUALITY_SCORE" in data.columns:
        quality = data["QUALITY_SCORE"].to_numpy()
        signal_on = signal_on & (quality >= min_quality)

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
                        stop_loss_pct, take_profit_pct, sizing, annual_factor, min_quality=0):
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
            stop_loss_pct, take_profit_pct, sizing, annual_factor, min_quality,
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


def score_history_chart(components, score_col, title, threshold=None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=components.index, y=components[score_col], name=title,
            line=dict(width=2), fill="tozeroy",
        )
    )
    if threshold is not None:
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="orange",
            annotation_text=f"Seuil d'entrée ({threshold})",
        )
    fig.update_layout(
        title=f"Évolution du score {title} dans le temps",
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

st.sidebar.subheader("📋 Actions suivies")

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = list(DEFAULT_TICKERS)


def _add_ticker(raw_value):
    """Ajoute un ou plusieurs tickers (séparés par virgule/saut de ligne) à la watchlist."""
    added, already_present = [], []
    for candidate in raw_value.replace("\n", ",").split(","):
        candidate = candidate.strip().upper()
        if not candidate:
            continue
        if candidate in st.session_state["watchlist"]:
            already_present.append(candidate)
        else:
            st.session_state["watchlist"].append(candidate)
            added.append(candidate)
    return added, already_present


# --- Ajout rapide d'un ticker ---
with st.sidebar.form("add_ticker_form", clear_on_submit=True):
    new_ticker = st.text_input(
        "Ajouter un ticker",
        placeholder="ex : AAPL, MC.PA, ^FCHI...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("➕ Ajouter à la liste", use_container_width=True)
    if submitted and new_ticker.strip():
        added, already_present = _add_ticker(new_ticker)
        if added:
            st.toast(f"Ajouté : {', '.join(added)}")
        if already_present:
            st.toast(f"Déjà dans la liste : {', '.join(already_present)}")

# --- Liste actuelle, avec bouton de suppression par ticker ---
if st.session_state["watchlist"]:
    for t in list(st.session_state["watchlist"]):
        col_name, col_remove = st.sidebar.columns([4, 1])
        col_name.markdown(f"`{t}`")
        if col_remove.button("✕", key=f"remove_ticker_{t}", help=f"Retirer {t}"):
            st.session_state["watchlist"].remove(t)
            st.rerun()
else:
    st.sidebar.caption("Liste vide — ajoutez au moins un ticker ci-dessus.")

# --- Import en masse / réinitialisation ---
with st.sidebar.expander("📥 Import en masse / réinitialiser"):
    bulk_input = st.text_area(
        "Coller une liste (virgules ou sauts de ligne)",
        placeholder="AAPL, MSFT, MC.PA",
        height=90,
        key="bulk_ticker_input",
    )
    col_import, col_reset = st.columns(2)
    with col_import:
        if st.button("Importer", use_container_width=True) and bulk_input.strip():
            added, already_present = _add_ticker(bulk_input)
            if added:
                st.toast(f"Ajouté : {', '.join(added)}")
            st.rerun()
    with col_reset:
        if st.button("↺ Réinitialiser", use_container_width=True):
            st.session_state["watchlist"] = list(DEFAULT_TICKERS)
            st.rerun()

tickers = list(st.session_state["watchlist"])

st.sidebar.markdown("---")
st.sidebar.subheader("💼 Mes positions")
st.sidebar.caption(
    "Renseignez vos positions détenues (prix d'achat, quantité) pour obtenir une "
    "recommandation Garder / Renforcer / Vendre plutôt qu'une simple lecture Direction."
)

if "positions" not in st.session_state:
    st.session_state["positions"] = {}

with st.sidebar.form("add_position_form", clear_on_submit=True):
    pos_ticker = st.text_input("Ticker détenu", placeholder="ex : SOI.PA")
    pos_price = st.number_input("Prix d'achat moyen", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    pos_qty = st.number_input("Quantité", min_value=0.0, value=0.0, step=1.0)
    pos_submitted = st.form_submit_button("💾 Enregistrer la position", use_container_width=True)
    if pos_submitted and pos_ticker.strip() and pos_price > 0 and pos_qty > 0:
        pos_ticker_clean = pos_ticker.strip().upper()
        st.session_state["positions"][pos_ticker_clean] = {"prix_achat": pos_price, "quantite": pos_qty}
        if pos_ticker_clean not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(pos_ticker_clean)
        st.toast(f"Position enregistrée : {pos_ticker_clean}")
        st.rerun()

if st.session_state["positions"]:
    for pos_ticker, pos_info in list(st.session_state["positions"].items()):
        col_pos, col_pos_remove = st.sidebar.columns([4, 1])
        col_pos.markdown(f"`{pos_ticker}` — {pos_info['quantite']:g} @ {pos_info['prix_achat']:.2f}")
        if col_pos_remove.button("✕", key=f"remove_position_{pos_ticker}", help=f"Retirer la position {pos_ticker}"):
            del st.session_state["positions"][pos_ticker]
            st.rerun()
else:
    st.sidebar.caption("Aucune position enregistrée.")

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

threshold = st.sidebar.slider("Seuil d'entrée (score Direction)", 40, 90, 65, 5,
                               help="Position ouverte quand le score Direction de la veille dépasse ce seuil.")

min_quality = st.sidebar.slider(
    "Qualité minimale (optionnel)", 0, 90, 0, 5,
    help=(
        "0 = désactivé. Si > 0, un signal Direction n'ouvre une position que si le score "
        "Qualité de la veille est aussi au-dessus de ce seuil — filtre les signaux "
        "directionnels forts mais bruités (ADX faible, volume absent, indicateurs en "
        "désaccord)."
    ),
)

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
st.sidebar.subheader("🎚️ Seuils de recommandation")
st.sidebar.caption(
    "Réglages de l'arbre de décision Garder / Renforcer / Vendre (voir 💼 Mes positions). "
    "Le stop-loss ci-dessus reste toujours prioritaire sur ces seuils."
)

direction_vente_seuil = st.sidebar.slider(
    "Direction sous laquelle → lecture baissière", 20, 60, 50, 5,
    help="Sous ce score Direction (avec un signal jugé fiable), la recommandation penche vers VENDRE.",
)
qualite_fiable_seuil = st.sidebar.slider(
    "Qualité minimale pour agir", 20, 60, 40, 5,
    help="Sous ce score Qualité, le signal est jugé trop bruité pour déclencher un achat ou une vente — la recommandation reste GARDER.",
)
direction_haussier_seuil = st.sidebar.slider(
    "Direction à partir de laquelle → lecture haussière", 50, 85, 65, 5,
    help="À partir de ce score Direction, la structure est jugée favorable à un renfort (sous réserve de Qualité et Risque).",
)
qualite_propre_seuil = st.sidebar.slider(
    "Qualité minimale pour renforcer", 50, 90, 70, 5,
    help="À partir de ce score Qualité, le signal est jugé assez propre pour justifier un renfort de position.",
)
risque_eleve_seuil = st.sidebar.slider(
    "Risque à partir duquel jugé élevé", 40, 90, 70, 5,
    help="À partir de ce score Risque, la position est jugée trop dangereuse pour être renforcée, voire à vendre si la Direction n'est pas franchement haussière.",
)

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
st.markdown("Analyse technique • Scanner • Direction / Qualité / Risque • Backtest robuste • Portefeuille")


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

        components = calculate_all_scores(df)
        df = pd.concat([df, components], axis=1)

        slope_stats = rolling_regression_stats(df, window=regression_window)
        df["REG_SLOPE_PCT"] = slope_stats["SLOPE_PCT"]
        df["REG_R2"] = slope_stats["R2"]

        all_data[ticker] = df

        direction_score, direction_signal, direction_details = get_latest_score(
            components, "DIRECTION_SCORE", "DIRECTION_SIGNAL",
            {"Tendance": "DIR_Tendance", "RSI": "DIR_RSI", "MACD": "DIR_MACD", "Momentum": "DIR_Momentum"},
        )
        quality_score, quality_signal, quality_details = get_latest_score(
            components, "QUALITY_SCORE", "QUALITY_SIGNAL",
            {"ADX": "QUAL_ADX", "Volume": "QUAL_Volume", "Cohérence": "QUAL_Coherence"},
        )
        risk_score, risk_signal, risk_details = get_latest_score(
            components, "RISK_SCORE", "RISK_SIGNAL",
            {"Volatilité": "RISK_Volatilite", "ATR": "RISK_ATR", "Distance SMA200": "RISK_Distance_SMA200"},
        )
        direction_delta, _ = get_score_delta(components, "DIRECTION_SCORE", lookback=1)
        trend = detect_trend(df)
        row = df.iloc[-1]
        close = safe_float(row["Close"])
        one_month_return = np.nan

        if len(df) >= 21:
            old = safe_float(df["Close"].iloc[-21])
            if pd.notna(old) and old != 0:
                one_month_return = (close / old - 1) * 100

        if pd.isna(direction_delta):
            direction_trend_icon = "⚪"
        elif direction_delta > 0:
            direction_trend_icon = "🔼"
        elif direction_delta < 0:
            direction_trend_icon = "🔽"
        else:
            direction_trend_icon = "➖"

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
                "Direction": direction_score,
                "Δ Direction (veille)": direction_delta,
                "Tendance direction": direction_trend_icon,
                "Qualité": quality_score,
                "Risque": risk_score,
                "Signal": direction_signal,
                "Signal qualité": quality_signal,
                "Signal risque": risk_signal,
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
results_df = results_df.sort_values("Direction", ascending=False).reset_index(drop=True)


# ============================================================
# VUE GENERALE
# ============================================================

st.header("🏆 Vue d'ensemble")
best = results_df.iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Actions analysées", len(results_df))
with c2:
    st.metric(
        "Meilleure Direction", f"{best['Ticker']} — {int(best['Direction'])}/100",
        help="Trié par score Direction. Vérifiez aussi sa Qualité avant d'agir — un score Direction élevé sur un signal bruité (🔴) est moins fiable.",
    )
with c3:
    st.metric("Direction ≥ 65", int((results_df["Direction"] >= 65).sum()))
with c4:
    st.metric("Qualité ≥ 70 (signal propre)", int((results_df["Qualité"] >= 70).sum()))


# ============================================================
# CLASSEMENT
# ============================================================

st.subheader("🏆 Classement technique")
st.caption(
    "**Direction** : la structure favorise-t-elle une hausse ? • **Qualité** : ce signal "
    "est-il propre ou bruité (0-100, plus haut = plus fiable) ? • **Risque** : dangerosité "
    "de la position (0-100, plus haut = plus risqué). Les trois sont volontairement séparés "
    "plutôt que fondus en un seul chiffre — voir la section Limites pour le détail."
)

numeric_cols = results_df.select_dtypes(include=[np.number]).columns
display_df = results_df.copy()
display_df[numeric_cols] = display_df[numeric_cols].round(2)

st.dataframe(display_df, use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_bar(x=results_df["Ticker"], y=results_df["Direction"], name="Direction", marker_color="#1f77b4")
fig.add_bar(x=results_df["Ticker"], y=results_df["Qualité"], name="Qualité", marker_color="#2ca02c")
fig.add_bar(x=results_df["Ticker"], y=results_df["Risque"], name="Risque", marker_color="#d62728")
fig.update_layout(
    title="Direction / Qualité / Risque par action",
    yaxis=dict(range=[0, 100]), height=420, barmode="group",
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MES POSITIONS — RECOMMANDATIONS
# ============================================================

tracked_positions = {
    t: p for t, p in st.session_state.get("positions", {}).items() if t in all_data
}

if tracked_positions:
    st.markdown("---")
    st.header("💼 Mes positions — recommandations")
    st.caption(
        "Recommandation par règles explicites (pas un score composite caché) : le "
        "stop-loss configuré dans la sidebar protège le capital en priorité, puis la "
        "combinaison Direction / Qualité / Risque détermine Garder, Renforcer ou Vendre. "
        "Cliquez sur une ligne dans « Analyse détaillée » pour voir le détail de la règle "
        "appliquée à un titre précis."
    )

    position_rows = []
    for pos_ticker, pos_info in tracked_positions.items():
        pos_df = all_data[pos_ticker]
        pos_row = pos_df.iloc[-1]
        current_price = safe_float(pos_row["Close"])
        entry_price = pos_info["prix_achat"]
        qty = pos_info["quantite"]

        pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else np.nan
        pnl_eur = (current_price - entry_price) * qty if pd.notna(current_price) else np.nan

        d_score = int(pos_row["DIRECTION_SCORE"])
        q_score = int(pos_row["QUALITY_SCORE"])
        r_score = int(pos_row["RISK_SCORE"])

        reco_label, reco_color, reco_reason = generate_position_recommendation(
            d_score, q_score, r_score, pnl_pct, stop_loss_pct,
            direction_vente_seuil, qualite_fiable_seuil,
            direction_haussier_seuil, qualite_propre_seuil, risque_eleve_seuil,
        )

        position_rows.append(
            {
                "Ticker": pos_ticker,
                "Prix achat": entry_price,
                "Qté": qty,
                "Prix actuel": current_price,
                "P&L %": pnl_pct,
                "P&L (€)": pnl_eur,
                "Direction": d_score,
                "Qualité": q_score,
                "Risque": r_score,
                "Recommandation": reco_label,
            }
        )

    positions_df = pd.DataFrame(position_rows)
    numeric_pos_cols = positions_df.select_dtypes(include=[np.number]).columns
    positions_display = positions_df.copy()
    positions_display[numeric_pos_cols] = positions_display[numeric_pos_cols].round(2)
    st.dataframe(positions_display, use_container_width=True, hide_index=True)

    n_sell = int((positions_df["Recommandation"] == "🔴 VENDRE").sum())
    n_add = int((positions_df["Recommandation"] == "🟢 RENFORCER").sum())
    n_hold = int((positions_df["Recommandation"] == "🟡 GARDER").sum())
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🟢 À renforcer", n_add)
    with c2:
        st.metric("🟡 À garder", n_hold)
    with c3:
        st.metric("🔴 À vendre", n_sell)


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

components = df[[
    "DIRECTION_SCORE", "DIRECTION_SIGNAL", "DIR_Tendance", "DIR_RSI", "DIR_MACD", "DIR_Momentum",
    "QUALITY_SCORE", "QUALITY_SIGNAL", "QUAL_ADX", "QUAL_Volume", "QUAL_Coherence",
    "RISK_SCORE", "RISK_SIGNAL", "RISK_Volatilite", "RISK_ATR", "RISK_Distance_SMA200",
]]

direction_score, direction_signal, direction_details = get_latest_score(
    components, "DIRECTION_SCORE", "DIRECTION_SIGNAL",
    {"Tendance": "DIR_Tendance", "RSI": "DIR_RSI", "MACD": "DIR_MACD", "Momentum": "DIR_Momentum"},
)
quality_score, quality_signal, quality_details = get_latest_score(
    components, "QUALITY_SCORE", "QUALITY_SIGNAL",
    {"ADX": "QUAL_ADX", "Volume": "QUAL_Volume", "Cohérence": "QUAL_Coherence"},
)
risk_score, risk_signal, risk_details = get_latest_score(
    components, "RISK_SCORE", "RISK_SIGNAL",
    {"Volatilité": "RISK_Volatilite", "ATR": "RISK_ATR", "Distance SMA200": "RISK_Distance_SMA200"},
)
direction_delta, _ = get_score_delta(components, "DIRECTION_SCORE", lookback=1)
quality_delta, _ = get_score_delta(components, "QUALITY_SCORE", lookback=1)
risk_delta, _ = get_score_delta(components, "RISK_SCORE", lookback=1)
trend = detect_trend(df)
row = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Prix", fmt(row["Close"]))
with c2:
    st.metric("RSI 14", fmt(row["RSI"], 1), help="< 30 : survente. > 70 : surachat.")
with c3:
    st.metric("SMA 50", fmt(row["SMA_50"]))
with c4:
    st.metric("SMA 200", fmt(row["SMA_200"]))

st.markdown("**Les trois scores** — séparés plutôt que fondus en un seul chiffre : un score Direction élevé n'a de sens que si la Qualité est bonne (signal propre, pas bruité), et le Risque s'apprécie indépendamment des deux.")
c5, c6, c7 = st.columns(3)
with c5:
    st.metric(
        "🟢 Direction", f"{direction_score}/100",
        delta=None if pd.isna(direction_delta) else f"{direction_delta:+d} vs veille",
        help="La structure actuelle (tendance SMA, RSI, MACD, momentum) favorise-t-elle une hausse ? Heuristique non calibrée — voir Limites.",
    )
    st.caption(direction_signal)
with c6:
    st.metric(
        "🔵 Qualité", f"{quality_score}/100",
        delta=None if pd.isna(quality_delta) else f"{quality_delta:+d} vs veille",
        help="Ce signal Direction est-il propre (ADX élevé, volume qui confirme, indicateurs d'accord entre eux) ou bruité ?",
    )
    st.caption(quality_signal)
with c7:
    st.metric(
        "🟠 Risque", f"{risk_score}/100",
        delta=None if pd.isna(risk_delta) else f"{risk_delta:+d} vs veille",
        delta_color="inverse",  # une hausse du risque n'est pas une "bonne" variation
        help="Dangerosité de la position (volatilité, ATR relatif, écart à la SMA200). Plus haut = plus risqué.",
    )
    st.caption(risk_signal)

st.info(f"**Tendance :** {trend}")

st.subheader("💼 Ma position sur ce titre")
existing_position = st.session_state.get("positions", {}).get(selected)

if existing_position:
    entry_price = existing_position["prix_achat"]
    qty = existing_position["quantite"]
    current_price = safe_float(row["Close"])
    pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else np.nan
    pnl_eur = (current_price - entry_price) * qty if pd.notna(current_price) else np.nan

    reco_label, reco_color, reco_reason = generate_position_recommendation(
        direction_score, quality_score, risk_score, pnl_pct, stop_loss_pct,
        direction_vente_seuil, qualite_fiable_seuil,
        direction_haussier_seuil, qualite_propre_seuil, risque_eleve_seuil,
    )

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.metric("Prix d'achat", fmt(entry_price))
    with pc2:
        st.metric("Quantité", f"{qty:g}")
    with pc3:
        st.metric("P&L", f"{pnl_pct:+.1f}%" if pd.notna(pnl_pct) else "N/A", delta=fmt(pnl_eur, 2, " €") if pd.notna(pnl_eur) else None)
    with pc4:
        st.metric("Recommandation", reco_label)

    if reco_color == "red":
        st.error(f"**{reco_label}** — {reco_reason}")
    elif reco_color == "green":
        st.success(f"**{reco_label}** — {reco_reason}")
    else:
        st.warning(f"**{reco_label}** — {reco_reason}")

    if st.button("🗑️ Retirer cette position", key=f"remove_pos_detail_{selected}"):
        del st.session_state["positions"][selected]
        st.rerun()
else:
    st.caption("Aucune position enregistrée pour ce titre.")
    with st.form(f"add_position_detail_{selected}", clear_on_submit=True):
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            detail_pos_price = st.number_input("Prix d'achat moyen", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        with dc2:
            detail_pos_qty = st.number_input("Quantité", min_value=0.0, value=0.0, step=1.0)
        with dc3:
            st.write("")
            detail_pos_submit = st.form_submit_button("💾 Enregistrer")
        if detail_pos_submit and detail_pos_price > 0 and detail_pos_qty > 0:
            st.session_state["positions"][selected] = {
                "prix_achat": detail_pos_price, "quantite": detail_pos_qty,
            }
            st.rerun()

st.subheader("Décomposition des scores")
detail_df = pd.DataFrame(
    {
        "Score": (
            ["Direction"] * len(direction_details)
            + ["Qualité"] * len(quality_details)
            + ["Risque"] * len(risk_details)
        ),
        "Composante": (
            list(direction_details.keys()) + list(quality_details.keys()) + list(risk_details.keys())
        ),
        "Points": (
            list(direction_details.values()) + list(quality_details.values()) + list(risk_details.values())
        ),
    }
)
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
    score_choice = st.radio(
        "Score affiché",
        ["🟢 Direction", "🔵 Qualité", "🟠 Risque"],
        index=0,
        horizontal=True,
        key="score_history_choice",
    )
    score_col_map = {
        "🟢 Direction": ("DIRECTION_SCORE", "DIRECTION_SIGNAL", "Direction"),
        "🔵 Qualité": ("QUALITY_SCORE", "QUALITY_SIGNAL", "Qualité"),
        "🟠 Risque": ("RISK_SCORE", "RISK_SIGNAL", "Risque"),
    }
    score_col, signal_col, score_title = score_col_map[score_choice]

    window_choice = st.radio(
        "Fenêtre affichée",
        ["10 séances", "20 séances", "60 séances", "Tout l'historique"],
        index=1,
        horizontal=True,
        help="Sur tout l'historique (plusieurs mois/années), les variations jour à jour sont écrasées visuellement — réduisez la fenêtre pour les voir clairement.",
    )
    window_map = {"10 séances": 10, "20 séances": 20, "60 séances": 60, "Tout l'historique": None}
    n_sessions = window_map[window_choice]
    components_window = components if n_sessions is None else components.tail(n_sessions)

    chart_threshold = threshold if score_col == "DIRECTION_SCORE" else None
    st.plotly_chart(
        score_history_chart(components_window, score_col, score_title, threshold=chart_threshold),
        use_container_width=True,
    )
    st.caption(f"Permet de voir si le score {score_title} est stable dans le temps ou s'il oscille beaucoup.")

    st.markdown("**Détail jour par jour**")
    table_n = min(n_sessions or 30, 30)  # la table reste lisible même si le graphique montre tout
    daily = components[score_col].tail(table_n + 1).to_frame()
    daily["Δ vs veille"] = daily[score_col].diff()
    daily = daily.iloc[1:].sort_index(ascending=False)  # la 1ère ligne sert seulement au calcul du 1er delta
    daily_display = pd.DataFrame(
        {
            "Date": daily.index.strftime("%Y-%m-%d"),
            score_title: daily[score_col].astype(int),
            "Δ vs veille": daily["Δ vs veille"].apply(
                lambda v: "—" if pd.isna(v) else f"{int(v):+d}"
            ),
            "Signal": components.loc[daily.index, signal_col].values,
        }
    )
    st.dataframe(daily_display, use_container_width=True, hide_index=True)

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

    st.markdown("---")
    st.markdown("**Pente de régression dans le temps**")
    st.caption(
        "La pente ci-dessus n'est qu'un instantané. En la recalculant à chaque séance "
        "(même fenêtre glissante), on peut voir si elle accélère, ralentit, ou change de "
        "signe — un changement de signe signale une inflexion potentielle de la tendance "
        "récente, souvent plus tôt qu'un croisement de moyennes mobiles (qui est lissé sur "
        "une fenêtre plus longue). Le R² glissant indique en parallèle si la tendance reste "
        "linéaire ou si le titre entre en range (R² qui chute)."
    )

    slope_window_choice = st.radio(
        "Fenêtre affichée",
        ["20 séances", "60 séances", "120 séances", "Tout l'historique"],
        index=1,
        horizontal=True,
        key="slope_window_choice",
    )
    slope_window_map = {
        "20 séances": 20, "60 séances": 60, "120 séances": 120, "Tout l'historique": None,
    }
    n_slope_sessions = slope_window_map[slope_window_choice]

    slope_stats = rolling_regression_stats(df, window=regression_window)
    slope_stats_display = slope_stats if n_slope_sessions is None else slope_stats.tail(n_slope_sessions)

    if slope_stats["SLOPE"].dropna().empty:
        st.caption("Historique insuffisant pour calculer la pente glissante sur cette fenêtre.")
    else:
        st.plotly_chart(regression_slope_chart(slope_stats_display, regression_window), use_container_width=True)

        inflections = detect_regression_inflections(slope_stats, lookback=30)
        if inflections:
            st.markdown("**Inflexions récentes (changement de signe de la pente, 30 dernières séances)**")
            inflections_df = pd.DataFrame(inflections, columns=["Date", "Événement"])
            inflections_df["Date"] = inflections_df["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(inflections_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Aucun changement de signe de la pente sur les 30 dernières séances.")

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

        compare_slope_features = st.checkbox(
            "Comparer avec / sans les features de pente de régression glissante",
            value=True,
            help=(
                "Teste empiriquement si la pente glissante (% du prix) et son R² "
                "apportent une information marginale par rapport à RSI/MACD/ADX déjà "
                "présents, plutôt que de le supposer à l'œil sur un graphique."
            ),
        )

        if st.button("▶️ Entraîner et évaluer le modèle", type="primary"):
            with st.spinner("Construction du jeu de données et validation walk-forward..."):
                dataset, feature_cols = build_ml_dataset(
                    df, horizon=ml_horizon, include_regression_features=False
                )
                eval_result = walk_forward_evaluation(dataset, feature_cols)

                eval_ext = None
                if compare_slope_features:
                    dataset_ext, feature_cols_ext = build_ml_dataset(
                        df, horizon=ml_horizon, include_regression_features=True
                    )
                    eval_ext = walk_forward_evaluation(dataset_ext, feature_cols_ext)

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

                if compare_slope_features and eval_ext is not None:
                    folds_ext = eval_ext["folds"]
                    mean_auc_ext = folds_ext["AUC"].mean()
                    mean_brier_ext = folds_ext["Brier"].mean()

                    st.markdown("**Sans pente glissante vs avec pente glissante**")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("AUC — sans pente", fmt(mean_auc, 3))
                    with c2:
                        st.metric(
                            "AUC — avec pente", fmt(mean_auc_ext, 3),
                            delta=f"{mean_auc_ext - mean_auc:+.3f}",
                        )
                    with c3:
                        st.metric("Brier — sans pente", fmt(mean_brier, 3))
                    with c4:
                        st.metric(
                            "Brier — avec pente", fmt(mean_brier_ext, 3),
                            delta=f"{mean_brier_ext - mean_brier:+.3f}",
                            delta_color="inverse",  # un Brier plus bas est meilleur
                        )

                    auc_gain = mean_auc_ext - mean_auc
                    if abs(auc_gain) < 0.01:
                        st.info(
                            f"Écart d'AUC de {auc_gain:+.3f} sur {selected} à cet horizon : "
                            "négligeable. La pente glissante n'apporte pas d'information "
                            "supplémentaire mesurable par rapport à RSI/MACD/ADX déjà "
                            "présents — l'hypothèse de redondance se confirme empiriquement "
                            "sur ce titre et cette période."
                        )
                    elif auc_gain >= 0.01:
                        st.success(
                            f"Écart d'AUC de {auc_gain:+.3f} : la pente glissante apporte un "
                            "gain mesurable ici. À vérifier sur d'autres titres/horizons "
                            "avant d'en tirer une conclusion générale — un seul test n'est "
                            "pas une preuve robuste."
                        )
                    else:
                        st.warning(
                            f"Écart d'AUC de {auc_gain:+.3f} : le modèle avec pente glissante "
                            "fait *moins bien* ici, probablement parce qu'elle ajoute du bruit "
                            "corrélé aux features existantes sans info nouvelle (dimension "
                            "supplémentaire à estimer, pour un jeu de données déjà limité)."
                        )

                    # La suite du détail (calibration, probabilité) porte sur le modèle
                    # "avec pente" pour rester cohérent avec la comparaison ci-dessus.
                    active_eval = eval_ext
                    active_dataset, active_feature_cols = dataset_ext, feature_cols_ext
                    detail_label = "avec pente glissante"
                else:
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

                    active_eval = eval_result
                    active_dataset, active_feature_cols = dataset, feature_cols
                    detail_label = "sans pente glissante"

                st.plotly_chart(
                    reliability_diagram(active_eval["test_probs"], active_eval["test_true"]),
                    use_container_width=True,
                )

                with st.expander(f"Détail par pli de validation (walk-forward, modèle {detail_label})"):
                    st.dataframe(active_eval["folds"].round(3), use_container_width=True, hide_index=True)

                st.markdown("---")
                proba = predict_latest_probability(active_dataset, active_feature_cols)
                if pd.notna(proba):
                    st.metric(
                        f"Probabilité estimée de hausse à {ml_horizon} période(s) (modèle {detail_label})",
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

quality_filter_text = (
    f" • Qualité minimale : **{min_quality}/100**" if min_quality > 0 else " • Pas de filtre Qualité"
)
st.write(
    f"""
    Position ouverte lorsque le score **Direction** de la veille atteint **{threshold}/100**
    (signal décalé d'une période pour éviter tout effet de bord){quality_filter_text}.
    Frais simulés : **{transaction_cost:.2f}%** par changement de position.
    Stop-loss : **{stop_loss_pct:.1f}%** • Take-profit : **{take_profit_pct:.1f}%**
    (0 = désactivé) • Sizing : **{sizing_label}**.
    """
)

if st.button("▶️ Lancer le backtest", type="primary"):
    with st.spinner("Calcul du backtest..."):
        bt, metrics = backtest_strategy(
            df, initial_capital, transaction_cost / 100, threshold,
            stop_loss_pct, take_profit_pct, sizing, annual_factor, min_quality,
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
            stop_loss_pct, take_profit_pct, sizing, annual_factor, min_quality,
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
- **Les trois scores (Direction/Qualité/Risque) sont des heuristiques, pas des modèles
  validés statistiquement.** Les regrouper en trois catégories plutôt qu'un seul chiffre
  réduit le mélange d'informations différentes dans un même nombre, et chaque composant
  n'apparaît que dans un seul score pour éviter de compter la même information plusieurs
  fois (ex. ADX n'est utilisé que dans Qualité, pas aussi dans Direction). Mais les poids
  et seuils à l'intérieur de chaque score restent choisis à la main, pas calibrés sur des
  données — un test walk-forward a par exemple montré que la pente de régression apportait
  une information marginale négligeable une fois RSI/MACD/ADX déjà présents, ce qui a guidé
  son exclusion des scores plutôt qu'une intuition. Rien ne garantit un pouvoir prédictif
  réel des trois scores eux-mêmes : seul le module ML (walk-forward, AUC affiché) est
  réellement calibré sur les prix passés.
- **Le backtest n'est pas out-of-sample.** Le seuil d'entrée (et le filtre Qualité) sont
  réglables librement en observant les résultats passés, ce qui invite à les ajuster jusqu'à
  trouver ce qui a "bien marché" — un biais de surapprentissage classique. Une validation
  sérieuse nécessiterait un découpage entraînement / test (walk-forward) sur plusieurs
  sous-périodes.
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
