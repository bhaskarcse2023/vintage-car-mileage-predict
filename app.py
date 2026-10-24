"""
AutoMPG Predictor — a Streamlit app that predicts a car's fuel efficiency (MPG).

Pipeline
--------
1. Extract    : read auto-mpg.csv (or a CSV uploaded by the user)
2. Clean      : normalise column names, turn '?' into NaN, coerce types, drop duplicates
3. Split      : 80 / 20 train-test split
4. Preprocess : median imputation + StandardScaler (numeric), mode imputation + One-Hot (origin)
5. Model      : Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting
                (each tuned with 5-fold GridSearchCV inside a leakage-free sklearn Pipeline)
6. Select     : best model by cross-validated RMSE, then re-fit on all data for serving
7. Serve      : single prediction, what-if analysis, batch CSV prediction
"""
from __future__ import annotations

import html as _html
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AutoMPG Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
DATA_PATH = Path(__file__).parent / "auto-mpg.csv"

NUM_FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
CAT_FEATURES = ["origin"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "mpg"

ORIGIN_NAMES = {1: "USA", 2: "Europe", 3: "Japan"}
NAME_TO_ORIGIN = {
    "usa": 1, "us": 1, "united states": 1, "america": 1, "american": 1,
    "europe": 2, "european": 2, "eu": 2, "germany": 2, "france": 2, "italy": 2, "sweden": 2, "uk": 2,
    "japan": 3, "japanese": 3, "asia": 3, "jp": 3,
}
COLUMN_ALIASES = {
    "model year": "model_year", "year": "model_year", "modelyear": "model_year",
    "car name": "car_name", "name": "car_name", "hp": "horsepower", "cyl": "cylinders",
    "disp": "displacement", "accel": "acceleration", "wt": "weight",
}
MISSING_TOKENS = ["?", "", " ", "NA", "N/A", "na", "n/a", "nan", "NaN", "null", "None", "-"]

FEATURE_LABELS = {
    "cylinders": "Cylinders", "displacement": "Displacement (cu in)", "horsepower": "Horsepower (hp)",
    "weight": "Weight (lb)", "acceleration": "0-60 mph time (s)", "model_year": "Model year (19xx)",
    "origin": "Origin",
}

UNITS = {
    "MPG (US)": ("mpg", lambda m: m),
    "km / litre": ("km/l", lambda m: m * 0.425144),
    "L / 100 km": ("L/100km", lambda m: 235.215 / m),
}

PRESETS = {
    "🏎️ Muscle car (V8)": dict(cylinders=8, displacement=350, horsepower=165, weight=3700, acceleration=11.5, model_year=72, origin=1),
    "🚙 Family sedan": dict(cylinders=6, displacement=232, horsepower=100, weight=3200, acceleration=15.5, model_year=77, origin=1),
    "🏁 European sport": dict(cylinders=4, displacement=121, horsepower=110, weight=2800, acceleration=15.0, model_year=78, origin=2),
    "🚗 City compact": dict(cylinders=4, displacement=98, horsepower=70, weight=2100, acceleration=16.0, model_year=80, origin=3),
    "🌱 Fuel sipper": dict(cylinders=4, displacement=85, horsepower=65, weight=1900, acceleration=17.0, model_year=81, origin=3),
}
PRESET_PLACEHOLDER = "— Quick presets —"

# ──────────────────────────────────────────────────────────────────────────────
# Themes
# ──────────────────────────────────────────────────────────────────────────────
THEMES = {
    "🌙 Midnight Garage": dict(
        mode="dark", plotly="plotly_dark", bg="#0a0e16", bg2="#0f1522", card="#141b2b", card2="#1a2338",
        text="#e9eef7", muted="#8d9ab2", accent="#ff5a1f", accent2="#ffb020", border="#273250",
        glow="rgba(255,90,31,.16)", glow2="rgba(59,130,246,.12)", grid="rgba(141,154,178,.18)",
        shadow="0 10px 30px rgba(0,0,0,.45)",
        palette=["#ff5a1f", "#3b82f6", "#10b981", "#ffb020", "#a78bfa"],
    ),
    "☀️ Showroom White": dict(
        mode="light", plotly="plotly_white", bg="#f3f5fa", bg2="#ffffff", card="#ffffff", card2="#eef1f8",
        text="#101828", muted="#5b6678", accent="#e5440b", accent2="#d98800", border="#dde3ee",
        glow="rgba(229,68,11,.10)", glow2="rgba(59,130,246,.10)", grid="rgba(91,102,120,.18)",
        shadow="0 10px 28px rgba(16,24,40,.10)",
        palette=["#e5440b", "#2563eb", "#059669", "#d98800", "#7c3aed"],
    ),
    "🏁 Racing Red": dict(
        mode="dark", plotly="plotly_dark", bg="#0c0506", bg2="#140809", card="#1b0c0e", card2="#271214",
        text="#fbeeee", muted="#b99a9c", accent="#ff2d3d", accent2="#ffd23f", border="#40191d",
        glow="rgba(255,45,61,.18)", glow2="rgba(255,210,63,.08)", grid="rgba(185,154,156,.18)",
        shadow="0 10px 30px rgba(0,0,0,.55)",
        palette=["#ff2d3d", "#ffd23f", "#f97316", "#e5e7eb", "#38bdf8"],
    ),
    "🏜️ Desert Sunset": dict(
        mode="light", plotly="plotly_white", bg="#fbf3e8", bg2="#fff8ee", card="#fffaf2", card2="#f7ebd9",
        text="#2b1d12", muted="#7a6552", accent="#d9480f", accent2="#b45309", border="#ecd9bd",
        glow="rgba(217,72,15,.12)", glow2="rgba(180,83,9,.10)", grid="rgba(122,101,82,.20)",
        shadow="0 10px 28px rgba(122,80,30,.14)",
        palette=["#d9480f", "#0f766e", "#b45309", "#7c3aed", "#2563eb"],
    ),
}

STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp {
  background: radial-gradient(900px 520px at 92% -8%, var(--glow), transparent 60%) fixed,
              radial-gradient(760px 480px at -8% 108%, var(--glow2), transparent 60%) fixed,
              var(--bg);
  color: var(--text);
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { visibility: hidden; height: 0; }
[data-testid="stSidebarCollapsedControl"] svg, [data-testid="stSidebarCollapseButton"] svg,
[data-testid="stExpandSidebarButton"] svg { color: var(--text) !important; fill: var(--text) !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }

/* text */
h1, h2, h3, h4, h5, h6, p, li, label, summary,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
a { color: var(--accent) !important; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] hr { border-color: var(--border); }

/* bordered containers act as cards */
div[class*="st-key-card"] {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 20px !important; box-shadow: var(--shadow); padding: 6px 10px;
  animation: fadeUp .55s ease both;
}

/* inputs */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div,
div[data-baseweb="textarea"], input, textarea {
  background: var(--card2) !important; color: var(--text) !important; border-color: var(--border) !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg {
  color: var(--text) !important; fill: var(--text) !important;
}
div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul, ul[role="listbox"] {
  background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--border);
}
li[role="option"], li[role="option"] * { color: var(--text) !important; }
li[role="option"]:hover, li[aria-selected="true"] { background: var(--card2) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--text) !important; }
div[data-baseweb="slider"] [role="slider"] { box-shadow: 0 0 0 4px var(--glow); }

/* buttons */
.stButton > button, [data-testid="stDownloadButton"] > button {
  width: 100%; border: 0; border-radius: 12px; padding: .6rem 1rem; font-weight: 700; letter-spacing: .02em;
  background: var(--grad); box-shadow: 0 8px 22px var(--glow); transition: transform .15s ease, filter .15s ease;
}
.stButton > button *, [data-testid="stDownloadButton"] > button * { color: #fff !important; }
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover { transform: translateY(-2px); filter: brightness(1.08); }
[data-testid="stFileUploaderDropzone"] {
  background: var(--card2) !important; border: 1.5px dashed var(--accent) !important; border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button { background: transparent !important; border: 1px solid var(--accent) !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: 46px; border-radius: 12px 12px 0 0; padding: 0 18px; background: transparent; }
.stTabs [data-baseweb="tab"] p { color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] p { color: var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 3px; }
.stTabs [data-baseweb="tab-border"] { background: transparent !important; }

/* misc widgets */
[data-testid="stAlert"] { background: var(--card) !important; border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 12px; }
[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--border) !important; border-radius: 14px; }
[data-testid="stExpander"] summary:hover { color: var(--accent); }
[data-testid="stSpinner"] * { color: var(--text) !important; }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }

/* hero */
.hero {
  position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between; gap: 24px;
  padding: 28px 34px 46px; border-radius: 24px; border: 1px solid var(--border);
  background: linear-gradient(135deg, var(--card), var(--card2)); box-shadow: var(--shadow);
  animation: fadeUp .6s ease both; margin-bottom: 18px;
}
.hero h1 { font-family: 'Orbitron', sans-serif; font-weight: 800; font-size: 2.15rem; margin: 0; letter-spacing: .04em; line-height: 1.15; }
.hero h1 span { background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent !important; -webkit-text-fill-color: transparent; }
.hero p { color: var(--muted) !important; margin: .55rem 0 0; max-width: 620px; font-size: .98rem; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.chip { padding: 5px 12px; border-radius: 999px; font-size: .74rem; font-weight: 600; border: 1px solid var(--border); color: var(--muted) !important; background: var(--card); }
.road { position: absolute; left: 0; right: 0; bottom: 18px; height: 3px; opacity: .55;
  background: repeating-linear-gradient(90deg, var(--accent) 0 26px, transparent 26px 52px); animation: road 1s linear infinite; }
.streak { position: absolute; bottom: 17px; left: -140px; width: 120px; height: 5px; border-radius: 6px;
  background: linear-gradient(90deg, transparent, var(--accent2)); box-shadow: 0 0 16px var(--accent2); animation: drive 6.5s linear infinite; }
.speedo { width: 132px; height: 132px; flex: 0 0 auto; }
.needle { transform-origin: 60px 60px; animation: sweep 4s ease-in-out infinite alternate; }
@keyframes road { to { background-position: -52px 0; } }
@keyframes drive { to { left: 105%; } }
@keyframes sweep { 0% { transform: rotate(-115deg); } 55% { transform: rotate(70deg); } 100% { transform: rotate(25deg); } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 var(--glow); } 50% { box-shadow: 0 0 0 10px transparent; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
@media (max-width: 760px) { .hero { flex-direction: column; align-items: flex-start; } .speedo { display: none; } .hero h1 { font-size: 1.5rem; } }

/* section titles, KPI cards, tables */
.sec { font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 1.02rem; letter-spacing: .06em; text-transform: uppercase; margin: 6px 0 10px; display: flex; align-items: center; gap: 10px; }
.sec::before { content: ""; width: 5px; height: 20px; border-radius: 4px; background: var(--grad); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 14px; }
.kpi { position: relative; overflow: hidden; background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px 14px 20px; animation: fadeUp .5s ease both; }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--grad); }
.kpi .l { font-size: .68rem; letter-spacing: .13em; text-transform: uppercase; color: var(--muted) !important; font-weight: 700; }
.kpi .v { font-family: 'Orbitron', sans-serif; font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
.kpi .s { font-size: .76rem; color: var(--muted) !important; margin-top: 2px; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 700; font-size: .82rem; letter-spacing: .03em; }
.insight { border: 1px solid var(--border); border-left: 4px solid var(--accent2); border-radius: 14px; padding: 12px 16px; background: var(--card); margin: 10px 0; font-size: .92rem; }
.tbl-wrap { overflow: auto; max-height: 430px; border: 1px solid var(--border); border-radius: 14px; background: var(--card); }
.tbl { border-collapse: collapse; width: 100%; font-size: .84rem; }
.tbl th { position: sticky; top: 0; background: var(--card2); color: var(--muted) !important; text-transform: uppercase; font-size: .68rem; letter-spacing: .09em; padding: 10px 12px; text-align: right; white-space: nowrap; }
.tbl td { padding: 8px 12px; border-top: 1px solid var(--border); text-align: right; white-space: nowrap; }
.tbl th:first-child, .tbl td:first-child { text-align: left; }
.tbl tr.best td { background: var(--glow); font-weight: 700; }
.tbl tr:hover td { background: var(--card2); }
.flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 8px 0 14px; }
.step { position: relative; background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px; animation: fadeUp .5s ease both; }
.step b { font-family: 'Orbitron', sans-serif; color: var(--accent) !important; font-size: .8rem; letter-spacing: .08em; }
.step div { font-size: .84rem; color: var(--muted) !important; margin-top: 4px; }
.brand { display: flex; align-items: center; gap: 12px; padding: 4px 2px 10px; }
.brand .logo { width: 44px; height: 44px; border-radius: 14px; display: grid; place-items: center; font-size: 1.5rem; background: var(--grad); animation: pulse 2.6s ease-in-out infinite; }
.brand .t { font-family: 'Orbitron', sans-serif; font-weight: 800; letter-spacing: .05em; }
.brand .s { font-size: .72rem; color: var(--muted) !important; }
.foot { text-align: center; color: var(--muted) !important; font-size: .78rem; margin-top: 26px; }
"""


def inject_theme(theme_name: str) -> dict:
    T = THEMES[theme_name]
    root = (
        ":root{"
        f"--bg:{T['bg']};--bg2:{T['bg2']};--card:{T['card']};--card2:{T['card2']};--text:{T['text']};"
        f"--muted:{T['muted']};--accent:{T['accent']};--accent2:{T['accent2']};--border:{T['border']};"
        f"--glow:{T['glow']};--glow2:{T['glow2']};--shadow:{T['shadow']};"
        f"--grad:linear-gradient(135deg,{T['accent']},{T['accent2']});"
        f"color-scheme:{T['mode']};"
        "}"
    )
    first, rest = STATIC_CSS.strip().split("\n", 1)  # first line is the font @import, which must lead the sheet
    css = "\n".join(l.strip() for l in (first + "\n" + root + "\n" + rest).splitlines() if l.strip())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    return T


def html(block: str) -> None:
    """Render an HTML snippet (strips indentation/blank lines so Markdown never treats it as code)."""
    clean = "\n".join(line.strip() for line in block.strip().splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


def hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def section(title: str) -> None:
    html(f'<div class="sec">{_html.escape(title)}</div>')


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f'<div class="kpi"><div class="l">{_html.escape(l)}</div><div class="v">{_html.escape(str(v))}</div>'
        f'<div class="s">{_html.escape(s)}</div></div>'
        for l, v, s in items
    )
    html(f'<div class="kpis">{cards}</div>')


def html_table(df: pd.DataFrame, highlight_first: bool = False, decimals: int = 3, max_rows: int = 200) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else f"{v:.{decimals}f}"
        return _html.escape(str(v))

    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for i, (_, row) in enumerate(df.head(max_rows).iterrows()):
        cls = ' class="best"' if (highlight_first and i == 0) else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>")
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def style_fig(fig: go.Figure, T: dict, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=T["plotly"], height=height, margin=dict(l=10, r=10, t=46 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"], family="Inter, sans-serif"), colorway=T["palette"],
        legend=dict(bgcolor="rgba(0,0,0,0)"), title=dict(text=title, font=dict(size=15)) if title else None,
    )
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"])
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"])
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Step 1-2 · Data extraction & cleaning
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame | None:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        key = re.sub(r"[\s\-]+", " ", str(c).strip().lower())
        cols.append(COLUMN_ALIASES.get(key, key).replace(" ", "_"))
    out = df.copy()
    out.columns = cols
    return out.loc[:, ~out.columns.duplicated()]


def _origin_to_code(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    named = series.astype(str).str.strip().str.lower().map(NAME_TO_ORIGIN)
    return named.fillna(numeric).astype(float)


def coerce_features(df: pd.DataFrame, report: list[str] | None = None) -> pd.DataFrame:
    """Turn '?'-style placeholders into NaN and force numeric dtypes."""
    df = df.copy()
    for c in NUM_FEATURES:
        if c in df.columns:
            before = int(df[c].isna().sum())
            df[c] = pd.to_numeric(df[c].replace(MISSING_TOKENS, np.nan), errors="coerce").astype(float)
            fixed = int(df[c].isna().sum()) - before
            if fixed > 0 and report is not None:
                report.append(f"'{c}': {fixed} placeholder value(s) such as '?' converted to missing.")
    if "origin" in df.columns:
        df["origin"] = _origin_to_code(df["origin"])
        df.loc[~df["origin"].isin(list(ORIGIN_NAMES)), "origin"] = np.nan
    return df


def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = normalize_columns(raw)
    missing = [c for c in [TARGET] + FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing) +
            ". Expected: mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin."
        )
    report: list[str] = []
    n0 = len(df)
    df = coerce_features(df, report)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    bad = int((df[TARGET].isna() | (df[TARGET] <= 0)).sum())
    if bad:
        df = df[df[TARGET].notna() & (df[TARGET] > 0)]
        report.append(f"Dropped {bad} row(s) with a missing/invalid target (mpg).")
    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates()
        report.append(f"Dropped {dups} duplicate row(s).")
    if len(df) < 50:
        raise ValueError(f"Only {len(df)} usable rows after cleaning — at least 50 are needed to train.")
    miss = df[FEATURES].isna().sum()
    for c, n in miss[miss > 0].items():
        how = "most frequent category" if c in CAT_FEATURES else "median"
        report.append(f"'{c}': {int(n)} missing value(s) will be imputed with the {how} (inside the pipeline, no leakage).")
    report.append("Numeric features are standardised; 'origin' is one-hot encoded.")
    report.insert(0, f"Loaded {n0} rows × {raw.shape[1]} columns.")
    df = df.reset_index(drop=True)
    df["origin_name"] = df["origin"].map(ORIGIN_NAMES).fillna("Unknown")
    return df, report


def prepare_for_prediction(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Your file is missing column(s): " + ", ".join(missing))
    return coerce_features(df)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3-6 · Pipelines, training, model selection
# ──────────────────────────────────────────────────────────────────────────────
def build_pipeline(estimator) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([("num", numeric, NUM_FEATURES), ("cat", categorical, CAT_FEATURES)])
    return Pipeline([("prep", prep), ("model", estimator)])


def model_specs() -> dict:
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=SEED),
            {"model__max_depth": [4, 6, 8, 10], "model__min_samples_leaf": [2, 4, 8]},
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
            {"model__max_features": [0.6, 1.0], "model__min_samples_leaf": [1, 2]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=SEED, subsample=0.8),
            {"model__n_estimators": [200, 400], "model__learning_rate": [0.03, 0.05], "model__max_depth": [2, 3]},
        ),
    }


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame) -> dict:
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, full_models, split_models, preds, importance, params = [], {}, {}, {}, {}, {}
    for name, (est, grid) in model_specs().items():
        gs = GridSearchCV(build_pipeline(est), grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1, refit=True)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        p = best.predict(X_te)
        rows.append({
            "Model": name,
            "CV RMSE": -gs.best_score_,
            "Test R²": r2_score(y_te, p),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_te, p))),
            "Test MAE": mean_absolute_error(y_te, p),
            "Train R²": r2_score(y_tr, best.predict(X_tr)),
        })
        split_models[name] = best
        preds[name] = p
        params[name] = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        pi = permutation_importance(best, X_te, y_te, n_repeats=15, random_state=SEED,
                                    scoring="neg_root_mean_squared_error")
        importance[name] = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
        full_models[name] = clone(best).fit(X, y)  # final model sees all data

    results = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)
    num = df[NUM_FEATURES]
    defaults = {f"f_{c}": float(round(num[c].median(), 1)) for c in NUM_FEATURES}
    bounds = {c: (float(np.floor(num[c].min())), float(np.ceil(num[c].max()))) for c in NUM_FEATURES}
    for c in ("cylinders", "displacement", "horsepower", "weight", "model_year"):
        defaults[f"f_{c}"] = int(round(defaults[f"f_{c}"]))
        bounds[c] = (int(bounds[c][0]), int(bounds[c][1]))
    cyl_options = sorted(int(v) for v in df["cylinders"].dropna().unique())
    origin_options = sorted(float(v) for v in df["origin"].dropna().unique())
    defaults["f_cylinders"] = min(cyl_options, key=lambda v: abs(v - defaults["f_cylinders"]))
    defaults["f_origin"] = float(df["origin"].mode().iloc[0])
    meta = {
        "sig": f"{len(df)}_{df[TARGET].sum():.3f}", "bounds": bounds, "defaults": defaults,
        "cyl_options": cyl_options, "origin_options": origin_options,
    }
    return dict(
        results=results, models=full_models, split_models=split_models, best_name=results.loc[0, "Model"],
        y_test=y_te.to_numpy(), preds=preds, importance=importance, params=params, meta=meta,
        n_train=len(X_tr), n_test=len(X_te),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_unit(mpg: float, unit: str) -> float:
    return float(UNITS[unit][1](max(float(mpg), 1.0)))


def predict_one(model, values: dict) -> float:
    row = pd.DataFrame([{k: float(values[k]) for k in FEATURES}])[FEATURES]
    return float(np.clip(model.predict(row)[0], 1.0, 80.0))


def efficiency_class(mpg: float) -> tuple[str, str]:
    if mpg < 15:
        return "⛽ Gas guzzler", "#ef4444"
    if mpg < 22:
        return "🚙 Thirsty", "#f59e0b"
    if mpg < 30:
        return "🚗 Balanced", "#84cc16"
    if mpg < 38:
        return "🌿 Efficient", "#10b981"
    return "🌱 Eco champion", "#14b8a6"


def make_gauge(mpg: float, unit: str, T: dict) -> go.Figure:
    short = UNITS[unit][0]
    lo_mpg, hi_mpg = 8.0, 48.0
    axis = sorted((to_unit(lo_mpg, unit), to_unit(hi_mpg, unit)))
    bands = [(8, 15, "#ef4444"), (15, 22, "#f59e0b"), (22, 30, "#84cc16"), (30, 48, "#10b981")]
    steps = []
    for lo, hi, col in bands:
        a, b = sorted((to_unit(lo, unit), to_unit(hi, unit)))
        steps.append({"range": [a, b], "color": hex_alpha(col, 0.5)})
    val = float(np.clip(to_unit(mpg, unit), axis[0], axis[1]))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={"suffix": f" {short}", "valueformat": ".1f", "font": {"size": 40, "color": T["text"], "family": "Orbitron, sans-serif"}},
        gauge={
            "axis": {"range": axis, "tickcolor": T["muted"], "tickfont": {"color": T["muted"]}},
            "bar": {"color": T["accent"], "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0, "steps": steps,
        },
    ))
    fig.update_layout(height=290, margin=dict(l=24, r=24, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=T["text"]))
    return fig


def speedometer_svg() -> str:
    ticks = []
    for i in range(0, 28):
        ang = np.deg2rad(135 + i * 270 / 27)
        r1, r2 = (44, 52) if i % 3 == 0 else (48, 52)
        x1, y1 = 60 + r1 * np.cos(ang), 60 + r1 * np.sin(ang)
        x2, y2 = 60 + r2 * np.cos(ang), 60 + r2 * np.sin(ang)
        ticks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:var(--muted);stroke-width:{2 if i % 3 == 0 else 1}"/>')
    return (
        '<svg class="speedo" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="60" cy="60" r="57" style="fill:none;stroke:var(--border);stroke-width:2"/>'
        + "".join(ticks) +
        '<g class="needle"><line x1="60" y1="60" x2="60" y2="22" style="stroke:var(--accent);stroke-width:3;stroke-linecap:round"/></g>'
        '<circle cx="60" cy="60" r="6" style="fill:var(--accent2)"/>'
        '</svg>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# UI pieces
# ──────────────────────────────────────────────────────────────────────────────
def hero(n_rows: int, best: str, r2: float) -> None:
    html(f"""
    <div class="hero">
      <div>
        <h1>AUTO<span>MPG</span> PREDICTOR</h1>
        <p>Estimate a car's fuel efficiency from its engine, weight and age — powered by a tuned machine-learning pipeline.</p>
        <div class="chips">
          <span class="chip">📦 {n_rows} cars</span>
          <span class="chip">🏆 Best model: {_html.escape(best)}</span>
          <span class="chip">🎯 Test R² {r2:.2f}</span>
          <span class="chip">⚙️ 5 algorithms compared</span>
        </div>
      </div>
      {speedometer_svg()}
      <div class="road"></div><div class="streak"></div>
    </div>
    """)


def apply_preset() -> None:
    name = st.session_state.get("preset")
    meta = st.session_state["_meta"]
    if name in PRESETS:
        for feat, val in PRESETS[name].items():
            if feat == "cylinders":
                val = min(meta["cyl_options"], key=lambda v: abs(v - val))
            elif feat == "origin":
                val = min(meta["origin_options"], key=lambda v: abs(v - val))
            else:
                lo, hi = meta["bounds"][feat]
                val = type(lo)(min(max(val, lo), hi))
            st.session_state[f"f_{feat}"] = val
    st.session_state["preset"] = PRESET_PLACEHOLDER


def input_panel(meta: dict) -> dict:
    b = meta["bounds"]
    st.selectbox("Quick presets", [PRESET_PLACEHOLDER] + list(PRESETS), key="preset", on_change=apply_preset,
                 label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox(FEATURE_LABELS["cylinders"], meta["cyl_options"], key="f_cylinders")
    with c2:
        st.selectbox(FEATURE_LABELS["origin"], meta["origin_options"], key="f_origin",
                     format_func=lambda c: ORIGIN_NAMES.get(int(c), str(c)))
    st.slider(FEATURE_LABELS["displacement"], b["displacement"][0], b["displacement"][1], key="f_displacement")
    st.slider(FEATURE_LABELS["horsepower"], b["horsepower"][0], b["horsepower"][1], key="f_horsepower")
    st.slider(FEATURE_LABELS["weight"], b["weight"][0], b["weight"][1], step=10, key="f_weight")
    c3, c4 = st.columns(2)
    with c3:
        st.slider(FEATURE_LABELS["acceleration"], b["acceleration"][0], b["acceleration"][1], step=0.1, key="f_acceleration")
    with c4:
        st.slider(FEATURE_LABELS["model_year"], b["model_year"][0], b["model_year"][1], key="f_model_year")
    return {c: st.session_state[f"f_{c}"] for c in FEATURES}


def tab_predict(art: dict, df: pd.DataFrame, model_name: str, unit: str, T: dict) -> None:
    meta = art["meta"]
    model = art["models"][model_name]
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(border=True, key="card_inputs"):
            section("Vehicle specs")
            values = input_panel(meta)
    mpg = predict_one(model, values)
    label, colour = efficiency_class(mpg)
    pct = float((df[TARGET] < mpg).mean() * 100)
    err = float(art["results"].set_index("Model").loc[model_name, "Test RMSE"])
    l_per_100 = to_unit(mpg, "L / 100 km")
    co2 = l_per_100 / 100 * 2310  # g CO₂ per km for petrol (≈2.31 kg per litre)

    with right:
        with st.container(border=True, key="card_result"):
            section("Predicted mileage")
            st.plotly_chart(make_gauge(mpg, unit, T), theme=None)
            html(f'<div style="text-align:center;margin-top:-6px"><span class="badge" style="background:{hex_alpha(colour, .16)};'
                 f'color:{colour};border:1px solid {hex_alpha(colour, .5)}">{label}</span></div>')
        kpi_row([
            ("MPG (US)", f"{mpg:.1f}", f"± {err:.1f} typical error"),
            ("Kilometres / litre", f"{to_unit(mpg, 'km / litre'):.1f}", "km per litre"),
            ("Fuel / 100 km", f"{l_per_100:.1f} L", "litres"),
            ("CO₂ (petrol)", f"{co2:.0f} g/km", "approximate"),
            ("Rank", f"Top {max(1, 100 - pct):.0f}%", f"beats {pct:.0f}% of dataset"),
        ])

    # Insight: effect of a 10 % lighter car
    w_lo = meta["bounds"]["weight"][0]
    lighter = dict(values, weight=max(w_lo, values["weight"] * 0.9))
    gain = predict_one(model, lighter) - mpg
    if gain > 0.15:
        tip = (f"Trimming ~{values['weight'] - lighter['weight']:.0f} lb (10 % of weight) is predicted to improve mileage "
               f"by <b>{gain:.1f} MPG</b> ({gain / mpg * 100:.0f} %).")
    else:
        tip = "In this configuration, a 10 % weight reduction barely changes the predicted mileage."
    html(f'<div class="insight">💡 <b>Insight:</b> {tip}</div>')

    c_a, c_b = st.columns(2, gap="large")
    with c_a:
        with st.container(border=True, key="card_compare"):
            section("All models, same car")
            names = list(art["models"])
            vals = [to_unit(predict_one(art["models"][n], values), unit) for n in names]
            fig = go.Figure(go.Bar(
                x=vals, y=names, orientation="h",
                marker_color=[T["accent"] if n == model_name else hex_alpha(T["muted"], .55) for n in names],
                text=[f"{v:.1f}" for v in vals], textposition="outside", cliponaxis=False,
            ))
            style_fig(fig, T, 300)
            fig.update_xaxes(title=UNITS[unit][0])
            st.plotly_chart(fig, theme=None)
    with c_b:
        with st.container(border=True, key="card_whatif"):
            section("What-if explorer")
            feat = st.selectbox("Vary", ["weight", "horsepower", "displacement", "model_year", "acceleration"],
                                format_func=lambda f: FEATURE_LABELS[f], key="whatif_feat")
            lo, hi = meta["bounds"][feat]
            grid = np.linspace(lo, hi, 40)
            sweep = pd.DataFrame([dict(values, **{feat: g}) for g in grid])[FEATURES].astype(float)
            ys = [to_unit(v, unit) for v in np.clip(model.predict(sweep), 1.0, 80.0)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid, y=ys, mode="lines", line=dict(color=T["accent"], width=3), name="Prediction"))
            fig.add_trace(go.Scatter(x=[values[feat]], y=[to_unit(mpg, unit)], mode="markers",
                                     marker=dict(size=14, color=T["accent2"], line=dict(color=T["text"], width=2)), name="Your car"))
            style_fig(fig, T, 300)
            fig.update_xaxes(title=FEATURE_LABELS[feat])
            fig.update_yaxes(title=UNITS[unit][0])
            st.plotly_chart(fig, theme=None)


def tab_batch(art: dict, model_name: str, unit: str, T: dict) -> None:
    model = art["models"][model_name]
    with st.container(border=True, key="card_batch"):
        section("Batch prediction from CSV")
        st.caption("Upload a CSV with columns: cylinders, displacement, horsepower, weight, acceleration, model year, origin "
                   "(origin = 1/2/3 or USA/Europe/Japan). Missing values and '?' are handled automatically.")
        template = pd.DataFrame({
            "cylinders": [4, 8, 6], "displacement": [105, 350, 232], "horsepower": [70, 165, "?"],
            "weight": [2200, 3700, 3200], "acceleration": [16.5, 11.5, 15.5], "model year": [80, 72, 77],
            "origin": ["Japan", "USA", "USA"],
        })
        c1, c2 = st.columns([1, 2])
        with c1:
            st.download_button("⬇️ Download template", template.to_csv(index=False).encode(), "mpg_template.csv", "text/csv")
        up = st.file_uploader("Upload CSV", type=["csv"], key="batch_file", label_visibility="collapsed")
    if up is None:
        return
    try:
        raw = pd.read_csv(up)
        feats = prepare_for_prediction(raw)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file: {e}")
        return
    miss_cells = int(feats[FEATURES].isna().sum().sum())
    preds = np.clip(model.predict(feats[FEATURES]), 1.0, 80.0)
    out = feats.copy()
    out["predicted_mpg"] = preds.round(2)
    out["predicted_km_per_l"] = (preds * 0.425144).round(2)
    out["predicted_l_per_100km"] = (235.215 / preds).round(2)
    kpi_row([
        ("Cars scored", f"{len(out)}", f"model: {model_name}"),
        ("Average", f"{to_unit(preds.mean(), unit):.1f}", UNITS[unit][0]),
        ("Best", f"{to_unit(preds.max(), unit):.1f}", UNITS[unit][0]),
        ("Auto-filled cells", f"{miss_cells}", "imputed by pipeline"),
    ])
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        section("Results")
        st.markdown(html_table(out.drop(columns=["origin_name"], errors="ignore"), decimals=2), unsafe_allow_html=True)
    with c2:
        section("Distribution")
        fig = px.histogram(x=[to_unit(p, unit) for p in preds], nbins=20, color_discrete_sequence=[T["accent"]])
        style_fig(fig, T, 330)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title=UNITS[unit][0])
        st.plotly_chart(fig, theme=None)
    st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(), "mpg_predictions.csv", "text/csv")


def tab_explore(df: pd.DataFrame, report: list[str], T: dict) -> None:
    kpi_row([
        ("Cars", f"{len(df)}", "rows after cleaning"),
        ("Avg MPG", f"{df[TARGET].mean():.1f}", f"range {df[TARGET].min():.0f}–{df[TARGET].max():.0f}"),
        ("Heaviest", f"{df['weight'].max():.0f} lb", "max weight"),
        ("Years", f"19{int(df['model_year'].min())}–{int(df['model_year'].max())}", "model years"),
        ("Missing cells", f"{int(df[FEATURES].isna().sum().sum())}", "imputed in pipeline"),
    ])
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        with st.container(border=True, key="card_preview"):
            section("Data preview")
            show = df.drop(columns=["origin_name"], errors="ignore").head(60)
            st.markdown(html_table(show, decimals=1), unsafe_allow_html=True)
    with c2:
        with st.container(border=True, key="card_quality"):
            section("Data quality log")
            for line in report:
                st.markdown(f"✅ {line}")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True, key="card_hist"):
            section("MPG distribution")
            fig = px.histogram(df, x=TARGET, nbins=30, color="origin_name", barmode="overlay", opacity=.75)
            style_fig(fig, T, 330).update_layout(legend_title_text="Origin")
            st.plotly_chart(fig, theme=None)
    with b:
        with st.container(border=True, key="card_corr"):
            section("Correlation heatmap")
            corr = df[NUM_FEATURES + [TARGET]].corr()
            fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                            color_continuous_scale=[[0, "#3b82f6"], [.5, hex_alpha(T["muted"], .15)], [1, T["accent"]]])
            style_fig(fig, T, 330).update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, theme=None)
    with st.container(border=True, key="card_scatter"):
        section("Feature vs MPG")
        feat = st.selectbox("X axis", NUM_FEATURES, index=3, format_func=lambda f: FEATURE_LABELS[f], key="scatter_feat")
        fig = px.scatter(df, x=feat, y=TARGET, color="origin_name", opacity=.8,
                         hover_data=[c for c in ["car_name"] if c in df.columns])
        fig.update_traces(marker=dict(size=9, line=dict(width=0)))
        style_fig(fig, T, 380).update_layout(legend_title_text="Origin")
        fig.update_xaxes(title=FEATURE_LABELS[feat])
        st.plotly_chart(fig, theme=None)
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True, key="card_box"):
            section("MPG by origin")
            fig = px.box(df, x="origin_name", y=TARGET, color="origin_name", points="outliers")
            style_fig(fig, T, 330).update_layout(showlegend=False)
            fig.update_xaxes(title=None)
            st.plotly_chart(fig, theme=None)
    with b:
        with st.container(border=True, key="card_trend"):
            section("Fuel economy over the years")
            tr = df.groupby(["model_year", "origin_name"], as_index=False)[TARGET].mean()
            fig = px.line(tr, x="model_year", y=TARGET, color="origin_name", markers=True)
            style_fig(fig, T, 330).update_layout(legend_title_text="Origin")
            fig.update_xaxes(title="Model year (19xx)")
            st.plotly_chart(fig, theme=None)


def tab_lab(art: dict, T: dict) -> None:
    res = art["results"]
    best = art["best_name"]
    section("Pipeline")
    steps = [
        ("1 · EXTRACT", "Read CSV (bundled or uploaded)"),
        ("2 · CLEAN", "'?' → missing, type coercion, drop duplicates"),
        ("3 · SPLIT", f"{art['n_train']} train / {art['n_test']} test"),
        ("4 · PREPROCESS", "Median impute + scale · mode impute + one-hot"),
        ("5 · TUNE", "5-fold GridSearchCV × 5 algorithms"),
        ("6 · SELECT", f"Lowest CV RMSE → {best}"),
        ("7 · SERVE", "Refit on all data, predict live"),
    ]
    html('<div class="flow">' + "".join(f'<div class="step"><b>{a}</b><div>{_html.escape(b)}</div></div>' for a, b in steps) + "</div>")

    with st.container(border=True, key="card_leader"):
        section("Leaderboard")
        st.markdown(html_table(res, highlight_first=True, decimals=3), unsafe_allow_html=True)
        st.caption("Ranked by cross-validated RMSE on the training split (lower is better). "
                   "Test metrics come from the untouched 20 % hold-out.")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True, key="card_r2"):
            section("Test R² by model")
            fig = go.Figure(go.Bar(
                x=res["Model"], y=res["Test R²"],
                marker_color=[T["accent"] if m == best else hex_alpha(T["muted"], .55) for m in res["Model"]],
                text=[f"{v:.3f}" for v in res["Test R²"]], textposition="outside", cliponaxis=False,
            ))
            style_fig(fig, T, 340)
            fig.update_yaxes(range=[max(0, res["Test R²"].min() - 0.1), min(1.0, res["Test R²"].max() + 0.06)])
            st.plotly_chart(fig, theme=None)
    with b:
        with st.container(border=True, key="card_rmse"):
            section("Error comparison (lower = better)")
            long = res.melt(id_vars="Model", value_vars=["CV RMSE", "Test RMSE", "Test MAE"], var_name="Metric", value_name="Value")
            fig = px.bar(long, x="Model", y="Value", color="Metric", barmode="group")
            style_fig(fig, T, 340)
            st.plotly_chart(fig, theme=None)

    with st.container(border=True, key="card_inspect"):
        section("Model inspector")
        pick = st.selectbox("Inspect model", list(art["models"]), index=list(art["models"]).index(best), key="inspect_model")
        c1, c2, c3 = st.columns(3, gap="large")
        y_true, y_pred = art["y_test"], art["preds"][pick]
        with c1:
            fig = go.Figure()
            lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=T["muted"], dash="dash"), name="Perfect"))
            fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(color=T["accent"], size=8, opacity=.8), name="Cars"))
            style_fig(fig, T, 320, "Actual vs predicted").update_layout(showlegend=False)
            fig.update_xaxes(title="Actual MPG")
            fig.update_yaxes(title="Predicted MPG")
            st.plotly_chart(fig, theme=None)
        with c2:
            fig = px.histogram(x=y_true - y_pred, nbins=20, color_discrete_sequence=[T["accent2"]])
            style_fig(fig, T, 320, "Residuals (actual − predicted)").update_layout(showlegend=False)
            st.plotly_chart(fig, theme=None)
        with c3:
            imp = art["importance"][pick]
            fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_LABELS[i].split(" (")[0] for i in imp.index],
                                   orientation="h", marker_color=T["accent"]))
            style_fig(fig, T, 320, "Permutation importance")
            fig.update_xaxes(title="RMSE increase when shuffled")
            st.plotly_chart(fig, theme=None)
        with st.expander("Best hyper-parameters found"):
            st.json(art["params"][pick] or {"info": "No hyper-parameters tuned for this model."})


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Sidebar: theme first so the CSS is injected before anything renders ──
    with st.sidebar:
        theme_name = st.selectbox("🎨 Theme", list(THEMES), key="theme")
    T = inject_theme(theme_name)

    with st.sidebar:
        html('<div class="brand"><div class="logo">🏎️</div><div><div class="t">AUTOMPG</div>'
             '<div class="s">Mileage intelligence</div></div></div>')
        unit = st.selectbox("📏 Display unit", list(UNITS), key="unit")
        st.markdown("---")
        up = st.file_uploader("📂 Train on your own CSV (optional)", type=["csv"], key="train_file",
                              help="Needs: mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin")

    # ── Extract + clean ──
    raw, source = None, "Bundled Auto-MPG dataset"
    if up is not None:
        try:
            raw, source = pd.read_csv(up), f"Uploaded: {up.name}"
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Could not read your CSV: {e}")
    if raw is None:
        raw = load_default_data()
    if raw is None:
        st.error("`auto-mpg.csv` was not found next to app.py. Add it to the repository or upload a CSV in the sidebar.")
        st.stop()
    try:
        df, report = clean_dataframe(raw)
    except ValueError as e:
        st.sidebar.error(str(e))
        fallback = load_default_data()
        if fallback is None:
            st.stop()
        df, report = clean_dataframe(fallback)
        source = "Bundled Auto-MPG dataset (your file was rejected)"

    # ── Train (cached across reruns and users) ──
    with st.spinner("Training & tuning 5 models… this only happens once ⚙️"):
        art = train_models(df)
    meta = art["meta"]
    st.session_state["_meta"] = meta
    if st.session_state.get("_sig") != meta["sig"]:
        st.session_state["_sig"] = meta["sig"]
        st.session_state.update(meta["defaults"])
        st.session_state["preset"] = PRESET_PLACEHOLDER

    best = art["best_name"]
    names = list(art["models"])
    with st.sidebar:
        choice = st.selectbox("🧠 Prediction model", [f"⭐ Auto · best ({best})"] + names, key="model_choice")
        model_name = best if choice.startswith("⭐") else choice
        st.markdown("---")
        st.caption(f"**Data:** {source}")
        st.caption(f"**Rows:** {len(df)} · **Train/Test:** {art['n_train']}/{art['n_test']}")
        r = art["results"].set_index("Model").loc[model_name]
        st.caption(f"**{model_name}** → R² {r['Test R²']:.3f} · RMSE {r['Test RMSE']:.2f} MPG")

    hero(len(df), best, float(art["results"].loc[0, "Test R²"]))
    t1, t2, t3, t4 = st.tabs(["🎯 Predict", "📁 Batch", "📊 Explore data", "🧪 Model lab"])
    with t1:
        tab_predict(art, df, model_name, unit, T)
    with t2:
        tab_batch(art, model_name, unit, T)
    with t3:
        tab_explore(df, report, T)
    with t4:
        tab_lab(art, T)
    html('<div class="foot">Built with scikit-learn · Streamlit · Plotly — predictions are estimates based on 1970-82 vehicles.</div>')


main()
