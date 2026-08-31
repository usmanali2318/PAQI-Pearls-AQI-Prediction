import os, base64, joblib, json, tempfile
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap
import hopsworks
from training_pipeline import aggregate_daily, HORIZONS

st.set_page_config(page_title="Pakistan AQI Forecast", layout="wide")

DARK = {"accent": "#07F3F4", "mid": "#10BDC2", "mid2": "#14969C", "border": "#187A83", "card": "#17535D", "text": "#E8FBFC", "shade": "#187A83"}
LIGHT = {"accent": "#284539", "mid": "#526A60", "mid2": "#6E8279", "border": "#9FB2A8", "card": "#ECF0EC", "text": "#284539", "shade": "#BCCCC3"}

# must match secondaryBackgroundColor/textColor in .streamlit/config.toml
DROPDOWN_BG = "#7A8B7F"
DROPDOWN_TEXT = "#E8FBFC"

# fixed so the same city gets the same color on every chart
CITY_COLORS = {
    "islamabad": "#4C72B0",
    "karachi": "#C44E52",
    "lahore": "#55A868",
    "peshawar": "#DD8452",
    "quetta": "#8172B2",
}

CATEGORIES = [(50, "Good", "#2ECC71"), (100, "Moderate", "#F1C40F"), (150, "Unhealthy for Sensitive Groups", "#E67E22"),
              (200, "Unhealthy", "#E74C3C"), (300, "Very Unhealthy", "#8E44AD"), (10_000, "Hazardous", "#7B241C")]

FEATURE_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "co": "Carbon Monoxide", "no2": "Nitrogen Dioxide", "so2": "Sulfur Dioxide",
    "aqi": "Recent AQI Trend", "temp": "Temperature", "humidity": "Humidity", "pressure": "Pressure",
    "wind_speed": "Wind Speed", "precip": "Precipitation", "wind_dir_sin": "Wind Direction", "wind_dir_cos": "Wind Direction",
    "aqi_hourly_std": "AQI Volatility", "aqi_hourly_max": "Day's Peak AQI", "aqi_change_rate": "AQI Change Rate",
    "dry_spell_days": "Dry Spell Length", "month": "Month", "day_of_week": "Day of Week",
    "lag_1d": "AQI 1 Day Ago", "lag_2d": "AQI 2 Days Ago", "lag_3d": "AQI 3 Days Ago", "lag_7d": "AQI 1 Week Ago",
    "rolling_mean_3d": "3-Day Avg AQI", "rolling_mean_7d": "7-Day Avg AQI", "rolling_mean_14d": "14-Day Avg AQI",
    "rolling_std_7d": "7-Day AQI Volatility",
    "is_winter_smog": "Winter Smog Season", "is_dust_season": "Dust Season", "is_monsoon": "Monsoon Season",
    "month_sin": "Seasonal Cycle", "month_cos": "Seasonal Cycle",
    "city_aqi_median": "City Median AQI", "city_aqi_max": "City Peak AQI", "city_aqi_min": "City Best AQI",
    "city_aqi_spread": "City AQI Spread",
    "district_1": "District 1 AQI", "district_2": "District 2 AQI", "district_3": "District 3 AQI",
    "district_4": "District 4 AQI", "district_5": "District 5 AQI",
}

def friendly_feature_name(col):
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    if col.startswith("city_"):
        return f"{col.removeprefix('city_').title()} (City)"
    return col.replace("_", " ").title()

def aqi_band(value):
    return next((band for band in CATEGORIES if value <= band[0]), CATEGORIES[-1])

def aqi_category(value):
    return aqi_band(value)[1:]

def aqi_band_ceiling(value):
    # anchors the y-axis to the real AQI scale so small dips don't look like cliffs
    return aqi_band(value)[0]

def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def style_fig(fig, palette):
    # plotly text is SVG, so the page's CSS never reaches it - set colors here instead
    grid = hex_to_rgba(palette["text"], 0.12)
    fig.update_layout(
        title=dict(text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"]),
        title_font=dict(color=palette["text"]),
        legend=dict(font=dict(color=palette["text"])),
        hoverlabel=dict(
            bgcolor=hex_to_rgba(palette["card"], 0.95),
            bordercolor=palette["accent"],
            font=dict(color=palette["text"], family="Arial, sans-serif", size=13),
            align="left",
        ),
    )
    fig.update_xaxes(tickfont=dict(color=palette["text"]), title_font=dict(color=palette["text"]),
                      gridcolor=grid, zerolinecolor=grid, linecolor=grid)
    fig.update_yaxes(tickfont=dict(color=palette["text"]), title_font=dict(color=palette["text"]),
                      gridcolor=grid, zerolinecolor=grid, linecolor=grid)
    return fig

def inject_theme(city_key, dark_mode):
    p = DARK if dark_mode else LIGHT
    variant = "dark" if dark_mode else "light"
    with open(f"assets/{city_key}{variant}.webp", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{b64}");
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    header[data-testid="stHeader"] {{ background: transparent; display: none !important; }}
    #MainMenu {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none !important; }}
    [data-testid="stDecoration"] {{ display: none !important; }}
    [data-testid="stStatusWidget"] {{ display: none !important; }}
    .glass-card {{
        background: {p['card']}88; border: 1px solid {p['border']}88; border-radius: 14px;
        padding: 18px; backdrop-filter: blur(12px); margin-bottom: 14px;
    }}
    .plotly-chart-card {{
        background: {p['card']}88; border: 1px solid {p['border']}88; border-radius: 14px;
        padding: 14px; backdrop-filter: blur(12px); margin-bottom: 14px;
    }}
    [data-testid="stMetric"] {{
        background: {p['card']}88; border: 1px solid {p['border']}88; border-radius: 14px;
        padding: 14px; backdrop-filter: blur(12px);
    }}
    [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {{
        background: {p['card']}88; border: 1px solid {p['border']}88;
        border-radius: 14px; padding: 10px; backdrop-filter: blur(12px);
    }}
    h1, h2, h3, h4, p, label, span:not(.badge) {{ color: {p['text']} !important; }}
    .badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85em; }}

    /* Pull everything up so there's no dead space under the ribbon */
    .block-container {{ padding-top: 5.25rem !important; padding-bottom: 2rem !important; }}

    /* Ribbon banner: fixed + width:100% (not 100vw, which overshoots by the
       scrollbar's width and gets clipped) so it's flush with all 3 edges */
    .st-key-ribbon {{
        position: fixed; top: 0; left: 0; right: 0; z-index: 999;
        background: {p['card']}66; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid {p['border']}66;
        padding: 18px calc(2rem + 3vw);
        display: flex; align-items: center;
    }}
    .st-key-ribbon > div {{ width: 100%; }}
    .st-key-ribbon [data-testid="stHorizontalBlock"] {{
        display: flex; align-items: center; justify-content: space-between; flex-wrap: nowrap; gap: 1rem; width: 100%;
    }}
    .st-key-ribbon [data-testid="stColumn"] {{ width: auto !important; flex: 0 0 auto !important; min-width: 0 !important; }}
    .ribbon-clock {{ white-space: nowrap; opacity: 0.75; font-size: 0.9rem; line-height: 1; color: {p['text']} !important; }}
    .ribbon-title {{
        display: flex; align-items: center; line-height: 1; margin: 0;
        font-size: 1.5rem; font-weight: 800; color: {p['text']} !important; letter-spacing: 0.3px;
    }}
    .ribbon-title span {{ font-weight: 400; opacity: 0.85; font-size: 1rem; line-height: 1; margin-left: 10px; color: {p['text']} !important; }}
    .st-key-ribbon [data-testid="stMarkdownContainer"] {{ margin: 0; padding: 0; }}

    /* Theme toggle, now inline with the city dropdown row, aligned to its control */
    .st-key-city_row [data-testid="stButton"] {{ display: flex; justify-content: flex-end; }}
    .st-key-city_row [data-testid="stButton"] button {{
        background: transparent !important; border: none !important; box-shadow: none !important;
        color: {p['text']} !important; font-weight: 600; padding: 0 0 8px 0; white-space: nowrap; opacity: 0.85;
    }}
    .st-key-city_row [data-testid="stButton"] button:hover {{ text-decoration: underline; opacity: 1; }}

    .app-blurb {{ opacity: 0.85; margin: 0 0 10px 0; font-size: 0.95rem; }}

    html {{ scroll-behavior: smooth; }}
    #eda-anchor {{ scroll-margin-top: 90px; display: block; }}
    [id^="nav-"] {{ scroll-margin-top: 90px; }}

    /* Side nav: labels always visible (dimmed), not hover-only, to match the
       reference design. True "current section" highlighting needs scroll
       position, which is JS-only - skipped per request to keep this CSS-only. */
    .side-nav {{
        position: fixed; right: 18px; top: 50%; transform: translateY(-50%); z-index: 900;
        display: flex; flex-direction: column; align-items: flex-end; gap: 0;
    }}
    .side-nav a {{
        display: flex; flex-direction: row-reverse; align-items: center; gap: 10px; text-decoration: none;
        padding: 9px 0; position: relative;
    }}
    .side-nav a::before {{
        content: ""; width: 9px; height: 9px; border-radius: 50%;
        background: {p['card']}; border: 2px solid {p['accent']}; flex-shrink: 0;
        transition: background 0.15s ease, border-color 0.15s ease;
    }}
    .side-nav a:hover::before {{ background: {p['accent']}; }}
    .side-nav a span {{
        font-size: 0.75rem; color: {p['text']} !important; opacity: 0.55; white-space: nowrap;
        transition: opacity 0.15s ease, font-weight 0.15s ease;
    }}
    .side-nav a:hover span {{ opacity: 1; font-weight: 600; }}
    .side-nav::before {{
        content: ""; position: absolute; right: 4px; top: 4px; bottom: 4px; width: 1px;
        background: {p['shade']}; z-index: -1;
    }}
    @media (max-width: 900px) {{ .side-nav {{ display: none; }} }}
    .jump-link {{ color: {p['accent']} !important; font-size: 0.85rem; text-decoration: none; }}
    .jump-link:hover {{ text-decoration: underline; }}

    /* Theme the city dropdown (closed control + open menu) and shrink it a bit.
       BaseWeb nests several divs inside [data-baseweb="select"] and the exact
       depth of the one actually carrying the background varies, so reset
       everything to transparent first, then paint every plausible depth the
       same color (harmless if more than one matches — they're identical). */
    div[data-baseweb="select"] * {{ background-color: transparent !important; box-shadow: none !important; }}
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div,
    div[data-baseweb="select"] > div > div > div {{
        background-color: {p['card']} !important;
    }}
    div[data-baseweb="select"] {{
        border: 1px solid {p['shade']} !important; border-radius: 10px !important;
        min-height: 36px !important; overflow: hidden;
    }}
    div[data-baseweb="select"] * {{ color: {p['text']} !important; font-size: 0.88rem !important; }}
    div[data-baseweb="select"] svg {{ fill: {p['text']} !important; }}
    ul[data-baseweb="menu"] {{ background: {DROPDOWN_BG} !important; border: 1px solid {p['shade']} !important; }}
    ul[data-baseweb="menu"] li {{ color: {DROPDOWN_TEXT} !important; background: transparent !important; font-size: 0.88rem !important; }}

    /* Streamlit's default primaryColor (red) drives input focus rings — override
       with the theme's own accent. (Radio dot color comes from .streamlit/config.toml's
       primaryColor instead — hand-guessing BaseWeb's internal DOM for that broke the
       label text, since a broad `> div` selector painted over the text wrapper too.) */
    div[data-baseweb="select"]:focus-within {{
        border-color: {p['accent']} !important; box-shadow: 0 0 0 1px {p['accent']} !important;
    }}
    li[role="option"]:hover, li[aria-selected="true"] {{ background: {p['accent']}33 !important; }}

    /* Merged current-AQI card (gauge + category badge as one unit) */
    .st-key-current_aqi_card {{
        background: {p['card']}88; border: 1px solid {p['border']}88; border-radius: 14px;
        padding: 18px; backdrop-filter: blur(12px); margin-bottom: 14px;
        display: flex; align-items: center;
    }}
    .st-key-current_aqi_card > div {{ width: 100%; }}
    .st-key-current_aqi_card [data-testid="stHorizontalBlock"] {{
        display: flex; align-items: center; gap: 2rem; width: 100%;
    }}
    .st-key-current_aqi_card [data-testid="stColumn"] {{ flex: 1 1 0; }}
    .st-key-current_aqi_card [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {{
        background: transparent !important; border: none !important; padding: 0 !important;
        border-radius: 0 !important; backdrop-filter: none !important;
    }}
    .aqi-category-block {{ display: flex; flex-direction: column; align-items: flex-start; }}
    </style>
    """, unsafe_allow_html=True)
    return p

@st.cache_resource(ttl=3600)
def load_model():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"])
    mr = project.get_model_registry()
    m = mr.get_model("multi_city_aqi_daily_model", version=1)
    path = m.download(local_path=tempfile.mkdtemp())
    bundle = joblib.load(f"{path}/model.pkl")
    try:
        holdout_preds = pd.read_csv(f"{path}/holdout_predictions.csv")
    except FileNotFoundError:
        holdout_preds = None  # older model bundle, predates this file being saved
    try:
        with open(f"{path}/eval_scores.json") as f:
            eval_scores = json.load(f)
    except FileNotFoundError:
        eval_scores = None  # older model bundle, predates this file being saved
    return bundle["point_model"], bundle["quantile_models"], project, holdout_preds, eval_scores

@st.cache_data(ttl=600)
def load_recent_data(_project):
    fg = _project.get_feature_store().get_feature_group("multi_city_aqi_features", version=1)
    return fg.read()

def build_live_features(daily):
    g = daily.groupby("city")
    for lag in [1, 2, 3, 7]:
        daily[f"lag_{lag}d"] = g["aqi"].shift(lag)
    daily["rolling_mean_3d"] = g["aqi"].transform(lambda s: s.rolling(3).mean())
    daily["rolling_mean_7d"] = g["aqi"].transform(lambda s: s.rolling(7).mean())
    daily["rolling_std_7d"] = g["aqi"].transform(lambda s: s.rolling(7).std())
    daily["rolling_mean_14d"] = g["aqi"].transform(lambda s: s.rolling(14).mean())
    daily["aqi_change_rate"] = g["aqi"].diff()

    def dry_spell(s):
        rain_group = (s > 0).cumsum()
        return (s == 0).astype(int).groupby(rain_group).cumsum()
    daily["dry_spell_days"] = g["precip"].transform(dry_spell)

    daily["is_winter_smog"] = daily["month"].isin([11, 12, 1, 2]).astype(int)
    daily["is_dust_season"] = daily["month"].isin([3, 4, 5]).astype(int)
    daily["is_monsoon"] = daily["month"].isin([6, 7, 8, 9]).astype(int)
    daily["month_sin"], daily["month_cos"] = np.sin(2*np.pi*daily["month"]/12), np.cos(2*np.pi*daily["month"]/12)

    dist_cols = [f"district_{i+1}" for i in range(5)]
    daily["city_aqi_median"] = daily[dist_cols].median(axis=1)
    daily["city_aqi_max"] = daily[dist_cols].max(axis=1)
    daily["city_aqi_min"] = daily[dist_cols].min(axis=1)
    daily["city_aqi_spread"] = daily["city_aqi_max"] - daily["city_aqi_min"]

    daily = pd.concat([daily, pd.get_dummies(daily["city"], prefix="city").astype(int)], axis=1)
    return daily.dropna(subset=["lag_7d", "rolling_mean_14d"]).reset_index(drop=True)

point_model, quantile_models, project, holdout_preds, eval_scores = load_model()
raw_df = load_recent_data(project)
cities = sorted(raw_df["city"].unique())

st.session_state.setdefault("dark_mode", False)
with st.container(key="ribbon"):
    rb1, rb2 = st.columns([5, 1], gap="small", vertical_alignment="center")
    with rb1:
        st.markdown(
            '<div class="ribbon-title">PAQI <span>Pearls AQI Predictor</span></div>',
            unsafe_allow_html=True,
        )
    with rb2:
        now_pk = dt.datetime.now(ZoneInfo("Asia/Karachi"))
        st.markdown(
            f'<span class="ribbon-clock">{now_pk.strftime("%b %d, %Y &middot; %I:%M %p PKT")}</span>',
            unsafe_allow_html=True,
        )
dark_mode = st.session_state.dark_mode

st.markdown("""
<div class="side-nav">
    <a href="#nav-home"><span>Home</span></a>
    <a href="#nav-trend"><span>24h Trend</span></a>
    <a href="#nav-forecast"><span>Forecast</span></a>
    <a href="#nav-why"><span>Why This Prediction</span></a>
    <a href="#eda-anchor"><span>EDA</span></a>
    <a href="#nav-models"><span>Model Comparison</span></a>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="app-blurb">Machine-learning powered 3-day air quality forecasts for major '
    "Pakistani cities, combining live pollutant readings with weather and seasonal patterns.</p>",
    unsafe_allow_html=True,
)

st.markdown('<div id="nav-home"></div>', unsafe_allow_html=True)
with st.container(key="city_row"):
    title_col, sel_col, btn_col = st.columns([3, 1, 1], vertical_alignment="center")
    with sel_col:
        city = st.selectbox("Change city", [c.title() for c in cities])
    with btn_col:
        theme_label = "Dark theme" if not st.session_state.dark_mode else "Light theme"
        if st.button(theme_label, key="theme_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with title_col:
        st.title(f"{city} Air Quality")
city_key = city.lower()
palette = inject_theme(city_key, dark_mode)

daily = aggregate_daily(raw_df)
daily = build_live_features(daily)
city_rows = daily[daily["city"] == city_key].sort_values("timestamp")
latest = city_rows.iloc[[-1]]
feature_cols = [c for c in daily.columns if c not in ["timestamp", "city", "date"]]
X_latest = latest[feature_cols]

preds_log = point_model.predict(X_latest)[0]
preds = np.expm1(preds_log)

city_hourly = raw_df[raw_df["city"] == city_key].sort_values("timestamp")
current_aqi = city_hourly["aqi"].iloc[-1]
prev_aqi = city_hourly["aqi"].iloc[-2]
cat_label, cat_color = aqi_category(current_aqi)


# --- Current AQI gauge + category, merged into one card ---
with st.container(key="current_aqi_card"):
    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=current_aqi,
            delta={"reference": prev_aqi, "increasing": {"color": "#E74C3C"}, "decreasing": {"color": "#2ECC71"}},
            number={"font": {"color": palette["text"]}},
            gauge={"axis": {"range": [0, 300], "tickcolor": palette["text"], "tickfont": {"color": palette["text"]}},
                   "bar": {"color": palette["accent"]},
                   "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0}))
        fig.update_layout(title=dict(text=""), paper_bgcolor="rgba(0,0,0,0)", font={"color": palette["text"]}, height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        stored_dt = pd.to_datetime(city_hourly["timestamp"].iloc[-1], unit="s", utc=True).tz_convert(ZoneInfo("Asia/Karachi"))
        st.markdown(f"""<div class="aqi-category-block">
            <h3>Current Air Quality</h3>
            <span class="badge" style="background:{cat_color}33; color:{cat_color};">{cat_label}</span>
            <p style="margin-top:14px;">Updated {stored_dt.strftime("%b %d, %Y at %I:%M %p")} PKT</p>
            </div>""", unsafe_allow_html=True)

# --- Pollutant tiles ---
st.markdown("#### Current Pollutants")
pollutants = [("PM2.5", "pm2_5"), ("PM10", "pm10"), ("O3", "o3"), ("NO2", "no2"), ("SO2", "so2"), ("CO", "co")]
pcols = st.columns(len(pollutants))
for col, (label, key) in zip(pcols, pollutants):
    with col:
        st.metric(label, f"{city_hourly[key].iloc[-1]:.1f}")

# --- 24h trend + current conditions ---
c3, c4 = st.columns([2, 1])
with c3:
    st.markdown('<div id="nav-trend"></div>', unsafe_allow_html=True)
    st.markdown("#### 24-Hour AQI Trend")
    last24 = city_hourly.tail(24)
    x_pkt = pd.to_datetime(last24["timestamp"], unit="s", utc=True).dt.tz_convert(ZoneInfo("Asia/Karachi"))
    fig24 = px.area(last24, x=x_pkt, y="aqi")
    fig24.update_traces(
        line_color=palette["accent"], fillcolor=hex_to_rgba(palette["accent"], 0.2),
        hovertemplate="%{x|%b %d, %Y \u00b7 %I:%M %p}<br><b>AQI %{y}</b><extra></extra>",
    )
    style_fig(fig24, palette)
    fig24.update_yaxes(range=[0, aqi_band_ceiling(last24["aqi"].max())])
    st.plotly_chart(fig24, use_container_width=True)
    st.markdown('<a href="#eda-anchor" class="jump-link">More detailed analysis</a>', unsafe_allow_html=True)
with c4:
    st.markdown("#### Current Conditions")
    st.metric("Temperature", f"{city_hourly['temp'].iloc[-1]:.1f} C")
    st.metric("Humidity", f"{city_hourly['humidity'].iloc[-1]:.0f}%")
    st.metric("Pressure", f"{city_hourly['pressure'].iloc[-1]:.1f} hPa")

# --- 3-day forecast cards ---
st.markdown('<div id="nav-forecast"></div>', unsafe_allow_html=True)
st.markdown("#### 3-Day Forecast")
fcols = st.columns(len(HORIZONS))
for i, h in enumerate(HORIZONS):
    label, color = aqi_category(preds[i])
    rmse_h = eval_scores.get("day6_split", {}).get("per_horizon", {}).get(str(h), {}).get("rmse") if eval_scores else None
    rmse_line = f"Model RMSE: ± {rmse_h}" if rmse_h is not None else "Model RMSE: unavailable"
    with fcols[i]:
        st.markdown(f"""<div class="glass-card">
            <p>+{h} day{'s' if h > 1 else ''}</p>
            <span class="badge" style="background:{color}33; color:{color};">{label}</span>
            <h2 style="margin:10px 0;">{preds[i]:.1f}</h2>
            <p style="opacity:0.8;">{rmse_line}</p>
            </div>""", unsafe_allow_html=True)

st.markdown("#### Predicted AQI Trend")
trend_df = pd.DataFrame({"day": ["Today"] + [f"+{h}d" for h in HORIZONS],
                          "aqi": [current_aqi] + list(preds)})
fig_trend = px.line(trend_df, x="day", y="aqi", markers=True)
fig_trend.update_traces(line_color=palette["accent"])
style_fig(fig_trend, palette)
fig_trend.update_yaxes(range=[0, aqi_band_ceiling(trend_df["aqi"].max())])
st.plotly_chart(fig_trend, use_container_width=True)

# --- SHAP explainability ---
st.markdown('<div id="nav-why"></div>', unsafe_allow_html=True)
st.markdown("#### Why This Prediction")
horizon_choice = st.radio("Horizon", HORIZONS, format_func=lambda h: f"+{h} day{'s' if h > 1 else ''}", horizontal=True)
h_idx = HORIZONS.index(horizon_choice)
sub_model = point_model.estimators_[h_idx]
background = X_latest if len(city_rows) < 30 else city_rows[feature_cols].sample(30, random_state=42)
explainer = shap.Explainer(sub_model.predict, background)
shap_values = explainer(X_latest)
shap_df = pd.DataFrame({"feature": [friendly_feature_name(c) for c in feature_cols], "value": shap_values.values[0]}).sort_values("value")
top_increase = shap_df.iloc[-1]
top_decrease = shap_df.iloc[0]

sc1, sc2 = st.columns(2)
sc1.metric("Top increase", top_increase["feature"], f"+{top_increase['value']:.2f}")
sc2.metric("Top decrease", top_decrease["feature"], f"{top_decrease['value']:.2f}")

fig_shap = px.bar(shap_df.tail(15), x="value", y="feature", orientation="h",
                   color=shap_df.tail(15)["value"] > 0,
                   color_discrete_map={True: palette["mid2"], False: palette["accent"]})
fig_shap.update_layout(showlegend=False)
style_fig(fig_shap, palette)
st.plotly_chart(fig_shap, use_container_width=True)

def show_only_selected_city(fig, selected):
    # legendonly keeps other cities clickable to add back in for comparison
    for trace in fig.data:
        trace.visible = True if trace.name == selected else "legendonly"
    return fig

st.markdown('<div id="eda-anchor"></div>', unsafe_allow_html=True)
st.markdown("##### Data Visualization and EDA")
st.markdown(
    '<p class="app-blurb">Historical patterns behind the forecast above. '
    f"Showing {city} by default; click a city name in any legend to add it for comparison.</p>",
    unsafe_allow_html=True,
)
eda_daily = aggregate_daily(raw_df)

st.markdown("##### AQI trend over time")
eda_daily["date_pkt"] = pd.to_datetime(eda_daily["timestamp"], unit="s", utc=True).dt.tz_convert(ZoneInfo("Asia/Karachi"))
fig_line = px.line(eda_daily, x="date_pkt", y="aqi", color="city", color_discrete_map=CITY_COLORS)
fig_line.update_traces(hovertemplate="%{fullData.name}: <b>%{y:.0f}</b><extra></extra>")
fig_line.update_layout(hovermode="x unified", xaxis=dict(hoverformat="%b %d, %Y"))
style_fig(fig_line, palette)
show_only_selected_city(fig_line, city_key)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("##### PM2.5 distribution")
fig_hist = px.histogram(raw_df, x="pm2_5", color="city", barmode="overlay", opacity=0.6, nbins=50, color_discrete_map=CITY_COLORS)
style_fig(fig_hist, palette)
show_only_selected_city(fig_hist, city_key)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("##### Pollutant correlation")
corr_cols = ["pm2_5", "pm10", "co", "no2", "so2", "aqi", "temp", "humidity", "pressure", "wind_speed", "precip"]
corr = raw_df[raw_df["city"] == city_key][corr_cols].corr().round(2)
fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale=[palette["card"], palette["accent"]], zmin=-1, zmax=1)
style_fig(fig_heat, palette)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("##### AQI spread and outliers by city")
fig_box = px.box(raw_df, x="city", y="aqi", color="city", color_discrete_map=CITY_COLORS)
style_fig(fig_box, palette)
show_only_selected_city(fig_box, city_key)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("##### Predicted vs actual, last 90 days (+1 day horizon)")
if holdout_preds is not None:
    city_holdout = holdout_preds[holdout_preds["city"] == city_key].copy()
    city_holdout["date"] = pd.to_datetime(city_holdout["date"])
    compare_df = city_holdout.rename(columns={"actual": "Actual", "predicted_1d": "Predicted"})
    fig_compare = px.line(compare_df, x="date", y=["Actual", "Predicted"])
    fig_compare.data[0].line.update(color=palette["accent"], width=2.5)
    fig_compare.data[1].line.update(color=palette["border"], width=2, dash="dash")
    fig_compare.update_traces(hovertemplate="%{fullData.name}: <b>%{y:.0f}</b><extra></extra>")
    fig_compare.update_layout(hovermode="x unified", xaxis=dict(hoverformat="%b %d, %Y"))
    style_fig(fig_compare, palette)
    st.plotly_chart(fig_compare, use_container_width=True)
    st.markdown(
        '<p class="app-blurb">These are genuine holdout predictions: the deployed model was trained '
        "excluding this 90-day window entirely, so this reflects real forecasting performance, not the "
        "model recalling data it was trained on.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<p class="app-blurb">Holdout predictions aren\'t available for the currently deployed model. '
        "Rerun training_pipeline.py to generate them.</p>",
        unsafe_allow_html=True,
    )

# --- Model comparison: top 3 models from training, scores across all 3 horizons ---
st.markdown('<div id="nav-models"></div>', unsafe_allow_html=True)
st.markdown("##### Model Comparison")
model_comparison = eval_scores.get("model_comparison") if eval_scores else None
if model_comparison:
    st.markdown(
        '<p class="app-blurb">Top 3 models from training, evaluated on the same held-out split.</p>',
        unsafe_allow_html=True,
    )
    mcols = st.columns(len(model_comparison))
    for col, m in zip(mcols, model_comparison):
        with col:
            st.markdown(f"""<div class="glass-card">
                <p style="opacity:0.8; margin-bottom:4px;">{m['name']}</p>
                <h3 style="margin:0 0 10px 0;">R2 {m['r2']:.3f}</h3>
                <p style="margin:0;">RMSE {m['rmse']} &nbsp;·&nbsp; MAE {m['mae']}</p>
                </div>""", unsafe_allow_html=True)

    comp_rows = [
        {"Model": m["name"], "Day": f"+{day}d", "RMSE": s["rmse"], "MAE": s["mae"], "R2": s["r2"]}
        for m in model_comparison for day, s in m["per_horizon"].items()
    ]
    comp_df = pd.DataFrame(comp_rows)
    model_order = [m["name"] for m in model_comparison]
    for metric in ["RMSE", "MAE", "R2"]:
        fig_comp = px.bar(comp_df, x="Day", y=metric, color="Model", barmode="group",
                           category_orders={"Model": model_order},
                           color_discrete_sequence=[palette["accent"], palette["mid2"], palette["border"]])
        fig_comp.update_layout(legend_title_text="")
        style_fig(fig_comp, palette)
        st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.markdown(
        '<p class="app-blurb">Model comparison data isn\'t available for the currently deployed model. '
        "Rerun training_pipeline.py to generate it.</p>",
        unsafe_allow_html=True,
    )