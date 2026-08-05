import os, base64, joblib
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

DARK = {"accent": "#07F3F4", "mid": "#10BDC2", "mid2": "#14969C", "border": "#187A83", "card": "#17535D", "text": "#E8FBFC"}
LIGHT = {"accent": "#284539", "mid": "#526A60", "mid2": "#6E8279", "border": "#9FB2A8", "card": "#ECF0EC", "text": "#284539"}

CATEGORIES = [(50, "Good", "#2ECC71"), (100, "Moderate", "#F1C40F"), (150, "Unhealthy for Sensitive Groups", "#E67E22"),
              (200, "Unhealthy", "#E74C3C"), (300, "Very Unhealthy", "#8E44AD"), (10_000, "Hazardous", "#7B241C")]

def aqi_band(value):
    """First (cutoff, label, color) band whose cutoff covers `value`."""
    return next((band for band in CATEGORIES if value <= band[0]), CATEGORIES[-1])

def aqi_category(value):
    return aqi_band(value)[1:]

def aqi_band_ceiling(value):
    """Category cutoff for `value`, so a y-axis anchored at [0, ceiling] reflects
    its real position on the AQI scale instead of autoscaling tightly to the data
    range (which makes a 2-point wobble look like a cliff)."""
    return aqi_band(value)[0]

def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

LAST_KNOWN_RMSE = {1: 8.73, 2: 16.73, 3: 19.23}  # from most recent training_pipeline.py evaluation run

def style_fig(fig, palette):
    """Force every text element (title, legend, axis titles/ticks) to a theme-aware
    color. Plotly text lives in SVG <text>/<tspan> nodes, which the page-level
    `span`/`p` CSS rule never reaches, so this has to be set on the figure itself
    or labels silently default to a color that can vanish in light mode."""
    grid = hex_to_rgba(palette["text"], 0.12)
    fig.update_layout(
        title=dict(text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"]),
        title_font=dict(color=palette["text"]),
        legend=dict(font=dict(color=palette["text"])),
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
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu {{ visibility: hidden; }}
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
    header[data-testid="stHeader"] {{ height: 0; min-height: 0; }}

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

    /* Theme the city dropdown (closed control + open menu) and shrink it a bit */
    div[data-baseweb="select"] > div {{
        background: {p['card']}dd !important; border-color: {p['border']}88 !important;
        color: {p['text']} !important; min-height: 36px !important; font-size: 0.88rem !important;
    }}
    div[data-baseweb="select"] svg {{ fill: {p['text']} !important; }}
    ul[data-baseweb="menu"] {{ background: {p['card']}f2 !important; }}
    ul[data-baseweb="menu"] li {{ color: {p['text']} !important; font-size: 0.88rem !important; }}
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

@st.cache_resource
def load_model():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"])
    mr = project.get_model_registry()
    m = mr.get_model("multi_city_aqi_daily_model", version=1)
    path = m.download()
    bundle = joblib.load(f"{path}/model.pkl")
    return bundle["point_model"], bundle["quantile_models"], project

@st.cache_data(ttl=3600)
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

point_model, quantile_models, project = load_model()
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

st.markdown(
    '<p class="app-blurb">Machine-learning powered 3-day air quality forecasts for major '
    "Pakistani cities, combining live pollutant readings with weather and seasonal patterns.</p>",
    unsafe_allow_html=True,
)

with st.container(key="city_row"):
    sel_col, _, btn_col = st.columns([1, 2, 1], vertical_alignment="bottom")
    with sel_col:
        city = st.selectbox("Select city", [c.title() for c in cities])
    with btn_col:
        theme_label = "Dark theme" if not st.session_state.dark_mode else "Light theme"
        if st.button(theme_label, key="theme_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
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

st.title(f"{city} Air Quality")

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
        st.markdown(f"""<div class="aqi-category-block">
            <h3>Current Air Quality</h3>
            <span class="badge" style="background:{cat_color}33; color:{cat_color};">{cat_label}</span>
            <p style="margin-top:14px;">Updated at hour {int(city_hourly['hour'].iloc[-1])}:00</p>
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
    st.markdown("#### 24-Hour AQI Trend")
    last24 = city_hourly.tail(24)
    fig24 = px.area(last24, x=pd.to_datetime(last24["timestamp"], unit="s"), y="aqi")
    fig24.update_traces(line_color=palette["accent"], fillcolor=hex_to_rgba(palette["accent"], 0.2))
    style_fig(fig24, palette)
    fig24.update_yaxes(range=[0, aqi_band_ceiling(last24["aqi"].max())])
    st.plotly_chart(fig24, use_container_width=True)
with c4:
    st.markdown("#### Current Conditions")
    st.metric("Temperature", f"{city_hourly['temp'].iloc[-1]:.1f} C")
    st.metric("Humidity", f"{city_hourly['humidity'].iloc[-1]:.0f}%")
    st.metric("Pressure", f"{city_hourly['pressure'].iloc[-1]:.1f} hPa")

# --- 3-day forecast cards ---
st.markdown("#### 3-Day Forecast")
fcols = st.columns(len(HORIZONS))
for i, h in enumerate(HORIZONS):
    label, color = aqi_category(preds[i])
    with fcols[i]:
        st.markdown(f"""<div class="glass-card">
            <p>+{h} day{'s' if h > 1 else ''}</p>
            <span class="badge" style="background:{color}33; color:{color};">{label}</span>
            <h2 style="margin:10px 0;">{preds[i]:.1f}</h2>
            <p style="opacity:0.8;">Model RMSE: ± {LAST_KNOWN_RMSE[h]}</p>
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
st.markdown("#### Why This Prediction")
horizon_choice = st.radio("Horizon", HORIZONS, format_func=lambda h: f"+{h} day{'s' if h > 1 else ''}", horizontal=True)
h_idx = HORIZONS.index(horizon_choice)
sub_model = point_model.estimators_[h_idx]
background = X_latest if len(city_rows) < 30 else city_rows[feature_cols].sample(30, random_state=42)
explainer = shap.Explainer(sub_model.predict, background)
shap_values = explainer(X_latest)
shap_df = pd.DataFrame({"feature": feature_cols, "value": shap_values.values[0]}).sort_values("value")
top_increase = shap_df.iloc[-1]
top_decrease = shap_df.iloc[0]

sc1, sc2 = st.columns(2)
sc1.metric("Top increase", top_increase["feature"], f"+{top_increase['value']:.2f}")
sc2.metric("Top decrease", top_decrease["feature"], f"{top_decrease['value']:.2f}")

fig_shap = px.bar(shap_df.tail(15), x="value", y="feature", orientation="h",
                   color=shap_df.tail(15)["value"] > 0, color_discrete_map={True: "#E67E22", False: "#2ECC71"})
fig_shap.update_layout(showlegend=False)
style_fig(fig_shap, palette)
st.plotly_chart(fig_shap, use_container_width=True)