# PAQI - Pearls AQI Predictor

A dashboard that forecasts air quality (AQI) 1 to 3 days ahead for 5 Pakistani cities:
Karachi, Lahore, Islamabad, Peshawar, and Quetta.

Live app: https://paqi-pearls-aqi-prediction.streamlit.app

## What it does

For each city it shows:
- current AQI (US EPA scale) and category (Good, Moderate, Unhealthy, etc.)
- a 3-day forecast with per-horizon model RMSE
- 24-hour AQI trend and current pollutant readings
- SHAP explanation of what's driving the forecast
- an EDA section (trend over time, PM2.5 distribution, pollutant correlation,
  AQI outliers by city, and predicted vs actual on a genuine 90-day holdout)

## How it's built

Four pipelines, one dashboard:

- **backfill.py** - one-time (or occasional) pull of 3 years of hourly AQI and
  weather history from Open-Meteo, for all 5 cities and their district points,
  into a Hopsworks feature store.
- **feature_pipeline.py** - runs hourly (GitHub Actions), fetches the latest
  hour's readings from Open-Meteo and appends them to the same feature store.
  OpenWeather is only used as a last-resort fallback if Open-Meteo doesn't
  respond after 3 retries, for both pollution and weather. In normal operation
  it isn't called at all.
- **training_pipeline.py** - runs on a schedule (GitHub Actions), trains
  several model families (RandomForest, HistGB, XGBoost, LightGBM, CatBoost,
  Ridge, a small feedforward NN, and a stacked ensemble), picks the best one,
  and saves it to the Hopsworks Model Registry along with its evaluation scores
  and holdout predictions.
- **main.py** - the Streamlit app. Loads the latest model and its scores
  straight from the registry on every run, so the numbers shown always match
  whatever model is actually deployed.

## Data source

Everything - training, the hourly pipeline, and what's shown as "current AQI" -
comes from Open-Meteo's Air Quality and Weather APIs, using the official US AQI
methodology (evaluated across all 6 pollutants, not just PM2.5/PM10). OpenWeather
exists only as an emergency fallback in feature_pipeline.py in case Open-Meteo is
unreachable from GitHub's runners, which happens occasionally.

## Model validation

Two checks, both against a "tomorrow = today" persistence baseline:

1. **Every 6th day held out**, scattered across the full training period, so
   the model is tested across every season, not just one part of the year.
2. **The last 90 days held out entirely.** These days are excluded from the
   training pool before the model family search even runs, so the deployed
   model has genuinely never seen them. This is the stricter test.

Current best model: LightGBM.

| | RMSE | MAE | R2 |
|---|---|---|---|
| Every-6th-day split | 15.70 | 11.13 | 0.827 |
| Last-90-days holdout | 18.24 | 13.10 | 0.647 |

Both beat the persistence baseline on their respective test sets.

## Running it locally

```
pip install -r requirements.txt
streamlit run main.py
```

Needs `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` set as environment variables.
`backfill.py` and `feature_pipeline.py` also need `OPENWEATHER_API_KEY` for the
fallback path.

## Project structure

```
main.py                  Streamlit dashboard
feature_pipeline.py      hourly data fetch, runs via GitHub Actions
training_pipeline.py     model training and evaluation
backfill.py               historical data backfill
.github/workflows/       GitHub Actions schedules for the above
.streamlit/config.toml   app theme
assets/                  city background images
```

## Known limitations

- Hopsworks' free tier has a monthly compute budget, and reads from the
  feature store (via the Feature Query Service) are the main thing that eats
  into it. Retraining or backfilling too often will burn through it faster
  than normal hourly operation does.
- The dashboard's live "current AQI" reflects the last hourly pipeline run,
  not a live-at-page-load fetch - it's as fresh as the most recent hour, not
  the second.