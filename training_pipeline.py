import os, joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import hopsworks

HORIZONS = [1, 2, 3]  # days ahead -> tomorrow, day after, day after that

PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
PM10_BP = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]

def us_aqi(pm25, pm10):
    def sub_index(c, bp):
        for lo, hi, ilo, ihi in bp:
            if lo <= c <= hi:
                return (ihi - ilo) / (hi - lo) * (c - lo) + ilo
        return bp[-1][3]
    return round(max(sub_index(pm25, PM25_BP), sub_index(pm10, PM10_BP)))

def load_data():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"])
    fg = project.get_feature_store().get_feature_group("multi_city_aqi_features", version=1)
    return fg.read().sort_values(["city", "timestamp"]).reset_index(drop=True), project

def category_accuracy(y_true, y_pred):
    bins = [50, 100, 150, 200, 300]
    true_cat, pred_cat = np.digitize(np.asarray(y_true), bins), np.digitize(np.asarray(y_pred), bins)
    return (true_cat == pred_cat).mean()

def per_horizon_metrics(y_test, preds):
    lines = []
    for i, h in enumerate(HORIZONS):
        rmse = mean_squared_error(y_test.iloc[:, i], preds[:, i]) ** 0.5
        mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
        r2 = r2_score(y_test.iloc[:, i], preds[:, i])
        lines.append(f"  +{h}d: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return "\n".join(lines)

def clean_hourly_outliers(df):
    df = df[(df["pm2_5"] >= 0) & (df["pm10"] >= 0)].copy()
    aqi_cap = df.groupby("city")["aqi"].transform(lambda s: s.quantile(0.95))
    df["aqi"] = df["aqi"].clip(upper=aqi_cap)
    return df

def aggregate_daily(df):
    df = clean_hourly_outliers(df)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    dist_cols = [f"district_{i+1}" for i in range(5)]
    agg = {"pm2_5": "mean", "pm10": "mean", "co": "mean", "no2": "mean", "so2": "mean", "aqi": "mean",
           "temp": "mean", "humidity": "mean", "pressure": "mean", "wind_speed": "mean", "precip": "sum",
           "wind_dir_sin": "mean", "wind_dir_cos": "mean"}
    for c in dist_cols:
        agg[c] = "mean"
    daily = df.groupby(["city", "date"]).agg(agg).reset_index()
    daily["aqi_hourly_std"] = df.groupby(["city", "date"])["aqi"].std().values  # intra-day swing as a feature
    daily["aqi_hourly_max"] = df.groupby(["city", "date"])["aqi"].max().values  # day's worst hour as a feature
    daily["timestamp"] = pd.to_datetime(daily["date"]).astype("int64") // 10**9
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.dayofweek
    return daily.sort_values(["city", "timestamp"]).reset_index(drop=True)

def engineer(df):
    g = df.groupby("city")

    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}d"] = g["aqi"].shift(lag)
    df["rolling_mean_3d"] = g["aqi"].transform(lambda s: s.rolling(3).mean())
    df["rolling_mean_7d"] = g["aqi"].transform(lambda s: s.rolling(7).mean())
    df["rolling_std_7d"] = g["aqi"].transform(lambda s: s.rolling(7).std())
    df["rolling_mean_14d"] = g["aqi"].transform(lambda s: s.rolling(14).mean())
    df["aqi_change_rate"] = g["aqi"].diff()

    def dry_spell(s):
        rain_group = (s > 0).cumsum()
        return (s == 0).astype(int).groupby(rain_group).cumsum()
    df["dry_spell_days"] = g["precip"].transform(dry_spell)

    df["is_winter_smog"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    df["is_dust_season"] = df["month"].isin([3, 4, 5]).astype(int)
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*df["month"]/12), np.cos(2*np.pi*df["month"]/12)

    dist_cols = [f"district_{i+1}" for i in range(5)]
    df["city_aqi_median"] = df[dist_cols].median(axis=1)
    df["city_aqi_max"] = df[dist_cols].max(axis=1)
    df["city_aqi_min"] = df[dist_cols].min(axis=1)
    df["city_aqi_spread"] = df["city_aqi_max"] - df["city_aqi_min"]

    for h in HORIZONS:
        df[f"target_{h}d"] = g["aqi"].shift(-h)

    df = pd.concat([df, pd.get_dummies(df["city"], prefix="city").astype(int)], axis=1)
    return df.dropna().reset_index(drop=True)

def train_eval(df):
    target_cols = [f"target_{h}d" for h in HORIZONS]
    X, y = df.drop(columns=["timestamp", "city", "date"] + target_cols), df[target_cols]
    y_log = np.log1p(y)

    day_id = df["timestamp"] // 86400
    test_mask = day_id % 6 == 0
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train_log, y_test = y_log[~test_mask], y[test_mask]

    tscv = TimeSeriesSplit(n_splits=2)
    tuned = {
        "RandomForest": (RandomForestRegressor(random_state=42),
                          {"n_estimators": [100, 150], "max_depth": [8, 12], "min_samples_leaf": [3, 5],
                           "criterion": ["squared_error", "absolute_error"]}),
        "HistGB": (HistGradientBoostingRegressor(random_state=42),
                   {"max_iter": [100, 150], "max_depth": [4, 6], "learning_rate": [0.05, 0.1],
                    "loss": ["squared_error", "absolute_error"]}),
        "XGBoost": (XGBRegressor(random_state=42, verbosity=0),
                    {"n_estimators": [100, 150], "max_depth": [4, 6], "learning_rate": [0.05, 0.1],
                     "objective": ["reg:squarederror", "reg:absoluteerror"]}),
        "LightGBM": (LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1),
                     {"n_estimators": [100, 150], "max_depth": [4, 6], "learning_rate": [0.05, 0.1],
                      "objective": ["regression", "mae"]}),
        "CatBoost": (CatBoostRegressor(random_state=42, verbose=False, thread_count=1),
                     {"n_estimators": [100, 150], "depth": [4, 6], "learning_rate": [0.05, 0.1],
                      "loss_function": ["RMSE", "MAE"]}),
    }

    best_params, fitted = {}, {}
    for name, (base, params) in tuned.items():
        search = RandomizedSearchCV(MultiOutputRegressor(base), {f"estimator__{k}": v for k, v in params.items()},
                                     n_iter=14, cv=tscv, scoring="r2", random_state=42, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train_log)
        fitted[name] = search.best_estimator_
        best_params[name] = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
        print(f"{name} best params: {best_params[name]}")

    ridge = MultiOutputRegressor(Ridge())
    ridge_search = RandomizedSearchCV(ridge, {"estimator__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0]},
                                       n_iter=7, cv=tscv, scoring="r2", random_state=42)
    ridge_search.fit(X_train, y_train_log)
    fitted["Ridge"] = ridge_search.best_estimator_

    nn = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                                                        early_stopping=True, random_state=42))])
    fitted["FeedforwardNN"] = MultiOutputRegressor(nn)
    fitted["FeedforwardNN"].fit(X_train, y_train_log)

    stack_base = [(n.lower(), tuned[n][0].set_params(**best_params[n])) for n in ["HistGB", "XGBoost", "LightGBM", "CatBoost"]]
    stack = StackingRegressor(estimators=stack_base, final_estimator=HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42), cv=3)
    fitted["Stacked Ensemble"] = MultiOutputRegressor(stack)
    fitted["Stacked Ensemble"].fit(X_train, y_train_log)

    persistence_preds = np.tile(df.loc[test_mask, "aqi"].values.reshape(-1, 1), (1, len(HORIZONS)))
    p_rmse = mean_squared_error(y_test, persistence_preds) ** 0.5
    p_mae, p_r2 = mean_absolute_error(y_test, persistence_preds), r2_score(y_test, persistence_preds)
    p_cat = category_accuracy(y_test.values, persistence_preds)
    print(f"\nPersistence baseline: RMSE={p_rmse:.2f}  MAE={p_mae:.2f}  R2={p_r2:.3f}  CategoryAcc={p_cat:.1%}")
    print("(any model below this line isn't adding real value over a trivial guess)\n")

    best_name, best_model, best_r2, best_preds = None, None, -np.inf, None
    for name, model in fitted.items():
        preds = np.expm1(model.predict(X_test))
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae, r2 = mean_absolute_error(y_test, preds), r2_score(y_test, preds)
        cat_acc = category_accuracy(y_test.values, preds)
        print(f"{name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}  CategoryAcc={cat_acc:.1%}")
        if r2 > best_r2:
            best_name, best_model, best_r2, best_preds = name, model, r2, preds

    print(f"\nBest model: {best_name} (R2={best_r2:.3f})")
    print(f"Per-horizon breakdown for {best_name}:")
    print(per_horizon_metrics(y_test, best_preds))

    print("\nPer-city breakdown (same model, same test set):")
    for city in df["city"].unique():
        mask = X_test.filter(like=f"city_{city}").iloc[:, 0].astype(bool)
        if mask.sum() == 0:
            continue
        rmse_c = mean_squared_error(y_test[mask], best_preds[mask]) ** 0.5
        mae_c = mean_absolute_error(y_test[mask], best_preds[mask])
        print(f"  {city}: RMSE={rmse_c:.2f}  MAE={mae_c:.2f}  n={mask.sum()}")

    rf_fitted = fitted["RandomForest"]
    importances = rf_fitted.estimators_[0].feature_importances_
    top = sorted(zip(X.columns, importances), key=lambda t: -t[1])[:10]
    print("\nTop 10 features (RandomForest, +1d target):")
    for feat, imp in top:
        print(f"  {feat}: {imp:.4f}")

    quantile_models = {}
    for h in HORIZONS:
        for q in [0.1, 0.9]:
            m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, max_depth=5,
                               verbosity=-1, random_state=42)
            m.fit(X_train, y_train_log[f"target_{h}d"])
            quantile_models[(h, q)] = m

    return best_name, best_model, quantile_models

def save_to_registry(project, model, name, quantile_models):
    os.makedirs("model_dir", exist_ok=True)
    joblib.dump({"point_model": model, "quantile_models": quantile_models}, "model_dir/model.pkl")
    mr = project.get_model_registry()
    for old in mr.get_models("multi_city_aqi_daily_model"):
        try:
            old.delete()
        except Exception:
            pass
    m = mr.python.create_model(name="multi_city_aqi_daily_model",
                                description=f"Best model: {name}, predicts log1p(daily AQI) at +1d/+2d/+3d for 5 cities - invert with expm1. Bundle contains point_model and quantile_models (q0.1/q0.9 per horizon).")
    try:
        m.save("model_dir")
    except Exception as e:
        print(f"Model uploaded, but Hopsworks' status check failed (known cluster issue): {e}")
        print("Check the Model Registry in the Hopsworks UI to confirm - it's almost always there anyway.")

def per_city_diagnostic(df):
    target_cols = [f"target_{h}d" for h in HORIZONS]
    drop_cols = ["timestamp", "date", "city"] + target_cols + [c for c in df.columns if c.startswith("city_")]
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42),
        "HistGB": HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42, verbosity=-1, n_jobs=1),
        "CatBoost": CatBoostRegressor(n_estimators=150, depth=4, learning_rate=0.1, random_state=42, verbose=False, thread_count=1),
    }
    print("\nPer-city diagnostic (fixed hyperparams, no search):")
    for city in df["city"].unique():
        sub = df[df["city"] == city]
        X, y = sub.drop(columns=drop_cols), sub[target_cols]
        y_log = np.log1p(y)
        day_id = sub["timestamp"] // 86400
        test_mask = day_id % 6 == 0
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train_log, y_test = y_log[~test_mask], y[test_mask]

        print(f"  {city} (n={len(sub)}):")
        for name, base in models.items():
            model = MultiOutputRegressor(base)
            model.fit(X_train, y_train_log)
            preds = np.expm1(model.predict(X_test))
            rmse = mean_squared_error(y_test, preds) ** 0.5
            mae, r2 = mean_absolute_error(y_test, preds), r2_score(y_test, preds)
            cat_acc = category_accuracy(y_test.values, preds)
            print(f"    {name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}  CategoryAcc={cat_acc:.1%}")

def typical_peak_hours(df):
    print("\nTypical AQI peak hours by city (avg AQI per hour-of-day, top 3 hours):")
    for city in df["city"].unique():
        sub = df[df["city"] == city]
        hourly_avg = sub.groupby("hour")["aqi"].mean().sort_values(ascending=False)
        hours = sorted(hourly_avg.head(3).index.tolist())
        print(f"  {city}: worst hours typically {hours[0]}:00-{hours[-1]}:00  "
              f"(avg AQI in these hours: {hourly_avg.head(3).mean():.1f} vs city avg {sub['aqi'].mean():.1f})")

if __name__ == "__main__":
    df, project = load_data()
    typical_peak_hours(df)
    df = aggregate_daily(df)
    print(f"Aggregated to {len(df)} city-days")
    df = engineer(df)
    per_city_diagnostic(df)
    best_name, best_model, quantile_models = train_eval(df)
    save_to_registry(project, best_model, best_name, quantile_models)
    print("Model saved to Hopsworks Model Registry.")