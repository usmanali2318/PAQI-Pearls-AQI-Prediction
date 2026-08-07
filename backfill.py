import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
import hopsworks

DAYS_BACK = 1095

CITIES = {
    "karachi": {"south": (24.8608, 67.0104), "keamari": (24.8944, 66.9874), "korangi": (24.8504, 67.1999),
                "malir": (24.8929, 67.1953), "central": (24.9002, 67.0446)},
    "lahore": {"gulberg": (31.5497, 74.3436), "iqbal_town": (31.5111, 74.2839), "walled_city": (31.5862, 74.3098),
               "cantt": (31.5170, 74.3830), "shalimar": (31.5850, 74.3500)},
    "islamabad": {"blue_area": (33.7100, 73.0550), "f10_f11": (33.6989, 73.0114), "g9_g10": (33.6795, 73.0169),
                  "i10": (33.6469, 73.0362), "margalla": (33.7218, 72.9962)},
    "peshawar": {"saddar": (34.0083, 71.5615), "university_town": (34.0044, 71.5064), "hayatabad": (33.9800, 71.4600),
                 "old_city": (34.0125, 71.5730), "gulbahar": (34.0000, 71.6000)},
    "quetta": {"saddar": (30.1798, 66.9750), "satellite_town": (30.1812, 67.0331), "sariab": (30.1300, 66.9800),
               "pashtunabad": (30.1900, 66.9500), "samungli": (30.2500, 66.9400)},
}

POLLUTANTS = ["us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
RENAME = {"carbon_monoxide": "co", "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2", "ozone": "o3", "us_aqi": "aqi"}

def fetch_pollution(lat, lon, start, end, retries=5):
    for attempt in range(retries):
        r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
            "latitude": lat, "longitude": lon, "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"), "hourly": ",".join(POLLUTANTS)})
        body = r.json()
        if "hourly" in body:
            time.sleep(2)
            r = body["hourly"]
            break
        if r.status_code == 429 and attempt < retries - 1:
            print(f"Rate limited, waiting 65s (attempt {attempt+1}/{retries})...")
            time.sleep(65)
            continue
        raise RuntimeError(f"Open-Meteo error (status {r.status_code}): {body}")
    df = pd.DataFrame({"timestamp": pd.to_datetime(r["time"]).astype("int64") // 10**9})
    for p in POLLUTANTS:
        df[RENAME.get(p, p)] = r[p]
    # Open-Meteo's air-quality endpoint pads past the requested end_date with
    # forecast-model output regardless of what end_date says, so clip to real
    # "now" here — otherwise forecast rows silently end up in training history.
    now_ts = int(datetime.now(timezone.utc).timestamp())
    return df[df["timestamp"] <= now_ts].reset_index(drop=True)

def fetch_weather_history(lat, lon, start, end):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon, "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
        "timezone": "UTC"}).json()["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(r["time"]).astype("int64") // 10**9,
        "temp": r["temperature_2m"], "humidity": r["relative_humidity_2m"],
        "pressure": r["surface_pressure"], "wind_speed": r["wind_speed_10m"],
        "precip": r["precipitation"]})
    wind_dir_rad = np.radians(r["wind_direction_10m"])
    df["wind_dir_sin"], df["wind_dir_cos"] = np.sin(wind_dir_rad), np.cos(wind_dir_rad)
    return df

def build_city_dataset(city, points, start, end):
    names = list(points.keys())
    primary_name, primary_coords = names[0], points[names[0]]

    df = fetch_pollution(*primary_coords, start, end)
    for i, name in enumerate(names):
        d = fetch_pollution(*points[name], start, end)[["timestamp", "aqi"]].rename(columns={"aqi": f"district_{i+1}"})
        df = pd.merge_asof(df.sort_values("timestamp"), d.sort_values("timestamp"), on="timestamp", direction="nearest")

    dist_cols = [f"district_{i+1}" for i in range(len(names))]
    df["city_aqi_mean"] = df[dist_cols].mean(axis=1)
    df["city_aqi_max"] = df[dist_cols].max(axis=1)
    df["city_aqi_min"] = df[dist_cols].min(axis=1)
    df["city_aqi_spread"] = df["city_aqi_max"] - df["city_aqi_min"]

    wx = fetch_weather_history(*primary_coords, start, end)
    df = pd.merge_asof(df.sort_values("timestamp"), wx.sort_values("timestamp"), on="timestamp", direction="nearest")
    df["city"] = city
    return df

def build_dataset():
    end, start = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    frames = []
    for city, points in CITIES.items():
        print(f"Fetching {city}...")
        frames.append(build_city_dataset(city, points, start, end))
    df = pd.concat(frames, ignore_index=True)

    ts = pd.to_datetime(df["timestamp"], unit="s")
    df["hour"], df["day"], df["month"], df["day_of_week"] = ts.dt.hour, ts.dt.day, ts.dt.month, ts.dt.dayofweek
    df = df.dropna()

    dist_cols = [f"district_{i+1}" for i in range(5)]
    float_cols = ["aqi", "co", "no2", "o3", "so2", "pm2_5", "pm10", "temp", "humidity", "pressure",
                  "wind_speed", "precip", "wind_dir_sin", "wind_dir_cos",
                  "city_aqi_mean", "city_aqi_max", "city_aqi_min", "city_aqi_spread"] + dist_cols
    df[float_cols] = df[float_cols].astype("float64")
    df[["timestamp", "hour", "day", "month", "day_of_week"]] = df[["timestamp", "hour", "day", "month", "day_of_week"]].astype("int64")
    return df

def push_to_hopsworks(df):
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"])
    fg = project.get_feature_store().get_or_create_feature_group(
        name="multi_city_aqi_features", version=1, primary_key=["timestamp", "city"], event_time="timestamp")
    fg.insert(df)

if __name__ == "__main__":
    df = build_dataset()
    print(f"Backfilled {len(df)} rows across {df['city'].nunique()} cities: "
          f"{pd.to_datetime(df['timestamp'].min(), unit='s')} -> {pd.to_datetime(df['timestamp'].max(), unit='s')}")
    push_to_hopsworks(df)
    print("Backfill inserted into Hopsworks.")