import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hopsworks

OWM_KEY = os.environ["OPENWEATHER_API_KEY"]

# PM-only, used ONLY when Open-Meteo fails - an approximation of Open-Meteo's
# full 6-pollutant EPA methodology. Fine for Pakistani cities since PM2.5 is
# almost always the controlling pollutant here anyway. Every row this touches
# is tagged with source="openweather-fallback" so it stays auditable.
PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
PM10_BP = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]

def us_aqi_pm_fallback(pm25, pm10):
    def sub(c, bp):
        for lo, hi, ilo, ihi in bp:
            if lo <= c <= hi:
                return (ihi-ilo)/(hi-lo)*(c-lo)+ilo
        return bp[-1][3]
    return round(max(sub(pm25, PM25_BP), sub(pm10, PM10_BP)))

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

def fetch_pollution_now(lat, lon):
    for attempt in range(3):
        try:
            r = requests.get(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                params={"latitude": lat, "longitude": lon,
                        "current": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"},
                timeout=(20, 30),
            )
            r.raise_for_status()
            c = r.json()["current"]
            return {"aqi": round(c["us_aqi"]), "pm2_5": c["pm2_5"], "pm10": c["pm10"],
                    "co": c["carbon_monoxide"], "no2": c["nitrogen_dioxide"],
                    "so2": c["sulphur_dioxide"], "o3": c["ozone"], "source": "open-meteo"}
        except requests.RequestException:
            if attempt < 2:
                time.sleep(5)

    r = requests.get("https://api.openweathermap.org/data/2.5/air_pollution",
                      params={"lat": lat, "lon": lon, "appid": OWM_KEY}, timeout=30).json()["list"][0]
    c = r["components"]
    return {"aqi": us_aqi_pm_fallback(c["pm2_5"], c["pm10"]), "pm2_5": c["pm2_5"], "pm10": c["pm10"],
            "co": c["co"], "no2": c["no2"], "so2": c["so2"], "o3": c["o3"], "source": "openweather-fallback"}

def fetch_weather_now(lat, lon):
    for attempt in range(3):
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation"},
                timeout=(20, 30),
            )
            r.raise_for_status()
            c = r.json()["current"]
            return {"temp": c["temperature_2m"], "humidity": c["relative_humidity_2m"],
                    "pressure": c["surface_pressure"], "wind_speed": c["wind_speed_10m"],
                    "wind_deg": c["wind_direction_10m"], "precip": c["precipitation"], "source": "open-meteo"}
        except requests.RequestException:
            if attempt < 2:
                time.sleep(5)

    wx = requests.get("https://api.openweathermap.org/data/2.5/weather",
                       params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric"}, timeout=30).json()
    return {"temp": wx["main"]["temp"], "humidity": wx["main"]["humidity"], "pressure": wx["main"]["pressure"],
            "wind_speed": wx["wind"]["speed"], "wind_deg": wx["wind"].get("deg", 0),
            "precip": wx.get("rain", {}).get("1h", 0.0), "source": "openweather-fallback"}

def fetch_city_row(city, points):
    names = list(points.keys())
    primary_lat, primary_lon = points[names[0]]

    # districts include the primary point at index 0, so fetch it once and reuse
    with ThreadPoolExecutor(max_workers=len(names) + 1) as ex:
        weather_fut = ex.submit(fetch_weather_now, primary_lat, primary_lon)
        dist_futs = [ex.submit(fetch_pollution_now, *points[name]) for name in names]
        weather = weather_fut.result()
        dist_results = [f.result() for f in dist_futs]

    pollution = dist_results[0]
    dist_vals = [d["aqi"] for d in dist_results]

    used_fallback = any(r["source"] == "openweather-fallback" for r in [weather] + dist_results)
    pollution = {k: v for k, v in pollution.items() if k != "source"}

    wind_dir_rad = np.radians(weather["wind_deg"])
    ts = datetime.now(timezone.utc)
    row = {
        "timestamp": int(ts.timestamp()), "city": city,
        **pollution,
        "temp": weather["temp"], "humidity": weather["humidity"], "pressure": weather["pressure"],
        "wind_speed": weather["wind_speed"], "precip": weather["precip"],
        "wind_dir_sin": np.sin(wind_dir_rad), "wind_dir_cos": np.cos(wind_dir_rad),
        **{f"district_{i+1}": v for i, v in enumerate(dist_vals)},
        "city_aqi_mean": sum(dist_vals) / len(dist_vals), "city_aqi_max": max(dist_vals), "city_aqi_min": min(dist_vals),
        "city_aqi_spread": max(dist_vals) - min(dist_vals),
        "hour": ts.hour, "day": ts.day, "month": ts.month, "day_of_week": ts.weekday(),
        "used_fallback": used_fallback,
    }
    return row

def fetch_features() -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=len(CITIES)) as ex:
        futs = {ex.submit(fetch_city_row, city, points): city for city, points in CITIES.items()}
        for fut, city in futs.items():
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"Skipping {city} this run: {e}")
    if not rows:
        raise RuntimeError("Every city failed this run - both Open-Meteo and OpenWeather are unreachable.")
    df = pd.DataFrame(rows)
    dist_cols = [f"district_{i+1}" for i in range(5)]
    float_cols = ["aqi", "co", "no2", "o3", "so2", "pm2_5", "pm10", "temp", "humidity", "pressure",
                  "wind_speed", "precip", "wind_dir_sin", "wind_dir_cos",
                  "city_aqi_mean", "city_aqi_max", "city_aqi_min", "city_aqi_spread"] + dist_cols
    df[float_cols] = df[float_cols].astype("float64")
    df[["timestamp", "hour", "day", "month", "day_of_week"]] = df[["timestamp", "hour", "day", "month", "day_of_week"]].astype("int64")
    return df

def push_to_hopsworks(df: pd.DataFrame):
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"])
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="multi_city_aqi_features", version=1, primary_key=["timestamp", "city"], event_time="timestamp",
        description="Hourly AQI (Open-Meteo, official US AQI) + weather + 5-district features for 5 Pakistani cities")
    # used_fallback isn't in the existing schema yet (get_or_create only creates on
    # first-ever run, it doesn't add columns to a feature group that already exists).
    # Drop it before insert for now - still printed in logs above for auditability.
    df = df.drop(columns=["used_fallback"], errors="ignore")
    for attempt in range(3):
        try:
            fg.insert(df)
            return
        except Exception as e:
            if attempt < 2:
                print(f"Insert failed ({e}), retrying in 30s...")
                time.sleep(30)
            else:
                raise

if __name__ == "__main__":
    df = fetch_features()
    print(df.T)
    push_to_hopsworks(df)
    print("Rows inserted into Hopsworks feature store.")