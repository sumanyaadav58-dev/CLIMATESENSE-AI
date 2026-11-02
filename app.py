import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client as api
from api_client import APIError, GEOCODING_URL, FORECAST_URL, AIR_URL, ARCHIVE_URL  # noqa: F401

st.set_page_config(page_title="ClimateSense AI", page_icon="🌍", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "data" / "climatesense.db")

# Cache lifetimes (seconds). Use the "Refresh Data" button to bypass them.
TTL_GEOCODE = 24 * 3600
TTL_WEATHER = 10 * 60
TTL_AIR = 10 * 60
TTL_HISTORY = 6 * 3600


# --------------------------------------------------------------------------- #
# Streamlit version compatibility (use_container_width -> width="stretch")
# --------------------------------------------------------------------------- #
def _st_version():
    nums = []
    for part in st.__version__.split(".")[:2]:
        m = re.match(r"\d+", part)
        nums.append(int(m.group()) if m else 0)
    return tuple(nums)


NEW_WIDTH_API = _st_version() >= (1, 50)


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element full-width on both old and new Streamlit versions."""
    if NEW_WIDTH_API:
        try:
            return fn(*args, width="stretch", **kwargs)
        except Exception:
            pass
    return fn(*args, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
@st.cache_resource
def db_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT,
            city TEXT,
            country TEXT,
            latitude REAL,
            longitude REAL,
            temperature REAL,
            humidity REAL,
            precipitation REAL,
            wind_speed REAL,
            pm2_5 REAL,
            pm10 REAL,
            us_aqi REAL
        )
    """)
    conn.commit()
    return conn


# --------------------------------------------------------------------------- #
# API request functions (cached wrappers around the resilient api_client).
# Exceptions are NOT cached by Streamlit, so a failed call is retried on the
# next refresh while successful calls are reused until their TTL expires.
# --------------------------------------------------------------------------- #
def get_json(url, params, api_name="API"):
    """Resilient GET: 20 s connect / 90 s read timeout, 3 attempts, exponential backoff."""
    return api.get_json(url, params, api_name=api_name)


@st.cache_data(ttl=TTL_GEOCODE, show_spinner=False)
def geocode(place):
    return api.geocode(place)


@st.cache_data(ttl=TTL_WEATHER, show_spinner=False)
def fetch_weather(lat, lon):
    return api.fetch_weather(lat, lon)


@st.cache_data(ttl=TTL_AIR, show_spinner=False)
def fetch_air(lat, lon):
    return api.fetch_air(lat, lon)


@st.cache_data(ttl=TTL_HISTORY, show_spinner=False)
def fetch_history(lat, lon, end_date=None):
    # end_date is part of the cache key so the cache rolls over at midnight.
    return api.fetch_history(lat, lon, end_date)


def clear_api_caches():
    for fn in (geocode, fetch_weather, fetch_air, fetch_history):
        fn.clear()


def safe_fetch(source, fn, *args):
    """Run an API call; never raise. Returns (data, issue_dict_or_None)."""
    try:
        return fn(*args), None
    except APIError as exc:
        return None, {"source": source, "label": exc.label, "message": exc.message,
                      "detail": exc.detail, "attempts": exc.attempts, "kind": exc.kind, "stale": False}
    except Exception as exc:  # unexpected bug - still do not crash the dashboard
        return None, {"source": source, "label": "Unexpected error", "message": str(exc),
                      "detail": type(exc).__name__, "attempts": 1, "kind": "request", "stale": False}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def weather_label(code):
    mapping = {
        0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
        45:"Fog",48:"Depositing rime fog",51:"Light drizzle",53:"Drizzle",
        55:"Dense drizzle",61:"Slight rain",63:"Moderate rain",65:"Heavy rain",
        71:"Slight snow",73:"Moderate snow",75:"Heavy snow",80:"Rain showers",
        81:"Moderate rain showers",82:"Violent rain showers",95:"Thunderstorm",
        96:"Thunderstorm with hail",99:"Thunderstorm with heavy hail"
    }
    try:
        return mapping.get(int(code), "Weather conditions")
    except (TypeError, ValueError):
        return "Weather conditions"


def to_float(value):
    """Convert to float; None / NaN / bad values become None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def fmt(value, unit="", default="—"):
    return f"{value} {unit}".strip() if value is not None and not (isinstance(value, float) and np.isnan(value)) else default


def save_snapshot(conn, place, w, air):
    c = w["current"]
    a = (air or {}).get("current") or {}
    exists = conn.execute(
        "SELECT 1 FROM snapshots WHERE captured_at IS ? AND city IS ? AND latitude IS ? LIMIT 1",
        (c.get("time"), place.get("name"), place.get("latitude"))).fetchone()
    if exists:  # identical refresh -> do not store duplicate rows
        return
    conn.execute(
        """INSERT INTO snapshots
        (captured_at,city,country,latitude,longitude,temperature,humidity,precipitation,wind_speed,pm2_5,pm10,us_aqi)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (c.get("time"), place.get("name"), place.get("country"),
         place.get("latitude"), place.get("longitude"), c.get("temperature_2m"),
         c.get("relative_humidity_2m"), c.get("precipitation"), c.get("wind_speed_10m"),
         a.get("pm2_5"), a.get("pm10"), a.get("us_aqi"))
    )
    conn.commit()


def make_risk(temp, rain_prob, aqi):
    if temp is None:
        heat = ("Unknown", "Current temperature is unavailable because the weather API could not be reached.")
    elif temp >= 40:
        heat = ("High", "Very high temperature. Reduce prolonged outdoor exposure and follow local advisories.")
    elif temp >= 35:
        heat = ("Moderate", "Warm conditions. Stay hydrated and monitor official heat advisories.")
    else:
        heat = ("Low", "No high-temperature threshold crossed in the current reading.")
    if rain_prob is None:
        rain = ("Unknown", "Rain probability is unavailable for this refresh.")
    elif rain_prob >= 70:
        rain = ("High", "High forecast precipitation probability. Check local weather alerts and drainage conditions.")
    elif rain_prob >= 40:
        rain = ("Moderate", "Rain is possible. Keep an eye on updated forecasts.")
    else:
        rain = ("Low", "No high rainfall-probability threshold crossed.")
    if aqi is None:
        pollution = ("Unknown", "US AQI is unavailable for this location or the air-quality API could not be reached.")
    elif aqi >= 151:
        pollution = ("High", "Elevated AQI. Sensitive groups should follow local health guidance.")
    elif aqi >= 101:
        pollution = ("Moderate", "AQI may affect sensitive groups. Check the local AQI category.")
    else:
        pollution = ("Low", "No elevated US AQI threshold crossed in the available reading.")
    return [("Heat risk", *heat), ("Rain risk", *rain), ("Air-quality risk", *pollution)]


def train_rain_model(history):
    # A small baseline classifier trained on observed historical daily weather.
    try:
        from sklearn.ensemble import RandomForestClassifier
        df = pd.DataFrame(history["daily"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 25:
            return None, "Not enough historical records to train a baseline model."
        df["date"] = pd.to_datetime(df["time"])
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        X = df[["temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max", "month", "day_of_year"]]
        y = (df["precipitation_sum"] >= 1.0).astype(int)
        if y.nunique() < 2:
            return None, "Historical data has only one rainfall class; model training was skipped."
        model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
        model.fit(X, y)
        return model, "Model trained on recent historical daily weather records."
    except Exception as e:
        return None, f"Model unavailable: {e}"


def get_rain_model(history, stamp):
    """Train once per data load instead of on every Streamlit rerun."""
    cached = st.session_state.get("rain_model")
    if cached and cached[0] == stamp:
        return cached[1], cached[2]
    model, message = train_rain_model(history)
    st.session_state["rain_model"] = (stamp, model, message)
    return model, message


# --------------------------------------------------------------------------- #
# Data loading (each API is independent: one failure never blocks the others)
# --------------------------------------------------------------------------- #
def load_dashboard(city_name, force=False):
    """Fetch everything and store it in session_state. Returns True if a location is available."""
    if force:
        clear_api_caches()
    prev_place = st.session_state.get("loaded_city")
    issues = []

    with st.spinner("Fetching live weather, air quality and historical data... "
                    "(slow networks are retried automatically, up to 3 attempts per API)"):
        try:
            place = geocode(city_name)
        except ValueError as exc:  # city not found
            st.session_state["issues"] = [{"source": "City search", "label": "Location not found",
                                           "message": str(exc), "detail": "", "attempts": 1,
                                           "kind": "request", "stale": False}]
            return prev_place is not None
        except APIError as exc:
            st.session_state["issues"] = [{"source": "City search (Geocoding API)", "label": exc.label,
                                           "message": exc.message, "detail": exc.detail,
                                           "attempts": exc.attempts, "kind": exc.kind, "stale": False}]
            return prev_place is not None
        except Exception as exc:
            st.session_state["issues"] = [{"source": "City search", "label": "Unexpected error",
                                           "message": str(exc), "detail": type(exc).__name__,
                                           "attempts": 1, "kind": "request", "stale": False}]
            return prev_place is not None

        lat, lon = place["latitude"], place["longitude"]
        same_place = bool(prev_place) and prev_place.get("latitude") == lat and prev_place.get("longitude") == lon
        end_date = (date.today() - timedelta(days=5)).isoformat()

        results = {
            "weather": safe_fetch("Weather API", fetch_weather, lat, lon),
            "air": safe_fetch("Air-quality API", fetch_air, lat, lon),
            "history": safe_fetch("Historical archive API", fetch_history, lat, lon, end_date),
        }

    stamps = dict(st.session_state.get("fetched_at", {})) if same_place else {}
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    for key, (data, issue) in results.items():
        if issue is None:
            st.session_state[key] = data
            stamps[key] = now
        else:
            old = st.session_state.get(key) if same_place else None
            if old is not None:  # keep the last good data, clearly labelled as stale
                issue["stale"] = True
                issue["stale_since"] = stamps.get(key, "an earlier refresh")
            else:
                st.session_state[key] = None
                stamps.pop(key, None)
            issues.append(issue)

    st.session_state.update({"loaded_city": place, "issues": issues, "fetched_at": stamps,
                             "last_load": now})
    weather, air = st.session_state.get("weather"), st.session_state.get("air")
    if weather is not None and results["weather"][1] is None:
        try:
            save_snapshot(db_connection(), place, weather, air)
        except Exception as exc:
            st.session_state["issues"].append({"source": "Local database", "label": "Request failure",
                                               "message": f"Snapshot could not be saved: {exc}",
                                               "detail": "", "attempts": 1, "kind": "request", "stale": False})
    return True


def render_issues():
    issues = st.session_state.get("issues") or []
    if not issues:
        return
    with st.expander(f"⚠️ {len(issues)} data source issue(s) — showing everything that is still available",
                     expanded=True):
        for it in issues:
            extra = f" Showing the last successful data from {it['stale_since']}." if it.get("stale") else ""
            st.warning(f"**{it['source']} — {it['label']}.** {it['message']}{extra}")
            bits = []
            if it.get("attempts", 1) > 1:
                bits.append(f"{it['attempts']} attempts made")
            if it.get("detail"):
                bits.append(it["detail"][:300])
            if bits:
                st.caption("Technical detail: " + " | ".join(bits))
        st.caption("Use **🔄 Refresh Data** in the sidebar to retry once the connection or API has recovered.")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🌍 ClimateSense AI")
st.caption("Live Climate Impact & Environmental Risk Analytics | MCA Data Analytics Project")
with st.sidebar:
    st.header("Location")
    city_input = st.text_input("Enter city", "Ludhiana")
    run = stretch(st.button, "Load climate dashboard", type="primary")
    refresh = stretch(st.button, "🔄 Refresh Data", help="Discard cached results and fetch everything again.")
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- Open-Meteo Weather API\n- Open-Meteo Air Quality API\n- Open-Meteo Historical Archive")
    st.caption("Internet connection is required for live data. API access is free for non-commercial use subject to provider terms.")
    st.caption("Results are cached (weather/air 10 min, history 6 h) to avoid repeated API calls. "
               "Failed requests are retried up to 3 times with exponential backoff.")

if "loaded_city" not in st.session_state:
    st.session_state.loaded_city = None

if run or refresh or st.session_state.loaded_city is None:
    ok = load_dashboard(city_input.strip() or "Ludhiana", force=bool(refresh))
    if not ok:
        render_issues()
        st.info("The dashboard could not be loaded. Check your internet connection and try a city name such as "
                "Ludhiana, Chandigarh, Delhi or Mumbai, then press **Load climate dashboard** or **🔄 Refresh Data**.")
        st.stop()

place = st.session_state.loaded_city
weather = st.session_state.get("weather")
air = st.session_state.get("air")
history = st.session_state.get("history")
last_load = st.session_state.get("last_load", "N/A")
cur = (weather or {}).get("current") or {}
aq = ((air or {}).get("current") or {})
aq_has_data = any(v is not None for k, v in aq.items() if k != "time" and k != "interval")

daily = None
if weather is not None:
    daily = pd.DataFrame(weather["daily"])
    daily["date"] = pd.to_datetime(daily["time"])
hist = None
if history is not None:
    hist = pd.DataFrame(history["daily"])
    hist["date"] = pd.to_datetime(hist["time"])
    if hist.empty:
        hist = None

city_name = f'{place.get("name")}, {place.get("country", "")}'

st.subheader(f"Climate overview — {city_name}")
st.caption(f"Coordinates: {place['latitude']:.3f}, {place['longitude']:.3f} | Last refreshed: {last_load} | Local time: {cur.get('time','N/A')}")
render_issues()

m1, m2, m3, m4, m5 = st.columns(5)
feels = to_float(cur.get("apparent_temperature"))
m1.metric("Temperature", fmt(cur.get("temperature_2m"), "°C"),
          f"Feels like {feels} °C" if feels is not None else None)
m2.metric("Humidity", fmt(cur.get("relative_humidity_2m"), "%"))
m3.metric("Rainfall now", fmt(cur.get("precipitation"), "mm"))
m4.metric("Wind speed", fmt(cur.get("wind_speed_10m"), "km/h"))
m5.metric("US AQI", f"{aq.get('us_aqi') if aq.get('us_aqi') is not None else 'N/A'}")

if weather is not None:
    st.info(f"Current conditions: **{weather_label(cur.get('weather_code', 0))}**. These are live weather observations/estimates from the selected API grid point, not a direct sensor at your exact address.")
else:
    st.warning("Live weather data is currently unavailable. Other sections (historical analytics, air quality) are shown where possible.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Weather", "Climate Analytics", "AI & Risk", "Air Quality", "Saved Reports"])

with tab1:
    if daily is None:
        st.warning("Weather forecast is unavailable right now (see the issue details above). "
                   "Press **🔄 Refresh Data** to try again.")
    else:
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown("### 7-day forecast")
            fig = px.line(daily, x="date", y=["temperature_2m_max", "temperature_2m_min"],
                          markers=True, labels={"value":"Temperature (°C)", "date":"Date", "variable":"Metric"},
                          title="Daily maximum and minimum temperature")
            stretch(st.plotly_chart, fig)
            fig2 = px.bar(daily, x="date", y="precipitation_sum", labels={"precipitation_sum":"Rainfall (mm)", "date":"Date"},
                          title="Forecast daily precipitation")
            stretch(st.plotly_chart, fig2)
        with right:
            st.markdown("### Forecast details")
            view = daily[["time","temperature_2m_max","temperature_2m_min","precipitation_sum","precipitation_probability_max"]].copy()
            view.columns = ["Date","Max °C","Min °C","Rain mm","Rain chance %"]
            stretch(st.dataframe, view, hide_index=True)
            st.markdown("### Current conditions")
            st.write(f"**Cloud cover:** {cur.get('cloud_cover','N/A')}%")
            st.write(f"**Pressure:** {cur.get('pressure_msl','N/A')} hPa")
            st.write(f"**Wind direction:** {cur.get('wind_direction_10m','N/A')}°")

with tab2:
    st.markdown("### Recent climate pattern (historical archive)")
    if hist is None:
        st.info("Historical climate data is currently unavailable, so the 90-day analytics cannot be shown. "
                "Live weather, risk screening and air-quality sections still work where their data loaded. "
                "Press **🔄 Refresh Data** to try again.")
    else:
        h1, h2, h3 = st.columns(3)
        mean_max = to_float(hist["temperature_2m_max"].mean())
        h1.metric("90-day mean max temp", f"{mean_max:.1f} °C" if mean_max is not None else "N/A")
        h2.metric("90-day total rainfall", f"{hist['precipitation_sum'].sum():.1f} mm")
        h3.metric("Historical records", f"{len(hist)} days")
        fig = px.line(hist, x="date", y=["temperature_2m_max","temperature_2m_min"],
                      labels={"value":"Temperature (°C)", "date":"Date", "variable":"Metric"},
                      title="Historical daily temperature")
        stretch(st.plotly_chart, fig)
        fig = px.bar(hist, x="date", y="precipitation_sum",
                     labels={"precipitation_sum":"Rainfall (mm)", "date":"Date"},
                     title="Historical daily rainfall")
        stretch(st.plotly_chart, fig)
        recent_mean = hist.tail(30)["temperature_2m_max"].mean()
        earlier_mean = hist.iloc[:-30].head(60)["temperature_2m_max"].mean() if len(hist) > 30 else np.nan
        anomaly = recent_mean - earlier_mean
        if np.isnan(anomaly):
            st.metric("Recent 30-day max-temperature difference vs preceding 60 days", "N/A")
        else:
            st.metric("Recent 30-day max-temperature difference vs preceding 60 days", f"{anomaly:+.2f} °C")
        st.caption("This is a short-period comparison, not a formal long-term climate-change attribution or 30-year climate normal.")

with tab3:
    st.markdown("### Environmental risk screening")
    tomorrow = None
    if daily is not None and len(daily) > 0:
        tomorrow = daily.iloc[1] if len(daily) > 1 else daily.iloc[0]
    risks = make_risk(to_float(cur.get("temperature_2m")),
                      to_float(tomorrow.get("precipitation_probability_max")) if tomorrow is not None else None,
                      to_float(aq.get("us_aqi")))
    for name, level, detail in risks:
        color = {"Low":"🟢","Moderate":"🟡","High":"🔴"}.get(level, "⚪")
        with st.container(border=True):
            st.markdown(f"**{color} {name}: {level}**")
            st.write(detail)
    st.markdown("### Baseline ML rainfall model")
    if history is None:
        st.info("The ML rainfall model needs historical data, which is currently unavailable.")
    else:
        model, message = get_rain_model(history, st.session_state.get("fetched_at", {}).get("history", last_load))
        st.write(message)
        if model is not None:
            if tomorrow is None:
                st.info("Tomorrow's forecast is unavailable, so the model cannot produce a rainfall estimate right now.")
            else:
                feats = {
                    "temperature_2m_max": to_float(tomorrow.get("temperature_2m_max")),
                    "temperature_2m_min": to_float(tomorrow.get("temperature_2m_min")),
                    "wind_speed_10m_max": to_float(tomorrow.get("wind_speed_10m_max")),
                    "month": pd.to_datetime(tomorrow["time"]).month,
                    "day_of_year": pd.to_datetime(tomorrow["time"]).dayofyear,
                }
                if any(v is None for v in feats.values()):
                    st.info("Tomorrow's forecast has missing values, so the model estimate is skipped.")
                else:
                    prob = model.predict_proba(pd.DataFrame([feats]))[0][1] * 100
                    st.metric("ML estimated probability of measurable rain tomorrow", f"{prob:.1f}%")
                    st.caption("Educational baseline model trained on the fetched recent historical data. Not an official warning or a substitute for meteorological forecasts.")
    st.markdown("### Tomorrow's API forecast")
    if tomorrow is None:
        st.info("Tomorrow's forecast is unavailable because the weather API could not be reached.")
    else:
        st.write(f"Maximum temperature: **{tomorrow['temperature_2m_max']} °C**")
        st.write(f"Rain probability: **{tomorrow['precipitation_probability_max']}%**")

with tab4:
    st.markdown("### Air quality details")
    if air is None:
        st.warning("Air-quality data could not be loaded (see the issue details above). "
                   "Weather and climate analytics are unaffected. Press **🔄 Refresh Data** to retry.")
    elif aq_has_data:
        cols = st.columns(5)
        for col, key, label, unit in zip(cols, ["pm2_5","pm10","nitrogen_dioxide","ozone","carbon_monoxide"],
                                          ["PM2.5","PM10","NO₂","O₃","CO"], ["µg/m³","µg/m³","µg/m³","µg/m³","µg/m³"]):
            val = aq.get(key)
            col.metric(label, f"{val} {unit}" if val is not None else "N/A")
        st.caption("AQI shown above is US AQI. Do not compare it directly with India's CPCB AQI categories; the scales and breakpoints differ.")
        air_hourly = pd.DataFrame(air.get("hourly") or {})
        if "time" in air_hourly and "us_aqi" in air_hourly:
            air_hourly["time"] = pd.to_datetime(air_hourly["time"])
            fig = px.line(air_hourly, x="time", y="us_aqi", title="US AQI forecast / hourly series")
            stretch(st.plotly_chart, fig)
    else:
        st.warning("Air quality data is not available for this location from the selected source.")

with tab5:
    st.markdown("### Saved refresh history")
    try:
        conn = db_connection()
        saved = pd.read_sql_query("SELECT captured_at, city, country, temperature, humidity, precipitation, wind_speed, pm2_5, pm10, us_aqi FROM snapshots ORDER BY id DESC LIMIT 100", conn)
    except Exception as exc:
        saved = pd.DataFrame()
        st.warning(f"Saved snapshots could not be read: {exc}")
    if saved.empty:
        st.info("No saved snapshots yet.")
    else:
        stretch(st.dataframe, saved, hide_index=True)
        csv = saved.to_csv(index=False).encode("utf-8")
        st.download_button("Download snapshots as CSV", csv, "climatesense_snapshots.csv", "text/csv")
    report = pd.DataFrame([{
        "City": city_name, "Fetched at": last_load,
        "Temperature C": cur.get("temperature_2m"), "Humidity %": cur.get("relative_humidity_2m"),
        "Rainfall mm": cur.get("precipitation"), "Wind km/h": cur.get("wind_speed_10m"),
        "US AQI": aq.get("us_aqi"),
        "90-day rainfall mm": round(hist["precipitation_sum"].sum(), 2) if hist is not None else None,
        "90-day mean max temp C": round(hist["temperature_2m_max"].mean(), 2) if hist is not None else None
    }])
    st.download_button("Download current climate summary CSV", report.to_csv(index=False).encode("utf-8"),
                       "climatesense_summary.csv", "text/csv")

st.divider()
st.caption("ClimateSense AI | Academic prototype. Data availability and update frequency depend on the external API. Risk indicators are educational screening rules, not official emergency alerts.")
