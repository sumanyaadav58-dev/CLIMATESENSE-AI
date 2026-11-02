# ClimateSense AI – Live Climate Impact & Environmental Risk Analytics

A runnable MCA-level academic prototype built with Streamlit, Python, Plotly, SQLite and scikit-learn.

## What it includes

- City search using Open-Meteo Geocoding.
- Current weather conditions and 7-day forecast.
- Air-quality values and US AQI where the source has coverage.
- Recent 90-day historical weather analysis.
- Temperature and rainfall charts.
- Short-period temperature comparison.
- Rule-based heat, rain and air-quality risk screening.
- A baseline Random Forest rainfall classifier trained on retrieved historical weather.
- SQLite storage of refresh snapshots.
- CSV downloads for saved snapshots and current summary.
- **Resilient API layer (v1.1):** 20 s connect timeout, 90 s read timeout, up to 3 attempts with exponential backoff (2 s, 4 s), automatic IPv4 fallback, and clear error messages that say whether a failure is an internet problem, an API availability problem, a rejected request or an invalid response.
- **Partial-failure tolerance:** if one API fails the rest of the dashboard keeps working (e.g. air-quality down -> weather and climate analytics still shown; no history -> an informative message instead of a crash). Last good data is kept and labelled if a refresh fails.
- **🔄 Refresh Data** button and result caching (weather/air 10 min, history 6 h, city search 24 h).

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (UI, analytics, ML, SQLite) |
| `api_client.py` | Timeouts, retries, backoff, error classification, Open-Meteo request functions |
| `tests/test_api_client.py` | Offline unit tests for the API layer (no internet needed) |
| `requirements.txt` | Python dependencies |
| `run_mac_linux.sh` / `run_windows.bat` | One-command launchers |
| `.vscode/` | VS Code interpreter + debug configuration |

## Requirements

- Python 3.10 or newer (3.9 works but 3.10+ is recommended)
- Internet connection (the app fetches data from Open-Meteo APIs)
- macOS, Linux or Windows

## Run on macOS

1. Unzip the project.
2. Open Terminal.
3. Move into the extracted folder, for example:

   `cd ~/Downloads/ClimateSense_AI`

4. Run:

   `chmod +x run_mac_linux.sh`

   `./run_mac_linux.sh`

5. Your browser should open. If not, open:

   `http://localhost:8501`

## Run on macOS from VS Code (step by step)

```bash
# 1. In Terminal
cd ~/Downloads/ClimateSense_AI
code .                       # opens the folder in VS Code
```
Then in the VS Code terminal (Terminal > New Terminal):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v   # optional: offline self-test (19 tests)
streamlit run app.py
```
In VS Code press `Cmd+Shift+P` -> **Python: Select Interpreter** -> choose `.venv`. You can also press **F5** (the "ClimateSense AI (Streamlit)" debug configuration is included).
If `code .` is not found: VS Code -> `Cmd+Shift+P` -> *Shell Command: Install 'code' command in PATH*.
The dashboard opens at `http://localhost:8501`. Stop it with `Ctrl+C`.

## Run on Windows

Double-click `run_windows.bat`, or open Command Prompt in the extracted folder and run it.

Then open `http://localhost:8501` if the browser does not open automatically.

## Run manually (all platforms)

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

- Open-Meteo Geocoding: https://geocoding-api.open-meteo.com/
- Open-Meteo Forecast: https://open-meteo.com/
- Open-Meteo Air Quality: https://air-quality-api.open-meteo.com/
- Open-Meteo Historical Archive: https://archive-api.open-meteo.com/

Review the providers' current terms and attribution requirements before public or commercial deployment.

## Important academic limitations

- Live weather is API-provided grid-point data, not a local physical sensor reading.
- Air-quality coverage varies by location. The displayed AQI is US AQI and is not the Indian CPCB AQI scale.
- The rainfall classifier is an educational baseline, not an operational weather forecast. Its training data is limited to the fetched recent historical period.
- Risk categories are simple screening thresholds for demonstration. They are not official warnings, medical advice, flood forecasts or crop advisories.
- A short 90-day comparison is not sufficient to establish long-term climate change. A research-grade climate anomaly analysis should use consistent 30-year climate normals and quality-controlled observations.

## How API failures are handled

| Situation | What the app does |
|---|---|
| Cannot connect (offline, DNS, VPN, firewall, broken IPv6) | Retries up to 3 times (wait 2 s, then 4 s), switches to IPv4 after the first failure, then shows **Internet connectivity problem** |
| Server accepts connection but is slow (read timeout > 90 s), HTTP 5xx, HTTP 429 | Retries with backoff, then shows **API availability problem** |
| HTTP 400/404 or `{"error": true}` | Not retried; shows **Request failure** with the API's reason |
| Empty / invalid JSON or missing fields | Retried (invalid JSON) then shows **Invalid API response** |
| Air-quality API fails | Air-quality tab shows a message; weather, climate analytics and risk screening continue (air risk shown as "Unknown") |
| Historical API fails | Climate Analytics and ML sections show an informative message; everything else continues |
| Weather API fails | Historical analytics and air quality continue |
| City search fails | Shows the error; keeps the previously loaded dashboard if there is one |

Tune the behaviour in `api_client.py` (`CONNECT_TIMEOUT`, `READ_TIMEOUT`, `MAX_ATTEMPTS`, `BACKOFF_BASE_SECONDS`) and the cache lifetimes at the top of `app.py` (`TTL_*`). Set `CLIMATESENSE_FORCE_IPV4=1` before launching to use IPv4 from the first attempt:

```bash
CLIMATESENSE_FORCE_IPV4=1 streamlit run app.py
```

## Troubleshooting

- **`ConnectTimeoutError ... api.open-meteo.com`**: your machine cannot open a connection to the API. Test in Terminal: `curl -v "https://api.open-meteo.com/v1/forecast?latitude=30.9&longitude=75.85&current=temperature_2m"`. If curl also hangs, the cause is your network (turn off VPN/proxy, switch Wi-Fi or use a mobile hotspot, try DNS 1.1.1.1 or 8.8.8.8, and check firewall/antivirus). If `curl -4` works but plain `curl` hangs, IPv6 is broken on your network - the app's IPv4 fallback handles this automatically.
- **SSL / certificate errors on macOS (python.org Python)**: run `/Applications/Python 3.x/Install Certificates.command`.
- After the connection recovers, press **🔄 Refresh Data** in the sidebar.

- If city search fails, try a larger nearby city or check internet connectivity.
- If air quality is unavailable, the selected provider may not have coverage for that location.
- If package installation fails, update Python and pip, then retry.
- The first launch can take several minutes while packages install.
