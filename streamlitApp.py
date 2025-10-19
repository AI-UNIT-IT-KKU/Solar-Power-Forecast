# app.py — Enhanced Solar Forecast UI with improved features
# requirements: streamlit, requests, pandas, numpy, xgboost, joblib, pvlib, matplotlib, scikit-learn

import streamlit as st
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import pvlib
from datetime import date, timedelta
from typing import Tuple, Optional

# -------------------- Configuration --------------------
MODEL_PATH = "models/xgb_ig_model.json"
SCALER_PATH = "models/scaler_ig.pkl"

FEATS = [
    "Total Solar Irradiance on Horizontal Plane GHI(W/m2)",
    "Total Solar Irradiance on Inclined Plane POA1(W/m2)",
    "Total Solar Irradiance on Inclined Plane POA2(W/m2)",
    "Module Surface Temperature1(degree centigrade)",
    "Module Surface Temperature2(degree centigrade)",
    "Soiling Loss Index Isc(%)",
    "Soiling Loss Index Geff(%)",
    "Isc Test(Amp)",
    "Isc Ref(Amp)",
    "Geff Test(W/M2)",
    "Geff Reference(W/M2)",
    "Temperature Reference Cell(Deg C)",
]

# -------------------- UI Configuration --------------------
st.set_page_config(
    page_title="Solar Forecast MVP",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.title("☀️ Solar Power Forecast System")
st.markdown("Advanced solar irradiance prediction with machine learning")

# -------------------- Sidebar Configuration --------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📍 Location")
    lat = st.number_input("Latitude", value=24.7136, format="%.6f", 
                         help="Decimal degrees (e.g., 24.7136 for Riyadh)")
    lon = st.number_input("Longitude", value=46.6753, format="%.6f",
                         help="Decimal degrees (e.g., 46.6753 for Riyadh)")
    tz = st.text_input("Timezone", value="Asia/Riyadh",
                      help="IANA timezone name")
    
    st.subheader("📅 Forecast Period")
    start_date = st.date_input("Start Date", value=date.today())
    horizon_days = st.number_input("Forecast Days", min_value=1, max_value=16, value=7,
                                   help="Maximum 16 days for forecast")
    
    st.subheader("🔧 Panel Configuration")
    tilt = st.number_input("Panel Tilt (°)", value=25.0, min_value=0.0, max_value=90.0,
                          help="Angle from horizontal (0°=flat, 90°=vertical)")
    azim = st.number_input("Panel Azimuth (°)", value=180.0, min_value=0.0, max_value=360.0,
                          help="0°=North, 90°=East, 180°=South, 270°=West")
    noct = st.number_input("NOCT (°C)", value=45.0, min_value=30.0, max_value=60.0,
                          help="Nominal Operating Cell Temperature")
    
    st.subheader("🎛️ Processing Options")
    use_minutely = st.checkbox("Upsample to 1-minute resolution", value=True,
                              help="Increases temporal resolution for better accuracy")
    smooth_hours = st.slider("Smoothing Window (hours)", 0, 6, 3,
                            help="Rolling average to reduce noise")
    show_daily_energy = st.checkbox("Show Daily Energy Chart", value=True)
    show_weather = st.checkbox("Show Weather Data", value=False)

# -------------------- Caching Functions --------------------
@st.cache_resource(show_spinner=False)
def load_model_and_scaler(model_path: str, scaler_path: str) -> Tuple:
    """Load XGBoost model and scaler with error handling"""
    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
        scaler = joblib.load(scaler_path)
        feats_fit = list(getattr(scaler, "feature_names_in_", [])) or FEATS
        return booster, scaler, feats_fit
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure {model_path} and {scaler_path} exist.")
        raise e
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_any(lat: float, lon: float, start_date: date, end_date: date, tz: str) -> pd.DataFrame:
    """Fetch weather data with improved error handling and caching"""
    today = date.today()
    
    # Adjust end_date to be exclusive (subtract 1 day)
    end_date = end_date - timedelta(days=1)
    
    # Determine API endpoint
    if start_date >= today:
        max_end = start_date + timedelta(days=15)  # 16 days total including start
        if end_date > max_end:
            end_date = max_end
            st.warning(f"Forecast horizon limited to 16 days. End date adjusted to {end_date}")
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "shortwave_radiation,temperature_2m,windspeed_10m,relative_humidity_2m,cloudcover",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": tz
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        if "hourly" not in data:
            raise ValueError(f"Invalid API response structure: {list(data.keys())}")
        
        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        
        return df
        
    except requests.Timeout:
        raise RuntimeError("Request timed out. Please try again.")
    except requests.HTTPError as e:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    except Exception as e:
        raise RuntimeError(f"Weather data fetch failed: {str(e)}")

# -------------------- Processing Functions --------------------
def ghi_to_poa(df_api: pd.DataFrame, lat: float, lon: float, tz: str, 
               tilt: float, azim: float) -> pd.DataFrame:
    """Convert GHI to POA with proper timezone handling"""
    df = df_api.copy()
    
    # Ensure timezone-aware index
    if df.index.tz is None:
        df = df.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:
        df = df.tz_convert(tz)
    
    # Get GHI and solar position
    ghi = df["shortwave_radiation"].astype(float).rename("ghi")
    solpos = pvlib.solarposition.get_solarposition(
        time=df.index, latitude=lat, longitude=lon
    )
    app_zen = solpos["apparent_zenith"].clip(0, 90)
    
    # Decompose GHI into DNI and DHI using Erbs model
    decomp = pvlib.irradiance.erbs(
        ghi=ghi, zenith=app_zen, datetime_or_doy=df.index
    )
    dhi = decomp["dhi"].clip(lower=0)
    dni = decomp["dni"].clip(lower=0)
    
    # Transpose to POA using Perez model
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azim,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solpos["zenith"], 
        solar_azimuth=solpos["azimuth"]
    )
    
    # Create output dataframe
    out = pd.DataFrame({
        "ghi": ghi,
        "poa1": poa["poa_global"].clip(lower=0),
        "poa2": poa["poa_global"].clip(lower=0),
        "temp2m": df["temperature_2m"].astype(float),
        "wind10m": df.get("windspeed_10m", pd.Series(index=df.index, data=np.nan)).astype(float),
        "rh": df.get("relative_humidity_2m", pd.Series(index=df.index, data=np.nan)).astype(float),
    })
    
    return out

def upsample_to_minute(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Upsample hourly data to 1-minute resolution with interpolation"""
    df_min = (df_hourly
              .resample("T")
              .interpolate(method="time")
              .bfill()
              .ffill())
    return df_min

def build_feature_matrix(df_min: pd.DataFrame, noct: float, feats: list) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix for model prediction"""
    # Calculate module temperature from NOCT model
    t_amb = df_min["temp2m"]
    t_mod = t_amb + (noct - 20.0) / 800.0 * df_min["poa1"]
    
    # Create feature matrix
    X = pd.DataFrame({
        feats[0]: df_min["ghi"].values,
        feats[1]: df_min["poa1"].values,
        feats[2]: df_min["poa2"].values,
        feats[3]: t_mod.values,
        feats[4]: t_mod.values,
        feats[5]: np.zeros(len(df_min)),
        feats[6]: np.zeros(len(df_min)),
        feats[7]: np.ones(len(df_min)),
        feats[8]: np.ones(len(df_min)),
        feats[9]: df_min["poa1"].values,
        feats[10]: df_min["poa1"].values,
        feats[11]: df_min["temp2m"].values,
    }, index=df_min.index)
    
    # Zero out nighttime irradiance values
    day_mask = df_min["ghi"] > 1.0
    irrad_cols = [feats[0], feats[1], feats[2], feats[9], feats[10]]
    X.loc[~day_mask, irrad_cols] = 0.0
    
    return X, day_mask

def create_forecast_plot(pred: pd.Series, pred_smooth: pd.Series, 
                        daily_mean: pd.Series, smooth_hours: int) -> plt.Figure:
    """Create enhanced forecast visualization"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot raw and smoothed forecasts
    ax.plot(pred.index, pred, label="Raw Forecast", alpha=0.3, linewidth=1, color='#3498db')
    ax.plot(pred_smooth.index, pred_smooth, label=f"Smoothed ({smooth_hours}h window)", 
            linewidth=2, color='#e74c3c')
    
    # Plot daily averages at noon
    daily_mean_noon_idx = daily_mean.index + pd.Timedelta(hours=12)
    ax.plot(daily_mean_noon_idx, daily_mean, "o-", label="Daily Average", 
            markersize=8, linewidth=2, color='#2ecc71')
    
    ax.set_title("Solar Power Forecast", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Predicted Power (W)", fontsize=12)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig

def create_energy_bar_chart(daily_energy_kwh: pd.Series) -> plt.Figure:
    """Create daily energy bar chart"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    bars = ax.bar(daily_energy_kwh.index, daily_energy_kwh.values, 
                  width=0.8, align="center", color='#f39c12', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_title("Daily Energy Production", fontsize=16, fontweight='bold')
    ax.set_ylabel("Energy (kWh)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

# -------------------- Main Application --------------------
def main():
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        run_forecast = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
    
    if not run_forecast:
        # Show welcome message and instructions
        st.info("👈 Configure your parameters in the sidebar and click 'Run Forecast' to begin")
        
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            This application provides advanced solar power forecasting using:
            - **Machine Learning**: XGBoost model trained on historical data
            - **Weather API**: Real-time and forecast weather data from Open-Meteo
            - **PV Modeling**: POA (Plane of Array) irradiance calculations using pvlib
            - **High Resolution**: Optional 1-minute temporal resolution
            
            **Use Cases:**
            - Solar plant operators planning generation
            - Grid operators balancing supply and demand
            - Energy traders optimizing market participation
            """)
        
        return
    
    # Run forecast workflow
    try:
        # Calculate end date
        end_date = start_date + timedelta(days=int(horizon_days))
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch weather data
        status_text.text("📡 Fetching weather data...")
        progress_bar.progress(20)
        df_api = fetch_weather_any(lat, lon, start_date, end_date, tz)
        
        if show_weather:
            with st.expander("🌤️ Raw Weather Data"):
                st.dataframe(df_api.head(24), use_container_width=True)
        
        # Step 2: Compute POA
        status_text.text("☀️ Computing plane-of-array irradiance...")
        progress_bar.progress(40)
        df_hourly = ghi_to_poa(df_api, lat, lon, tz, tilt, azim)
        
        # Step 3: Resample
        status_text.text("⏱️ Resampling to target resolution...")
        progress_bar.progress(50)
        if use_minutely:
            df_input = upsample_to_minute(df_hourly)
        else:
            # Keep hourly but resample to ensure exactly 24 hours per day
            df_input = df_hourly.resample("H").mean().ffill().bfill()
        
        # Step 4: Build features
        status_text.text("🔧 Building feature matrix...")
        progress_bar.progress(60)
        X_future_raw, day_mask = build_feature_matrix(df_input, noct, FEATS)
        
        # Step 5: Load model and predict
        status_text.text("🤖 Loading model and generating predictions...")
        progress_bar.progress(75)
        booster, scaler, feats_fit = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        
        # Ensure column order matches training
        X_future_raw = X_future_raw.reindex(columns=feats_fit).astype(float)
        
        # Scale features
        X_scaled = pd.DataFrame(
            scaler.transform(X_future_raw),
            index=X_future_raw.index,
            columns=feats_fit
        )
        
        # Generate predictions
        dmat = xgb.DMatrix(X_scaled, feature_names=feats_fit)
        yhat = booster.predict(dmat)
        
        # Post-process predictions
        pred = pd.Series(yhat, index=X_scaled.index, name="forecast")
        pred[~day_mask] = 0.0  # Zero out nighttime
        pred = pred.clip(lower=0)  # Ensure non-negative
        
        # Apply smoothing
        if smooth_hours > 0:
            pred_smooth = pred.rolling(f"{smooth_hours}h", min_periods=1).mean()
        else:
            pred_smooth = pred
        
        # Calculate daily statistics
        daily_mean = pred_smooth.resample("D").mean()
        daily_max = pred_smooth.resample("D").max()
        
        # Calculate daily energy (kWh)
        if use_minutely:
            # For minute data: sum minutes, divide by 60 (to get Wh), divide by 1000 (to get kWh)
            daily_energy_kwh = pred.resample("D").sum() / 60.0 / 1000.0
        else:
            # For hourly data: sum hours, divide by 1000 (to get kWh)
            daily_energy_kwh = pred.resample("D").sum() / 1000.0
        
        # Get exactly 24 hourly values per day for display
        pred_hourly_display = pred.resample("H").mean()
        pred_smooth_hourly = pred_smooth.resample("H").mean()
        
        # Complete progress
        status_text.text("✅ Forecast complete!")
        progress_bar.progress(100)
        
        # Display summary metrics
        st.success("Forecast generated successfully!")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Energy", f"{daily_energy_kwh.sum():.1f} kWh")
        with metric_cols[1]:
            st.metric("Peak Power", f"{pred_smooth.max():.0f} W")
        with metric_cols[2]:
            st.metric("Avg Daily Energy", f"{daily_energy_kwh.mean():.1f} kWh")
        with metric_cols[3]:
            st.metric("Days Forecasted", len(daily_energy_kwh))
        
        # Display forecast plot
        st.subheader("📊 Power Forecast")
        fig1 = create_forecast_plot(pred, pred_smooth, daily_mean, smooth_hours)
        st.pyplot(fig1)
        
        # Display energy chart
        if show_daily_energy:
            st.subheader("⚡ Daily Energy Production")
            fig2 = create_energy_bar_chart(daily_energy_kwh)
            st.pyplot(fig2)
        
        # Display data tables
        tab1, tab2, tab3 = st.tabs(["📅 Daily Summary", "📈 Hourly Data", "📋 Statistics"])
        
        with tab1:
            daily_summary = pd.DataFrame({
                "Mean Power (W)": daily_mean.round(2),
                "Peak Power (W)": daily_max.round(2),
                "Energy (kWh)": daily_energy_kwh.round(2)
            })
            st.dataframe(daily_summary, use_container_width=True)
        
        with tab2:
            # Display exactly 24 hours per day
            hourly_display = pd.DataFrame({
                "Time": pred_hourly_display.index.strftime("%Y-%m-%d %H:%M"),
                "Power (W)": pred_hourly_display.round(2)
            })
            st.dataframe(hourly_display, use_container_width=True)
            st.caption(f"Total records: {len(hourly_display)} (24 hours × {len(daily_energy_kwh)} days)")
        
        with tab3:
            st.markdown("**Overall Statistics**")
            stats_data = {
                "Metric": ["Total Energy", "Mean Power", "Peak Power", "Capacity Factor"],
                "Value": [
                    f"{daily_energy_kwh.sum():.2f} kWh",
                    f"{pred_smooth.mean():.2f} W",
                    f"{pred_smooth.max():.2f} W",
                    f"{(pred_smooth.mean() / pred_smooth.max() * 100):.1f}%"
                ]
            }
            st.table(pd.DataFrame(stats_data))
        
        # Download section
        st.subheader("💾 Download Results")
        
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            # Download hourly data (exactly 24 per day)
            hourly_csv = pd.DataFrame({
                "time": pred_hourly_display.index.astype(str),
                "forecast_W": pred_hourly_display.values
            }).to_csv(index=False)
            st.download_button(
                "📄 Hourly Data (CSV)",
                data=hourly_csv,
                file_name=f"forecast_hourly_{start_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with dl_col2:
            daily_csv = daily_summary.reset_index().rename(
                columns={"index": "date"}
            ).to_csv(index=False)
            st.download_button(
                "📊 Daily Summary (CSV)",
                data=daily_csv,
                file_name=f"forecast_daily_{start_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with dl_col3:
            energy_csv = daily_energy_kwh.rename("kWh").to_frame().reset_index().rename(
                columns={"index": "date"}
            ).to_csv(index=False)
            st.download_button(
                "⚡ Daily Energy (CSV)",
                data=energy_csv,
                file_name=f"forecast_energy_{start_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
        st.stop()

# -------------------- Run Application --------------------
if __name__ == "__main__":
    main()