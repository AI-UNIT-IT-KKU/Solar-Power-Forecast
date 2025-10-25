# =============================================================================
# FILE 1: api.py — FastAPI Backend
# =============================================================================
"""
FastAPI backend for solar power forecasting
Run with: uvicorn api:app --reload --port 8000
Requirements: fastapi, uvicorn, pydantic, requests, pandas, numpy, xgboost, joblib, pvlib
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime, timedelta
import requests
import pandas as pd
import numpy as np
# The original implementation used XGBoost; this has been replaced
# with a PyTorch-based CNN/LSTM model.  Torch is an optional
# dependency and must be installed in the runtime environment.
import torch
import torch.nn as nn
import joblib
import pvlib
import os

# -----------------------------------------------------------------------------
# Model definition: CNN + LSTM regression network
#
# The training script saved only the model's ``state_dict`` and metadata.  To
# reconstruct the model at runtime we need its architecture.  This class
# implements the architecture used during training.  See the training code for
# details.
class CNNLSTMReg(nn.Module):
    """A convolutional plus LSTM model for time‑series regression.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    conv_channels : int, optional
        Number of convolutional filters in the Conv1d layers.  Defaults to 64.
    kernel_size : int, optional
        Size of the convolutional kernel.  Defaults to 3.
    pool : int, optional
        Pooling size for the MaxPool1d layer.  Defaults to 2.
    lstm_hidden : int, optional
        Number of hidden units in the LSTM.  Defaults to 64.
    dropout : float, optional
        Dropout rate applied before the final linear layer.  Defaults to 0.2.
    """
    def __init__(
        self,
        n_features: int,
        conv_channels: int = 64,
        kernel_size: int = 3,
        pool: int = 2,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Convolutional layers expect (B, C, L).  We treat the features as
        # channels and the sequence length as L.  ``pad`` approximates
        # ``padding='same'`` behaviour.
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=pool)
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.relu2 = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (B, W, F).  Transpose to (B, F, W) for Conv1d.
        z = x.transpose(1, 2)
        z = self.relu1(self.conv1(z))
        z = self.pool1(z)
        z = self.relu2(self.conv2(z))
        # Back to (B, W', C) for LSTM
        z = z.transpose(1, 2)
        out, _ = self.lstm(z)
        last = out[:, -1, :]
        last = self.do(last)
        return self.fc(last).squeeze(-1)

# -------------------- Configuration --------------------
# Update paths to the CNN‑LSTM model and scalers.  These should point
# to the files provided in your environment.  Absolute paths are
# recommended if the working directory is uncertain.
MODEL_PATH = "models/CNN_LSTM_IG120.pt"
X_SCALER_PATH = "models/x_scaler.pkl"
Y_SCALER_PATH = "models/y_scaler.pkl"

# Feature list used during training of the neural network.  The order
# must match exactly the columns expected by the x‑scaler.  If the
# scaler exposes feature_names_in_ this list will be overwritten in
# code below.  See ``features_ig`` provided by the user.
FEATS = [
    "Total Solar Irradiance on Horizontal Plane GHI(Wh/m2)",
    "Soiling Loss Index Isc(%)",
    "Soiling Loss Index Geff(%)",
    "Total Solar Irradiance on Horizontal Plane GHI(W/m2)",
    "Module Surface Temperature2(degree centigrade)",
    "Module Surface Temperature1(degree centigrade)",
    "Temperature Reference Cell(Deg C)",
    "Temperature Test(Deg C)",
    "Ambient Temp.(degree centigrade)",
    "Relative Humidity(%)",
]

# -------------------- FastAPI App --------------------
app = FastAPI(
    title="Solar Forecast API",
    description="API for solar power forecasting using machine learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Models --------------------
class ForecastRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    horizon_days: int = Field(..., ge=1, le=16, description="Number of days to forecast")
    timezone: str = Field(default="UTC", description="IANA timezone name")
    tilt: float = Field(default=25.0, ge=0, le=90, description="Panel tilt angle in degrees")
    azimuth: float = Field(default=180.0, ge=0, le=360, description="Panel azimuth in degrees")
    noct: float = Field(default=45.0, ge=30, le=60, description="NOCT temperature in °C")
    use_minutely: bool = Field(default=True, description="Upsample to 1-minute resolution")
    smooth_hours: int = Field(default=3, ge=0, le=6, description="Smoothing window in hours")

class HourlyDataPoint(BaseModel):
    timestamp: str
    power_w: float

class DailyDataPoint(BaseModel):
    date: str
    mean_power_w: float
    peak_power_w: float
    energy_kwh: float

class ForecastResponse(BaseModel):
    status: str
    total_energy_kwh: float
    peak_power_w: float
    avg_daily_energy_kwh: float
    days_forecasted: int
    hourly_data: List[HourlyDataPoint]
    daily_data: List[DailyDataPoint]

# -------------------- Load Model --------------------
# Attempt to load the PyTorch model state and reconstruct the CNN‑LSTM
# architecture along with associated scalers.  The saved model file is
# expected to be a checkpoint dictionary with keys ``state_dict``,
# ``n_features``, ``window``, ``features`` and ``scalers``.  See
# training script for details.
try:
    # Load the checkpoint (dict) saved during training
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Extract metadata
        n_features_ckpt = checkpoint.get("n_features")
        window_size_ckpt = checkpoint.get("window", 1)
        feats_from_ckpt = checkpoint.get("features", FEATS)
        scalers_info = checkpoint.get("scalers", {})
        # Instantiate the model with same parameters used in training.
        model_loaded_inst = CNNLSTMReg(n_features=n_features_ckpt)
        model_loaded_inst.load_state_dict(checkpoint["state_dict"])
        model_loaded_inst.eval()
        booster = model_loaded_inst
        # Use feature order from checkpoint
        feats_fit = feats_from_ckpt
        # Determine scaler paths from checkpoint.  Resolve relative paths
        # relative to the directory containing the model checkpoint.  If
        # paths are absolute or missing, fall back to the configured
        # defaults.
        base_dir = os.path.dirname(MODEL_PATH)
        x_scaler_path_ck = scalers_info.get("x_scaler_path")
        y_scaler_path_ck = scalers_info.get("y_scaler_path")
        if x_scaler_path_ck:
            x_scaler_path_full = x_scaler_path_ck if os.path.isabs(x_scaler_path_ck) else os.path.join(base_dir, x_scaler_path_ck)
        else:
            x_scaler_path_full = X_SCALER_PATH
        if y_scaler_path_ck:
            y_scaler_path_full = y_scaler_path_ck if os.path.isabs(y_scaler_path_ck) else os.path.join(base_dir, y_scaler_path_ck)
        else:
            y_scaler_path_full = Y_SCALER_PATH
        scaler = joblib.load(x_scaler_path_full)
        y_scaler = joblib.load(y_scaler_path_full)
        # Store window size for sequence creation in inference
        seq_window = window_size_ckpt
        print("✅ CNN‑LSTM model and scalers loaded successfully")
    else:
        # Fallback: assume model was saved as a full torch module
        model_loaded_inst = checkpoint
        model_loaded_inst.eval()
        booster = model_loaded_inst
        scaler = joblib.load(X_SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        feats_fit = list(getattr(scaler, "feature_names_in_", [])) or FEATS
        seq_window = 1
        print("✅ Model and scalers loaded successfully (full model)")
except Exception as e:
    print(f"❌ Error loading model or scalers: {e}")
    booster = None
    scaler = None
    y_scaler = None
    feats_fit = FEATS
    seq_window = 1

# -------------------- Helper Functions --------------------
def fetch_weather_data(lat: float, lon: float, start_date: date, end_date: date, tz: str) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo API"""
    today = date.today()
    end_date = end_date - timedelta(days=1)
    
    if start_date >= today:
        max_end = start_date + timedelta(days=15)
        if end_date > max_end:
            end_date = max_end
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
            raise ValueError(f"Invalid API response structure")
        
        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather data fetch failed: {str(e)}")

def ghi_to_poa(df_api: pd.DataFrame, lat: float, lon: float, tz: str, 
               tilt: float, azim: float) -> pd.DataFrame:
    """Convert GHI to POA irradiance"""
    df = df_api.copy()
    
    if df.index.tz is None:
        df = df.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:
        df = df.tz_convert(tz)
    
    ghi = df["shortwave_radiation"].astype(float).rename("ghi")
    solpos = pvlib.solarposition.get_solarposition(time=df.index, latitude=lat, longitude=lon)
    app_zen = solpos["apparent_zenith"].clip(0, 90)
    
    decomp = pvlib.irradiance.erbs(ghi=ghi, zenith=app_zen, datetime_or_doy=df.index)
    dhi = decomp["dhi"].clip(lower=0)
    dni = decomp["dni"].clip(lower=0)
    
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azim,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solpos["zenith"], 
        solar_azimuth=solpos["azimuth"]
    )
    
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
    """Upsample to 1-minute resolution"""
    return df_hourly.resample("T").interpolate(method="time").bfill().ffill()

def build_feature_matrix(df_min: pd.DataFrame, noct: float, feats: list):
    """Build feature matrix for prediction.

    This function constructs the input features in the order defined by
    ``feats``.  The new feature set differs from the original XGBoost
    implementation and is tailored to the CNN‑LSTM model.  It includes
    an approximation of energy (Wh/m²) per time step, placeholders for
    soiling indices, direct GHI, module temperatures, reference and test
    cell temperatures, ambient temperature and relative humidity.
    """
    # Ambient air temperature
    t_amb = df_min["temp2m"]
    # Estimate module temperature based on NOCT and plane-of-array
    t_mod = t_amb + (noct - 20.0) / 800.0 * df_min["poa1"]
    # Approximate energy in Wh/m².  Compute time delta in hours between
    # consecutive samples; default to 1/60 h for minute data or 1 h for
    # hourly data.
    if len(df_min.index) > 1:
        delta_hours = (df_min.index.to_series().diff().dt.total_seconds() / 3600.0).fillna(0)
        delta_hours = delta_hours.replace(0, 1.0 / 60.0)
    else:
        delta_hours = pd.Series([1.0], index=df_min.index)
    energy_wh = df_min["ghi"] * delta_hours
    # Build DataFrame with the correct column order
    X = pd.DataFrame(
        {
            feats[0]: energy_wh.values,
            feats[1]: np.zeros(len(df_min)),  # Soiling Loss Index Isc (%), placeholder
            feats[2]: np.zeros(len(df_min)),  # Soiling Loss Index Geff (%), placeholder
            feats[3]: df_min["ghi"].values,  # Direct GHI (W/m2)
            feats[4]: t_mod.values,  # Module Surface Temperature2
            feats[5]: t_mod.values,  # Module Surface Temperature1
            feats[6]: df_min["temp2m"].values,  # Temperature Reference Cell (approx)
            feats[7]: df_min["temp2m"].values,  # Temperature Test (approx)
            feats[8]: df_min["temp2m"].values,  # Ambient Temp.
            feats[9]: df_min["rh"].values,  # Relative Humidity
        },
        index=df_min.index,
    )
    # Identify daytime samples; night-time irradiance features are zeroed
    day_mask = df_min["ghi"] > 1.0
    irrad_cols = [feats[0], feats[3]]
    X.loc[~day_mask, irrad_cols] = 0.0
    return X, day_mask

# -------------------- API Endpoints --------------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Solar Forecast API",
        "version": "1.0.0",
        "endpoints": {
            "/forecast": "POST - Generate solar power forecast",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = booster is not None and scaler is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate solar power forecast"""
    
    # Ensure the model and scalers are loaded.  All three must be
    # present: the neural network (booster), the feature scaler and
    # the target scaler.
    if booster is None or scaler is None or y_scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = start_date + timedelta(days=request.horizon_days)
        
        # Fetch weather data
        df_api = fetch_weather_data(
            request.latitude, request.longitude, 
            start_date, end_date, request.timezone
        )
        
        # Compute POA
        df_hourly = ghi_to_poa(
            df_api, request.latitude, request.longitude, 
            request.timezone, request.tilt, request.azimuth
        )
        
        # Resample
        if request.use_minutely:
            df_input = upsample_to_minute(df_hourly)
        else:
            df_input = df_hourly.resample("H").mean().ffill().bfill()
        
        # Build feature matrix in the order expected by the scaler
        X_future_raw, day_mask = build_feature_matrix(df_input, request.noct, FEATS)
        X_future_raw = X_future_raw.reindex(columns=feats_fit).astype(float)
        # Scale inputs
        X_scaled_arr = scaler.transform(X_future_raw)
        # Create sliding windows for sequence input.  The model expects
        # sequences of length ``seq_window`` and will produce one
        # prediction per sequence.  Align predictions with the last
        # timestamp of each window.  If there are fewer samples than
        # required by the window, no prediction can be made.
        n_steps = len(X_scaled_arr)
        if n_steps < seq_window:
            raise HTTPException(status_code=500, detail=f"Not enough data to build a sequence of length {seq_window}")
        sequences = []
        for i in range(n_steps - seq_window + 1):
            sequences.append(X_scaled_arr[i : i + seq_window])
        sequences = np.stack(sequences)
        X_tensor_seq = torch.tensor(sequences, dtype=torch.float32)
        # Predict with the neural network
        with torch.no_grad():
            yhat_seq = booster(X_tensor_seq).cpu().numpy()
        # Build full-length prediction vector, aligning each prediction
        # with the end of its window.  Prepend zeros for the initial
        # seq_window-1 time steps.
        full_pred_scaled = np.zeros(n_steps)
        full_pred_scaled[seq_window - 1 :] = yhat_seq
        # Inverse transform predictions back to watts
        yhat_inv = y_scaler.inverse_transform(full_pred_scaled.reshape(-1, 1)).ravel()
        # Construct prediction series
        pred = pd.Series(yhat_inv, index=X_future_raw.index, name="forecast")
        # Zero out nighttime values and clip negatives
        pred.loc[~day_mask] = 0.0
        pred = pred.clip(lower=0)
        
        # Smoothing
        if request.smooth_hours > 0:
            pred_smooth = pred.rolling(f"{request.smooth_hours}h", min_periods=1).mean()
        else:
            pred_smooth = pred
        
        # Resample to hourly for output
        pred_hourly = pred_smooth.resample("H").mean()
        
        # Calculate daily statistics
        daily_mean = pred_smooth.resample("D").mean()
        daily_max = pred_smooth.resample("D").max()
        
        if request.use_minutely:
            daily_energy_kwh = pred.resample("D").sum() / 60.0 / 1000.0
        else:
            daily_energy_kwh = pred.resample("D").sum() / 1000.0
        
        # Prepare response
        hourly_data = [
            HourlyDataPoint(
                timestamp=ts.isoformat(),
                power_w=float(val)
            )
            for ts, val in pred_hourly.items()
        ]
        
        daily_data = [
            DailyDataPoint(
                date=dt.strftime("%Y-%m-%d"),
                mean_power_w=float(daily_mean.loc[dt]),
                peak_power_w=float(daily_max.loc[dt]),
                energy_kwh=float(daily_energy_kwh.loc[dt])
            )
            for dt in daily_energy_kwh.index
        ]
        
        return ForecastResponse(
            status="success",
            total_energy_kwh=float(daily_energy_kwh.sum()),
            peak_power_w=float(pred_smooth.max()),
            avg_daily_energy_kwh=float(daily_energy_kwh.mean()),
            days_forecasted=len(daily_energy_kwh),
            hourly_data=hourly_data,
            daily_data=daily_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


# =============================================================================
# FILE 2: app_with_api.py — Streamlit Frontend with API Integration
# =============================================================================
"""
Streamlit frontend that connects to FastAPI backend
Run FastAPI first: uvicorn api:app --reload --port 8000
Then run Streamlit: streamlit run app_with_api.py
"""

# Uncomment and use this in a separate file named app_with_api.py:
"""
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

# -------------------- Configuration --------------------
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Solar Forecast MVP",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
    <style>
    .main { padding: 1rem 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
''', unsafe_allow_html=True)

# -------------------- Header --------------------
st.title("☀️ Solar Power Forecast System")
st.markdown("Advanced solar irradiance prediction with FastAPI backend")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📍 Location")
    lat = st.number_input("Latitude", value=24.7136, format="%.6f")
    lon = st.number_input("Longitude", value=46.6753, format="%.6f")
    tz = st.text_input("Timezone", value="Asia/Riyadh")
    
    st.subheader("📅 Forecast Period")
    start_date = st.date_input("Start Date", value=date.today())
    horizon_days = st.number_input("Forecast Days", min_value=1, max_value=16, value=7)
    
    st.subheader("🔧 Panel Configuration")
    tilt = st.number_input("Panel Tilt (°)", value=25.0, min_value=0.0, max_value=90.0)
    azim = st.number_input("Panel Azimuth (°)", value=180.0, min_value=0.0, max_value=360.0)
    noct = st.number_input("NOCT (°C)", value=45.0, min_value=30.0, max_value=60.0)
    
    st.subheader("🎛️ Processing Options")
    use_minutely = st.checkbox("Upsample to 1-minute", value=True)
    smooth_hours = st.slider("Smoothing Window (h)", 0, 6, 3)
    show_daily_energy = st.checkbox("Show Daily Energy", value=True)

# -------------------- Main Content --------------------
col1, col2 = st.columns([2, 1])

with col2:
    run_forecast = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

if not run_forecast:
    st.info("👈 Configure parameters and click 'Run Forecast'")
    with st.expander("ℹ️ About This Tool"):
        st.markdown('''
        - **FastAPI Backend**: Scalable REST API architecture
        - **Machine Learning**: XGBoost model for predictions
        - **Real-time Data**: Weather forecasts from Open-Meteo
        - **PV Modeling**: Advanced POA calculations
        ''')
else:
    try:
        # Prepare request
        payload = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "horizon_days": horizon_days,
            "timezone": tz,
            "tilt": tilt,
            "azimuth": azim,
            "noct": noct,
            "use_minutely": use_minutely,
            "smooth_hours": smooth_hours
        }
        
        # Call API
        with st.spinner("🔄 Generating forecast..."):
            response = requests.post(f"{API_URL}/forecast", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
        
        st.success("✅ Forecast generated successfully!")
        
        # Display metrics
        metrics = st.columns(4)
        with metrics[0]:
            st.metric("Total Energy", f"{data['total_energy_kwh']:.1f} kWh")
        with metrics[1]:
            st.metric("Peak Power", f"{data['peak_power_w']:.0f} W")
        with metrics[2]:
            st.metric("Avg Daily", f"{data['avg_daily_energy_kwh']:.1f} kWh")
        with metrics[3]:
            st.metric("Days", data['days_forecasted'])
        
        # Convert to DataFrames
        hourly_df = pd.DataFrame([
            {"timestamp": pd.to_datetime(h["timestamp"]), "power_w": h["power_w"]}
            for h in data["hourly_data"]
        ]).set_index("timestamp")
        
        daily_df = pd.DataFrame(data["daily_data"])
        
        # Plot forecast
        st.subheader("📊 Power Forecast")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(hourly_df.index, hourly_df["power_w"], linewidth=2, color='#e74c3c')
        ax.set_title("Solar Power Forecast", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Power (W)", fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Daily energy chart
        if show_daily_energy:
            st.subheader("⚡ Daily Energy Production")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.bar(daily_df["date"], daily_df["energy_kwh"], 
                   width=0.8, color='#f39c12', alpha=0.8)
            for i, row in daily_df.iterrows():
                ax2.text(i, row["energy_kwh"], f'{row["energy_kwh"]:.1f}',
                        ha='center', va='bottom', fontsize=10)
            ax2.set_title("Daily Energy Production", fontsize=16, fontweight='bold')
            ax2.set_ylabel("Energy (kWh)", fontsize=12)
            ax2.grid(True, axis="y", alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Data tables
        tab1, tab2 = st.tabs(["📅 Daily Summary", "📈 Hourly Data"])
        
        with tab1:
            st.dataframe(daily_df, use_container_width=True)
        
        with tab2:
            hourly_display = hourly_df.reset_index()
            hourly_display["timestamp"] = hourly_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(hourly_display, use_container_width=True)
        
        # Downloads
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_csv = hourly_display.to_csv(index=False)
            st.download_button(
                "📄 Hourly Data (CSV)",
                data=hourly_csv,
                file_name=f"forecast_hourly_{start_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            daily_csv = daily_df.to_csv(index=False)
            st.download_button(
                "📊 Daily Summary (CSV)",
                data=daily_csv,
                file_name=f"forecast_daily_{start_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI is running on http://localhost:8000")
        st.code("uvicorn api:app --reload --port 8000")
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.text}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
"""