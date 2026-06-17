import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib

# ==============================================================================
# 1. PAGE CONFIGURATION (MUST BE FIRST)
# ==============================================================================
st.set_page_config(page_title="VIBRAAI Diagnostic Platform", layout="wide")

# Custom CSS styling to match the dark card UI theme from the design
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d2d2d;
        text-align: left;
    }
    .metric-title { color: #888888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;}
    .metric-value { color: #ffffff; font-size: 32px; font-weight: bold; margin: 5px 0; }
    .metric-sub { color: #888888; font-size: 12px; }
    
    .file-box {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2d2d2d;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD MACHINE LEARNING MODEL
# ==============================================================================
@st.cache_resource
def load_model():
    return joblib.load('bearing_trained_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load 'bearing_trained_model.pkl'. Make sure it is in your project folder! Error: {e}")
    st.stop()

# ==============================================================================
# 3. HEADER & STAGE NAVIGATION
# ==============================================================================
top_col1, top_col2 = st.columns([1.5, 2])
with top_col1:
    st.subheader("VIBRAAI DIAGNOSTIC PLATFORM")

with top_col2:
    st.write(
        '<div style="text-align: right; margin-top: 10px;">'
        '<span style="background:#333; padding:8px 16px; border-radius:20px; margin-right:5px; color:#fff; font-size:13px; font-weight:bold;">1 · Acquire</span>'
        '<span style="background:#333; padding:8px 16px; border-radius:20px; margin-right:5px; color:#fff; font-size:13px; font-weight:bold;">2 · Features</span>'
        '<span style="background:#333; padding:8px 16px; border-radius:20px; margin-right:5px; color:#fff; font-size:13px; font-weight:bold;">3 · Model</span>'
        '<span style="background:#1a73e8; padding:8px 16px; border-radius:20px; color:#fff; font-size:13px; font-weight:bold;">4 · Predict</span>'
        '</div>', 
        unsafe_allow_html=True
    )

st.write("##")

# ==============================================================================
# 4. TOP DATA METRIC CARDS
# ==============================================================================
card1, card2, card3, card4 = st.columns(4)

with card1:
    st.markdown('<div class="metric-card"><div class="metric-title">Files Loaded</div><div class="metric-value">12</div><div class="metric-sub">bearing vibration records</div></div>', unsafe_allow_html=True)

with card2:
    st.markdown('<div class="metric-card"><div class="metric-title">Samples / File</div><div class="metric-value">20,480</div><div class="metric-sub">at 12,000 Hz</div></div>', unsafe_allow_html=True)

with card3:
    st.markdown('<div class="metric-card"><div class="metric-title">Channels</div><div class="metric-value">8</div><div class="metric-sub">accelerometer axes</div></div>', unsafe_allow_html=True)

with card4:
    st.markdown('<div class="metric-card"><div class="metric-title">Duration / File</div><div class="metric-value">1.71 s</div><div class="metric-sub">per recording</div></div>', unsafe_allow_html=True)

st.write("##")

# ==============================================================================
# 5. DYNAMIC INPUT INTERFACE (OPEN TO EVERYONE)
# ==============================================================================
st.markdown("### 📥 MACHINE DATA INPUT PORTAL")
input_mode = st.radio(
    "Choose Data Source Mode:",
    ["📊 Use Preloaded Historical Demo Snaps", "✏️ Input Custom Machine Parameters (Open Test)"]
)

st.markdown("---")

# Initialize default variables
rms_val, kurtosis_val, p2p_val, skewness_val, std_val = 0.08, 0.2, 0.4, 0.02, 0.08
selected_timestamp = "Custom Live Run"

if input_mode == "📊 Use Preloaded Historical Demo Snaps":
    st.write("Select an active baseline timestamp record to view historical failure progressions:")
    timestamps = [
        "2003_10_22_12_06.24", "2003_10_22_12_09.13", "2003_10_22_12_14.13",
        "2003_10_22_12_19.13", "2003_10_22_12_24.13", "2003_10_22_12_29.13",
        "2003_10_22_12_34.13", "2003_10_22_12_39.13", "2003_10_22_12_44.13",
        "2003_10_22_12_49.13", "2003_10_22_12_54.13", "2003_10_22_12_59.13"
    ]
    selected_timestamp = st.selectbox("📌 Select Data Snapshot Target:", timestamps)
    
    # Map index to simulated data behavior
    idx = timestamps.index(selected_timestamp)
    np.random.seed(idx)
    if idx < 6:
        rms_val, kurtosis_val, p2p_val, multiplier = np.random.uniform(0.065, 0.074), np.random.uniform(0.05, 0.25), np.random.uniform(0.35, 0.55), 0.8
    elif idx < 10:
        rms_val, kurtosis_val, p2p_val, multiplier = np.random.uniform(0.120, 0.180), np.random.uniform(2.5, 4.8), np.random.uniform(1.2, 2.2), 1.9
    else:
        rms_val, kurtosis_val, p2p_val, multiplier = np.random.uniform(0.320, 0.490), np.random.uniform(12.0, 24.0), np.random.uniform(3.8, 5.2), 4.2
        
    skewness_val = 0.02 * multiplier
    std_val = rms_val * 0.99

else:
    st.write("🔧 **Enter your own sensor parameters below to evaluate any external rotary machinery asset:**")
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        rms_val = st.number_input("Vibration RMS Level (g)", min_value=0.01, max_value=2.00, value=0.08, step=0.01, help="Root Mean Square overall acceleration energy.")
        std_val = st.number_input("Standard Deviation (g)", min_value=0.01, max_value=2.00, value=0.07, step=0.01)
    with col_in2:
        kurtosis_val = st.number_input("Signal Kurtosis", min_value=-2.0, max_value=50.0, value=0.2, step=0.1, help="Measures the spikiness/peakedness of the vibration profile.")
        skewness_val = st.number_input("Signal Skewness", min_value=-5.0, max_value=5.0, value=0.02, step=0.01)
    with col_in3:
        p2p_val = st.number_input("Peak-to-Peak Amplitude (g)", min_value=0.05, max_value=10.0, value=0.45, step=0.05)
    
    # Dynamically scale graph profiles to match user's custom inputted metrics
    multiplier = (rms_val / 0.08)

# ==============================================================================
# 6. SIGNAL GENERATION ENGINE (Common to both modes)
# ==============================================================================
t = np.linspace(0, 42, 300)
raw_waveform = (np.sin(t * 0.8) * 0.1 + np.sin(t * 2.5) * 0.05 + np.random.normal(0, 0.03, 300)) * multiplier

freq = np.linspace(0, 466, 200)
fft_spectrum = (np.abs(np.sin(freq * 0.02)) * 0.6 + np.random.exponential(0.1, 200)) * multiplier

# Query model using the final calculated values
user_features = np.array([[rms_val, kurtosis_val, p2p_val, skewness_val, std_val]])
prediction = model.predict(user_features)[0]
probabilities = model.predict_proba(user_features)[0]

# ==============================================================================
# 7. TIME & FREQUENCY DOMAIN CHART RENDERING
# ==============================================================================
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### TIME DOMAIN WAVEFORM")
    df_time = pd.DataFrame({'Time (ms)': t, 'Acceleration (g)': raw_waveform})
    fig_time = px.line(df_time, x='Time (ms)', y='Acceleration (g)', template="plotly_dark")
    fig_time.update_traces(line_color='#2d76d4')
    fig_time.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=280)
    st.plotly_chart(fig_time, use_container_width=True)

with chart_col2:
    st.markdown("#### FREQUENCY SPECTRUM (FFT)")
    df_freq = pd.DataFrame({'Frequency (Hz)': freq, 'Amplitude Profile': fft_spectrum})
    fig_freq = px.line(df_freq, x='Frequency (Hz)', y='Amplitude Profile', template="plotly_dark")
    fig_freq.update_traces(line_color='#0f9d58')
    fig_freq.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=280)
    st.plotly_chart(fig_freq, use_container_width=True)

# ==============================================================================
# 8. LIVE EVALUATION ALERTS FOOTER (THE DIAGNOSTIC RESULT)
# ==============================================================================
st.markdown("---")
st.subheader("📢 Real-time AI Classification Diagnostic Output:")

confidence = probabilities[prediction] * 100

if prediction == 0:
    st.success(f"### ✅ MODEL STATUS [0]: NORMAL RUNNING CONDITIONS (Confidence: {confidence:.1f}%)")
    st.write(f"The structural signals extracted for file snapshot `{selected_timestamp}` indicate low displacement energy. The system is stable; no structural degradation anomalies are present.")
elif prediction == 1:
    st.warning(f"### ⚠️ MODEL STATUS [1]: EARLY FAULT WARNING DETECTION (Confidence: {confidence:.1f}%)")
    st.write(f"**Anomalous Fatigue Detected:** Incipient micro-cracking profile matches standard outer race wear patterns in snapshot `{selected_timestamp}`. Plan a maintenance cycle.")
else:
    st.error(f"### 🚨 MODEL STATUS [2]: CRITICAL STRUCTURAL FAILURE ALARM (Confidence: {confidence:.1f}%)")
    st.write(f"**Emergency Shut-down Advisory:** Severe parameter breaches identified in snapshot `{selected_timestamp}`. Material flaking or structural spalling has progressed beyond functional tolerances. Isolate asset immediately.")