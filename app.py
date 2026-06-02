import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from numba import njit

st.set_page_config(page_title='BESS Ramp Compliance Simulator', layout='wide')

# --- Professional Engineering CSS ---
st.markdown("""
    <style>
    [data-testid='stMetricValue'] { font-size: 1.8rem; color: #58a6ff; }
    .main-header { font-size: 28px; font-weight: bold; color: #58a6ff; margin-bottom: 20px; }
    .section-header { font-size: 14px; font-weight: bold; color: #8b949e; text-transform: uppercase; margin-top: 30px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    .stNumberInput div[data-baseweb='input'] { background-color: #0d1117; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">BESS Ramp Compliance Simulator</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Physical Parameters")
pwr_cap = st.sidebar.number_input('BESS Power Limit (MW)', 0, 200, 50)
enr_cap = st.sidebar.number_input('BESS Energy Capacity (MWh)', 1, 400, 100)
init_soc_pct = st.sidebar.number_input('Initial SOC (%)', 0, 100, 50)
eff_one_way = st.sidebar.number_input('One-Way Efficiency', 0.80, 1.00, 0.95, step=0.01)

soc_choice = st.sidebar.selectbox('Operating SOC Window', ['30% - 70%', '20% - 80%', '10% - 90%'])
soc_min, soc_max = [float(x.replace('%', '').strip())/100 for x in soc_choice.split('-')]
ramp_thresh = st.sidebar.number_input('Compliance Threshold (MW/min)', 0.0, 10.0, 3.0)

@st.cache_data
def load_data():
    csv_path = "power time series 1 min (1)(in).csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)['Solar_MW'].values
    return np.zeros(525600)

@njit
def run_sim_engine(pv_data, p_cap, e_cap, s_min, s_max, start_soc_pct, eff, threshold):
    n = len(pv_data)
    grid_export, bess_pwr, soc_history = np.zeros(n), np.zeros(n), np.zeros(n)
    curr_energy = (start_soc_pct / 100) * e_cap
    e_min, e_max = s_min * e_cap, s_max * e_cap
    violations, daytime_mins = 0, 0
    prev_export = pv_data[0] if pv_data[0] < 100.0 else 100.0

    for t in range(n):
        pv = pv_data[t]
        is_day = pv > 0.5
        if is_day: daytime_mins += 1
        pv_capped = min(pv, 100.0)
        raw_ramp = pv_capped - prev_export
        target = 0.0
        if raw_ramp > threshold: target = raw_ramp - threshold
        elif raw_ramp < -threshold: target = raw_ramp + threshold

        actual_bess = 0.0
        if target > 0: # Charge
            space = (e_max - curr_energy) * 60.0 / eff
            actual_bess = min(target, p_cap, space)
            curr_energy += (actual_bess * eff) / 60.0
        elif target < 0: # Discharge
            avail = (curr_energy - e_min) * 60.0 * eff
            actual_bess = -min(abs(target), p_cap, avail)
            curr_energy += (actual_bess / eff) / 60.0

        exp = pv_capped - actual_bess
        grid_export[t], bess_pwr[t], soc_history[t] = exp, actual_bess, (curr_energy / e_cap) * 100.0
        if is_day and abs(exp - prev_export) > (threshold + 0.001): violations += 1
        prev_export = exp
    return grid_export, bess_pwr, soc_history, violations, daytime_mins

pv_signal = load_data()
export, bess, soc, v_count, d_mins = run_sim_engine(
    pv_signal, float(pwr_cap), float(enr_cap), soc_min, soc_max, float(init_soc_pct), float(eff_one_way), float(ramp_thresh)
)

# --- Metrics ---
c1, c2, c3, c4 = st.columns(4)
comp_rate = ((d_mins - v_count) / d_mins * 100) if d_mins > 0 else 0
c1.metric('Ramp Compliance', f'{comp_rate:.2f}%')
c2.metric('Annual Violations', f'{v_count:,} min')
c3.metric('Total BESS Dispatch', f'{np.sum(np.abs(bess))/60:,.0f} MWh')
efc = (np.sum(np.abs(bess))/60) / (2 * enr_cap * (soc_max-soc_min))
c4.metric('Annual EFC', f'{efc:.1f}')

# --- Charts ---
st.markdown('<div class="section-header">Annual State-of-Charge (WebGL Optimized)</div>', unsafe_allow_html=True)

# --- Annual SOC with Conditional Highlighting ---
st.markdown('<div class="section-header">Annual SOC Diagnostic (Red = At Limit)</div>', unsafe_allow_html=True)

# Create a color array: Red if at/outside limits, yellow otherwise
# Using a small epsilon for floating point comparison
soc_colors = np.where((soc >= (soc_max*100 - 0.1)) | (soc <= (soc_min*100 + 0.1)), '#f85149', '#f2cc60')

fig_annual = go.Figure()
fig_annual.add_trace(go.Scattergl(
    y=soc, 
    mode='markers', 
    marker=dict(color=soc_colors, size=2), 
    name='Annual SOC %'
))
fig_annual.update_layout(
    template='plotly_dark', 
    height=400, 
    xaxis=dict(title='Minutes of Year'), 
    yaxis=dict(title='SOC (%)', range=[0, 100]),
    showlegend=False
)
st.plotly_chart(fig_annual, use_container_width=True)

