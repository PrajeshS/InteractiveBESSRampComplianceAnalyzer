import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title='BESS Ramp Compliance Simulator', layout='wide')

# --- Professional Engineering CSS ---
st.markdown("""
    <style>
    [data-testid='stMetricValue'] { font-size: 1.8rem; color: #58a6ff; }
    .main-header { font-size: 28px; font-weight: bold; color: #58a6ff; margin-bottom: 20px; }
    .section-header { font-size: 14px; font-weight: bold; color: #8b949e; text-transform: uppercase; margin-top: 30px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    path = 'power time series 1 min (1)(in).csv'
    if os.path.exists(path):
        return pd.read_csv(path)['Solar_MW'].values
    return np.zeros(525600)

def run_sim_engine(pv_data, p_cap, e_cap, s_min, s_max, start_soc_pct, eff, threshold):
    n = len(pv_data)
    grid_export = np.zeros(n)
    bess_pwr = np.zeros(n)
    soc_history = np.zeros(n)
    curr_energy = (start_soc_pct / 100) * e_cap
    e_min, e_max = s_min * e_cap, s_max * e_cap
    violations, daytime_mins = 0, 0
    total_bess_mwh = 0.0
    ramp_curtail_mwh = 0.0
    
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
            # AC limit based on DC space: Power = (dE / dt) / eff
            space_ac = (e_max - curr_energy) * 60.0 / eff
            actual_bess = min(target, p_cap, space_ac)
            curr_energy += (actual_bess * eff) / 60.0
        elif target < 0: # Discharge
            # AC limit based on DC avail: Power = (dE / dt) * eff
            avail_ac = (curr_energy - e_min) * 60.0 * eff
            actual_bess = -min(abs(target), p_cap, avail_ac)
            curr_energy += (actual_bess / eff) / 60.0

        exp = pv_capped - actual_bess
        grid_export[t], bess_pwr[t], soc_history[t] = exp, actual_bess, (curr_energy / e_cap) * 100.0
        
        # Track solar energy lost due to BESS inability to manage ramp
        if target > 0:
            ramp_curtail_mwh += (target - actual_bess) / 60.0

        if is_day and abs(exp - prev_export) > (threshold + 0.001): 
            violations += 1
        
        prev_export = exp
        total_bess_mwh += abs(actual_bess) / 60.0
        
    return grid_export, bess_pwr, soc_history, violations, daytime_mins, total_bess_mwh, ramp_curtail_mwh

# UI Controls
st.sidebar.header('⚙ BESS Configuration')
pwr = st.sidebar.number_input('Power (MW)', 0, 200, 50)
enr = st.sidebar.number_input('Energy (MWh)', 1, 400, 100)
win = st.sidebar.selectbox('SOC Window', ['30% - 70%', '20% - 80%', '10% - 90%'])
s_min, s_max = [float(x.replace('%',''))/100 for x in win.split('-')]

pv_signal = load_data()
export, bess, soc, v_count, d_mins, a_bess_mwh, r_curt_mwh = run_sim_engine(pv_signal, float(pwr), float(enr), s_min, s_max, 50.0, 0.96, 3.0)

st.markdown('<div class="main-header">BESS Ramp Compliance Dashboard</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
comp_pct = ((d_mins - v_count) / d_mins * 100) if d_mins > 0 else 0
c1.metric('Annual Compliance', f'{comp_pct:.2f}%')
c2.metric('Annual Violations', f'{v_count:,} min')
c3.metric('Annual EFC', f'{a_bess_mwh / (2 * enr):.1f}')
c4.metric('Ramp Curtailment', f'{r_curt_mwh:,.0f} MWh')

st.markdown('<div class="section-header">Annual SOC Diagnostic (Red = At Limit)</div>', unsafe_allow_html=True)
soc_colors = np.where((soc >= (s_max*100-0.1)) | (soc <= (s_min*100+0.1)), '#f85149', '#f2cc60')
fig = go.Figure()
fig.add_trace(go.Scattergl(y=soc, mode='markers', marker=dict(color=soc_colors, size=2), name='SOC %'))
fig.update_layout(template='plotly_dark', height=400, xaxis=dict(title='Minutes of Year'), yaxis=dict(title='SOC (%)', range=[0, 100]))
st.plotly_chart(fig, use_container_width=True)
