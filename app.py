import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title='BESS Ramp Compliance Simulator', layout='wide')

# --- Custom CSS ---
st.markdown("""
    <style>
    [data-testid='stMetricValue'] { font-size: 1.8rem; color: #58a6ff; }
    .main-header { font-size: 24px; font-weight: bold; color: #58a6ff; margin-bottom: 20px; }
    .caption { font-size: 0.85rem; color: #8b949e; margin-top: -15px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">Interactive BESS Ramp Compliance Analyzer</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header('Simulation Parameters')
pwr_cap = st.sidebar.number_input('BESS Power (MW)', 0, 200, 50, step=1)
enr_cap = st.sidebar.number_input('BESS Energy (MWh)', 1, 400, 100, step=5)
soc_choice = st.sidebar.selectbox('SOC Window', ['30% - 70%', '20% - 80%', '10% - 90%'])
try:
    soc_min, soc_max = [float(x.replace('%', '').strip())/100 for x in soc_choice.split('-')]
except:
    soc_min, soc_max = 0.3, 0.7

st.sidebar.markdown('---')
day_options = {
    'Worst Ramp Stress (06-06)': 156,
    'Highest Variability (02-26)': 56,
    'Largest SOC Swing (11-28)': 331
}
selected_day_label = st.sidebar.selectbox('Select Event Day', list(day_options.keys()))
selected_day = day_options[selected_day_label]

if selected_day == 156:
    st.sidebar.markdown('<div class="caption">Note: Highest cumulative daily violation severity (RSI).</div>', unsafe_allow_html=True)
elif selected_day == 56:
    st.sidebar.markdown('<div class="caption">Note: Single largest 1-minute solar power jump (97 MW/min).</div>', unsafe_allow_html=True)
elif selected_day == 331:
    st.sidebar.markdown('<div class="caption">Note: Maximum peak-to-peak depth of discharge (22.68% SOC Swing).</div>', unsafe_allow_html=True)

# --- Simulation Engine ---
@st.cache_data
def load_data():
    csv_path = 'power time series 1 min (1)(in).csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df['Solar_MW'].values
    return np.zeros(525600)

def run_sim(pv_data, p_cap, e_cap, s_min, s_max, s_day):
    n = len(pv_data)
    grid_export = np.zeros(n)
    bess_pwr = np.zeros(n)
    soc_history = np.zeros(n)
    curr_energy = 0.5 * e_cap
    violations = 0
    day_mins = 0
    
    # Budget counters
    total_solar_mwh = 0
    total_export_mwh = 0
    total_curtail_mwh = 0
    
    # Daily tracking for selected day
    d_start, d_end = s_day * 1440, (s_day + 1) * 1440
    d_metrics = {'solar': 0, 'export': 0, 'curtail': 0, 'throughput': 0}

    for t in range(n):
        pv = pv_data[t]
        total_solar_mwh += pv / 60
        if pv > 0.5: day_mins += 1
        
        prev_export = grid_export[t-1] if t > 0 else pv
        eff_pv = min(pv, 100.0)
        raw_ramp = eff_pv - prev_export
        target = (raw_ramp - 3.0) if raw_ramp > 3.0 else ((raw_ramp + 3.0) if raw_ramp < -3.0 else 0)
        
        actual_bess = 0
        if target > 0:
            space = (s_max * e_cap - curr_energy) * 60 / 0.92
            actual_bess = min(target, p_cap, space)
            curr_energy += (actual_bess * 0.92) / 60
        elif target < 0:
            avail = (curr_energy - s_min * e_cap) * 60 * 0.92
            actual_bess = -min(abs(target), p_cap, avail)
            curr_energy += (actual_bess / 0.92) / 60
            
        export = min(eff_pv - actual_bess, 100.0)
        curtail = max(0, pv - export - actual_bess)
        
        grid_export[t] = export
        total_export_mwh += export / 60
        total_curtail_mwh += curtail / 60
        
        bess_pwr[t] = actual_bess
        soc_history[t] = (curr_energy / e_cap) * 100
        
        if pv > 0.5 and abs(export - prev_export) > 3.001: violations += 1
        
        if t >= d_start and t < d_end:
            d_metrics['solar'] += pv / 60
            d_metrics['export'] += export / 60
            d_metrics['curtail'] += curtail / 60
            d_metrics['throughput'] += abs(actual_bess) / 60
            
    return grid_export, bess_pwr, soc_history, violations, day_mins, d_metrics, total_solar_mwh, total_export_mwh, total_curtail_mwh

pv_signal = load_data()
export, bess, soc, v_count, d_mins, d_stats, a_solar, a_export, a_curtail = run_sim(pv_signal, pwr_cap, enr_cap, soc_min, soc_max, selected_day)

# --- Display Results ---
st.subheader('Annual Performance & Reliability')
c1, c2, c3, c4 = st.columns(4)
comp_pct = ((d_mins - v_count) / d_mins * 100) if d_mins > 0 else 100
c1.metric('Ramp Compliance', f'{comp_pct:.2f}%')
c2.metric('Violations', f'{v_count:,}')
c3.metric('Annual BESS Throughput', f'{np.sum(np.abs(bess))/60:,.0f} MWh')
efc = (np.sum(np.abs(bess))/60) / (2 * enr_cap * (soc_max - soc_min))
c4.metric('Annual EFC', f'{efc:.1f}')

st.subheader('Annual Energy Budget (100 MW POC Export Limit)')
e1, e2, e3, e4 = st.columns(4)
e1.metric('Total Solar', f'{a_solar:,.0f} MWh')
e2.metric('Grid Export', f'{a_export:,.0f} MWh')
e3.metric('Curtailment', f'{a_curtail:,.0f} MWh')
e4.metric('Curtailed %', f'{(a_curtail/a_solar*100):.1f}%')

st.subheader('Daily Energy Budget (Selected Day)')
d1, d2, d3, d4 = st.columns(4)
d1.metric('Daily Solar', f"{d_stats['solar']:.1f} MWh")
d2.metric('Daily Export', f"{d_stats['export']:.1f} MWh")
d3.metric('Daily Curtailment', f"{d_stats['curtail']:.1f} MWh")
d4.metric('Daily BESS Throughput', f"{d_stats['throughput']:.1f} MWh")

s, e = selected_day * 1440, (selected_day + 1) * 1440
fig = go.Figure()
fig.add_trace(go.Scatter(y=pv_signal[s:e], name='Raw Solar', line=dict(dash='dash', color='gray')))
fig.add_trace(go.Scatter(y=export[s:e], name='Grid Export', fill='tozeroy', line=dict(color='#58a6ff')))
fig.add_trace(go.Scatter(y=soc[s:e], name='BESS SOC %', yaxis='y2', line=dict(color='#f2cc60')))
fig.update_layout(hovermode='x unified', height=550, yaxis=dict(title='Power (MW)'), yaxis2=dict(title='SOC (%)', overlaying='y', side='right', range=[0, 100], showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)
