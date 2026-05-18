import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title='BESS Ramp Compliance Simulator', layout='wide')

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    [data-testid='stMetricValue'] { font-size: 1.8rem; color: #58a6ff; }
    .main-header { font-size: 24px; font-weight: bold; color: #58a6ff; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- Header & Logos ---
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="main-header">Interactive BESS Ramp Compliance Analyzer</div>', unsafe_allow_html=True)
with col2:
    logos = ['rivilogo.jpg', 'lakdhanavilogo.jpg', 'ltllogo.png']
    existing_logos = [img for img in logos if os.path.exists(img)]
    if existing_logos:
        st.image(existing_logos, width=80)

# --- Sidebar Controls ---
st.sidebar.header('Simulation Parameters')

# Power & Energy with Stepper functionality via number_input
pwr_cap = st.sidebar.number_input('BESS Power (MW)', 0, 200, 50, step=1)
enr_cap = st.sidebar.number_input('BESS Energy (MWh)', 1, 400, 100, step=5)

soc_choice = st.sidebar.selectbox('SOC Window', ['30% - 70%', '20% - 80%', '10% - 90%'])
try:
    soc_min, soc_max = [float(x.replace('%', '').strip())/100 for x in soc_choice.split('-')]
except:
    soc_min, soc_max = 0.3, 0.7

st.sidebar.markdown('---')
# Specific Stress Event Days from original simulator
day_options = {
    'Worst Ramp Stress (06-06)': 156,
    'High Variability (02-24)': 54,
    'Largest SOC Swing (05-07)': 126
}
selected_day_label = st.sidebar.selectbox('Select Event Day', list(day_options.keys()))
selected_day = day_options[selected_day_label]

# --- Simulation Engine ---
@st.cache_data
def load_data():
    csv_path = 'power time series 1 min (1)(in).csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df['Solar_MW'].values
    return np.zeros(525600)

def run_sim(pv_data, p_cap, e_cap, s_min, s_max):
    n = len(pv_data)
    grid_export = np.zeros(n)
    bess_pwr = np.zeros(n)
    soc_history = np.zeros(n)
    curr_energy = 0.5 * e_cap
    violations = 0
    day_mins = 0

    for t in range(n):
        pv = pv_data[t]
        if pv > 0.5: day_mins += 1
        
        prev_export = grid_export[t-1] if t > 0 else pv
        eff_pv = min(pv, 100.0) # POC limit
        raw_ramp = eff_pv - prev_export

        target = 0
        if abs(raw_ramp) > 3.0:
            target = (raw_ramp - 3.0) if raw_ramp > 0 else (raw_ramp + 3.0)

        actual_bess = 0
        if target > 0: # Charge
            space = (s_max * e_cap - curr_energy) * 60 / 0.92
            actual_bess = min(target, p_cap, space)
            curr_energy += (actual_bess * 0.92) / 60
        elif target < 0: # Discharge
            avail = (curr_energy - s_min * e_cap) * 60 * 0.92
            actual_bess = -min(abs(target), p_cap, avail)
            curr_energy += (actual_bess / 0.92) / 60

        export = min(eff_pv - actual_bess, 100.0)
        grid_export[t] = export
        bess_pwr[t] = actual_bess
        soc_history[t] = (curr_energy / e_cap) * 100
        
        if pv > 0.5 and abs(export - prev_export) > 3.001: 
            violations += 1
            
    return grid_export, bess_pwr, soc_history, violations, day_mins

# --- Execution ---
pv_signal = load_data()
export, bess, soc, v_count, d_mins = run_sim(pv_signal, pwr_cap, enr_cap, soc_min, soc_max)

# --- KPI Row 1: Compliance & BESS Health ---
st.subheader('Performance & Reliability KPIs')
c1, c2, c3, c4 = st.columns(4)
comp_pct = ((d_mins - v_count) / d_mins * 100) if d_mins > 0 else 100
c1.metric('Ramp Compliance', f'{comp_pct:.2f}%')
c2.metric('Violations', f'{v_count:,}')
c3.metric('BESS Energy Processed', f'{np.sum(np.abs(bess))/60:,.0f} MWh')
efc = (np.sum(np.abs(bess))/60) / (2 * enr_cap * (soc_max - soc_min))
c4.metric('Annual EFC', f'{efc:.1f}')

# --- KPI Row 2: Annual Energy Budget ---
st.subheader('Annual Energy Budget (100 MW POC Export Limit)')
e1, e2, e3, e4 = st.columns(4)
total_gen = np.sum(pv_signal)/60
total_exp = np.sum(export)/60
curtail = max(0, total_gen - total_exp - (np.sum(bess)/60))
e1.metric('Total Solar', f'{total_gen:,.0f} MWh')
e2.metric('Grid Export', f'{total_exp:,.0f} MWh')
e3.metric('Curtailment', f'{curtail:,.0f} MWh')
e4.metric('Curtailed %', f'{(curtail/total_gen*100):.1f}%')

# --- Charting ---
st.markdown('---')
s, e = selected_day * 1440, (selected_day + 1) * 1440
fig = go.Figure()
fig.add_trace(go.Scatter(y=pv_signal[s:e], name='Raw Solar Profile', line=dict(dash='dash', color='gray')))
fig.add_trace(go.Scatter(y=export[s:e], name='Grid Export (100MW Cap)', fill='tozeroy', line=dict(color='#58a6ff')))
fig.add_trace(go.Scatter(y=soc[s:e], name='BESS SOC %', yaxis='y2', line=dict(color='#f2cc60')))
fig.update_layout(
    hovermode='x unified', 
    height=550, 
    legend=dict(orientation='h', y=1.1),
    yaxis=dict(title='Power (MW)', gridcolor='#30363d'),
    yaxis2=dict(title='SOC (%)', overlaying='y', side='right', range=[0, 100], showgrid=False),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig, use_container_width=True)
