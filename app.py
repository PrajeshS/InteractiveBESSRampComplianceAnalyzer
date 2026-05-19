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
    .event-note { font-size: 0.85rem; color: #8b949e; margin-top: 10px; font-style: italic; line-height: 1.4; }
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

st.sidebar.markdown('---')
st.sidebar.header("📅 Event Timeline")
day_options = {
    'Worst Ramp Stress (06-06)': 156,
    'Highest Variability (02-26)': 56,
    'Largest SOC Swing (11-28)': 331,
    'Highest Solar Generation (03-17)': 75,
    'Lowest Solar Generation (12-13)': 346
}
selected_label = st.sidebar.selectbox('Select View Day', list(day_options.keys()))
selected_day = day_options[selected_label]

# Event Explanations (Engineering Context)
event_notes = {
    'Worst Ramp Stress (06-06)': "Tests cumulative BESS effort; contains the highest frequency of ramps requiring constant response.",
    'Highest Variability (02-26)': "Tests Power Capacity (MW); features a 97 MW/min drop due to extreme cloud cover.",
    'Largest SOC Swing (11-28)': "Tests Energy Capacity (MWh); requires the deepest sustained discharge cycle of the year.",
    'Highest Solar Generation (03-17)': "Tests 'Inherent Curtailment' risk when the 100 MW POC limit is frequently exceeded.",
    'Lowest Solar Generation (12-13)': "Baseline monitoring; minimal BESS intervention required."
}
if selected_label in event_notes:
    st.sidebar.markdown(f'<div class="event-note">💡 <b>Note:</b> {event_notes[selected_label]}</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_path = "power time series 1 min (1)(in).csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)['Solar_MW'].values
    return np.zeros(525600)

def run_sim(pv_data, p_cap, e_cap, s_min, s_max, s_day, start_soc_pct, eff, threshold):
    n = len(pv_data)
    grid_export, bess_pwr, soc_history = np.zeros(n), np.zeros(n), np.zeros(n)
    curr_energy = (start_soc_pct / 100) * e_cap
    violations, day_mins = 0, 0
    t_solar, t_export, t_curtail_inh, t_curtail_ramp, t_bess_mwh = 0, 0, 0, 0, 0
    d_solar, d_export, d_curtail, d_bess_mwh = 0, 0, 0, 0
    e_min, e_max = s_min * e_cap, s_max * e_cap
    d_start, d_end = s_day * 1440, (s_day + 1) * 1440

    for t in range(n):
        pv = pv_data[t]
        if pv > 0.5: day_mins += 1
        t_solar += pv / 60

        prev_export = grid_export[t-1] if t > 0 else pv
        raw_pv_capped = min(pv, 100.0)
        t_curtail_inh += max(0, pv - 100.0) / 60

        raw_ramp = raw_pv_capped - prev_export
        target = 0
        if raw_ramp > threshold: target = raw_ramp - threshold
        elif raw_ramp < -threshold: target = raw_ramp + threshold

        actual_bess = 0
        if target > 0: # Charge required
            available_pwr = ((e_max - curr_energy) * 60) / eff
            actual_bess = min(target, p_cap, available_pwr)
            curr_energy += (actual_bess * eff) / 60
            if target > actual_bess:
                t_curtail_ramp += (target - actual_bess) / 60
        elif target < 0: # Discharge required
            available_pwr = ((curr_energy - e_min) * 60) * eff
            actual_bess = -min(abs(target), p_cap, available_pwr)
            curr_energy += (actual_bess / eff) / 60

        exp = raw_pv_capped - actual_bess
        grid_export[t], bess_pwr[t], soc_history[t] = exp, actual_bess, (curr_energy / e_cap) * 100
        t_export += exp / 60
        t_bess_mwh += abs(actual_bess) / 60

        if t >= d_start and t < d_end:
            d_solar += pv / 60
            d_export += exp / 60
            d_curtail += (max(0, pv - exp - (actual_bess if actual_bess > 0 else 0))) / 60
            d_bess_mwh += abs(actual_bess) / 60

        if pv > 0.5 and abs(exp - prev_export) > (threshold + 0.001): violations += 1

    return grid_export, bess_pwr, soc_history, violations, day_mins, t_solar, t_export, t_curtail_inh, t_curtail_ramp, t_bess_mwh, d_solar, d_export, d_curtail, d_bess_mwh

pv_signal = load_data()
export, bess, soc, v_count, d_mins, a_solar, a_export, a_curt_inh, a_curt_ramp, a_bess_mwh, ds, de, dc, db = run_sim(
    pv_signal, pwr_cap, enr_cap, soc_min, soc_max, selected_day, init_soc_pct, eff_one_way, ramp_thresh
)

# --- UI Display ---
c1, c2, c3, c4 = st.columns(4)
c1.metric('Ramp Compliance', f'{(d_mins-v_count)/d_mins*100:.2f}%')
c2.metric('Annual Violations', f'{v_count:,} minutes')
c3.metric('Total BESS Effort', f'{a_bess_mwh:,.0f} MWh')
c4.metric('Annual EFC', f'{a_bess_mwh / (2 * enr_cap * (soc_max-soc_min)):.1f}')

st.markdown('<div class="section-header">Annual Energy Budget</div>', unsafe_allow_html=True)
c5, c6, c7, c8, c9 = st.columns(5)
c5.metric('Solar Generation', f'{a_solar:,.0f} MWh')
c6.metric('Grid Export', f'{a_export:,.0f} MWh')
c7.metric('Inherent Curtailment', f'{a_curt_inh:,.0f} MWh')
c8.metric('Ramp Curtailment', f'{a_curt_ramp:,.0f} MWh')
c9.metric('Total Curtailment', f'{((a_curt_inh + a_curt_ramp)/a_solar*100):.1f}%')

st.markdown('<div class="section-header">Daily Energy Budget (Selected Day)</div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4)
d1.metric('Daily Solar', f'{ds:,.1f} MWh')
d2.metric('Daily Export', f'{de:,.1f} MWh')
d3.metric('Inherent Curtailment', f'{dc:,.1f} MWh')
d4.metric('Daily BESS Effort', f'{db:,.1f} MWh')

s, e = selected_day * 1440, (selected_day + 1) * 1440
times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=pv_signal[s:e], name='Raw Solar', line=dict(color='#8b949e', dash='dot')))
fig.add_trace(go.Scatter(x=times, y=export[s:e], name='Net Export', line=dict(color='#58a6ff', width=2)))
fig.add_trace(go.Scatter(x=times, y=bess[s:e], name='BESS MW', fill='tozeroy', line=dict(color='#238636', width=1)))
fig.add_trace(go.Scatter(x=times, y=soc[s:e], name='SOC %', yaxis='y2', line=dict(color='#f2cc60', width=2)))
fig.update_layout(hovermode='x unified', yaxis=dict(title='Power (MW)'), yaxis2=dict(overlaying='y', side='right', range=[0,100], title='SOC %'), template='plotly_dark', height=550, legend=dict(orientation='h', y=1.1))
st.plotly_chart(fig, use_container_width=True)
