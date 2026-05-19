import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title='BESS Engineering Audit - Final', layout='wide')

# --- Professional Engineering CSS ---
st.markdown("""
    <style>
    [data-testid='stMetricValue'] { font-size: 1.8rem; color: #58a6ff; }
    .main-header { font-size: 28px; font-weight: bold; color: #58a6ff; margin-bottom: 20px; }
    .section-header { font-size: 14px; font-weight: bold; color: #8b949e; text-transform: uppercase; margin-top: 30px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    .audit-note { font-size: 0.85rem; color: #ffab70; background: #2a1b10; padding: 10px; border-radius: 5px; border: 1px solid #6e3621; margin-bottom: 20px; }
    .event-note { font-size: 0.9rem; color: #8b949e; margin-top: 10px; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">BESS Ramp Compliance Simulator (Physically Verified)</div>', unsafe_allow_html=True)
st.markdown('<div class="audit-note">⚠️ <b>Physical Audit Applied:</b> SOC is now continuous across the 365-day horizon. Sign consistency fixed for curtailment and discharge logic.</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Physical Parameters")
pwr_cap = st.sidebar.number_input('BESS Power Limit (MW)', 0, 200, 50)
enr_cap = st.sidebar.number_input('BESS Energy Capacity (MWh)', 1, 400, 100)
init_soc_pct = st.sidebar.number_input('Initial Year-Start SOC (%)', 0, 100, 50)
eff_one_way = st.sidebar.slider('One-Way Efficiency', 0.80, 0.99, 0.95, step=0.01)

soc_choice = st.sidebar.selectbox('Operating SOC Window', ['30% - 70%', '20% - 80%', '10% - 90%'])
soc_min, soc_max = [float(x.replace('%', '').strip())/100 for x in soc_choice.split('-')]

with st.sidebar.expander("Advanced Physics Settings"):
    ramp_thresh = st.sidebar.number_input('Compliance Threshold (MW/min)', 0.0, 10.0, 3.0)

st.sidebar.markdown('---')
st.sidebar.header("📅 Select View Day")
day_options = {
    'Worst Ramp Stress (06-06)': 156, 
    'Highest Variability (02-26)': 56, 
    'Largest SOC Swing (11-28)': 331,
    'Highest Solar Generation (01-14)': 13,
    'Lowest Solar Generation (12-25)': 358
}
selected_label = st.sidebar.selectbox('Event Timeline', list(day_options.keys()))
selected_day = day_options[selected_label]

# Event Day Notes
event_notes = {
    156: "<b>Worst Ramp Stress:</b> This day exhibits the highest cumulative volatility and frequency of solar fluctuations. It is used to stress-test the BESS's ability to switch rapidly between charge and discharge modes while maintaining grid stability.",
    56: "<b>Highest Variability:</b> This day contains the sharpest single-minute production drop recorded (approx. 97 MW). It is critical for verifying if the BESS Power Limit (MW) is sufficient to catch massive 'cliffs' in solar output.",
    331: "<b>Largest SOC Swing:</b> This day represents the most sustained energy transfer requirement. It pushes the battery to its deepest cycle of the year, testing if the Energy Capacity (MWh) is deep enough to handle long-duration ramp support.",
    13: "<b>Highest Solar Generation:</b> Peak production day. This scenario tests the 'Inherent Curtailment' levels and evaluates BESS saturation risks when solar energy is at its maximum and the grid limit is hit.",
    358: "<b>Lowest Solar Generation:</b> Deep winter/overcast scenario. This verifies BESS behavior during low-output periods where the ramp rate might naturally stay within limits but SOC management remains essential for next-day readiness."
}
st.sidebar.markdown(f'<div class="event-note">{event_notes[selected_day]}</div>', unsafe_allow_html=True)

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
    t_solar, t_export, t_curtail, t_bess_mwh = 0, 0, 0, 0
    d_solar, d_export, d_curtail, d_bess_mwh = 0, 0, 0, 0

    e_min, e_max = s_min * e_cap, s_max * e_cap
    d_start, d_end = s_day * 1440, (s_day + 1) * 1440

    for t in range(n):
        pv = pv_data[t]
        if pv > 0.5: day_mins += 1
        t_solar += pv / 60

        prev_export = grid_export[t-1] if t > 0 else pv
        raw_pv_capped = min(pv, 100.0)
        raw_ramp = raw_pv_capped - prev_export

        target = 0
        if raw_ramp > threshold: target = raw_ramp - threshold
        elif raw_ramp < -threshold: target = raw_ramp + threshold

        actual_bess = 0
        if target > 0:
            available_pwr = ((e_max - curr_energy) * 60) / eff
            actual_bess = min(target, p_cap, available_pwr)
            curr_energy += (actual_bess * eff) / 60
        elif target < 0:
            available_pwr = ((curr_energy - e_min) * 60) * eff
            actual_bess = -min(abs(target), p_cap, available_pwr)
            curr_energy += (actual_bess / eff) / 60

        exp = min(raw_pv_capped - actual_bess, 100.0)
        curt = max(0, pv - exp - (actual_bess if actual_bess > 0 else 0))

        grid_export[t], bess_pwr[t], soc_history[t] = exp, actual_bess, (curr_energy / e_cap) * 100
        t_export += exp / 60
        t_curtail += curt / 60
        t_bess_mwh += abs(actual_bess) / 60

        if t >= d_start and t < d_end:
            d_solar += pv / 60
            d_export += exp / 60
            d_curtail += curt / 60
            d_bess_mwh += abs(actual_bess) / 60

        if pv > 0.5 and abs(exp - prev_export) > (threshold + 0.001): violations += 1

    return grid_export, bess_pwr, soc_history, violations, day_mins, t_solar, t_export, t_curtail, t_bess_mwh, d_solar, d_export, d_curtail, d_bess_mwh

pv_signal = load_data()
export, bess, soc, v_count, d_mins, a_solar, a_export, a_curtail, a_bess_mwh, ds, de, dc, db = run_sim(
    pv_signal, pwr_cap, enr_cap, soc_min, soc_max, selected_day, init_soc_pct, eff_one_way, ramp_thresh
)

# --- UI Display ---
c1, c2, c3, c4 = st.columns(4)
c1.metric('Ramp Compliance', f'{(d_mins-v_count)/d_mins*100:.2f}%')
c2.metric('Annual Violations', f'{v_count:,} minutes')
c3.metric('Total BESS Effort', f'{a_bess_mwh:,.0f} MWh')
c4.metric('Annual Equivalent Full Cycles', f'{a_bess_mwh / (2 * enr_cap * (soc_max-soc_min)):.1f}')

st.markdown('<div class="section-header">Annual Energy Totals</div>', unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)
c5.metric('Solar Generation', f'{a_solar:,.0f} MWh')
c6.metric('Grid Exported', f'{a_export:,.0f} MWh')
c7.metric('Inherent Curtailment', f'{a_curtail:,.0f} MWh')
c8.metric('Curtailment %', f'{(a_curtail/a_solar*100):.1f}%')

st.markdown(f'<div class="section-header">Daily Energy Budget: {selected_label}</div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4)
d1.metric('Daily Solar', f'{ds:.1f} MWh')
d2.metric('Daily Export', f'{de:.1f} MWh')
d3.metric('Daily Curtail', f'{dc:.1f} MWh')
d4.metric('BESS Dispatch', f'{db:.1f} MWh')

# --- Plotting ---
s, e = selected_day * 1440, (selected_day + 1) * 1440
times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=pv_signal[s:e], name='Raw Solar', line=dict(color='#8b949e', dash='dot')))
fig.add_trace(go.Scatter(x=times, y=export[s:e], name='Net Export', line=dict(color='#58a6ff', width=2)))
fig.add_trace(go.Scatter(x=times, y=bess[s:e], name='BESS Power (MW)', fill='tozeroy', line=dict(color='#238636', width=1)))
fig.add_trace(go.Scatter(x=times, y=soc[s:e], name='BESS SOC %', yaxis='y2', line=dict(color='#f2cc60', width=2)))
fig.update_layout(hovermode='x unified', yaxis2=dict(overlaying='y', side='right', range=[0,100], title='SOC %'),
                  template='plotly_dark', height=550, legend=dict(orientation='h', y=1.1))
st.plotly_chart(fig, use_container_width=True)
