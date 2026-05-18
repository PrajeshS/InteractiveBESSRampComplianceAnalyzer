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
    .event-desc { font-size: 0.9rem; color: #c9d1d9; background: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; margin-top: 15px; line-height: 1.4; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">Interactive BESS Ramp Compliance Analyzer (RTE η=0.92)</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header('⚙️ System Configuration')
pwr_cap = st.sidebar.number_input('BESS Power Limit (MW)', 0, 200, 50, step=1)
enr_cap = st.sidebar.number_input('BESS Energy Capacity (MWh)', 1, 400, 100, step=5)
init_soc_pct = st.sidebar.number_input('Initial State of Charge (%)', 0, 100, 50, step=1)

soc_choice = st.sidebar.selectbox('Operating SOC Window', ['30% - 70%', '20% - 80%', '10% - 90%'])
try:
    soc_min, soc_max = [float(x.replace('%', '').strip())/100 for x in soc_choice.split('-')]
except:
    soc_min, soc_max = 0.3, 0.7

st.sidebar.markdown('---')
st.sidebar.subheader('📅 Event Stress Testing')
day_options = {
    'Worst Ramp Stress (06-06)': 156,
    'Highest Variability (02-26)': 56,
    'Largest SOC Swing (11-28)': 331
}
selected_day_label = st.sidebar.selectbox('Select Critical Event Day', list(day_options.keys()))
selected_day = day_options[selected_day_label]

descs = {
    'Worst Ramp Stress (06-06)': "June 6th (Day 157): This day represents the absolute peak of the 'Ramp Stress Index'. High-frequency cloud cover creates constant minute-by-minute volatility, forcing the BESS to cycle rapidly to maintain the 3MW/min limit.",
    'Highest Variability (02-26)': "Feb 26th (Day 57): Features the highest single-minute solar drop recorded (**97.09 MW/min**). This is a critical test of BESS Power (MW) response time and instantaneous discharge capability.",
    'Largest SOC Swing (11-28)': "Nov 28th (Day 332): The day with the largest cumulative energy throughput. It tests the Energy Capacity (MWh) limits by forcing the battery through its deepest single-day charge/discharge cycle."
}
st.sidebar.markdown(f'<div class="event-desc">{descs[selected_day_label]}</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_path = "power time series 1 min (1)(in).csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)['Solar_MW'].values
    return np.zeros(525600)

def run_sim(pv_data, p_cap, e_cap, s_min, s_max, s_day, start_soc):
    n = len(pv_data)
    grid_export, bess_pwr, soc_history = np.zeros(n), np.zeros(n), np.zeros(n)
    curr_energy = (start_soc / 100) * e_cap
    violations, day_mins = 0, 0
    t_solar, t_export, t_curtail, t_bess = 0, 0, 0, 0
    d_start, d_end = s_day * 1440, (s_day + 1) * 1440
    d_stats = {'solar': 0, 'export': 0, 'curtail': 0, 'bess': 0}

    for t in range(n):
        if t % 1440 == 0: curr_energy = (start_soc / 100) * e_cap
        pv = pv_data[t]
        t_solar += pv / 60
        if pv > 0.5: day_mins += 1
        prev = grid_export[t-1] if t > 0 else pv
        eff_pv = min(pv, 100.0)
        raw_ramp = eff_pv - prev
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
            
        exp = min(eff_pv - actual_bess, 100.0)
        curt = max(0, pv - exp - actual_bess)
        grid_export[t], bess_pwr[t], soc_history[t] = exp, actual_bess, (curr_energy / e_cap) * 100
        t_export += exp / 60
        t_curtail += curt / 60
        t_bess += abs(actual_bess) / 60
        
        if pv > 0.5 and abs(exp - prev) > 3.001: violations += 1
        if t >= d_start and t < d_end:
            d_stats['solar'] += pv / 60
            d_stats['export'] += exp / 60
            d_stats['curtail'] += curt / 60
            d_stats['bess'] += abs(actual_bess) / 60
            
    return grid_export, bess_pwr, soc_history, violations, day_mins, d_stats, t_solar, t_export, t_curtail, t_bess

pv_signal = load_data()
export, bess, soc, v_count, d_mins, d_stats, a_solar, a_export, a_curtail, a_bess = run_sim(pv_signal, pwr_cap, enr_cap, soc_min, soc_max, selected_day, init_soc_pct)

# --- KPI Dashboard ---
st.markdown('<div class="section-header">Performance & Reliability KPIs</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric('Ramp Compliance', f'{(d_mins - v_count) / d_mins * 100:.2f}%' if d_mins > 0 else '100%')
c2.metric('Violations Remaining', f'{v_count:,} mins')
c3.metric('Annual BESS Throughput', f'{a_bess:,.0f} MWh')
c4.metric('Annual Full Equivalent Cycle Count', f'{a_bess / (2 * enr_cap * (soc_max - soc_min)):.1f} cycles')

st.markdown('<div class="section-header">Annual Energy Budget (100 MW POC)</div>', unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)
c5.metric('Total Solar Generation', f'{a_solar:,.0f} MWh')
c6.metric('Grid Exported', f'{a_export:,.0f} MWh')
c7.metric('Inherent Curtailment', f'{a_curtail:,.0f} MWh')
c8.metric('Curtailment Ratio', f'{(a_curtail/a_solar*100):.1f}%')

st.markdown('<div class="section-header">Daily Energy Budget (Selected Event Day)</div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4)
d1.metric('Daily Solar Generation', f"{d_stats['solar']:.1f} MWh")
d2.metric('Daily Export', f"{d_stats['export']:.1f} MWh")
d3.metric('Daily Curtailment', f"{d_stats['curtail']:.1f} MWh")
d4.metric('Daily BESS Throughput', f"{d_stats['bess']:.1f} MWh")

# --- Main Plot ---
st.markdown('<div class="section-header">Engineering Detail: Minute-by-Minute Dispatch</div>', unsafe_allow_html=True)
s, e = selected_day * 1440, (selected_day + 1) * 1440
times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=pv_signal[s:e], name='Raw Solar Signal', line=dict(dash='dot', color='#8b949e', width=1)))
fig.add_trace(go.Scatter(x=times, y=bess[s:e], name='BESS Power (MW)', fill='tozeroy', fillcolor='rgba(63, 185, 80, 0.15)', line=dict(color='#3fb950', width=1.5)))
fig.add_trace(go.Scatter(x=times, y=export[s:e], name='Grid Export (POC Limit)', line=dict(color='#58a6ff', width=2.5)))
fig.add_trace(go.Scatter(x=times, y=soc[s:e], name='BESS SOC (%)', yaxis='y2', line=dict(color='#f2cc60', width=2)))

fig.update_layout(
    hovermode='x unified', height=600,
    xaxis=dict(title='Time of Day (HH:MM)', nticks=24, gridcolor='#30363d'),
    yaxis=dict(title='Power (MW)', gridcolor='#30363d'),
    yaxis2=dict(title='State of Charge (%)', overlaying='y', side='right', range=[0, 100], showgrid=False),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c9d1d9'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
st.plotly_chart(fig, use_container_width=True)
