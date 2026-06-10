import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title='BESS Ramp Compliance Simulator', layout='wide')

# --- Professional Engineering CSS ---
st.markdown("""
    <style>
    [data-testid='stMetricValue'] { font-size: 2.0rem; color: #58a6ff; }
    .main-header { font-size: 28px; font-weight: bold; color: #58a6ff; margin-bottom: 20px; }
    .section-header { font-size: 14px; font-weight: bold; color: #8b949e; text-transform: uppercase; margin-top: 30px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    .event-note { font-size: 1.0rem; color: #8b949e; margin-top: 10px; font-style: italic; line-height: 1.4; }
    .stNumberInput div[data-baseweb='input'] { background-color: #0d1117; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">BESS Ramp Compliance Simulator</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Physical & Simulation Parameters")
pwr_cap = st.sidebar.number_input('BESS Power Limit (MW)', 0, 2000, 50)
enr_cap = st.sidebar.number_input('BESS Energy Capacity (MWh)', 0, 4000, 100)
init_soc_pct = st.sidebar.number_input('Initial Year SOC (%)', 0, 100, 50)
eff_one_way = st.sidebar.number_input('One-Way Efficiency', 0.80, 1.00, 0.96, step=0.01)

soc_choice = st.sidebar.selectbox('Operating SOC Window', ['0% - 100%', '10% - 90%', '20% - 80%', '30% - 70%'])
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
    'Worst Ramp Stress (06-06)': "Most demanding ramp-compliance day of the year. Features continuous high-frequency solar fluctuations caused by rapidly changing cloud conditions, forcing the BESS to respond almost continuously to maintain export ramp limits. Primarily stresses cumulative battery utilization and overall ramp-control capability.",
    'Highest Variability (02-26)': "Contains the largest short-duration solar power fluctuation observed in the dataset (111.26 MW/min gain and 65.86 MW/min drop), including extreme MW/min ramps caused by dense transient cloud cover. Primarily stresses instantaneous BESS power capability (MW), fast-response performance, and short-term ramp absorption requirements.",
    'Largest SOC Swing (11-28)': "Produces the deepest sustained battery charge-discharge cycle of the year due to prolonged asymmetric solar variability. Primarily stresses usable energy capacity (MWh) and the BESS ability to maintain compliance during extended ramp events without energy depletion.",
    'Highest Solar Generation (03-17)': "Represents the highest overall solar production day of the year, with extended periods operating near or above the 100 MW export limit. Primarily stresses inherent clipping behavior and the interaction between export limiting and ramp-control operation.",
    'Lowest Solar Generation (12-13)': "Low-irradiance baseline operating day with minimal solar variability and limited BESS intervention. Useful for benchmarking idle system behavior, verifying low-stress operational stability, and comparing controller response under near-steady-state conditions."
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

# ------------------------------------------------------------------
# Annual Net Daily BESS Energy (Charge - Discharge)
# ------------------------------------------------------------------

st.markdown(
    '<div class="section-header">Annual Net BESS Energy Balance</div>',
    unsafe_allow_html=True
)

@st.cache_data
def compute_daily_net_bess_energy(bess_power):
    # Convert MW → MWh per minute timestep
    bess_mwh_series = bess_power / 60.0  # signed value

    # reshape into full-year days (365 x 1440)
    daily = bess_mwh_series.reshape(365, 1440)

    # sum per day (signed net energy)
    daily_net = daily.sum(axis=1)

    return daily_net

daily_net_bess_mwh = compute_daily_net_bess_energy(bess)

# --- UI Display ---
c1, c2, c3, c4 = st.columns(4)
c1.metric('Ramp Compliance', f'{(d_mins-v_count)/d_mins*100:.2f}%')
c2.metric('Annual Violations', f'{v_count:,} minutes')
c3.metric('Total BESS Effort', f'{a_bess_mwh:,.0f} MWh')
c4.metric('Annual Equivalent Full Cycles', f'{a_bess_mwh / (2 * enr_cap * (soc_max-soc_min)):.1f}')

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
fig.add_trace(go.Scatter(x=times, y=pv_signal[s:e], name='Raw Solar (MW)', line=dict(color='#8b949e', dash='dot')))
fig.add_trace(go.Scatter(x=times, y=export[s:e], name='Net Export (MW)', line=dict(color='#58a6ff', width=2)))
fig.add_trace(go.Scatter(x=times, y=bess[s:e], name='BESS (MW)', fill='tozeroy', line=dict(color='#238636', width=1)))
fig.add_trace(go.Scatter(x=times, y=soc[s:e], name='SOC (%)', yaxis='y2', line=dict(color='#f2cc60', width=2)))
fig.update_layout(hovermode='x unified', xaxis=dict(title='Time'), yaxis=dict(title='Power (MW)'), yaxis2=dict(overlaying='y', side='right', range=[0,100], title='SOC (%)'), template='plotly_dark', height=550, legend=dict(orientation='h', y=1.1))
st.plotly_chart(fig, use_container_width=True)
# ------------------------------------------------------------------
# Annual SOC Profile (Full-Year Simulation)
# ------------------------------------------------------------------

st.markdown(
    '<div class="section-header">Annual SOC Profile</div>',
    unsafe_allow_html=True
)

soc_lower = soc_min * 100
soc_upper = soc_max * 100



@st.cache_data
def get_annual_dates():
    return pd.date_range(
        start='2025-01-01',
        periods=525600,
        freq='min'
    )
annual_dates = get_annual_dates()

fig_soc = go.Figure()

# Continuous SOC line
fig_soc.add_trace(
    go.Scatter(
        x=annual_dates,
        y=soc,
        mode='lines',
        name='SOC (%)',
        line=dict(color='#f2cc60', width=1),
        hovertemplate='%{x|%d %b}<br>SOC %{y:.2f}%<extra></extra>'
    )
)

# Points where SOC is at a limit
limit_mask = (
    (soc <= soc_lower + 0.01) |
    (soc >= soc_upper - 0.01)
)

fig_soc.add_trace(
    go.Scatter(
        x=annual_dates[limit_mask],
        y=soc[limit_mask],
        mode='markers',
        name='SOC Limit Reached',
        marker=dict(
            color='red',
            size=4
        ),
        hovertemplate='%{x|%d %b}<br>SOC %{y:.2f}%<extra></extra>'
    )
)



fig_soc.add_hline(
    y=soc_lower,
    line_dash='dash',
    line_width=1,
    annotation_position='bottom right',
    annotation_text=f'Lower Limit ({soc_lower:.0f}%)'
)

fig_soc.add_hline(
    y=soc_upper,
    line_dash='dash',
    line_width=1,
    annotation_position='top right',
    annotation_text=f'Upper Limit ({soc_upper:.0f}%)'
)

fig_soc.update_layout(
    template='plotly_dark',
    height=500,
    hovermode='x unified',
    xaxis_title='Date',
    yaxis_title='SOC (%)',
    yaxis=dict(range=[0, 100]),
    legend=dict(
        orientation='h',
        y=1.05
    )
)

st.plotly_chart(fig_soc, use_container_width=True)
fig_daily = go.Figure()

fig_daily.add_trace(
    go.Bar(
        x=np.arange(1, 366),
        y=daily_bess_mwh,
        name="Daily BESS Energy (MWh)",
        marker_color="#58a6ff"
    )
)

fig_daily.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Day of Year",
    yaxis_title="BESS Energy Throughput (MWh/day)",
    hovermode="x unified"
)

st.plotly_chart(fig_daily, use_container_width=True)
