import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
RAMP_LIMIT_MW_PER_MIN = 3.0
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

st.markdown('<div class="main-header">BESS ±3 MW/min Ramp Compliance Simulator</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Physical & Simulation Parameters")
pwr_cap = st.sidebar.number_input('BESS Power Limit (MW)', 0, 4000, 20)
enr_cap = st.sidebar.number_input('BESS Energy Capacity (MWh)', 0, 8000, 40)
init_soc_pct = st.sidebar.number_input('Initial Year SOC (%)', 0, 100, 50)
eff_one_way = st.sidebar.number_input('One-Way Efficiency', 0.80, 1.00, 0.96, step=0.01)

soc_choice = st.sidebar.selectbox('Operating SOC Window', ['0% - 100%', '10% - 90%', '20% - 80%', '30% - 70%'])
soc_min, soc_max = [float(x.replace('%', '').strip())/100 for x in soc_choice.split('-')]
st.sidebar.markdown('---')
st.sidebar.header("📅 Key Reference Days")

st.sidebar.markdown("""
<div class="event-note">

<b>05-07 — Worst Ramp Stress</b><br>
Most demanding ramp-compliance day of the year with continuous high-frequency solar fluctuations.

<br>

<b>02-26 — Highest Variability</b><br>
Largest short-duration solar ramps observed, including extreme MW/min increases and decreases.

<br>

<b>03-17 — Largest SOC Swing</b><br>
Deepest sustained charge-discharge cycle and greatest energy utilization.

<br>

<b>03-17 — Highest Solar Generation</b><br>
Highest annual solar production with extended operation near or above the 100 MW export limit.

<br>

<b>12-13 — Lowest Solar Generation</b><br>
Lowest irradiance day and useful baseline for low-stress operation.

</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_path = "power time series 1 min (1)(in).csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)['Solar_MW'].values
    return np.zeros(525600)
@st.cache_data
def get_annual_dates():
    return pd.date_range(
        start='2025-01-01',
        periods=525600,
        freq='min'
    )
@st.cache_data
def calculate_ideal_bess(pv_data):

    pv_capped = np.minimum(pv_data, 100.0)

    ideal_export = np.zeros(len(pv_capped))
    ideal_bess = np.zeros(len(pv_capped))

    ideal_export[0] = pv_capped[0]

    for t in range(1, len(pv_capped)):

        ideal_export[t] = np.clip(
            pv_capped[t],
            ideal_export[t-1] - RAMP_LIMIT_MW_PER_MIN,
            ideal_export[t-1] + RAMP_LIMIT_MW_PER_MIN
        )

        ideal_bess[t] = pv_capped[t] - ideal_export[t]

    return ideal_export, ideal_bess
@st.cache_data
def calculate_required_initial_energy(pv_data, p_cap):

    required_energy = []
    energy_dates = []

    for day in range(365):

        start = day * 1440
        end = (day + 1) * 1440

        day_pv = pv_data[start:end]

        export = np.zeros(len(day_pv))
        bess = np.zeros(len(day_pv))

        export[0] = min(day_pv[0], 100.0)

        for t in range(1, len(day_pv)):

            pv_now = min(day_pv[t], 100.0)

            target_export = np.clip(
                pv_now,
                export[t-1] - RAMP_LIMIT_MW_PER_MIN,
                export[t-1] + RAMP_LIMIT_MW_PER_MIN
            )

            required_bess = pv_now - target_export

            required_bess = np.clip(
                required_bess,
                -p_cap,
                p_cap
            )

            export[t] = pv_now - required_bess
            bess[t] = required_bess

        cumulative_energy = np.cumsum(
            bess / 60.0
        )

        required_energy.append(max(0, -np.min(cumulative_energy)))

        energy_dates.append(
            annual_dates[start]
        )

    return energy_dates, required_energy
@st.cache_data
def run_sim(pv_data, p_cap, e_cap, s_min, s_max, start_soc_pct, eff):
    n = len(pv_data)
    grid_export, bess_pwr, soc_history = np.zeros(n), np.zeros(n), np.zeros(n)
    curr_energy = (start_soc_pct / 100) * e_cap
    violations, day_mins = 0, 0
    t_solar, t_export, t_curtail_inh, t_curtail_ramp, t_bess_mwh = 0, 0, 0, 0, 0

    e_min, e_max = s_min * e_cap, s_max * e_cap


    for t in range(n):
        pv = pv_data[t]
        if pv > 0.5: day_mins += 1
        t_solar += pv / 60

        prev_export = (
        grid_export[t-1]
        if t > 0
        else min(pv, 100.0)
        )
        raw_pv_capped = min(pv, 100.0)
        t_curtail_inh += max(0, pv - 100.0) / 60

               # =====================================================
        # TARGET SOC = UPPER SOC WINDOW LIMIT
        # =====================================================
        
        actual_bess = 0
        
        target_soc_energy = e_max
        
        clipped_power = max(0, pv - 100.0)
        
        # -----------------------------------------------------
        # 1. Charge from clipped energy first
        # -----------------------------------------------------
        
        if curr_energy < target_soc_energy and clipped_power > 0:
        
            available_charge_power = (
                (target_soc_energy - curr_energy) * 60
            ) / eff
        
            clip_charge = min(
                clipped_power,
                p_cap,
                available_charge_power
            )
        
            curr_energy += (clip_charge * eff) / 60
        
            actual_bess += clip_charge
        
        # -----------------------------------------------------
        # 2. Ramp calculation
        # -----------------------------------------------------
        
        raw_ramp = raw_pv_capped - prev_export
        
        remaining_absorption = 0
        
        # -----------------------------------------------------
        # RAMP UP
        # -----------------------------------------------------
        
        if raw_ramp > RAMP_LIMIT_MW_PER_MIN:
        
            required_absorption = (
                raw_ramp - RAMP_LIMIT_MW_PER_MIN
            )
        
            if curr_energy < target_soc_energy:
        
                available_charge_power = (
                    (target_soc_energy - curr_energy) * 60
                ) / eff
        
                remaining_power_capacity = max(0, p_cap - actual_bess)
                ramp_charge = min(required_absorption, remaining_power_capacity, available_charge_power)
        
                curr_energy += (ramp_charge * eff) / 60
        
                actual_bess += ramp_charge
        
                remaining_absorption = (
                    required_absorption - ramp_charge
                )
        
            else:
        
                remaining_absorption = (
                    required_absorption
                )
        
        # -----------------------------------------------------
        # RAMP DOWN
        # -----------------------------------------------------
        
        elif raw_ramp < -RAMP_LIMIT_MW_PER_MIN:
        
            required_discharge = abs(
                raw_ramp + RAMP_LIMIT_MW_PER_MIN
            )
        
            available_discharge_power = (
                (curr_energy - e_min) * 60
            ) * eff
        
            discharge = min(
                required_discharge,
                p_cap,
                available_discharge_power
            )
        
            curr_energy -= (
                discharge / eff
            ) / 60
        
            actual_bess -= discharge
        
        # -----------------------------------------------------
        # EXPORT CALCULATION
        # -----------------------------------------------------
        
        exp = raw_pv_capped
        
        # charging not supplied by clipping
        exp -= max(
            0,
            actual_bess - clipped_power
        )
        
        # discharge support
        if actual_bess < 0:
            exp += abs(actual_bess)
        
        # inverter clipping when SOC full
        exp -= remaining_absorption
        
        t_curtail_ramp += remaining_absorption / 60
        grid_export[t], bess_pwr[t], soc_history[t] = exp, actual_bess, (curr_energy / e_cap) * 100
        t_export += exp / 60
        t_bess_mwh += abs(actual_bess) / 60

        if pv > 0.5 and abs(exp - prev_export) > (RAMP_LIMIT_MW_PER_MIN + 0.001): violations += 1

    return grid_export, bess_pwr, soc_history, violations, day_mins, t_solar, t_export, t_curtail_inh, t_curtail_ramp, t_bess_mwh

pv_signal = load_data()
annual_dates = get_annual_dates()
ideal_export, ideal_bess = calculate_ideal_bess(pv_signal)
required_energy_dates, required_initial_energy = (calculate_required_initial_energy(pv_signal, pwr_cap))
export, bess, soc, v_count, d_mins, a_solar, a_export, a_curt_inh, a_curt_ramp, a_bess_mwh = run_sim(
    pv_signal,
    pwr_cap,
    enr_cap,
    soc_min,
    soc_max,
    init_soc_pct,
    eff_one_way
)
daily_net_energy = []
daily_dates = []

for day in range(365):

    start = day * 1440
    end = (day + 1) * 1440

    day_pv = pv_signal[start:end]

    (
        _,
        day_bess,
        _,
        *_
    ) = run_sim(
        day_pv,
        pwr_cap,
        enr_cap,
        soc_min,
        soc_max,
        init_soc_pct,
        eff_one_way
    )

    daily_net_energy.append(
        np.sum(day_bess) / 60
    )

    daily_dates.append(
        annual_dates[start]
    )
ideal_df = pd.DataFrame({
    "time": annual_dates,
    "energy": (ideal_bess) / 60.0
})

daily_ideal_energy = (
    ideal_df
    .resample("D", on="time")["energy"]
    .sum()
)

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

st.markdown(
    '<div class="section-header">Ideal BESS Daily Net Energy Required for ±3 MW/min Ramp Compliance</div>',
    unsafe_allow_html=True
)

fig_ideal = go.Figure()

fig_ideal.add_trace(
    go.Bar(
        x=daily_ideal_energy.index,
        y=daily_ideal_energy.values,
        name="Ideal BESS Activity (MWh/day)"
    )
)

fig_ideal.update_layout(
    template="plotly_dark",
    height=450,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Net BESS Energy (MWh)"
)
st.plotly_chart(fig_ideal, use_container_width=True)
st.markdown(
    '<div class="section-header">Daily Minimum Initial Stored Energy Required</div>',
    unsafe_allow_html=True
)

fig_energy = go.Figure()

fig_energy.add_trace(
    go.Bar(
        x=required_energy_dates,
        y=required_initial_energy,
        name='Required Initial Energy (MWh)'
    )
)

fig_energy.update_layout(
    template='plotly_dark',
    height=450,
    hovermode='x unified',
    xaxis_title='Date',
    yaxis_title='Required Initial Energy (MWh)'
)

st.plotly_chart(
    fig_energy,
    use_container_width=True
)
st.markdown(
    '<div class="section-header">BESS Daily Net Energy for ±3 MW/min Ramp Compliance</div>',
    unsafe_allow_html=True
)

fig_daily = go.Figure()

fig_daily.add_trace(
    go.Bar(
        x=daily_dates,
        y=daily_net_energy,
        name='Net BESS Energy (MWh/day)'
    )
)

fig_daily.update_layout(
    template='plotly_dark',
    height=450,
    hovermode='x unified',
    xaxis_title='Date',
    yaxis_title='Net BESS Energy (MWh)'
)

st.plotly_chart(
    fig_daily,
    use_container_width=True
)
st.markdown(
    '<div class="section-header">Annual: Solar, BESS Dispatch, and SOC Response for ±3 MW/min Ramp Compliance</div>',
    unsafe_allow_html=True
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=annual_dates, y=pv_signal, name='Raw Solar (MW)', line=dict(color='#8b949e', dash='dot')))
fig.add_trace(go.Scatter(x=annual_dates, y=export, name='Net Export (MW)', line=dict(color='#58a6ff', width=2)))
fig.add_trace(go.Scatter(x=annual_dates, y=bess, name='BESS (MW)', fill='tozeroy', line=dict(color='#238636', width=1)))
fig.add_trace(go.Scatter(x=annual_dates, y=soc, name='SOC (%)', yaxis='y2', line=dict(color='#f2cc60', width=2)))
fig.update_layout(hovermode='x unified', xaxis=dict(title='Date'), yaxis=dict(title='Power (MW)'), yaxis2=dict(overlaying='y', side='right', range=[0,100], title='SOC (%)'), template='plotly_dark', height=550, legend=dict(orientation='h', y=1.1))
st.plotly_chart(fig, use_container_width=True)
