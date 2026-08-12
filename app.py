import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
RAMP_LIMIT_MW_PER_MIN = 3

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

st.markdown('<div class="main-header">🔋 BESS ±3 MW/min Ramp Compliance Simulator</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Physical & Simulation Parameters")
pwr_cap = st.sidebar.number_input('BESS Power Limit (MW)', 0, 4000, 20)
enr_cap = st.sidebar.number_input('BESS Energy Capacity (MWh)', 1, 8000, 40)
init_soc_pct = st.sidebar.number_input('Initial Year SOC (%)', 0, 100, 50)
eff_one_way = st.sidebar.number_input('One-Way Efficiency', 0.80, 1.00, 0.97, step=0.01)

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
def calculate_no_bess_compliance(pv_data):

    no_bess_compliant_minutes = 0
    daytime_minutes = 0

    for t in range(1, len(pv_data)):

        # Only consider daytime operation
        if pv_data[t] > 0.1:

            daytime_minutes += 1

            # Apply the 100 MW plant export limit
            current_pv = min(pv_data[t], 100.0)
            previous_pv = min(pv_data[t-1], 100.0)

            # Natural solar ramp without BESS
            raw_ramp = current_pv - previous_pv

            # Check whether solar naturally satisfies ±3 MW/min
            if abs(raw_ramp) <= RAMP_LIMIT_MW_PER_MIN:
                no_bess_compliant_minutes += 1

    return no_bess_compliant_minutes, daytime_minutes
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
    charging_minutes, discharging_minutes = 0, 0
    manageable_minutes = 0
    up_ramp_violations = 0
    down_ramp_violations = 0
    largest_up_ramp = -np.inf
    largest_down_ramp = np.inf
    largest_up_ramp_idx = None
    largest_down_ramp_idx = None
    largest_raw_solar_up_ramp = -np.inf
    largest_raw_solar_down_ramp = np.inf 
    largest_raw_solar_up_ramp_idx = None
    largest_raw_solar_down_ramp_idx = None
    t_solar, t_export, t_curtail_inh, t_curtail_ramp, t_bess_mwh = 0, 0, 0, 0, 0
    e_min, e_max = s_min * e_cap, s_max * e_cap


    for t in range(n):
        pv = pv_data[t]
        if pv > 0.1: day_mins += 1
        t_solar += pv / 60

        prev_export = (
        grid_export[t-1]
        if t > 0
        else min(pv, 100.0)
        )
        raw_pv_capped = min(pv, 100.0)
        t_curtail_inh += max(0, pv - 100.0) / 60
        raw_ramp = raw_pv_capped - prev_export
        # --------------------------------------------------
        # Raw solar ramp before BESS action
        # --------------------------------------------------
        
        if t > 0:
        
            raw_solar_ramp = pv - pv_data[t-1]
        
            if raw_solar_ramp > largest_raw_solar_up_ramp:
                largest_raw_solar_up_ramp = raw_solar_ramp
                largest_raw_solar_up_ramp_idx = t
        
            if raw_solar_ramp < largest_raw_solar_down_ramp:
                largest_raw_solar_down_ramp = raw_solar_ramp
                largest_raw_solar_down_ramp_idx = t

        target = 0
        
        if raw_ramp > RAMP_LIMIT_MW_PER_MIN:
            target = (raw_ramp - RAMP_LIMIT_MW_PER_MIN)
        
        elif raw_ramp < -RAMP_LIMIT_MW_PER_MIN:
            target = (raw_ramp + RAMP_LIMIT_MW_PER_MIN)
        
        actual_bess = 0
        
        if target > 0:  # charge
        
            available_pwr = (
                (e_max - curr_energy) * 60
            ) / eff
        
            actual_bess = min(
                target,
                p_cap,
                available_pwr
            )
        
            curr_energy += (
                actual_bess * eff
            ) / 60
        
            if round(target,3) > round(actual_bess,3):
                t_curtail_ramp += (target - actual_bess) / 60
        
        elif target < 0:  # discharge
        
            available_pwr = (
                (curr_energy - e_min) * 60
            ) * eff
        
            actual_bess = -min(
                abs(target),
                p_cap,
                available_pwr
            )
            curr_energy += (
                actual_bess / eff
            ) / 60
        
        exp = raw_pv_capped - actual_bess
        grid_export[t], bess_pwr[t], soc_history[t] = exp, actual_bess, (curr_energy / e_cap) * 100
        t_export += exp / 60
        t_bess_mwh += abs(actual_bess) / 60

        
             # --------------------------------------------------
        # Check final grid-export ramp after BESS action
        # --------------------------------------------------

        final_ramp = exp - prev_export

        is_violation = (
            pv > 0.1
            and round(abs(final_ramp), 3) > round(RAMP_LIMIT_MW_PER_MIN, 3)
        )

        # --------------------------------------------------
        # Track up-ramp and down-ramp violations
        # --------------------------------------------------

        if is_violation:

            violations += 1

            # Up-ramp violation
            if final_ramp > RAMP_LIMIT_MW_PER_MIN:

                up_ramp_violations += 1

                if final_ramp > largest_up_ramp:
                    largest_up_ramp = final_ramp
                    largest_up_ramp_idx = t

            # Down-ramp violation
            elif final_ramp < -RAMP_LIMIT_MW_PER_MIN:

                down_ramp_violations += 1

                if final_ramp < largest_down_ramp:
                    largest_down_ramp = final_ramp
                    largest_down_ramp_idx = t

        else:

            # Final grid export is manageable/compliant
            if pv > 0.1:
                manageable_minutes += 1

                # Only count BESS operating minutes when compliant
                if actual_bess > 0:
                    charging_minutes += 1

                elif actual_bess < 0:
                    discharging_minutes += 1
         
    return grid_export, bess_pwr, soc_history, violations, day_mins, t_solar, t_export, t_curtail_inh, t_curtail_ramp, t_bess_mwh, charging_minutes, discharging_minutes, day_mins, manageable_minutes, up_ramp_violations, down_ramp_violations, largest_up_ramp, largest_down_ramp, largest_up_ramp_idx, largest_down_ramp_idx, largest_raw_solar_up_ramp, largest_raw_solar_down_ramp, largest_raw_solar_up_ramp_idx, largest_raw_solar_down_ramp_idx
@st.cache_data
def calculate_daily_max_energy_power_only(pv_data, p_cap):

    daily_max_energy = []
    daily_dates = []

    for day in range(365):

        start = day * 1440
        end = (day + 1) * 1440

        day_pv = pv_data[start:end]

        # --------------------------------------------------
        # Each day is completely independent
        # BESS starts at 0 MWh
        # --------------------------------------------------
        cumulative_energy = 0.0
        max_energy = 0.0
        min_energy = 0.0

        # First minute establishes the initial grid export
        previous_export = min(day_pv[0], 100.0)

        for t in range(1, len(day_pv)):

            # 100 MW plant export limit
            pv_capped = min(day_pv[t], 100.0)

            # Natural solar ramp relative to previous export
            raw_ramp = pv_capped - previous_export

            # --------------------------------------------------
            # BESS power required to maintain ±3 MW/min
            #
            # Positive = charging
            # Negative = discharging
            # --------------------------------------------------
            if raw_ramp > RAMP_LIMIT_MW_PER_MIN:

                required_bess = (
                    raw_ramp - RAMP_LIMIT_MW_PER_MIN
                )

            elif raw_ramp < -RAMP_LIMIT_MW_PER_MIN:

                required_bess = (
                    raw_ramp + RAMP_LIMIT_MW_PER_MIN
                )

            else:

                required_bess = 0.0

            # --------------------------------------------------
            # Apply ONLY the BESS power limit
            #
            # No:
            #   - energy capacity
            #   - SOC limit
            #   - efficiency
            #   - initial SOC
            # --------------------------------------------------
            actual_bess = np.clip(
                required_bess,
                -p_cap,
                p_cap
            )

            # --------------------------------------------------
            # Convert MW for one minute into MWh
            # --------------------------------------------------
            cumulative_energy += actual_bess / 60.0

            # Track maximum absolute cumulative movement
            max_energy = max(
                max_energy,
                cumulative_energy
            )
            
            min_energy = min(
                min_energy,
                cumulative_energy
            )

            # --------------------------------------------------
            # Resulting grid export becomes reference
            # for the next minute
            # --------------------------------------------------
            previous_export = (
                pv_capped - actual_bess
            )

        daily_max_energy.append(
            max_energy - min_energy
        )

        daily_dates.append(
            annual_dates[start]
        )

    return daily_dates, daily_max_energy
pv_signal = load_data()
annual_dates = get_annual_dates()
no_bess_compliant_minutes, daytime_minutes = calculate_no_bess_compliance(pv_signal)
ideal_export, ideal_bess = calculate_ideal_bess(pv_signal)
required_energy_dates, required_initial_energy = (calculate_required_initial_energy(pv_signal, pwr_cap))
export, bess, soc, v_count, d_mins, a_solar, a_export, a_curt_inh, a_curt_ramp, a_bess_mwh, charging_minutes, discharging_minutes, day_mins, manageable_minutes, up_ramp_violations, down_ramp_violations, largest_up_ramp, largest_down_ramp, largest_up_ramp_idx, largest_down_ramp_idx, largest_raw_solar_up_ramp, largest_raw_solar_down_ramp, largest_raw_solar_up_ramp_idx, largest_raw_solar_down_ramp_idx = run_sim(
pv_signal,
pwr_cap,
enr_cap,
soc_min,
soc_max,
init_soc_pct,
eff_one_way
)
# --------------------------------------------------
# Ramp event timestamps
# --------------------------------------------------

if largest_up_ramp_idx is not None:
    largest_up_ramp_time = annual_dates[
        largest_up_ramp_idx
    ]
else:
    largest_up_ramp_time = None


if largest_down_ramp_idx is not None:
    largest_down_ramp_time = annual_dates[
        largest_down_ramp_idx
    ]
else:
    largest_down_ramp_time = None


if largest_raw_solar_up_ramp_idx is not None:
    largest_raw_solar_up_ramp_time = annual_dates[
        largest_raw_solar_up_ramp_idx
    ]
else:
    largest_raw_solar_up_ramp_time = None


if largest_raw_solar_down_ramp_idx is not None:
    largest_raw_solar_down_ramp_time = annual_dates[
        largest_raw_solar_down_ramp_idx
    ]
else:
    largest_raw_solar_down_ramp_time = None

daily_net_energy = []
daily_dates = []
daily_dates, daily_max_energy = calculate_daily_max_energy_power_only(
    pv_signal,
    pwr_cap
)
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
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Ramp Compliance (> 0.1 MW)', f'{(d_mins-v_count)/d_mins*100:.2f}%')
c2.metric('Annual Violations', f'{v_count:,} mins')
c3.metric('Total BESS Effort (Throughput)', f'{a_bess_mwh:,.0f} MWh')
c4.metric('Annual Equivalent Full Cycles', f'{a_bess_mwh / (2 * enr_cap * (soc_max-soc_min)):.2f}')
c5.metric('Daytime Minutes (> 0.1 MW)', f'{day_mins:,} mins')

c6, c7, c8, c9, c10 = st.columns(5)

c6.metric('Naturally Compliant Minutes', f'{no_bess_compliant_minutes:,} mins')
c7.metric('Manageable Daytime Minutes',f'{manageable_minutes:,} mins')
c8.metric('Successful Charging Minutes',f'{charging_minutes:,} mins')
c9.metric('Successful Discharging Minutes',f'{discharging_minutes:,} mins')

st.markdown('<div class="section-header">Annual Energy Budget</div>', unsafe_allow_html=True)
c11, c12, c13, c14, c15 = st.columns(5)
c11.metric('Solar Generation', f'{a_solar:,.0f} MWh')
c12.metric('Grid Export', f'{a_export:,.0f} MWh')
c13.metric('Inherent Curtailment', f'{a_curt_inh:,.0f} MWh')
c14.metric('Ramp Curtailment', f'{a_curt_ramp:,.0f} MWh')
c15.metric('Total Curtailment', f'{((a_curt_inh + a_curt_ramp)/a_solar*100):.2f}%')
c10.metric('Successful BESS Operational Minutes',f'{charging_minutes + discharging_minutes:,} mins')
st.markdown(
    '<div class="section-header">Ramp Event Summary</div>',
    unsafe_allow_html=True
)

r1, r2 = st.columns(2)

r1.metric(
    'Up Ramp Violations',
    f'{up_ramp_violations:,} mins'
)

r2.metric(
    'Down Ramp Violations',
    f'{down_ramp_violations:,} mins'
)
r3, r4, r5, r6 = st.columns (4)
r3.metric(
    'Largest Net Export Up Ramp',
    f'+{largest_up_ramp:.2f} MW/min'
)

r4.metric(
    'Largest Net Export Down Ramp',
    f'{largest_down_ramp:.2f} MW/min'
)

r5.metric(
    'Net Export Up Ramp Time',
    largest_up_ramp_time.strftime('%Y-%m-%d %H:%M')
    if largest_up_ramp_time is not None
    else 'None'
)

r6.metric(
    'Net Export Down Ramp Time',
    largest_down_ramp_time.strftime('%Y-%m-%d %H:%M')
    if largest_down_ramp_time is not None
    else 'None'
)
r7, r8, r9, r10 = st.columns(4)

r7.metric(
    'Largest Raw Solar Up Ramp',
    f'+{largest_raw_solar_up_ramp:.2f} MW/min'
)

r8.metric(
    'Largest Raw Solar Down Ramp',
    f'{largest_raw_solar_down_ramp:.2f} MW/min'
)

r9.metric(
    'Raw Solar Up Ramp Time',
    largest_raw_solar_up_ramp_time.strftime('%Y-%m-%d %H:%M')
    if largest_raw_solar_up_ramp_time is not None
    else 'None'
)

r10.metric(
    'Raw Solar Down Ramp Time',
    largest_raw_solar_down_ramp_time.strftime('%Y-%m-%d %H:%M')
    if largest_raw_solar_down_ramp_time is not None
    else 'None'
)

st.markdown(
    '<div class="section-header">Daily Ideal BESS Net Energy Required for ±3 MW/min Ramp Compliance</div>',
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
    '<div class="section-header">Daily Minimum Initial Stored Energy Required for ±3 MW/min Ramp Compliance</div>',
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
    '<div class="section-header">Daily BESS Net Energy for ±3 MW/min Ramp Compliance</div>',
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
# ------------------------------------------
# Daily Maximum BESS Energy Movement
# Power-Limit-Only Calculation
# ------------------------------------------

daily_max_energy = np.array(daily_max_energy)

# --------------------------------------------------
# Create 5 MWh bins
#
# ≤5
# >5-10
# >10-15
# >15-20
# etc.
# --------------------------------------------------

max_energy = np.max(daily_max_energy)

max_bin = max(
    5,
    np.ceil(max_energy / 5) * 5
)

energy_bins = np.arange(
    0,
    max_bin + 5,
    5
)

# --------------------------------------------------
# Calculate frequency
# --------------------------------------------------

energy_counts, _ = np.histogram(
    daily_max_energy,
    bins=energy_bins
)

# --------------------------------------------------
# Create labels
# --------------------------------------------------

energy_labels = []

for i in range(len(energy_bins) - 1):

    lower = energy_bins[i]
    upper = energy_bins[i + 1]

    if i == 0:

        energy_labels.append(
            f"≤{upper:.0f}"
        )

    else:

        energy_labels.append(
            f">{lower:.0f}-{upper:.0f}"
        )

# --------------------------------------------------
# Plot histogram
# --------------------------------------------------

st.markdown(
    '<div class="section-header">'
    'Daily Maximum BESS Energy Movement Distribution'
    '</div>',
    unsafe_allow_html=True
)

fig_energy_hist = go.Figure()

fig_energy_hist.add_trace(
    go.Bar(
        x=energy_labels,
        y=energy_counts,
        text=energy_counts,
        textposition='outside',
        name='Number of Days'
    )
)

fig_energy_hist.update_layout(
    template='plotly_dark',
    height=450,
    bargap=0,
    showlegend=False,
    xaxis_title='Maximum Absolute Daily Energy Movement (MWh)',
    yaxis_title='Frequency (Days)'
)

st.plotly_chart(
    fig_energy_hist,
    use_container_width=True
)
# ------------------------------------------
# Grid Ramp Rate Distribution After BESS
# ------------------------------------------
# Only consider daytime operating periods
grid_ramps = np.abs(np.diff(export))
valid_mask = pv_signal[1:] > 0.1
grid_ramps_active = grid_ramps[valid_mask]
# Round ramps to 3 decimal places
grid_ramps_active = np.round(grid_ramps_active, 3)
# Determine maximum ramp for automatic bins
max_ramp = np.ceil(np.max(grid_ramps_active) / 5) * 5
# Create bins
ramp_bins = [0, 3.0, 5.0] + list(np.arange(10, max_ramp + 5, 5))
# Move exact bin-edge values into the lower bin
edges = np.array(ramp_bins[1:-1])

for edge in edges:
    grid_ramps_active[np.isclose(grid_ramps_active, edge, atol=5e-4)] -= 1e-9

# Labels
ramp_labels = ["≤3"] + [
    f">{ramp_bins[i]}-{ramp_bins[i+1]}"
    for i in range(1, len(ramp_bins)-1)
]

# Histogram
ramp_counts, _ = np.histogram(
    grid_ramps_active,
    bins=ramp_bins
)
st.markdown(
    '<div class="section-header">Grid Ramp Rate Distribution After BESS Action</div>',
    unsafe_allow_html=True
)

fig_ramp_hist = go.Figure()

fig_ramp_hist.add_trace(
    go.Bar(
        x=ramp_labels,
        y=ramp_counts,
        text=ramp_counts,
        textposition='outside',
        name='Frequency'
    )
)

fig_ramp_hist.update_layout(
    template='plotly_dark',
    height=450,
    bargap=0,
    showlegend=False,
    xaxis_title='Grid Ramp Rate Range (MW/min)',
    yaxis_title='Frequency (Minutes)'
)

st.plotly_chart(
    fig_ramp_hist,
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
