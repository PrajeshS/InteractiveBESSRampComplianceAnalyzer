import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title='BESS Ramp Optimizer', layout='wide')

# --- Header & Logos ---
col1, col2 = st.columns([2, 3])
with col1:
    st.title('BESS Ramp Compliance Simulator')
with col2:
    st.image(['rivilogo.jpg', 'lakdhanavilogo.jpg', 'ltllogo.png'], width=100)

# --- Sidebar Controls ---
st.sidebar.header('Simulation Parameters')
pwr_cap = st.sidebar.slider('BESS Power (MW)', 0, 200, 50)
enr_cap = st.sidebar.slider('BESS Energy (MWh)', 1, 400, 100)
soc_choice = st.sidebar.selectbox('SOC Window', ['30% - 70%', '20% - 80%', '10% - 90%'])
soc_min, soc_max = [float(x.strip('%'))/100 for x in soc_choice.split('-')]

# --- Simulation Logic ---
@st.cache_data
def load_data():
    df = pd.read_csv('power time series 1 min (1)(in).csv')
    return df['Solar_MW'].values

def run_sim(pv_data, p_cap, e_cap, s_min, s_max):
    n = len(pv_data)
    grid_export = np.zeros(n)
    bess_pwr = np.zeros(n)
    soc_history = np.zeros(n)
    curr_energy = 0.5 * e_cap
    
    for t in range(n):
        pv = pv_data[t]
        prev_export = grid_export[t-1] if t > 0 else pv
        eff_pv = min(pv, 100.0)
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
    return grid_export, bess_pwr, soc_history

# --- Execution (Auto-runs on any parameter change) ---
pv_signal = load_data()
export, bess, soc = run_sim(pv_signal, pwr_cap, enr_cap, soc_min, soc_max)

# --- Display Results ---
k1, k2, k3 = st.columns(3)
k1.metric('Annual Export', f'{np.sum(export)/60:,.0f} MWh')
k2.metric('BESS Energy Processed', f'{np.sum(np.abs(bess))/60:,.0f} MWh')
k3.metric('Peak SOC Reach', f'{np.max(soc):.1f}%')

# --- Plotting ---
st.subheader('Daily Operational Detail')
day = st.number_input('Select Day of Year', 0, 364, 156)
s, e = day*1440, (day+1)*1440

fig = go.Figure()
fig.add_trace(go.Scatter(y=pv_signal[s:e], name='Raw Solar', line=dict(dash='dash', color='gray')))
fig.add_trace(go.Scatter(y=export[s:e], name='Grid Export', fill='tozeroy', line=dict(color='#58a6ff')))
fig.add_trace(go.Scatter(y=soc[s:e], name='SOC %', yaxis='y2', line=dict(color='orange')))
fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='SOC %'), hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)
