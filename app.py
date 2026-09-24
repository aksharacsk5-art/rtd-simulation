import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Interactive RTD Simulation",
    page_icon="⚛️",
    layout="wide"
)


# ============================================================
# DARK MODE COLORS
# ============================================================

BACKGROUND = "#061522"
PANEL = "#0B1F2E"
GRID = "#29404F"
TEXT = "#E6F1F5"
CYAN = "#39D5E8"
ORANGE = "#FFB45C"
GREEN = "#58D68D"


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BACKGROUND};
        color: {TEXT};
    }}

    h1, h2, h3 {{
        color: {TEXT} !important;
    }}

    .stMarkdown {{
        color: {TEXT};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

hbar = 1.055e-34
m = 9.11e-31
e = 1.602e-19


# ============================================================
# ENERGY RANGE
# ============================================================

E_eV = np.linspace(0.001, 0.5, 1500)
E = E_eV * e


# ============================================================
# TRANSMISSION CALCULATION
# ============================================================

def calculate_transmission(V0_eV, a_nm, L_nm):

    V0 = V0_eV * e
    a = a_nm * 1e-9
    L = L_nm * 1e-9

    T = np.zeros(len(E))

    for i, energy in enumerate(E):

        k = np.sqrt(2 * m * energy) / hbar

        if energy < V0:

            alpha = np.sqrt(
                2 * m * (V0 - energy)
            ) / hbar

            Tb = np.exp(-2 * alpha * a)

        else:

            Tb = 1.0

        phase = k * L

        resonance = 1 / (
            1 + 20 * np.sin(phase) ** 2
        )

        T[i] = Tb * resonance

    if np.max(T) > 0:
        T = T / np.max(T)

    return T


# ============================================================
# FIND RESONANCE
# ============================================================

def find_resonance(T):

    index = np.argmax(T)

    return E_eV[index], T[index]


# ============================================================
# SIMPLIFIED RTD I-V MODEL
# ============================================================

def calculate_current(voltage, resonance_energy):

    V_peak = resonance_energy

    width = 0.035

    current = (
        voltage
        * np.exp(
            -((voltage - V_peak) ** 2)
            / (2 * width ** 2)
        )
    )

    current += 0.015 * voltage

    current = current / np.max(current)

    return current


# ============================================================
# TITLE
# ============================================================

st.title("INTERACTIVE RESONANT TUNNELING DIODE (RTD)")

st.markdown(
    "Adjust the parameters below and observe how resonant "
    "tunneling affects transmission and the RTD I–V characteristic."
)


# ============================================================
# SIDEBAR SLIDERS
# ============================================================

st.sidebar.header("RTD Parameters")

V0 = st.sidebar.slider(
    "Barrier Height V₀ (eV)",
    min_value=0.10,
    max_value=0.45,
    value=0.30,
    step=0.01
)

a = st.sidebar.slider(
    "Barrier Width a (nm)",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1
)

L = st.sidebar.slider(
    "Well Width L (nm)",
    min_value=2.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)


# ============================================================
# CALCULATIONS
# ============================================================

T = calculate_transmission(
    V0,
    a,
    L
)

resonance_energy, resonance_T = find_resonance(T)

voltage = np.linspace(0, 0.5, 800)

current = calculate_current(
    voltage,
    resonance_energy
)

peak_index = np.argmax(current)

peak_voltage = voltage[peak_index]

ndr_end_voltage = min(
    peak_voltage + 0.10,
    0.50
)

ndr_end_index = np.argmin(
    np.abs(
        voltage - ndr_end_voltage
    )
)


# ============================================================
# DISPLAY CURRENT PARAMETERS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Barrier Height",
        f"{V0:.2f} eV"
    )

with col2:
    st.metric(
        "Resonance Energy",
        f"{resonance_energy:.3f} eV"
    )

with col3:
    st.metric(
        "Peak Voltage",
        f"{peak_voltage:.3f} V"
    )


# ============================================================
# TRANSMISSION GRAPH
# ============================================================

fig1, ax1 = plt.subplots(
    figsize=(10, 4.5)
)

fig1.patch.set_facecolor(BACKGROUND)

ax1.set_facecolor(PANEL)

ax1.plot(
    E_eV,
    T,
    color=CYAN,
    linewidth=2.5
)

ax1.plot(
    resonance_energy,
    resonance_T,
    "o",
    color=ORANGE,
    markersize=8
)

ax1.set_xlabel(
    "Electron Energy (eV)",
    color=TEXT
)

ax1.set_ylabel(
    "Transmission Probability T(E)",
    color=TEXT
)

ax1.set_title(
    "Transmission Probability vs Electron Energy",
    color=TEXT
)

ax1.set_xlim(0, 0.50)
ax1.set_ylim(0, 1.05)

ax1.tick_params(
    colors=TEXT
)

for spine in ax1.spines.values():
    spine.set_color(GRID)

ax1.grid(
    True,
    color=GRID,
    alpha=0.5
)

st.pyplot(
    fig1,
    use_container_width=True
)

plt.close(fig1)


# ============================================================
# I-V GRAPH
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(10, 4.5)
)

fig2.patch.set_facecolor(BACKGROUND)

ax2.set_facecolor(PANEL)

ax2.plot(
    voltage,
    current,
    color=ORANGE,
    linewidth=2.5
)

# Peak marker

ax2.plot(
    peak_voltage,
    current[peak_index],
    "o",
    color=CYAN,
    markersize=8
)

# NDR end marker

ax2.plot(
    voltage[ndr_end_index],
    current[ndr_end_index],
    "o",
    color=GREEN,
    markersize=7
)

# NDR starting line

ax2.axvline(
    peak_voltage,
    color=CYAN,
    linestyle="--",
    linewidth=1.5
)

# NDR ending line

ax2.axvline(
    voltage[ndr_end_index],
    color=GREEN,
    linestyle="--",
    linewidth=1.5
)

# NDR label

ax2.text(
    (peak_voltage + voltage[ndr_end_index]) / 2,
    0.25,
    "NDR",
    color=TEXT,
    fontsize=12,
    fontweight="bold",
    ha="center"
)

ax2.set_xlabel(
    "Voltage (V)",
    color=TEXT
)

ax2.set_ylabel(
    "Normalized Current",
    color=TEXT
)

ax2.set_title(
    "RTD Current–Voltage Characteristic",
    color=TEXT
)

ax2.set_xlim(0, 0.50)
ax2.set_ylim(0, 1.05)

ax2.tick_params(
    colors=TEXT
)

for spine in ax2.spines.values():
    spine.set_color(GRID)

ax2.grid(
    True,
    color=GRID,
    alpha=0.5
)

st.pyplot(
    fig2,
    use_container_width=True
)

plt.close(fig2)


# ============================================================
# EXPLANATION
# ============================================================

st.markdown(
    f"""
    ### Simulation Result

    **Resonance energy:** {resonance_energy:.3f} eV

    At resonance, the transmission probability reaches a maximum.
    The corresponding I–V curve shows a peak followed by a
    **Negative Differential Resistance (NDR)** region.
    """
)