import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from prediction.confidence import compute_confidence  # removed duplicate import
from prediction.monte_carlo import MonteCarloImpact
from simulation.missile import Missile
from simulation.integrator import propagate
from radar.measurements import cartesian_to_radar
from radar.noise import add_measurement_noise
from tracking.ekf import ExtendedKalmanFilter
from prediction.impact_predictor import ImpactPredictor

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Air Defense Command Center",
    layout="wide"
)

st.title(
    "🛰️ Air Defense Command Center"
)

# ---------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------

missile = Missile()

initial_state = [
    0.0,
    0.0,
    0.0,
    np.random.uniform(400, 1200),   # vx
    np.random.uniform(-200, 200),   # vy
    np.random.uniform(400, 1200),   # vz
]

sol = propagate(
    missile,
    initial_state,
    duration=300,
    dt=0.5
)

# ---------------------------------------------------
# TRACK TARGET
# ---------------------------------------------------

ekf = ExtendedKalmanFilter()

ekf.x = np.array([
    sol.y[0][0],
    sol.y[1][0],
    sol.y[2][0],
    sol.y[3][0],
    sol.y[4][0],
    sol.y[5][0]
])

true_positions = []
estimated_positions = []

for i in range(len(sol.t)):

    true_state = np.array([
        sol.y[0][i],
        sol.y[1][i],
        sol.y[2][i],
        sol.y[3][i],
        sol.y[4][i],
        sol.y[5][i]
    ])

    measurement = cartesian_to_radar(true_state)
    measurement = add_measurement_noise(measurement)

    ekf.predict(0.5)
    ekf.update(measurement)

    true_positions.append(true_state[:3])
    estimated_positions.append(ekf.get_position())

true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# ---------------------------------------------------
# IMPACT PREDICTION
# ---------------------------------------------------

predictor = ImpactPredictor()

# Use state at 1/3 of track — missile still in flight, not yet at ground
mid = len(sol.t) // 3
mid_state = np.array([
    sol.y[0][mid], sol.y[1][mid], sol.y[2][mid],
    sol.y[3][mid], sol.y[4][mid], sol.y[5][mid]
])

impact = predictor.predict(mid_state)

impact_x = impact["impact_x"]
impact_y = impact["impact_y"]
time_to_impact = impact["time_to_impact"]

# ---------------------------------------------------
# MONTE CARLO IMPACT CLOUD
# ---------------------------------------------------

mc = MonteCarloImpact(samples=1000)

cloud = mc.run(mid_state, ekf.P)

radius = mc.confidence_radius(cloud)

cloud_x = cloud[:, 0]
cloud_y = cloud[:, 1]

# confidence computed from actual cloud, not stale empty array
confidence = compute_confidence(cloud)

# ---------------------------------------------------
# THREAT LEVEL
# ---------------------------------------------------

if time_to_impact < 30:
    threat = "CRITICAL"
elif time_to_impact < 60:
    threat = "HIGH"
elif time_to_impact < 120:
    threat = "MEDIUM"
else:
    threat = "LOW"

# ---------------------------------------------------
# KPI ROW
# ---------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Threat", threat)
c2.metric("TTI", f"{time_to_impact:.1f}s")
c3.metric("Confidence", f"{confidence:.0f}%")
c4.metric("Track Samples", len(sol.t))

# ---------------------------------------------------
# TRAJECTORY
# ---------------------------------------------------

left, right = st.columns(2)

with left:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=true_positions[:, 0],
            y=true_positions[:, 2],
            mode="lines",
            name="True"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=estimated_positions[:, 0],
            y=estimated_positions[:, 2],
            mode="lines",
            name="EKF"
        )
    )

    fig.update_layout(
        title="Trajectory Tracking",
        xaxis_title="Range (m)",
        yaxis_title="Altitude (m)"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    impact_fig = go.Figure()

    impact_fig.add_trace(
        go.Scatter(
            x=cloud_x,
            y=cloud_y,
            mode="markers",
            marker=dict(size=4),
            name="Impact Cloud"
        )
    )

    impact_fig.add_trace(
        go.Scatter(
            x=[impact_x],
            y=[impact_y],
            mode="markers+text",
            text=["Predicted Impact"],
            textposition="top center"
        )
    )

    impact_fig.update_layout(title="Impact Prediction")

    st.plotly_chart(impact_fig, use_container_width=True)

# ---------------------------------------------------
# TACTICAL MAP
# ---------------------------------------------------

st.subheader("Tactical Situation Display")

map_fig = go.Figure()

map_fig.add_trace(
    go.Scatter(
        x=[0],
        y=[0],
        mode="markers+text",
        text=["Radar"]
    )
)

map_fig.add_trace(
    go.Scatter(
        x=[estimated_positions[-1, 0]],
        y=[estimated_positions[-1, 1]],
        mode="markers+text",
        text=["Target"]
    )
)

map_fig.add_trace(
    go.Scatter(
        x=[impact_x],
        y=[impact_y],
        mode="markers+text",
        text=["Impact"]
    )
)

map_fig.update_layout(
    xaxis_title="East (m)",
    yaxis_title="North (m)",
    height=500
)

st.plotly_chart(map_fig, use_container_width=True)

# ---------------------------------------------------
# CURRENT TRACK
# ---------------------------------------------------

track_df = pd.DataFrame({
    "Parameter": ["X", "Y", "Altitude", "VX", "VY", "VZ"],
    "Value": np.round(ekf.get_state(), 2)
})

st.subheader("Current Track State")
st.dataframe(track_df, use_container_width=True)

# ---------------------------------------------------
# EVENT LOG
# ---------------------------------------------------

events = pd.DataFrame({
    "Event": [
        "Target Detected",
        "Track Initiated",
        "EKF Tracking Active",
        "Impact Prediction Updated",
        "Threat Assessment Generated"
    ]
})

st.subheader("Event Log")
st.dataframe(events, use_container_width=True)