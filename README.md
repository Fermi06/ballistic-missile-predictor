# 🛰️ Air Defense Command Center

A real-time missile tracking and impact prediction system built with Python. Simulates a physics-based ballistic trajectory, tracks it using an Extended Kalman Filter, predicts the impact zone via Monte Carlo analysis, and displays everything on a live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## Features

- **Physics simulation** — ballistic flight model with atmospheric drag and altitude-dependent wind
- **Radar modelling** — converts Cartesian state to range/azimuth/elevation with realistic measurement noise
- **Extended Kalman Filter (EKF)** — non-linear state estimation from noisy radar returns
- **Monte Carlo impact prediction** — 1000-sample covariance-weighted impact cloud with confidence scoring
- **Machine learning pipeline** — Random Forest trained on simulated trajectory data to predict impact coordinates
- **Live Streamlit dashboard** — trajectory plot, impact cloud, tactical map, threat level, and TTI updated on every run

---

## Project Structure

```
├── simulation/
│   ├── missile.py          # Physics model (drag, gravity, wind)
│   ├── integrator.py       # ODE integrator with ground-impact event
│   ├── atmosphere.py       # Exponential atmosphere density model
│   └── wind.py             # Altitude-layered wind profile
│
├── radar/
│   ├── measurements.py     # Cartesian → range/azimuth/elevation
│   └── noise.py            # Gaussian measurement noise
│
├── tracking/
│   ├── ekf.py              # Extended Kalman Filter with numerical Jacobian
│   └── fusion.py           # Covariance-weighted sensor fusion
│
├── prediction/
│   ├── impact_predictor.py # Euler integrator for fast impact estimation
│   ├── monte_carlo.py      # Monte Carlo impact cloud generator
│   └── confidence.py       # Spread-normalized confidence scoring
│
├── ml/
│   ├── dataset_generator.py # Simulates trajectories and labels impact points
│   ├── models.py            # Random Forest and Gradient Boosting regressors
│   └── train.py             # Training, evaluation, and model export
│
├── ui/
│   └── dashboard.py        # Streamlit dashboard
│
└── main.py                 # Dataset generation entry point
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate training data

```bash
python main.py
```

This simulates 5000 random trajectories and saves `missile_dataset.csv`.

### 3. Train the ML model

```bash
python -m ml.train
```

Trains a Random Forest regressor and saves `impact_predictor.pkl`.

### 4. Launch the dashboard

```bash
streamlit run ui/dashboard.py
```

Press **R** in the browser to re-simulate a new randomised threat scenario.

---

## How It Works

1. A missile is launched with randomised initial velocity components
2. The physics engine integrates the trajectory under gravity, atmospheric drag, and wind
3. Simulated radar measurements (range, azimuth, elevation) are generated with Gaussian noise
4. The EKF fuses noisy measurements into a clean state estimate at each timestep
5. Impact is predicted from the mid-flight state using a fast Euler integrator
6. Monte Carlo sampling perturbs the state by the EKF covariance to generate an impact cloud
7. Confidence is scored by how tightly clustered the cloud is (normalized spread)
8. Threat level is classified by time-to-impact (TTI)

---

## Techniques Used

| Area                  | Technique                                        |
| --------------------- | ------------------------------------------------ |
| State estimation      | Extended Kalman Filter (EKF)                     |
| Trajectory simulation | SciPy `solve_ivp` with event detection           |
| Impact prediction     | Monte Carlo sampling + Euler integration         |
| Machine learning      | Random Forest / Gradient Boosting (scikit-learn) |
| Sensor modelling      | Spherical coordinate radar with noise            |
| Visualisation         | Plotly + Streamlit                               |

---

## Requirements

```
numpy
scipy
pandas
scikit-learn
streamlit
plotly
joblib
```

---

## Disclaimer

This project is a simulation for educational and portfolio purposes. It does not represent any classified system or real operational capability.
# ballistic-missile-predictor
