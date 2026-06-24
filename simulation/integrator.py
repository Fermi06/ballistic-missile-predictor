from scipy.integrate import solve_ivp
import numpy as np


def ground_impact_event(t, state):
    return state[2]  # altitude

ground_impact_event.terminal = True
ground_impact_event.direction = -1


def propagate(
    missile,
    initial_state,
    duration=1000,
    dt=0.1
):

    t_eval = np.arange(
        0,
        duration,
        dt
    )

    solution = solve_ivp(
        missile.derivatives,
        [0, duration],
        initial_state,
        t_eval=t_eval,
        events=ground_impact_event,
        rtol=1e-6,
        atol=1e-6
    )

    return solution