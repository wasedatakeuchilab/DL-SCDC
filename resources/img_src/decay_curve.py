import pathlib

import numpy as np
import plotly.express as px
from scipy import integrate

random = np.random.RandomState(321)
normalize = lambda x: x / max(x)

t0 = 0.2
y0 = 0.1
sigma = 0.01
tau = 0.1
time = np.linspace(0.0, 1.0, 480)
sol = integrate.solve_ivp(
    lambda t, y: -y / tau + np.exp(-((t - t0) ** 2) / (sigma**2)),
    t_span=(min(time), max(time)),
    y0=[0.0],
    method="Radau",
    t_eval=time,
)
px.line(
    x=time,
    y=normalize(sol.y[0]) * (1.0 - y0)
    + y0
    + random.normal(0, (1.0 - y0) * 1e-2, size=time.shape),
).update_layout(xaxis_title="Time", yaxis_title="$y(t)$", showlegend=False).add_scatter(
    x=[t0 - 2 * sigma],
    y=[y0],
    mode="markers+text",
    marker=dict(color="black", size=8),
    text=["$\\text{SCDC}(t_0, y_0)$"],
    textposition="top right",
).add_hline(
    y0, line=dict(dash="dash")
).add_vline(
    t0 - 2 * sigma, line=dict(dash="dash")
).write_image(
    pathlib.Path(__file__).parent / "decay_curve.png", format="png"
)
