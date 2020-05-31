import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import tools

# set the shape of aray
s = 3

# initialize the scalar field
X, Y, Z = np.mgrid[-s:s:30j, -s:s:30j, -s:s:30j]
val = np.sin(X*Y*Z) / (X*Y*Z)

# gradient
dx = (val[:-2, 1:-1, 1:-1] - val[2:, 1:-1, 1:-1])*0.5
dy = (val[1:-1, :-2, 1:-1] - val[1:-1, 2:, 1:-1])*0.5
dz = (val[1:-1, 1:-1, :-2] - val[1:-1, 1:-1, 2:])*0.5


# Initialize figure with 1 subplot
fig = make_subplots(
    rows=1, cols=1)

# add the vector field as 3d cones
fig.add_trace(
    go.Cone(
        # position
        x=X[1:-1, 1:-1, 1:-1].flatten(),
        y=Y[1:-1, 1:-1, 1:-1].flatten(),
        z=Z[1:-1, 1:-1, 1:-1].flatten(),
        # vector value
        u=dx.flatten(),
        v=dy.flatten(),
        w=dz.flatten(),
        sizemode="absolute",
        colorscale='Blues',
        opacity=1,
        sizeref=0.4,
        colorbar=dict(len=1, x=1),
    )
)

# add the scalar field as volume
fig.add_trace(
    go.Volume(
        # position
        x=X[1:-1, 1:-1, 1:-1].flatten(),
        y=Y[1:-1, 1:-1, 1:-1].flatten(),
        z=Z[1:-1, 1:-1, 1:-1].flatten(),
        # scalar value
        value=val[1:-1, 1:-1, 1:-1].flatten(),
        # min iso value
        isomin=0,
        # max iso value
        isomax=1,
        colorscale='Reds',
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=17,  # needs to be a large number for good volume rendering
        colorbar=dict(len=1, x=0.95),
    )
)

# set to dark mode
fig.layout.template = 'plotly_dark'

# write plot to html
# html_path = "SETUPS/PY_Num_Differentiation/Numerical_Differentiation.html"
# fig.write_html(html_path)

# show the figure
fig.show()
