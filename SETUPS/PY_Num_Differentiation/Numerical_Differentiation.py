import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# set the shape of aray
s = 3

# set the sphere on the corner of array
X, Y, Z = np.mgrid[-s:s:40j, -s:s:40j, -s:s:40j]
val = np.sin(X*Y*Z) / (X*Y*Z)

# gradient
dx = (val[:-2, 1:-1, 1:-1] - val[2:, 1:-1, 1:-1])*0.5
dy = (val[1:-1, :-2, 1:-1] - val[1:-1, 2:, 1:-1])*0.5
dz = (val[1:-1, 1:-1, :-2] - val[1:-1, 1:-1, 2:])*0.5

# set the vector field
fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Cone(
    x=X[1:-1, 1:-1, 1:-1].flatten(),
    y=Y[1:-1, 1:-1, 1:-1].flatten(),
    z=Z[1:-1, 1:-1, 1:-1].flatten(),
    u=dx.flatten(),
    v=dy.flatten(),
    w=dz.flatten(),
    sizemode="absolute",
    colorscale='Blues',
    opacity=1,
    sizeref=0.5)
)
fig.add_trace(go.Volume(
    x=X[1:-1, 1:-1, 1:-1].flatten(),
    y=Y[1:-1, 1:-1, 1:-1].flatten(),
    z=Z[1:-1, 1:-1, 1:-1].flatten(),
    value=val[1:-1, 1:-1, 1:-1].flatten(),
    isomin=0,
    isomax=1,
    colorscale='Reds',
    opacity=0.1,  # needs to be small to see through all surfaces
    surface_count=17,  # needs to be a large number for good volume rendering
)
)

# set to dark mode
fig.layout.template = 'plotly_dark'

fig.update_layout(height=600, width=800,
                  title_text="Subplots with Annotations")
# show the figure
fig.show()
