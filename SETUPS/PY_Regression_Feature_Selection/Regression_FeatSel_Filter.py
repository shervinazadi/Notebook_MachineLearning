import plotly.figure_factory as pff
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

# Loading the dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)  # Feature Matrix
Y = pd.DataFrame(boston.target, columns=["MEDV"])  # Target Variable


# Filtering: Pearson correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
# We are lookinf for correlation above 0.5
Data = pd.concat([X, Y], axis=1)
CoRel = Data.corr()

# Plotting the heat map of Pearson Correlation
# fig = pff.create_annotated_heatmap(np.array(CoRel).round(decimals=2), colorscale='RdBu', x=list(Data.columns.values), y=list(Data.columns.values))
# fig.show()
