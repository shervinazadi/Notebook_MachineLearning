import plotly.figure_factory as pff
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

# Loading the dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)  # Feature Matrix
Y = pd.DataFrame(boston.target, columns=["MEDV"])  # Target Variable


# Filtering: Pearson correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

Data = pd.concat([X, Y], axis=1)
CoRel = Data.corr()

# Plotting the heat map of Pearson Correlation for all features in relation to each other
# fig_1 = pff.create_annotated_heatmap(np.array(CoRel).round(decimals=2), colorscale='RdBu', x=list(Data.columns.values), y=list(Data.columns.values))
# fig_1.show()

# We are looking for correlation above 0.5 or below -0.5
CoRel_Abs = abs(CoRel["MEDV"])
SelFeat = CoRel_Abs[CoRel_Abs > 0.5]

# We need to make sure that Selected Features are independant of each other
SelCoRel = Data[SelFeat.index.values].corr()

# Plotting the heat map of Selected Features Correlation
# fig_2 = pff.create_annotated_heatmap(np.array(SelCoRel).round(decimals=2), colorscale='RdBu', x=list(SelCoRel.columns.values), y=list(SelCoRel.columns.values))
# fig_2.show()

# Checking for correlation of Selected feature beaing above 0.5 or below -0.5
mask = np.array(0.5 < abs(SelCoRel.iloc[:-1, :-1]))  # droping target (MEDV)
# droping the diagonal elements (since they are meaningless) and upper triangle (since it is repetative)
mask *= np.tri(mask.shape[0], mask.shape[0], -1, dtype="bool")
# finding the location of coorelated selected features
SelCoRelInd = np.argwhere(mask)

# checking the correlation of them (correlated selected features) with target
TargetSelCoRel = np.array(abs(SelCoRel.iloc[-1]))

# find the index of minimum correlation in each row
MinInd = np.argmin(TargetSelCoRel[SelCoRelInd], axis=1)
# find the index of non independant features
NonIndFeat = SelCoRelInd[np.arange(MinInd.size), MinInd]

# set dependand features to NaN and drop them
SelFeat.iloc[NonIndFeat] = None
SelFeat = SelFeat.dropna()
print(SelFeat)
