
from sklearn.datasets import load_boston
import pandas as pd
import statsmodels.api as sm

# Loading the dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)  # Feature Matrix
Y = pd.DataFrame(boston.target, columns=["MEDV"])  # Target Variable


# Backward Elimination
# we assume that all the features (columns) in dataset are usefull. Therefore we put their names in Selected Features variable (SelFeat)
SelFeat = list(X.columns)

while (len(SelFeat) > 0):
    # First we make the regression model, in this case Ordinary Least Squares(OLS): https://en.wikipedia.org/wiki/Ordinary_least_squares

    # extracting the selected features (columns) and adding constant 1 to them
    X_1 = sm.add_constant(X[SelFeat])
    # fitting the model
    model = sm.OLS(Y, X_1).fit()

    # Second we extract the pvalues and eleminate the unncessary features

    # extracting pvalues: : https://www.statsdirect.com/help/basics/p_values.html
    pvals = model.pvalues

    # finding the feature with the maximum pvalue
    pmax_val = max(pvals)
    pmax_name = pvals.idxmax()

    # we are looking for pvalue > 0.05 to remove
    if (pmax_val > 0.05):
        # if the pmax value was greater than 0.05 remove it from the list of selected features
        SelFeat.remove(pmax_name)
    else:
        # if the pmax value was not greater than 0.05, break the loop
        break

# print the list of selected features
print(SelFeat)
