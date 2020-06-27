
import sklearn.datasets as skl_ds
import pandas as pd
import sklearn.model_selection as skl_ms
import sklearn.feature_selection as skl_fs
import sklearn.linear_model as skl_lm
import numpy as np

# Loading the dataset
boston = skl_ds.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)  # Feature Matrix
Y = pd.DataFrame(boston.target, columns=["MEDV"])  # Target Variable

# no of features
nof_list = np.arange(1, 13)
high_score = 0
# Variable to store the optimum features
nof = 0
score_list = []
for n in range(len(nof_list)):
    X_train, X_test, Y_train, Y_test = skl_ms.train_test_split(
        X, Y, test_size=0.3, random_state=0)
    model = skl_lm.LinearRegression()
    rfe = skl_fs.RFE(model, n_features_to_select=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train, Y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe, Y_train)
    score = model.score(X_test_rfe, Y_test)
    score_list.append(score)
    if(score > high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" % nof)
print("Score with %d features: %f" % (nof, high_score))
