
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


# Spliting the dataset into training subset (70%) and testinf subset (%30)
X_train, X_test, Y_train, Y_test = skl_ms.train_test_split(
    X, Y, test_size=0.3, random_state=0)

# initiating the score list
score_list = [0.0]
selected_feature_masks = [['']]
for n in range(1, len(X.columns)):
    # constructing the regression model
    model = skl_lm.LinearRegression()
    # initiate the RFE model
    rfe = skl_fs.RFE(model, n_features_to_select=n)
    # finding the most relevant features based on recursively fitting the "model" object passed in the previous step; and removing the non-selected features from X_train
    X_train_rfe = rfe.fit_transform(X_train, Y_train)
    # removing the non-selected features from X_test
    X_test_rfe = rfe.transform(X_test)
    # fitting the regression model only with the selected features
    model.fit(X_train_rfe, Y_train)
    # scoring the model with the test data
    score = model.score(X_test_rfe, Y_test)
    #  storing the score value
    score_list.append(score)
    # storing the feature mask
    selected_feature_masks.append(rfe.support_)

# retrieving the name of features
features = np.array(X.columns)
score_list = np.array(score_list)
# finding the index of the maximum score -> finding the feature mask used by RFE for the maximum score -> masking the features list with the corresponding mask
SelFeat = features[selected_feature_masks[score_list.argmax()]]
# print the list of selected features
print(SelFeat)
