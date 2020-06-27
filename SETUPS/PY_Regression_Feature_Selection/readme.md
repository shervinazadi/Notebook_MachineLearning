# Feature Selection

Feature selection in machine learning is the process of selecting relevant (differentiating and independent) features of a dataset for the creation of a model. There are three main approaches:

### 1. Filter Method

- Independant of the regression model
- mainly based on the correlation of the features (variables)

- In the example we utilize [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) as the measure for filtering the features. There are other measures that can be utilized in filtering methods such as [Pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) or [Relief](<https://en.wikipedia.org/wiki/Relief_(feature_selection)>)

### 2. Wrapper Method

trains a predictive model to evaluate different subsets of features.

- more specific selection with regards to the modelling technic
- higher risk of overfitting in case of limited input data
- In the examples we utilize [Backward Elimination](https://en.wikipedia.org/wiki/Stepwise_regression) and [RFE](https://link.springer.com/content/pdf/10.1023/A:1012487302797.pdf).

There are other approaches such as [Forward Selection, Bidirectional Elimination](https://en.wikipedia.org/wiki/Stepwise_regression)

### 3. Embedded Method

combining the advantages of previous approaches

---

For further reading you can refer to [Feature selection - Wikipedia](https://en.wikipedia.org/wiki/Feature_selection)

Libraries:[Plotly](https://plotly.com/python),[NumPy](https://numpy.org/), [Statsmodels](https://www.statsmodels.org/stable/index.html), [scikit-learn](https://scikit-learn.org/stable/index.html)
