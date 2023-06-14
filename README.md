<!-- #region -->
# Facies Classification

This repository contains a set of scripts for geological data preprocessing, visualization and Machine Learning (ML) model training.

The toolkit is designed to process geological data such as logs from wells, and apply machine learning models to categorize subsurface lithologies. 

Here, we use a set of commonly used ML algorithms to predict lithology (rock types) from well log data. 

## Preprocessing and Visualization

The scripts begin by importing necessary libraries and datasets. The provided dataset is then preprocessed through data cleaning, feature extraction, and visualization.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('yourDataset.csv')

# Preprocess the data
dfTrain, dfTest = preProcessing(df)

# Apply labels
dfTrain = labelAuto(dfTrain)
dfTest = labelAuto(dfTest)

# Visualize the data
logVisual(dfTest,10, facies_colors, formations_kay)
```

## Machine Learning Implementation

The machine learning part includes training several types of models including Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, K-Nearest Neighbors and Support Vector Machine classifiers.

Firstly, the data is standardized using StandardScaler and then split into training and test datasets.

```python
arrTrain, dfTrainStd = stdDataset(dfTrain)
arrTest, dfTestStd = stdDataset(dfTest)

#Train test split dataset
X_train, X_test, y_train, y_test = train_test_split(dfTrainStd[['DTC', 'GR', 'MSFL', 'ILD', 'NPHI', 'RHOB']], dfTrain['FACIES'].astype(float), test_size=0.2, random_state=10, stratify=dfTrain['FACIES'])
```

Next, classifiers are initialized and trained on the preprocessed dataset. Evaluation metrics such as accuracy, precision, recall and F1 score are calculated.

```python
evaluationFacies(X_test, y_test)
```

This step is then repeated with test data to evaluate the performance of each classifier.

```python
evaluationFacies(dfTestStd, y_test2)
```

## Statistical Evaluation

Finally, we use the trained classifiers to make predictions on the testing dataset. 

```python
predRes_tree = tree_clf.predict(dfTestStd)
predRes_rnd = rnd_clf.predict(dfTestStd)
predRes_gbst = gbst_clf.predict(dfTestStd)
predRes_lgst = lgst_clf.predict(dfTestStd)
predRes_knn = knn_clf.predict(dfTestStd)
predRes_svm = svm_clf.predict(dfTestStd)
```

## Conclusion

This is a high-level overview of the script. To fully understand and utilize the capabilities of the toolkit, users are encouraged to dive deeper into the code, tweak parameters, and modify the script as needed.
<!-- #endregion -->

```python

```
