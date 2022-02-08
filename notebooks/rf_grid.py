# %%
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from drowsiness_detection.data import get_train_test_splits
from drowsiness_detection.helpers import binarize

# %%

# %%
threshold = 7

# load train and test data
train, test = get_train_test_splits()

# X still contains NaNs
train = np.nan_to_num(train, nan=-1)
test = np.nan_to_num(test, nan=-1)

# split in full data for CV
full = np.concatenate([train, test])
X = full[:, :-1]
y = full[:, -1]
# binarize y to represent not drowsy vs drowsy
y = binarize(y, threshold)

# %%
model = RandomForestClassifier()
# define grid search
param_grid = {'n_estimators': [10, 100, 1000], 'max_depth': [2, 5, 10, None], 'min_samples_leaf': [1, 10, 100, 1000], "max_features": ['sqrt', 'log2'],
              'criterion': ['gini', 'entropy']}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X, y)
rf_grid_result = grid_result
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# %%
with open("rf_result.pkl", "wb") as fp:
    pickle.dump(file=fp, obj=rf_grid_result)
# %%
