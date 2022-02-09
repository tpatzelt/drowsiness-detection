import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from drowsiness_detection.data import get_data_not_splitted

X, y = get_data_not_splitted()

model = RandomForestClassifier()

# define grid search
param_grid = {'n_estimators': [10, 100, 1000], 'max_depth': [2, 5, 10, None],
              'min_samples_leaf': [1, 10, 100, 1000], "max_features": ['sqrt', 'log2'],
              'criterion': ['gini', 'entropy']}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv,
                           scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X, y)
rf_grid_result = grid_result

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

with open("rf_result.pkl", "wb") as fp:
    pickle.dump(file=fp, obj=rf_grid_result)
