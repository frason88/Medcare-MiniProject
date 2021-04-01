import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle


pd.set_option('max_columns',35)

df= pd.read_csv('cancer.csv')
print(df.head())
print(df.shape)

df1= df.drop(['Unnamed: 32', 'id'], axis=1)
df1.head()


diagnosis= {'M':0, 'B':1}
df1['diagnosis']= df1['diagnosis'].map(diagnosis)



X= df1.drop('diagnosis', axis= 1)
y= df1.diagnosis


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 10, stratify= y)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 110, num = 11)]
#max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

#Create the random grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



# Use the random grid to search for best hyperparameters

# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid,
                               n_iter = 100, cv = 3, verbose=2, random_state = 10, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_


# Manually provide the best parameters to model for training
model_1 = RandomForestClassifier(**{'n_estimators': 800,
 'min_samples_split': 2,
 'min_samples_leaf': 1,
 'max_features': 'auto',
 'max_depth': 77,
 'bootstrap': False})

result_1= model_1.fit(X_train, y_train)


pred_1 = result_1.predict(X_test)

from sklearn.metrics import accuracy_score, mean_squared_error
accuracy_score(y_test, pred_1)

print(np.sqrt(mean_squared_error(y_test, pred_1)))


# open a file, where you ant to store the data
file = open('Cancer_model.pkl', 'wb')


# dump information to that file
pickle.dump(model_1, file)








