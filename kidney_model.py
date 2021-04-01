
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('max_columns',30)

df= pd.read_csv('kidney_disease.csv')
print(df.head(5))

print(df.shape)


df.isnull().sum().sort_values(ascending=False)


df.classification.value_counts()


classification= {'ckd\t':1, 'ckd':1, 'notckd':0}
df['classification']= df['classification'].map(classification)

df.classification.value_counts()

df1= df.drop(['id','rbc','rc','wc','pot','sod','pcv'], axis=1)

df1= df1.dropna()
df1.shape

df1.head()

row_rep = {'yes':1,'no':0,'normal':1,'abnormal':0,'present':1,'notpresent':0,'good':1,'poor':0,'\tno':0,'\tyes':1,' yes':1}
df1 = df1.replace(row_rep)
df1.head()

X= df1.drop('classification', axis= 1)
y= df1.classification


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 20, stratify= y)


#Random Forest


from sklearn.ensemble import RandomForestClassifier


model_1 = RandomForestClassifier()
model_1.fit(X_train, y_train)
pred_1 = model_1.predict(X_test)

from sklearn.metrics import accuracy_score, mean_squared_error
print(accuracy_score(y_test, pred_1))




from sklearn.model_selection import RandomizedSearchCV

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
model_12 = RandomForestClassifier(**{'n_estimators': 600,
 'min_samples_split': 10,
 'min_samples_leaf': 2,
 'max_features': 'auto',
 'max_depth': 88,
 'bootstrap': False})

model_12= model_12.fit(X_train, y_train)

pred_12 = model_12.predict(X_test)

from sklearn.metrics import accuracy_score, mean_squared_error
print(accuracy_score(y_test, pred_12))



model_12.feature_importances_



import pickle

# open a file, where you ant to store the data
file = open('kidney_model.pkl', 'wb')

# dump information to that file
pickle.dump(model_12, file)




















































