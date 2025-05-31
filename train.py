import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import os
import pickle
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction=mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

os.makedirs('models', exist_ok=True)
with open('models/week2.pkl', 'wb') as f:
    pickle.dump(mod_dt, f)
print('Model saved')
