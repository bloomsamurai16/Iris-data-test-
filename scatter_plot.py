#!usr/bin/python3

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()
features=iris.data.T
sepal_length=features[0]
sepal_width=features[1]
petal_length=features[2]
petal_width=features[3]

sepal_length_label=iris.feature_names[0]
sepal_width_label=iris.feature_names[1]
petal_length_label=iris.feature_names[2]
petal_width_label=iris.feature_names[3]

X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=0)

plt.scatter(sepal_width,sepal_length,c=iris.target)
plt.xlabel(sepal_width_label)
plt.ylabel(sepal_length_label)
plt.show()
