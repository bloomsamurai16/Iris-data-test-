#!usr/bin/python3

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
features=iris.data.T
#print(iris.keys())
#print(iris.feature_names)
#print(iris.DESCR) 
#print(iris.target)
#print(iris.feature_names)
#print(iris.data)
#print(features[0])
#print(features[1])
#print(features[2])
#print(features[3])
sepal_length=features[0]
sepal_width=features[1]
petal_length=features[2]
petal_width=features[3]

sepal_length_label=iris.feature_names[0]
sepal_width_label=iris.feature_names[1]
petal_length_label=iris.feature_names[2]
petal_width_label=iris.feature_names[3]

plt.scatter(sepal_width,sepal_length,c=iris.target)
plt.xlabel(sepal_width_label)
plt.ylabel(sepal_length_label)
plt.show()
