#!usr/bin/python3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()
features=iris.data.T
print("DESCRIPTION: ",iris.DESCR) 
print("\nkeys: ",iris.keys())
print("\nfeature names: ",iris["feature_names"])
print("\niris data: \n",iris.data)
print("\ntarget names: ",iris.target_names)
print("\ntarget:\n",iris.target)
print("\nsepal length:\n",features[0])
print("\nsepal width:\n",features[1])
print("\npetal length:\n",features[2])
print("\npetal width:\n ",features[3])
