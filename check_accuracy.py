#!usr/bin/python3

from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

classifier=KNeighborsClassifier()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)

print(accuracy_score(y_test,predictions))
