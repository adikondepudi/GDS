import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV as LogReg
import seaborn as sns


X, y = make_moons(n_samples=500, noise=0.2)

X1 = X[:,0] #x value
Y1 = X[:,0] #y value
X2 = X[:,0]*2 #x value
Y2 = X[:,1]*2 #y value  
X3 = X[:,0]**2 #x value
Y3 = X[:,1]**2 #y value


XY12= np.column_stack([X1,Y2])
XY13= np.column_stack([X1,Y3])
XY22= np.column_stack([X2,Y2])
XY23= np.column_stack([X2,Y3])
XY32= np.column_stack([X3,Y2])
XY33= np.column_stack([X3,Y3])



X_train, X_test, y_train, y_test = train_test_split(XY13, y, test_size=0.30)
clf_mod = LogReg()
clf_mod.fit(X_train,y_train)
print("Risk:", (1 - clf_mod.score(X_test, y_test)))

#c
from sklearn.inspection import DecisionBoundaryDisplay
disp = DecisionBoundaryDisplay.from_estimator(clf_mod, X_test, response_method="auto")
disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
plt.show()