import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
import seaborn as sns

df1 = load_breast_cancer(as_frame=True)
X, y = df1.data, df1.target

knn_accuracy = []
perceptron_accuracy = []


X = preprocessing.normalize(X)
X = pd.DataFrame(X, columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',        'mean smoothness', 'mean compactness', 'mean concavity',        'mean concave points', 'mean symmetry', 'mean fractal dimension',        'radius error', 'texture error', 'perimeter error', 'area error',        'smoothness error', 'compactness error', 'concavity error',        'concave points error', 'symmetry error', 'fractal dimension error',        'worst radius', 'worst texture', 'worst perimeter', 'worst area',        'worst smoothness', 'worst compactness', 'worst concavity',        'worst concave points', 'worst symmetry', 'worst fractal dimension'])


#knn
for n in range(0,49):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = knn(3)
    model.fit(X_train,y_train)
    knn_accuracy.append(model.score(X_test,y_test))

#perceptron
for ln in range(0,49):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    p = Perceptron(penalty='l2', alpha=1e-6, tol=1e-10)
    p.fit(X_train, y_train)
    perceptron_accuracy.append(p.score(X_test,y_test))

d = {'knn_accuracy': knn_accuracy, 'perceptron_accuracy': perceptron_accuracy}
df = pd.DataFrame(data=d)

sns.boxplot(data=df)

#knn still seems to be better

sns.pairplot(data=X, vars=['mean radius', 'mean symmetry', 'mean concavity', 'mean fractal dimension'])



plt.show()

#the relationships between the data seem to be the same.
