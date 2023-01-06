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
dataset = load_breast_cancer()
df12 = pd.DataFrame(dataset.data, columns=dataset.feature_names)

#train the model on normal data, and test on noisy test data
#create split before loop


mu,sigma = 0,10000 #mean and std
noise = np.random.randint(mu, sigma, [569,30]) #look at doc
df12_noise = df12 + noise

X, y = df12_noise, df1.target

knn_accuracy = []
perceptron_accuracy = []
preprocessing.normalize(X)


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
plt.show()