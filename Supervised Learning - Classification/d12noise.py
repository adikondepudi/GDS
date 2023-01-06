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


mu,sigma = 0,10 #mean and std
noise = np.random.randint(mu, sigma, [569,30]) #look at doc
df12_noise = df12 + noise


print(df12.head())
print(df12_noise.head())