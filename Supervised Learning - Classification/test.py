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

print(X.columns)