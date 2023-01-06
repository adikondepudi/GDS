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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
clf = LogReg()
clf.fit(X_train,y_train)
# clf.score(X_test, y_test)

#a
# ypred = clf.predict(X_test)
# from sklearn import metrics
# print("confusion matrix")
# print(metrics.confusion_matrix(y_test,ypred)) 

# print("Risk:", (1 - clf.score(X_test, y_test)))

#b
#No, the points are in a curve and there isnt a single line that divides the points well.

#c
#UNCOMMENT THIS TO VIEW UNMODIFIED DATA POINTS
# from sklearn.inspection import DecisionBoundaryDisplay
# disp = DecisionBoundaryDisplay.from_estimator(clf, X_test, response_method="auto")
# disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")

#Commented out code was used to determine the best performing polynomial x^a + bx + c, where X is the data.
# Nested for loops were used to go through values 0-4 for a, b, and c.
# Dataframe was sorted and then outputted to a CSV file (the sort wasn't really working properly, so I sorted with excel)
# The best performing polynomial is coded below and a visual represetnation of the datapoints is shown below. 

# model_score = []
# i_value = []
# j_value = []
# k_value = []

# for i in range (0,5):
#     for j in range (0,5):
#         for k in range (0,5):
#             score = 0
#             for l in range (0,5):
#                 X1 = X**i + X*j + k
#                 X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.30)
#                 clf = LogReg()
#                 clf.fit(X_train,y_train)
#                 score += clf.score(X_test, y_test)
#             i_value.append(i)
#             j_value.append(j)
#             k_value.append(k)
#             model_score.append(score/5)

# d = {'model_score': model_score, 'i_value': i_value, 'j_value': j_value, 'k_value': k_value}
# df = pd.DataFrame(data=d)
# df.sort_values(by='model_score')
# df.to_csv("./out.csv")

X1 = X[:,0]
X2 = X[:,0]*2
X3 = X[:,1]*2
X4 = X[:,0]**2
X5 = X[:,1]**2

X13= np.column_stack([X4,X5])

# X1 = X1.tolist()
# X2 = X2.tolist()
# X3 = X3.tolist()
# X4 = X4.tolist()
# X5 = X5.tolist()

# d = {'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5}
# df = pd.DataFrame(data=d)
# df.to_csv("./please work.csv")

#make new dataframe with new features
# dX = pd.DataFrame(np.reshape(X1, (1,len(X1))))
# dX1 = {'X1': X1}
# dX2 = {'X2': X2}
# dX3 = {'X3': X3}
# dX4 = {'X4': X4}
# dX5 = {'X5': X5}
# df_merged = dX.append(dX1, dX2)
# df_merged = dX.append(dX3, dX4)
# df_merged = dX.append(dX5)
# df_merged.to_csv("./shut the fuck up.csv")



# X_best = X**3 + X*4 + 1
print("MOFO")
X1 = np.reshape(X1,(-1,1))
print("FUCK YOU")
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.30)
print("GO FUCK YOURSELF")
clf_mod = LogReg()
clf_mod.fit(X_train,y_train)
print("Risk:", (1 - clf_mod.score(X_test, y_test)))

#c
from sklearn.inspection import DecisionBoundaryDisplay
disp = DecisionBoundaryDisplay.from_estimator(clf, X_test, response_method="auto")
disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
plt.show()


# print(clf.n_features_in_)
# print(clf.coef_.shape)
# plt.stem(clf.coef_[0,:])
# plt.show()


#basically k fold cross validation, dviding data into difefrent sections and each cycle use different parts of the data to be the validation, switching up as validation data and others used as traiing data, if u do all those combos, you have a bunch of estimates of risk and if u take 