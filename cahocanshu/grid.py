# -*- coding: utf-8 -*-
"""
@file: grid.py
@author: tianfeihan
@time: 2019-03-13  21:08:22
@description: 
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV  # 导入网格搜索方法的包
import numpy as np
# Each datapoint is a 8x8 image of a digit. 每一个数据点都是 8x8 的像素点
digits = datasets.load_digits()
# print(digits.data.shape)  # (1797, 64)
# print(digits.data[:5, :])

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(digits.images[3])
# plt.show()
# n_samples = len(digits.images)
# print(n_samples)  # 1797
# X = digits.data
# y = digits.target
filename="../canshuData/breast_mass.csv"
data=np.loadtxt(filename,delimiter=",")

X=data[:,0:-1]
y=data[:,-1]
# print(y[:5])  # [0 1 2 3 4] 对应的数字

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

param_grid = [{
    'kernel': ['rbf'],
    'gamma': [1e-3, 1e-4],
    'C': [1, 10, 100, 1000]
}, {
    'kernel': ['linear'],
    'C': [1, 10, 100, 1000]
}]
scores = ['roc_auc']
for score in scores:
    print('score %s' % score)
    print('------')
    clf = GridSearchCV(SVC(), param_grid, cv=5, scoring='%s' % score)
    clf.fit(X_train, y_train)
    print('best params is :')
    print(clf.best_params_)
    print('grid score')
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%.3f (+-/%0.03f) for %r' % (mean, std * 2, params))

    print('-----')
    print('classification report')
    y_true, y_pred = y_test, clf.predict(X_test)
    report = classification_report(y_true, y_pred)
    print(report)
