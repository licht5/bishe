# -*- coding: utf-8 -*-
"""
@file: shuliang.py
@author: tianfeihan
@time: 2019-03-02  16:14:40
@description: 
"""
from sklearn.model_selection import learning_curve   #可视化学习的整个过程
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xlwt,xlrd
from general import shujuchuli
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn import svm
import general.pretreatmnet as  pretreatmnet
import general.unbalanceRate as unbalanceRate
import os
from xlutils.copy import copy
import general.globalVariable as gv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

filename="../dataFile/train.csv"
data=np.loadtxt(filename,delimiter=",")

X=data[:,1:]
y=data[:,0]


def ada(k):
    train_sizes, train_loss, test_loss = learning_curve(AdaBoostClassifier(n_estimators=k), X, y, cv=10,
                                                        scoring='neg_mean_squared_error',
                                                        train_sizes=[0.1, 0.25, 0.5, 0.75, 1])  # 记录的点是学习过程中的10%，25%等等的点
    return train_sizes, train_loss, test_loss
def knn(k):
    train_sizes, train_loss, test_loss = learning_curve(KNeighborsClassifier(n_estimators=k), X, y, cv=10,
                                                        scoring='neg_mean_squared_error',train_sizes=[0.1, 0.25, 0.5, 0.75, 1])


def huitu(train_sizes, train_loss, test_loss ,i):
    # 可视化展示
    train_loss_mean = -1 * np.mean(train_loss, axis=1)
    test_loss_mean = -1 * np.mean(test_loss, axis=1)
    plt.subplot(1, 2, i)
    plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='train')
    plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='cross_validation')
    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")

if __name__ == '__main__':
    train_sizes, train_loss, test_loss=ada(3)
    huitu(train_sizes, train_loss, test_loss ,1)
    train_sizes, train_loss, test_loss = ada(15)
    huitu(train_sizes, train_loss, test_loss, 2)
    plt.show()
    # train_sizes, train_loss, test_loss=knn(3)
    # huitu(train_sizes, train_loss, test_loss ,1)
    # train_sizes, train_loss, test_loss = ada(15)
    # huitu(train_sizes, train_loss, test_loss, 2)
