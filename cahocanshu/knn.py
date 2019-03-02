# -*- coding: utf-8 -*-
"""
@file: knn.py
@author: tianfeihan
@time: 2019-03-01  19:47:29
@description: 
"""

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
import numpy as np
from xlutils.copy import copy
import general.globalVariable as gv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
# filename="../dataFile/mushroom.csv"
import cahocanshu.bianliang as bl
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
def ceping(filename,att):
    project_data=np.loadtxt(filename,delimiter=",")
    if (att=="a"):
        project_X=project_data[:,:-1]
        project_y=project_data[:,-1]
    else:
        project_X = project_data[:, 1:]
        project_y = project_data[:, 0]
    print("总数目："+str(len(project_y))+"\n为1数量："+str(len(project_y)-project_y.sum()))

    k_range = range(1, 70,2)
    # learning_rate=[0.0001,0.001,0.005,0.1,0.2,0.3,0.4,0.5,1,1.5,2]
    k_loss = []
    k_auc = []
    for k in k_range:  # 对参数进行控制，选择参数表现好的，可视化展示
        knn = AdaBoostClassifier(n_estimators=k)
        auc = cross_val_score(knn, project_X, project_y, cv=10, scoring='roc_auc')  # for classification   精度
        loss = -cross_val_score(knn, project_X, project_y, cv=10,
                                scoring='neg_mean_squared_error')  # for regression    损失函数
        k_auc.append(auc.mean())  # 计算均值得分
        k_loss.append(loss.mean())
    return k_range,k_auc

# knn = KNeighborsClassifier(n_neighbors=5)
# score = cross_val_score(knn,mushroom_X,mushroom_y,cv=5,scoring='roc_auc')
# print(score)
# print(score.mean())



def huitu(i,k_range,k_auc):

    plt.plot(k_range, k_auc)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Cross-validates roc_auc")
    plt.title(bl.projicet_name)
    for a, b in zip(k_range, k_auc):
        plt.text(a, b + 0.001, '%.4f' % b, ha='center', va='bottom', fontsize=9)
    plt.savefig('../picture/saunfa/'+i+"_"+bl.projicet_name+'_estimator.png')
    plt.show()



if __name__ == '__main__':
    # for i in range(len(bl.projicet)):
    #     bl.filename="../canshuData/"+bl.projicet[i]+".csv"
    #     k_range, k_auc=ceping(bl.filename,bl.att[i])
    #     huitu(i,k_range, k_auc)
    saunfa="adaboost"
    bl.filename = "../canshuData/" + bl.projicet_name + ".csv"
    k_range, k_auc=ceping(bl.filename,"b")
    huitu(saunfa,k_range, k_auc)




