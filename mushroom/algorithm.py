# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-17  20:52:38
@description: 本项目主要算法部分
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
from xlutils.copy import copy
import general.globalVariable as gv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

def getData():
    train_filename = gv.train_filename
    test_filename = gv.test_filename
    train_data, train_target = shujuchuli.getdata(train_filename, gv.att_add, gv.att_type)
    ts_data, ts_target = shujuchuli.getdata(test_filename, gv.att_add, gv.att_type)
    if not gv.flag:
        shuju_1 = sum(train_target)
        print("种类1的训练样本:" + str(shuju_1))
        print("种类2的训练样本:" + str(len(train_target) - shuju_1))

        shuju_2 = sum(ts_target)
        print("种类1的测试样本:" + str(shuju_2))
        print("种类2的测试样本:" + str(len(ts_target) - shuju_2))
        gv.flag=True


    return train_data,ts_data,train_target,ts_target





# ******************************* execel *******************************
# excel_filename=gv.excel_filename
alg_name=gv.alg_name


# ******************************* 显示评价指标 *******************************
def showResult(result,i,ts_target):
    acc= metrics.accuracy_score(ts_target, result)
    pre= metrics.precision_score(ts_target, result)
    rec=metrics.recall_score(ts_target, result)
    f1= metrics.f1_score(ts_target, result)
    auc=metrics.roc_auc_score(ts_target, result)
    tem = []
    tem.append(acc)
    tem.append(pre)
    tem.append(rec)
    tem.append(f1)
    tem.append(auc)
    gv.algrithm[i].append(tem)

# ******************************* 逻辑斯特回归 *******************************
def LogistRe(train_data,ts_data, train_target,ts_target):
    clf = LogisticRegression(random_state=0)
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result, 0, ts_target)


def KNNClass(train_data,ts_data, train_target,ts_target):
    clf = KNeighborsClassifier(n_neighbors=2)
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result, 1, ts_target)



# *******************************支持向量机 70/345*******************************
def svcFunc(train_data,ts_data, train_target,ts_target):
    clf=svm.SVC(probability=True)
    clf=clf.fit(train_data,train_target)
    result=clf.predict(ts_data)
    showResult(result,2,ts_target)


#*******************************高斯朴素贝叶斯 77/345*******************************
def gsNB(train_data,ts_data, train_target,ts_target):
    gnb = GaussianNB()
    clf = gnb.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,3,ts_target)



#******************************* 决策树*******************************
def deTree(train_data,ts_data, train_target,ts_target):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,4,ts_target)

#******************************* 神经网络  *******************************
def MLPClass(train_data,ts_data, train_target,ts_target):
    clf = MLPClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,5,ts_target)

#******************************* adaboost 的集成方法 *******************************
def adaBoost(train_data,ts_data, train_target,ts_target):
    clf = AdaBoostClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,6,ts_target)

#******************************* GBDT 的集成方法 *******************************
def GBDT(train_data,ts_data, train_target,ts_target):
    clf = GradientBoostingClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    showResult(result,7,ts_target)

# *******************************随机森林的集成方法  n_estimators为5时，准确率最高 *******************************
def rdForest(train_data,ts_data, train_target,ts_target):
    clf = RandomForestClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,8,ts_target)

def  totalAlgrithon():
    print("==================="+str(gv.count)+"次算法 =====================")
    gv.count=gv.count+1
    train_data,ts_data, train_target,ts_target=getData()
    LogistRe(train_data, ts_data, train_target, ts_target)
    KNNClass(train_data, ts_data, train_target, ts_target)
    svcFunc(train_data,ts_data, train_target,ts_target)
    gsNB(train_data,ts_data, train_target,ts_target)
    deTree(train_data,ts_data, train_target,ts_target)
    MLPClass(train_data, ts_data, train_target, ts_target)
    adaBoost(train_data,ts_data, train_target,ts_target)
    GBDT(train_data, ts_data, train_target, ts_target)
    rdForest(train_data,ts_data, train_target,ts_target)
