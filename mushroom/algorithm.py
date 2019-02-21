# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-17  20:52:38
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
rate_x, rate_y = 7, 1
test_rate = 0.4
att_type="num"
att_add="a"
# filename="data_after.csv"
# filename = "../dataFile/out.csv"
filename = "../dataFile/Ionosphere.csv"
savename="../dataFile/ceshi.csv"
pretreatmnet.PreTreatmnet(filename,True,savename)
unbalanceRate.SetDataUnbalanced(savename, test_rate, rate_x, rate_y, att_type,att_add)
train_filename="../dataFile/train.csv"
test_filename="../dataFile/test.csv"
train_data,train_target= shujuchuli.getdata(train_filename, att_add,att_type)
ts_data,ts_target= shujuchuli.getdata(test_filename, att_add,att_type)

# *******************************统计不平衡下的数量*******************************
shuju_1=sum(train_target)
print("种类1的训练样本:"+str(shuju_1))
print("种类2的训练样本:"+str(len(train_target)-shuju_1))

shuju_2=sum(ts_target)
print("种类1的测试样本:"+str(shuju_2))
print("种类2的测试样本:"+str(len(ts_target)-shuju_2))



# ******************************* 显示评价指标 *******************************
def showResult(result):
    acc= metrics.accuracy_score(ts_target, result)
    print("acc",acc)
    pre= metrics.accuracy_score(ts_target, result)
    print("pre", pre)
    rec=metrics.recall_score(ts_target, result)
    print("rec", rec)
    f1= metrics.f1_score(ts_target, result)
    print("f1:",f1)
    auc=metrics.roc_auc_score(ts_target, result)
    print("auc:", auc)


    print("\n")

# *******************************支持向量机 70/345*******************************
def svcFunc():
    clf=svm.SVC(probability=True)
    clf=clf.fit(train_data,train_target)
    result=clf.predict(ts_data)
    print("svm:")
    showResult(result)


#*******************************高斯朴素贝叶斯 77/345*******************************
def gsNB():
    gnb = GaussianNB()
    clf = gnb.fit(train_data, train_target)
    result = gnb.predict(ts_data)
    print("guassNB:")
    showResult(result)



#******************************* 决策树*******************************
def deTree():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    print("DecisionTree:")
    showResult(result)


#******************************* adaboost 的集成方法 *******************************
def adaBoost():
    clf = AdaBoostClassifier(n_estimators=1)
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    print("adaboost")
    showResult(result)


# *******************************随机森林的集成方法  n_estimators为5时，准确率最高 *******************************
def rdForest():
    clf = RandomForestClassifier(n_estimators=1)
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    print("RandomForest:")
    showResult(result)



if __name__ == '__main__':
    svcFunc()
    gsNB()
    deTree()
    adaBoost()
    rdForest()



#     print("acc", metrics.accuracy_score(ts_target, result))
#     print("pre", metrics.precision_score(ts_target, result))
#     print("rec", metrics.recall_score(ts_target, result))
#     target_names = ['class 0', 'class 1']
#     print("f1:", metrics.f1_score(ts_target, result))
#     print("auc:", metrics.roc_auc_score(ts_target, result))
#     print(classification_report(ts_target
# , result, target_names=target_names))

# predict_prob_y = clf.predict_proba(ts_data)
# new_result=[]
# for i  in predict_prob_y:
#     new_result.append(i[1])
# print("auc:",metrics.roc_auc_score(ts_target,np.array(new_result)))
