# -*- coding: utf-8 -*-
"""
@file: globalVariable.py
@author: tianfeihan
@time: 2019-02-28  15:24:56
@description: 
"""
rate_x, rate_y = 1,6
test_rate = 0.4
project_name="mushroom"
att_type="num"
att_add="a"
filename="../dataFile/"+project_name+".csv"
savename="../dataFile/ceshi.csv"
train_filename = "../dataFile/train.csv"
test_filename = "../dataFile/test.csv"
excel_filename="../dataFile/"+project_name+".xls"
alg_name=["LR","KNN","svm","guassNB","DecisionTree","MLP","adaBoost","GBDT","rdForest"]
alg_num=len(alg_name)
evaluation=["acc","pre","rec","f1","auc"]
evaluation_num=len(evaluation)
algrithm=[[],[],[],[],[],[],[],[],[]]

