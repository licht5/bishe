# # -*- coding: utf-8 -*-
# """
# @file: Algrithom.py
# @author: tianfeihan
# @time: 2019-02-27  21:49:52
# @description:
# """
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# import xlwt,xlrd
# from general import shujuchuli
# from sklearn.metrics import classification_report
# from sklearn import metrics
# from sklearn import tree
# from sklearn import svm
# import general.pretreatmnet as  pretreatmnet
# import general.unbalanceRate as unbalanceRate
# import os
# from xlutils.copy import copy
# class Algrithom:
#
#     filename = "../dataFile/mushroom.csv"
#     savename = "../dataFile/ceshi.csv"
#     pretreatmnet.PreTreatmnet(filename, True, savename)
#     unbalanceRate.SetDataUnbalanced(savename, test_rate, rate_x, rate_y, att_type, att_add)
#     train_filename = "../dataFile/train.csv"
#     test_filename = "../dataFile/test.csv"
#     train_data, train_target = shujuchuli.getdata(train_filename, att_add, att_type)
#     ts_data, ts_target = shujuchuli.getdata(test_filename, att_add, att_type)
#     def __init__(self,rate_x,rate_y,test_rate,project_name,att_type,att_add):
#         self.rate_x=rate_x
#         self.rate_y=rate_y
#         self.test_rate=test_rate
#         self.project_name=project_name
#         self.att_type=att_type
#         self.att_add=att_add
#     def chushihua(self):
#         filename = filename = "../dataFile/" + self.project_name + ".csv"
#         savename = "../dataFile/ceshi.csv"
#         pretreatmnet.PreTreatmnet(filename, True, savename)
#         unbalanceRate.SetDataUnbalanced(savename, self.test_rate, self.rate_x, self.rate_y, self.att_type, self.att_add)
#         train_filename = "../dataFile/train.csv"
#         test_filename = "../dataFile/test.csv"
#         train_data, train_target = shujuchuli.getdata(train_filename, self.att_add, self.att_type)
#         ts_data, ts_target = shujuchuli.getdata(test_filename, self.att_add, self.att_type)
#
#     def showResult(result, i):
#         acc = metrics.accuracy_score(ts_target, result)
#         pre = metrics.precision_score(ts_target, result)
#         rec = metrics.recall_score(ts_target, result)
#         f1 = metrics.f1_score(ts_target, result)
#         auc = metrics.roc_auc_score(ts_target, result)
#         if os.path.exists(excel_filename):
#             excel_data = xlrd.open_workbook(excel_filename)
#             table = excel_data.sheet_by_name(alg_name[i])
#             rows = table.nrows
#             newWB = copy(excel_data)
#             sheet = newWB.get_sheet(i)
#             sheet.write(rows, 0, acc)
#             sheet.write(rows, 1, pre)
#             sheet.write(rows, 2, rec)
#             sheet.write(rows, 3, f1)
#             sheet.write(rows, 4, auc)
#             newWB.save(excel_filename)
#         else:
#             makeExists(acc, pre, rec, f1, auc)
#

a=[[]]*5
print(a)