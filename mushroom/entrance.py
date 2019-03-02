# -*- coding: utf-8 -*-
"""
@file: entrance.py
@author: tianfeihan
@time: 2019-02-28  15:12:46
@description: 本项目程序入口
"""
import os
import numpy
import xlrd
import xlwt
from xlutils.copy import copy
import mushroom.algorithm
import general.globalVariable as gv
from general import pretreatmnet, unbalanceRate
from keshihua.xiangxing_auc import DrwBox
from keshihua.zhexiantu import DrwZhexian


def func1():
    if gv.rate_y+gv.rate_x==2 and os.path.exists(gv.excel_filename):
        os.remove(gv.excel_filename)
    elif gv.rate_y+gv.rate_x<2:
        print("********************** wrong：不平衡率设置不对 ********************")
    else:
        pass
    for i in range(10):

        unbalanceRate.SetDataUnbalanced(gv.savename, gv.test_rate, gv.rate_x, gv.rate_y, gv.att_type, gv.att_add)
        mushroom.algorithm.totalAlgrithon()
        if (i == 9):
            gv.flag = False



def write_excel(data):
    print("========== run to write_excel====== ")
    data_tem = []
    for i in range(gv.alg_num):
        tem = numpy.array(data[i])
        tem_mean = tem.mean(axis=0).tolist()
        data_tem.append(tem_mean)
    if os.path.exists(gv.excel_filename):
        # excel_data = xlrd.open_workbook(gv.excel_filename)
        for inj in range(gv.alg_num):
            excel_data = xlrd.open_workbook(gv.excel_filename)
            table = excel_data.sheet_by_name(gv.alg_name[inj])
            rows = table.nrows
            newWB = copy(excel_data)
            sheet = newWB.get_sheet(inj)
            for j in range(gv.evaluation_num):
                sheet.write(rows, j, data_tem[inj][j])
            newWB.save(gv.excel_filename)
    else:
        workbook = xlwt.Workbook()
        for index in range(gv.alg_num):
            table = workbook.add_sheet(gv.alg_name[index])
            for e_num in range(gv.evaluation_num):
                table.write(0, e_num, gv.evaluation[e_num])
            for data_index in range(gv.evaluation_num):
                table.write(1,data_index,data_tem[index][data_index])
        workbook.save(gv.excel_filename)

if __name__ == '__main__':
    # for i in  range(len(gv.project)):
    #     gv.project_name=gv.project[i]
    #     gv.att_add=gv.att[i]

    #     DrwBox()
    #     DrwZhexian()
    pretreatmnet.PreTreatmnet(gv.filename, True, gv.savename)
    func1()
    write_excel(gv.algrithm)






    # for i in range(1,15):
    #     gv.rate_x=i
    #     pretreatmnet.PreTreatmnet(gv.filename, True, gv.savename)
    #     func1()
    #     write_excel(gv.algrithm)






    # print(os.path.exists(gv.excel_filename))



